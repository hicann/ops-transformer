/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "op_common/op_host/util/math_util.h"
#include "op_host_csv_case_loader.h"
#include "tiling_case_executor.h"

#include "../../../op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.h"
#include "../../../op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling.h"
#include "../../../op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_layout.h"

namespace {
using QkvTiling = optiling::QkvRmsNormRopeCacheWithKScale::QkvRmsNormRopeCacheWithKScaleBaseTiling;
using CompileInfo = optiling::QkvRmsNormRopeCacheWithKScaleCompileInfo;
using TensorDesc = gert::TilingContextPara::TensorDescription;
using TilingData = optiling::QkvRmsNormRopeCacheWithKScaleTilingData;

constexpr uint64_t TEST_OP_WORKSPACE_SIZE = 4096U;
constexpr uint64_t RESERVED_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t TEST_LAYOUT_NTD = 0U;
constexpr uint64_t TEST_LAYOUT_TND = 1U;
constexpr int64_t EXPECT_UNSET = -1;
constexpr float DEFAULT_EPSILON = 1e-6F;
constexpr uint64_t TEST_UB_SIZE = 262144U;
constexpr uint64_t TEST_L1_SIZE = 524288U;
constexpr uint64_t TEST_L0C_SIZE = 131072U;
constexpr uint64_t MROPE_MX_VECTOR_TILING_KEY = 302059776ULL;

int64_t ReadExpected(const csv_map &csvMap, const std::string &key)
{
    const std::string value = ReadMap(csvMap, key);
    return value.empty() ? EXPECT_UNSET : std::stoll(value);
}

bool IsNullValue(const std::string &value)
{
    return value.empty() || value == "<null>";
}

std::string DecodeString(const std::string &value)
{
    return value == "<empty>" ? std::string() : value;
}

std::vector<int64_t> ParseIntList(const std::string &value)
{
    if (IsNullValue(value) || value == "<empty>") {
        return {};
    }
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), '|', ' ');
    return GetShapeArr(normalized);
}

uint64_t CalcQkPreprocessNzBytes(uint64_t rowCount)
{
    const uint64_t rowStride = Ops::Base::CeilAlign(rowCount - 1, QkvTiling::QK_PREPROCESS_UB_NZ_STRIDE_ALIGN) + 1;
    const uint64_t blockCount = (QkvTiling::QK_PREPROCESS_NZ_D_BLOCKS - 1) * rowStride + rowCount;
    return blockCount * QkvTiling::QK_PREPROCESS_BLOCK_BYTES;
}

struct ExpectedTiling {
    int64_t tilingKey = EXPECT_UNSET;
    int64_t blockNum = EXPECT_UNSET;
    int64_t workspaceSize = EXPECT_UNSET;
    int64_t tilingDataZero = EXPECT_UNSET;
    int64_t tokenTile = EXPECT_UNSET;
    int64_t tokenTilePerAiv = EXPECT_UNSET;
    int64_t rowTile = EXPECT_UNSET;
    int64_t rowTileAligned = EXPECT_UNSET;
    int64_t coreTokenTile = EXPECT_UNSET;
    int64_t coreGroupNum = EXPECT_UNSET;
    int64_t kvStrideBlock = EXPECT_UNSET;
    int64_t kvStrideHead = EXPECT_UNSET;
    int64_t kvStrideToken = EXPECT_UNSET;
    int64_t kScaleStrideBlock = EXPECT_UNSET;
    int64_t kScaleStrideHead = EXPECT_UNSET;
    int64_t kScaleStrideToken = EXPECT_UNSET;
};

struct QkvTilingCase : public HostUtParamBase {
    TensorDesc qkv = TD_DEFAULT;
    TensorDesc qGamma = TD_DEFAULT;
    TensorDesc kGamma = TD_DEFAULT;
    TensorDesc cosSin = TD_DEFAULT;
    TensorDesc slotMapping = TD_DEFAULT;
    TensorDesc kCache = TD_DEFAULT;
    TensorDesc vCache = TD_DEFAULT;
    TensorDesc kScaleCache = TD_DEFAULT;
    TensorDesc queryStartLoc = TD_DEFAULT;
    TensorDesc seqLens = TD_DEFAULT;
    TensorDesc rotation = TD_DEFAULT;
    TensorDesc vScale = TD_DEFAULT;
    TensorDesc mropePosition = TD_DEFAULT;

    TensorDesc qOut = TD_DEFAULT;
    TensorDesc qScale = TD_DEFAULT;

    std::string headNums;
    std::string layoutQkv;
    std::string layoutQOut;
    std::string epsilon;
    std::string mropeSection;
    std::string qQuantMode;
    std::string kQuantMode;
    std::string qOutDtypeAttr;
    uint32_t aicNum = 32U;
    uint32_t aivNum = aicNum * QkvTiling::AIV_PER_AIC;
    uint64_t ubSize = TEST_UB_SIZE;
    uint64_t l1Size = TEST_L1_SIZE;
    uint64_t l0cSize = TEST_L0C_SIZE;
    uint64_t opWorkspaceSize = TEST_OP_WORKSPACE_SIZE;
    std::string socVersion = "Ascend950";
    ExpectedTiling expected;

    QkvTilingCase(const csv_map &csvMap)
        : HostUtParamBase(csvMap)
    {
        inputInstance.emplace_back(GetTensorGE(csvMap, "qkv_shape", "qkv_dtype", "qkv_format", qkv));
        inputInstance.emplace_back(GetTensorGE(csvMap, "qGamma_shape", "qGamma_dtype", "qGamma_format", qGamma));
        inputInstance.emplace_back(GetTensorGE(csvMap, "kGamma_shape", "kGamma_dtype", "kGamma_format", kGamma));
        inputInstance.emplace_back(GetTensorGE(csvMap, "cosSin_shape", "cosSin_dtype", "cosSin_format", cosSin));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "slotMapping_shape", "slotMapping_dtype", "slotMapping_format", slotMapping));
        inputInstance.emplace_back(GetTensorGE(csvMap, "kCache_shape", "kCache_dtype", "kCache_format", kCache));
        inputInstance.emplace_back(GetTensorGE(csvMap, "vCache_shape", "vCache_dtype", "vCache_format", vCache));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "kScaleCache_shape", "kScaleCache_dtype", "kScaleCache_format", kScaleCache));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "queryStartLoc_shape", "queryStartLoc_dtype", "queryStartLoc_format", queryStartLoc));
        inputInstance.emplace_back(GetTensorGE(csvMap, "seqLens_shape", "seqLens_dtype", "seqLens_format", seqLens));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "rotation_shape", "rotation_dtype", "rotation_format", rotation));
        inputInstance.emplace_back(GetTensorGE(csvMap, "vScale_shape", "vScale_dtype", "vScale_format", vScale));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "mropePosition_shape", "mropePosition_dtype", "mropePosition_format", mropePosition));
        ApplyStrideFromCsv(csvMap, "kCache_stride", kCache);
        ApplyStrideFromCsv(csvMap, "vCache_stride", vCache);
        ApplyStrideFromCsv(csvMap, "kScaleCache_stride", kScaleCache);

        outputInstance.emplace_back(GetTensorGE(csvMap, "qOut_shape", "qOut_dtype", "qOut_format", qOut));
        outputInstance.emplace_back(GetTensorGE(csvMap, "qScale_shape", "qScale_dtype", "qScale_format", qScale));
        outputInstance.emplace_back(inputInstance[5]);
        outputInstance.emplace_back(inputInstance[6]);
        outputInstance.emplace_back(inputInstance[7]);

        headNums = ReadMap(csvMap, "head_nums");
        layoutQkv = ReadMap(csvMap, "layout_qkv");
        layoutQOut = ReadMap(csvMap, "layout_q_out");
        epsilon = ReadMap(csvMap, "epsilon");
        mropeSection = ReadMap(csvMap, "mrope_section");
        qQuantMode = ReadMap(csvMap, "q_quant_mode");
        kQuantMode = ReadMap(csvMap, "k_quant_mode");
        qOutDtypeAttr = ReadMap(csvMap, "q_out_dtype_attr");
        aicNum = static_cast<uint32_t>(std::stoull(ReadMap(csvMap, "aic_num", "32")));
        const std::string aivNumValue = ReadMap(csvMap, "aiv_num");
        aivNum = IsNullValue(aivNumValue) ? aicNum * QkvTiling::AIV_PER_AIC :
                                            static_cast<uint32_t>(std::stoull(aivNumValue));
        ubSize = std::stoull(ReadMap(csvMap, "ub_size", std::to_string(TEST_UB_SIZE)));
        l1Size = std::stoull(ReadMap(csvMap, "l1_size", std::to_string(TEST_L1_SIZE)));
        l0cSize = std::stoull(ReadMap(csvMap, "l0c_size", std::to_string(TEST_L0C_SIZE)));
        opWorkspaceSize = std::stoull(ReadMap(csvMap, "op_workspace_size", std::to_string(TEST_OP_WORKSPACE_SIZE)));
        socVersion = ReadMap(csvMap, "soc_version", "Ascend950");

        expected.tilingKey = ReadExpected(csvMap, "expectTilingKey");
        expected.blockNum = ReadExpected(csvMap, "expectBlockNum");
        expected.workspaceSize = ReadExpected(csvMap, "expectWorkspaceSize");
        expected.tilingDataZero = ReadExpected(csvMap, "expectTilingDataZero");
        expected.tokenTile = ReadExpected(csvMap, "expectTokenTile");
        expected.tokenTilePerAiv = ReadExpected(csvMap, "expectTokenTilePerAiv");
        expected.rowTile = ReadExpected(csvMap, "expectRowTile");
        expected.rowTileAligned = ReadExpected(csvMap, "expectRowTileAligned");
        expected.coreTokenTile = ReadExpected(csvMap, "expectCoreTokenTile");
        expected.coreGroupNum = ReadExpected(csvMap, "expectCoreGroupNum");
        expected.kvStrideBlock = ReadExpected(csvMap, "expectKvStrideBlock");
        expected.kvStrideHead = ReadExpected(csvMap, "expectKvStrideHead");
        expected.kvStrideToken = ReadExpected(csvMap, "expectKvStrideToken");
        expected.kScaleStrideBlock = ReadExpected(csvMap, "expectKScaleStrideBlock");
        expected.kScaleStrideHead = ReadExpected(csvMap, "expectKScaleStrideHead");
        expected.kScaleStrideToken = ReadExpected(csvMap, "expectKScaleStrideToken");
    }
};

std::vector<gert::TilingContextPara::OpAttr> BuildAttrs(const QkvTilingCase &testCase)
{
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    if (IsNullValue(testCase.headNums)) {
        return attrs;
    }

    attrs.emplace_back("head_nums",
                       Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(ParseIntList(testCase.headNums)));
    if (IsNullValue(testCase.layoutQkv)) {
        return attrs;
    }
    attrs.emplace_back("layout_qkv",
                       Ops::Transformer::AnyValue::CreateFrom<std::string>(DecodeString(testCase.layoutQkv)));
    if (IsNullValue(testCase.layoutQOut)) {
        return attrs;
    }
    attrs.emplace_back("layout_q_out",
                       Ops::Transformer::AnyValue::CreateFrom<std::string>(DecodeString(testCase.layoutQOut)));
    if (IsNullValue(testCase.epsilon)) {
        return attrs;
    }
    attrs.emplace_back("epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(std::stof(testCase.epsilon)));
    if (IsNullValue(testCase.mropeSection)) {
        return attrs;
    }
    attrs.emplace_back("mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(
                                            ParseIntList(testCase.mropeSection)));
    if (IsNullValue(testCase.qQuantMode)) {
        return attrs;
    }
    attrs.emplace_back("q_quant_mode",
                       Ops::Transformer::AnyValue::CreateFrom<std::string>(DecodeString(testCase.qQuantMode)));
    if (IsNullValue(testCase.qOutDtypeAttr) && IsNullValue(testCase.kQuantMode)) {
        return attrs;
    }
    attrs.emplace_back("q_out_dtype",
                       Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(Str2DTypeGE(
                           IsNullValue(testCase.qOutDtypeAttr) ? "FLOAT8_E4M3FN" : testCase.qOutDtypeAttr))));
    if (IsNullValue(testCase.kQuantMode)) {
        return attrs;
    }
    attrs.emplace_back("k_quant_mode",
                       Ops::Transformer::AnyValue::CreateFrom<std::string>(DecodeString(testCase.kQuantMode)));
    return attrs;
}

CompileInfo BuildCompileInfo(const QkvTilingCase &testCase)
{
    CompileInfo compileInfo;
    compileInfo.aicNum = testCase.aicNum;
    compileInfo.aivNum = testCase.aivNum;
    compileInfo.ubSize = testCase.ubSize;
    compileInfo.l1Size = testCase.l1Size;
    compileInfo.l0cSize = testCase.l0cSize;
    compileInfo.opWorkspaceSize = testCase.opWorkspaceSize;
    return compileInfo;
}

gert::TilingContextPara BuildTilingContext(const QkvTilingCase &testCase, CompileInfo &compileInfo)
{
    return gert::TilingContextPara(
        "QkvRmsNormRopeCacheWithKScale",
        {testCase.qkv, testCase.qGamma, testCase.kGamma, testCase.cosSin, testCase.slotMapping, testCase.kCache,
         testCase.vCache, testCase.kScaleCache, testCase.queryStartLoc, testCase.seqLens, testCase.rotation,
         testCase.vScale, testCase.mropePosition},
        {testCase.qOut, testCase.qScale, testCase.kCache, testCase.vCache, testCase.kScaleCache}, BuildAttrs(testCase),
        testCase.inputInstance, testCase.outputInstance, &compileInfo, testCase.socVersion, testCase.aicNum,
        compileInfo.ubSize);
}

bool RunTiling(const QkvTilingCase &testCase, TilingInfo &tilingInfo)
{
    auto compileInfo = BuildCompileInfo(testCase);
    auto tilingContext = BuildTilingContext(testCase, compileInfo);
    return ExecuteTiling(tilingContext, tilingInfo);
}

uint64_t ParseLayout(const std::string &value, uint64_t defaultLayout)
{
    const std::string layout = DecodeString(value);
    if (layout == "NTD") {
        return TEST_LAYOUT_NTD;
    }
    if (layout == "TND") {
        return TEST_LAYOUT_TND;
    }
    return defaultLayout;
}

struct DerivedInput {
    uint64_t totalTokens = 0U;
    uint64_t batch = 0U;
    uint64_t numQHeads = 0U;
    uint64_t numKHeads = 0U;
    uint64_t numVHeads = 0U;
    uint64_t headDim = 0U;
    uint64_t blockSize = 0U;
    float epsilon = DEFAULT_EPSILON;
};

DerivedInput DeriveInput(const QkvTilingCase &testCase)
{
    DerivedInput input;
    const auto headNums = ParseIntList(testCase.headNums);
    input.numQHeads = static_cast<uint64_t>(headNums[0]);
    input.numKHeads = static_cast<uint64_t>(headNums[1]);
    input.numVHeads = static_cast<uint64_t>(headNums[2]);

    const uint64_t layoutQkv = ParseLayout(testCase.layoutQkv, TEST_LAYOUT_TND);
    const auto &qkvShape = testCase.qkv.shape_.GetStorageShape();
    input.totalTokens = static_cast<uint64_t>(qkvShape.GetDim(layoutQkv == TEST_LAYOUT_TND ? 0 : 1));
    input.headDim = static_cast<uint64_t>(qkvShape.GetDim(2));
    // M-RoPE deliberately leaves the RoPE sequence tensors absent.
    const bool isMrope = ParseIntList(testCase.mropeSection).size() == 3U;
    input.batch = isMrope ? 0U : static_cast<uint64_t>(testCase.queryStartLoc.shape_.GetStorageShape().GetDim(0) - 1);
    input.blockSize = static_cast<uint64_t>(testCase.kCache.shape_.GetStorageShape().GetDim(2));
    input.epsilon = IsNullValue(testCase.epsilon) ? DEFAULT_EPSILON : std::stof(testCase.epsilon);
    return input;
}

void ExpectU64(const std::string &caseName, const char *field, int64_t expected, uint64_t actual)
{
    if (expected >= 0) {
        EXPECT_EQ(actual, static_cast<uint64_t>(expected)) << "caseName=" << caseName << ", field=" << field;
    }
}

void ExpectTilingResult(const QkvTilingCase &testCase, const TilingInfo &tilingInfo)
{
    const auto &expected = testCase.expected;
    if (expected.tilingKey >= 0) {
        EXPECT_EQ(tilingInfo.tilingKey, expected.tilingKey) << "caseName=" << testCase.case_name;
    }
    ExpectU64(testCase.case_name, "blockNum", expected.blockNum, tilingInfo.blockNum);
    if (expected.workspaceSize >= 0) {
        ASSERT_FALSE(tilingInfo.workspaceSizes.empty()) << "caseName=" << testCase.case_name;
        EXPECT_EQ(tilingInfo.workspaceSizes[0], expected.workspaceSize) << "caseName=" << testCase.case_name;
    }

    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(TilingData)) << "caseName=" << testCase.case_name;
    ASSERT_NE(tilingInfo.tilingData, nullptr) << "caseName=" << testCase.case_name;
    if (expected.tilingDataZero == 1) {
        const auto *begin = tilingInfo.tilingData.get();
        EXPECT_TRUE(std::all_of(begin, begin + sizeof(TilingData), [](uint8_t value) { return value == 0U; }))
            << "caseName=" << testCase.case_name;
        return;
    }

    const auto &tiling = *reinterpret_cast<const TilingData *>(tilingInfo.tilingData.get());
    const auto input = DeriveInput(testCase);
    EXPECT_EQ(tiling.totalTokens, input.totalTokens) << "caseName=" << testCase.case_name;
    EXPECT_EQ(tiling.batch, input.batch) << "caseName=" << testCase.case_name;
    EXPECT_EQ(tiling.qHeadNum, input.numQHeads) << "caseName=" << testCase.case_name;
    EXPECT_EQ(tiling.kvHeadNum, input.numKHeads) << "caseName=" << testCase.case_name;
    EXPECT_EQ(input.numVHeads, input.numKHeads) << "caseName=" << testCase.case_name;
    EXPECT_EQ(tiling.headDim, input.headDim) << "caseName=" << testCase.case_name;
    EXPECT_EQ(tiling.blockSize, input.blockSize) << "caseName=" << testCase.case_name;

    const auto mropeSection = ParseIntList(testCase.mropeSection);
    if (mropeSection.size() == 3U) {
        EXPECT_EQ(tiling.mropeSectionH, static_cast<uint64_t>(mropeSection[1])) << "caseName=" << testCase.case_name;
        EXPECT_EQ(tiling.mropeSectionW, static_cast<uint64_t>(mropeSection[2])) << "caseName=" << testCase.case_name;
    }

    const bool isMropeMx = DecodeString(testCase.qQuantMode) == "Mx" && DecodeString(testCase.kQuantMode) == "Mx";
    const uint64_t tokenTilePerAiv = Ops::Base::CeilDiv(tiling.tokenTile, QkvTiling::AIV_PER_AIC);
    const uint64_t qkHeadNum = tiling.qHeadNum + tiling.kvHeadNum;
    const uint64_t rowTile = tiling.tokenTile * qkHeadNum;
    ExpectU64(testCase.case_name, "tokenTile", expected.tokenTile, tiling.tokenTile);
    ExpectU64(testCase.case_name, "tokenTilePerAiv", expected.tokenTilePerAiv, tokenTilePerAiv);
    ExpectU64(testCase.case_name, "rowTile", expected.rowTile, rowTile);
    ExpectU64(testCase.case_name, "rowTileAligned", expected.rowTileAligned, Ops::Base::CeilAlign(rowTile, 16UL));
    ExpectU64(testCase.case_name, "coreTokenTile", expected.coreTokenTile, tiling.coreTokenTile);
    ExpectU64(testCase.case_name, "coreGroupNum", expected.coreGroupNum, tiling.coreGroupNum);

    if (isMropeMx) {
        EXPECT_EQ(tiling.batch, 0U);
        EXPECT_EQ(tiling.coreTokenTile, 0U);
        EXPECT_EQ(tiling.coreGroupNum, 0U);
        EXPECT_EQ(tilingInfo.blockNum, std::min(input.totalTokens, static_cast<uint64_t>(testCase.aivNum)));
        ::QkvRmsNormRopeCacheWithKScale::MropeMxUbLayout layout{};
        const bool hasQTail = (tiling.qHeadNum & 15U) == 8U;
        const bool layoutOk = hasQTail ?
                                  ::QkvRmsNormRopeCacheWithKScale::TryMakeMropeMxQGlobalTileWaveUbLayout(
                                      tiling.tokenTile, tiling.qHeadNum, tiling.kvHeadNum, tiling.headDim, layout) :
                                  ::QkvRmsNormRopeCacheWithKScale::TryMakeMropeMxUbLayout(
                                      tiling.tokenTile, tiling.qHeadNum, tiling.kvHeadNum, tiling.headDim, layout);
        ASSERT_TRUE(layoutOk);
        EXPECT_LE(layout.totalBytes, testCase.ubSize);
        EXPECT_EQ(layout.totalBytes % 32U, 0U);
        const uint64_t kScaleTokenStrideBytes =
            Ops::Base::CeilAlign(tiling.kvHeadNum * ::QkvRmsNormRopeCacheWithKScale::MROPE_MX_SCALE_COUNT_D128,
                                 ::QkvRmsNormRopeCacheWithKScale::MROPE_MX_UB_ALIGN_BYTES);
        EXPECT_EQ(layout.kScaleOffsetBytes % ::QkvRmsNormRopeCacheWithKScale::MROPE_MX_UB_ALIGN_BYTES, 0U);
        EXPECT_EQ(kScaleTokenStrideBytes % ::QkvRmsNormRopeCacheWithKScale::MROPE_MX_UB_ALIGN_BYTES, 0U);
        EXPECT_LE(layout.kScaleOffsetBytes - layout.kDataOffsetBytes + tiling.tokenTile * kScaleTokenStrideBytes,
                  layout.kSlotBytes);
    } else {
        const uint64_t qPreprocessRows = tokenTilePerAiv * tiling.qHeadNum;
        const uint64_t kPreprocessRows = tokenTilePerAiv * tiling.kvHeadNum;
        EXPECT_LE(CalcQkPreprocessNzBytes(qPreprocessRows) + CalcQkPreprocessNzBytes(kPreprocessRows),
                  QkvTiling::QK_PREPROCESS_UB_BYTES);
        EXPECT_LE(tokenTilePerAiv * qkHeadNum, QkvTiling::QK_OUTPUT_ROWS_PER_AIV);
        const uint64_t qkvInputRowsPerAiv =
            mropeSection.size() == 3U ? QkvTiling::MROPE_COMPACT_INPUT_ROWS_PER_AIV : QkvTiling::QKV_INPUT_ROWS_PER_AIV;
        EXPECT_LE(tokenTilePerAiv * (qkHeadNum + tiling.kvHeadNum), qkvInputRowsPerAiv);
        EXPECT_LE(tokenTilePerAiv * tiling.kvHeadNum, QkvTiling::V_OUTPUT_ROWS_PER_AIV);
    }

    ExpectU64(testCase.case_name, "kvCacheStrideBlock", expected.kvStrideBlock, tiling.kvCacheStrideBlock);
    ExpectU64(testCase.case_name, "kvCacheStrideHead", expected.kvStrideHead, tiling.kvCacheStrideHead);
    ExpectU64(testCase.case_name, "kvCacheStrideToken", expected.kvStrideToken, tiling.kvCacheStrideToken);
    ExpectU64(testCase.case_name, "kScaleCacheStrideBlock", expected.kScaleStrideBlock, tiling.kScaleCacheStrideBlock);
    ExpectU64(testCase.case_name, "kScaleCacheStrideHead", expected.kScaleStrideHead, tiling.kScaleCacheStrideHead);
    ExpectU64(testCase.case_name, "kScaleCacheStrideToken", expected.kScaleStrideToken, tiling.kScaleCacheStrideToken);
    EXPECT_FLOAT_EQ(tiling.epsilon, input.epsilon) << "caseName=" << testCase.case_name;
}

const std::vector<QkvTilingCase> &GetTilingCases()
{
    static const auto cases = GetCasesFromCsv<QkvTilingCase>(ReplaceFileExtension2Csv(__FILE__));
    return cases;
}

const QkvTilingCase *FindTilingCase(const std::string &caseName)
{
    const auto &cases = GetTilingCases();
    const auto it = std::find_if(cases.begin(), cases.end(),
                                 [&](const QkvTilingCase &testCase) { return testCase.case_name == caseName; });
    return it == cases.end() ? nullptr : &*it;
}

bool MatchesExpectedResult(const QkvTilingCase &testCase, bool success, const TilingInfo &tilingInfo)
{
    if (success != (testCase.expectResult == ge::GRAPH_SUCCESS)) {
        return false;
    }
    if (!success) {
        return true;
    }
    if (tilingInfo.tilingDataSize < sizeof(TilingData) || tilingInfo.tilingData == nullptr) {
        return false;
    }
    const auto *tiling = reinterpret_cast<const TilingData *>(tilingInfo.tilingData.get());
    const auto matches = [](int64_t expected, uint64_t actual) {
        return expected < 0 || static_cast<uint64_t>(expected) == actual;
    };
    return matches(testCase.expected.tilingKey, static_cast<uint64_t>(tilingInfo.tilingKey)) &&
           matches(testCase.expected.blockNum, tilingInfo.blockNum) &&
           matches(testCase.expected.tokenTile, tiling->tokenTile) &&
           matches(testCase.expected.coreTokenTile, tiling->coreTokenTile) &&
           matches(testCase.expected.coreGroupNum, tiling->coreGroupNum);
}

struct ConcurrentTilingResult {
    bool ok = true;
    uint64_t failedCase = 0U;
    uint32_t failedIteration = 0U;
    bool success = false;
    uint64_t tokenTile = 0U;
    uint64_t coreTokenTile = 0U;
    uint64_t coreGroupNum = 0U;
    uint64_t blockNum = 0U;
};

class QkvRmsNormRopeCacheWithKScaleCsvTiling : public testing::TestWithParam<QkvTilingCase> {};

} // namespace

TEST_P(QkvRmsNormRopeCacheWithKScaleCsvTiling, RunsCase)
{
    const auto &testCase = GetParam();
    TilingInfo tilingInfo;
    const bool success = RunTiling(testCase, tilingInfo);
    const auto status = success ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    EXPECT_EQ(status, testCase.expectResult) << "caseName=" << testCase.case_name;
    if (status == ge::GRAPH_SUCCESS && testCase.expectResult == ge::GRAPH_SUCCESS) {
        ExpectTilingResult(testCase, tilingInfo);
    }
}

INSTANTIATE_TEST_SUITE_P(CsvCases, QkvRmsNormRopeCacheWithKScaleCsvTiling, testing::ValuesIn(GetTilingCases()),
                         PrintCaseInfoString<QkvTilingCase>);

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, MropeMxHeadRoutesSelectSingleVectorKey)
{
    const std::array<std::pair<const char *, uint64_t>, 8> routes = {{
        {"mrope_mx_h8_t129_current_route", MROPE_MX_VECTOR_TILING_KEY},
        {"mrope_mx_h16_t129_current_route", MROPE_MX_VECTOR_TILING_KEY},
        {"mrope_mx_h24_t129_current_route", MROPE_MX_VECTOR_TILING_KEY},
        {"mrope_mx_h32_t129_current_route", MROPE_MX_VECTOR_TILING_KEY},
        {"mrope_mx_h40_t129_current_route", MROPE_MX_VECTOR_TILING_KEY},
        {"mrope_mx_h48_t129_current_route", MROPE_MX_VECTOR_TILING_KEY},
        {"mrope_mx_h56_t129_current_route", MROPE_MX_VECTOR_TILING_KEY},
        {"mrope_mx_h64_t129_current_route", MROPE_MX_VECTOR_TILING_KEY},
    }};
    for (const auto &[caseName, expectedKey] : routes) {
        const auto *testCase = FindTilingCase(caseName);
        ASSERT_NE(testCase, nullptr) << "caseName=" << caseName;
        TilingInfo tilingInfo;
        ASSERT_TRUE(RunTiling(*testCase, tilingInfo)) << "caseName=" << caseName;
        EXPECT_EQ(tilingInfo.tilingKey, expectedKey) << "caseName=" << caseName;
    }
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, MropeMxQTailWaveFallsBackToSingleTokenUnderUbPressure)
{
    const auto *baseCase = FindTilingCase("mrope_mx_h8_t129_current_route");
    ASSERT_NE(baseCase, nullptr);
    ::QkvRmsNormRopeCacheWithKScale::MropeMxUbLayout oneToken{};
    ::QkvRmsNormRopeCacheWithKScale::MropeMxUbLayout twoTokens{};
    ASSERT_TRUE(::QkvRmsNormRopeCacheWithKScale::TryMakeMropeMxQGlobalTileWaveUbLayout(1U, 8U, 1U, 128U, oneToken));
    ASSERT_TRUE(::QkvRmsNormRopeCacheWithKScale::TryMakeMropeMxQGlobalTileWaveUbLayout(2U, 8U, 1U, 128U, twoTokens));
    ASSERT_LT(oneToken.totalBytes, twoTokens.totalBytes);

    QkvTilingCase lowUbCase = *baseCase;
    lowUbCase.ubSize = oneToken.totalBytes;
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunTiling(lowUbCase, tilingInfo));
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    const auto *tiling = reinterpret_cast<const TilingData *>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingInfo.tilingKey, MROPE_MX_VECTOR_TILING_KEY);
    EXPECT_EQ(tiling->tokenTile, 1U);
}

TEST(QkvRmsNormRopeCacheWithKScaleTilingDataAbi, Remains144BytesWithFrozenOffsets)
{
    EXPECT_EQ(sizeof(TilingData), 144U);
    EXPECT_EQ(offsetof(TilingData, tokenTile), 112U);
    EXPECT_EQ(offsetof(TilingData, epsilon), 120U);
    EXPECT_EQ(offsetof(TilingData, mropeSectionH), 128U);
}

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RealTilingIsStableAcrossThreads)
{
    constexpr uint32_t THREAD_COUNT = 8U;
    constexpr uint32_t ITERATIONS = 64U;
    const std::array<std::string, 6> caseNames = {
        "q16_k2_v2_basic",  "t512_q64_k8",          "t1024_q16_k2", "mrope_mx_h8_t129_current_route",
        "too_many_q_heads", "unsupported_head_dim",
    };
    std::array<const QkvTilingCase *, caseNames.size()> cases = {};
    for (size_t i = 0; i < caseNames.size(); ++i) {
        cases[i] = FindTilingCase(caseNames[i]);
        ASSERT_NE(cases[i], nullptr) << "caseName=" << caseNames[i];
    }

    std::atomic<bool> start(false);
    std::array<ConcurrentTilingResult, THREAD_COUNT> results;
    std::array<std::thread, THREAD_COUNT> threads;
    for (uint32_t threadIdx = 0U; threadIdx < THREAD_COUNT; ++threadIdx) {
        threads[threadIdx] = std::thread([threadIdx, &cases, &results, &start]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint32_t iter = 0U; iter < ITERATIONS; ++iter) {
                const uint64_t caseIdx = (static_cast<uint64_t>(threadIdx) + iter) % cases.size();
                const auto &testCase = *cases[caseIdx];
                TilingInfo tilingInfo;
                const bool success = RunTiling(testCase, tilingInfo);
                if (!MatchesExpectedResult(testCase, success, tilingInfo)) {
                    results[threadIdx].ok = false;
                    results[threadIdx].failedCase = caseIdx;
                    results[threadIdx].failedIteration = iter;
                    results[threadIdx].success = success;
                    results[threadIdx].blockNum = tilingInfo.blockNum;
                    if (success && tilingInfo.tilingDataSize >= sizeof(TilingData) &&
                        tilingInfo.tilingData != nullptr) {
                        const auto *tiling = reinterpret_cast<const TilingData *>(tilingInfo.tilingData.get());
                        results[threadIdx].tokenTile = tiling->tokenTile;
                        results[threadIdx].coreTokenTile = tiling->coreTokenTile;
                        results[threadIdx].coreGroupNum = tiling->coreGroupNum;
                    }
                    return;
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto &thread : threads) {
        thread.join();
    }

    for (uint32_t threadIdx = 0U; threadIdx < THREAD_COUNT; ++threadIdx) {
        EXPECT_TRUE(results[threadIdx].ok)
            << "threadIdx=" << threadIdx << " failedCase=" << results[threadIdx].failedCase
            << " failedIteration=" << results[threadIdx].failedIteration << " success=" << results[threadIdx].success
            << " blockNum=" << results[threadIdx].blockNum << " tokenTile=" << results[threadIdx].tokenTile
            << " coreTokenTile=" << results[threadIdx].coreTokenTile
            << " coreGroupNum=" << results[threadIdx].coreGroupNum;
    }
}
