/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_block_attn_res_update_tiling.cpp
 * \brief CSV-driven unit tests for BlockAttnResUpdate Ascend 950 tiling.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../op_host/op_tiling/arch35/block_attn_res_update_tiling.h"
#include "../../../op_kernel/arch35/block_attn_res_update_tiling_data.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "gmm_csv_ge_parse_utils.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {

constexpr const char *OP_NAME = "BlockAttnResUpdate";
constexpr const char *SOC_VERSION = "Ascend950";
constexpr const char *NPU_ARCH = "3510";
constexpr size_t MIN_INPUT_COLUMN_COUNT = 25UL;
constexpr size_t SUCCESS_COLUMN_COUNT = 36UL;
constexpr size_t INPUT_COUNT = 6UL;
constexpr size_t OUTPUT_COUNT = 2UL;
constexpr uint64_t VALID_AIC_NUM = 32UL;
constexpr uint64_t VALID_UB_SIZE = 245760UL;
constexpr uint64_t VALID_TILING_DATA_CAPACITY = 4096UL;

ge::DataType ParseDtype(const std::string &value)
{
    return ops::ut::ParseGeDtype(ops::ut::Trim(value));
}

gert::StorageShape ParseShape(const std::string &value)
{
    return ops::ut::MakeGertStorageShape(ops::ut::ParseDims(ops::ut::Trim(value)));
}

gert::TilingContextPara MakeValidTilingContextPara()
{
    return gert::TilingContextPara(OP_NAME,
                                   {
                                       {ParseShape("3:63"), ge::DT_FLOAT, ge::FORMAT_ND},
                                       {ParseShape("3:63"), ge::DT_BF16, ge::FORMAT_ND},
                                       {ParseShape("63"), ge::DT_FLOAT, ge::FORMAT_ND},
                                       {ParseShape("3:63"), ge::DT_FLOAT, ge::FORMAT_ND},
                                       {ParseShape("3"), ge::DT_FLOAT, ge::FORMAT_ND},
                                       {ParseShape("3"), ge::DT_FLOAT, ge::FORMAT_ND},
                                   },
                                   {
                                       {ParseShape("3:63"), ge::DT_FLOAT, ge::FORMAT_ND},
                                       {ParseShape("3:63"), ge::DT_BF16, ge::FORMAT_ND},
                                   },
                                   std::vector<gert::TilingContextPara::OpAttr>{}, nullptr, NPU_ARCH, VALID_AIC_NUM,
                                   VALID_UB_SIZE, VALID_TILING_DATA_CAPACITY);
}

struct BlockAttnResUpdateTilingCase {
    void Run() const
    {
        const uint32_t aicNum = aivNum / 2U;

        std::vector<gert::TilingContextPara::OpAttr> attrs;
        if (eps != "DEFAULT") {
            attrs.emplace_back("eps", Ops::Transformer::AnyValue::CreateFrom<float>(std::stof(eps)));
        }

        gert::TilingContextPara para(OP_NAME,
                                     {
                                         {ParseShape(partialShape), partialDtype, ge::FORMAT_ND},
                                         {ParseShape(deltaShape), deltaDtype, ge::FORMAT_ND},
                                         {ParseShape(pseudoQueryShape), pseudoQueryDtype, ge::FORMAT_ND},
                                         {ParseShape(numeratorShape), numeratorDtype, ge::FORMAT_ND},
                                         {ParseShape(logitMaxShape), logitMaxDtype, ge::FORMAT_ND},
                                         {ParseShape(expSumShape), expSumDtype, ge::FORMAT_ND},
                                     },
                                     {
                                         {ParseShape(partialRefShape), partialRefDtype, ge::FORMAT_ND},
                                         {ParseShape(hShape), hDtype, ge::FORMAT_ND},
                                     },
                                     attrs, nullptr, NPU_ARCH, aicNum, ubSize, tilingDataCapacity);

        TilingInfo actual;
        const bool actualResult = ExecuteTiling(para, actual);
        ASSERT_EQ(actualResult, expectSuccess) << "case=" << caseName;
        if (!expectSuccess) {
            return;
        }

        ASSERT_EQ(actual.blockNum, expectBlockDim) << "case=" << caseName;
        ASSERT_EQ(actual.tilingKey, expectTilingKey) << "case=" << caseName;
        ASSERT_EQ(actual.workspaceSizes.size(), 1UL) << "case=" << caseName;
        EXPECT_EQ(actual.workspaceSizes[0], expectWorkspace) << "case=" << caseName;
        ASSERT_EQ(actual.tilingDataSize, sizeof(BlockAttnResUpdateTilingData)) << "case=" << caseName;
        ASSERT_NE(actual.tilingData, nullptr) << "case=" << caseName;

        const auto *tilingData = reinterpret_cast<const BlockAttnResUpdateTilingData *>(actual.tilingData.get());
        EXPECT_EQ(tilingData->dSize, expectDSize) << "case=" << caseName;
        EXPECT_EQ(tilingData->tPerCore, expectTPerCore) << "case=" << caseName;
        EXPECT_EQ(tilingData->lastTPerCore, expectLastTPerCore) << "case=" << caseName;
        EXPECT_EQ(tilingData->tileT, expectTileT) << "case=" << caseName;
        EXPECT_EQ(tilingData->statsTStride, expectStatsTStride) << "case=" << caseName;
        EXPECT_FLOAT_EQ(tilingData->eps, expectEps) << "case=" << caseName;
        EXPECT_FLOAT_EQ(tilingData->invD, expectInvD) << "case=" << caseName;
        EXPECT_EQ(tilingData->usedCoreNum, expectUsedCoreNum) << "case=" << caseName;
    }

    std::string caseName;
    std::string prefix;
    uint32_t aivNum = 0U;
    uint64_t ubSize = 0UL;
    uint64_t tilingDataCapacity = 0UL;

    std::string partialShape;
    std::string deltaShape;
    std::string pseudoQueryShape;
    std::string numeratorShape;
    std::string logitMaxShape;
    std::string expSumShape;
    std::string partialRefShape;
    std::string hShape;

    ge::DataType partialDtype = ge::DT_UNDEFINED;
    ge::DataType deltaDtype = ge::DT_UNDEFINED;
    ge::DataType pseudoQueryDtype = ge::DT_UNDEFINED;
    ge::DataType numeratorDtype = ge::DT_UNDEFINED;
    ge::DataType logitMaxDtype = ge::DT_UNDEFINED;
    ge::DataType expSumDtype = ge::DT_UNDEFINED;
    ge::DataType partialRefDtype = ge::DT_UNDEFINED;
    ge::DataType hDtype = ge::DT_UNDEFINED;

    std::string eps;
    bool expectSuccess = false;
    uint64_t expectBlockDim = 0UL;
    uint64_t expectTilingKey = 0UL;
    int64_t expectWorkspace = 0;
    uint32_t expectDSize = 0U;
    uint32_t expectTPerCore = 0U;
    uint32_t expectLastTPerCore = 0U;
    uint32_t expectTileT = 0U;
    uint32_t expectStatsTStride = 0U;
    float expectEps = 0.0F;
    float expectInvD = 0.0F;
    uint16_t expectUsedCoreNum = 0U;
};

struct TilingCsvLoadResult {
    std::vector<BlockAttnResUpdateTilingCase> cases;
    std::vector<std::string> errors;
};

TilingCsvLoadResult LoadTilingCases()
{
    TilingCsvLoadResult result;
    const std::string csvPath = ops::ut::ResolveCsvPath("test_block_attn_res_update_tiling.csv",
                                                        "attention/block_attn_res_update/tests/ut/op_host", __FILE__);
    std::ifstream csvData(csvPath, std::ios::in);
    if (!csvData.is_open()) {
        result.errors.emplace_back("Cannot open tiling case file: " + csvPath);
        return result;
    }

    std::string line;
    size_t lineNo = 0UL;
    while (std::getline(csvData, line)) {
        ++lineNo;
        if (ops::ut::Trim(line).empty() || ops::ut::Trim(line).front() == '#') {
            continue;
        }

        std::vector<std::string> items;
        ops::ut::SplitStr2Vec(line, ",", items);
        if (items.empty() || ops::ut::Trim(items[0]) == "socVersion") {
            continue;
        }
        if (items.size() < MIN_INPUT_COLUMN_COUNT) {
            result.errors.emplace_back("Invalid tiling CSV column count at " + csvPath + ":" + std::to_string(lineNo));
            continue;
        }

        const std::string caseName = items.size() > 1UL ? ops::ut::Trim(items[1]) : "";
        try {
            size_t index = 0UL;
            const std::string socVersion = ops::ut::Trim(items[index++]);
            if (socVersion != SOC_VERSION) {
                continue;
            }

            BlockAttnResUpdateTilingCase testCase;
            testCase.caseName = ops::ut::Trim(items[index++]);
            const bool enable = ops::ut::ParseBool(items[index++]);
            if (!enable) {
                continue;
            }
            testCase.prefix = ops::ut::Trim(items[index++]);
            testCase.aivNum = static_cast<uint32_t>(std::stoul(ops::ut::Trim(items[index++])));
            testCase.ubSize = std::stoull(ops::ut::Trim(items[index++]));
            testCase.tilingDataCapacity = std::stoull(ops::ut::Trim(items[index++]));

            testCase.partialShape = ops::ut::Trim(items[index++]);
            testCase.deltaShape = ops::ut::Trim(items[index++]);
            testCase.pseudoQueryShape = ops::ut::Trim(items[index++]);
            testCase.numeratorShape = ops::ut::Trim(items[index++]);
            testCase.logitMaxShape = ops::ut::Trim(items[index++]);
            testCase.expSumShape = ops::ut::Trim(items[index++]);
            testCase.partialRefShape = ops::ut::Trim(items[index++]);
            testCase.hShape = ops::ut::Trim(items[index++]);

            testCase.partialDtype = ParseDtype(items[index++]);
            testCase.deltaDtype = ParseDtype(items[index++]);
            testCase.pseudoQueryDtype = ParseDtype(items[index++]);
            testCase.numeratorDtype = ParseDtype(items[index++]);
            testCase.logitMaxDtype = ParseDtype(items[index++]);
            testCase.expSumDtype = ParseDtype(items[index++]);
            testCase.partialRefDtype = ParseDtype(items[index++]);
            testCase.hDtype = ParseDtype(items[index++]);

            testCase.eps = ops::ut::Trim(items[index++]);
            testCase.expectSuccess = ops::ut::ParseBool(items[index++]);
            if (testCase.expectSuccess) {
                if (items.size() < SUCCESS_COLUMN_COUNT) {
                    throw std::runtime_error("successful case does not contain all expected tiling fields");
                }
                testCase.expectBlockDim = std::stoull(ops::ut::Trim(items[index++]));
                testCase.expectTilingKey = std::stoull(ops::ut::Trim(items[index++]));
                testCase.expectWorkspace = std::stoll(ops::ut::Trim(items[index++]));
                testCase.expectDSize = static_cast<uint32_t>(std::stoul(ops::ut::Trim(items[index++])));
                testCase.expectTPerCore = static_cast<uint32_t>(std::stoul(ops::ut::Trim(items[index++])));
                testCase.expectLastTPerCore = static_cast<uint32_t>(std::stoul(ops::ut::Trim(items[index++])));
                testCase.expectTileT = static_cast<uint32_t>(std::stoul(ops::ut::Trim(items[index++])));
                testCase.expectStatsTStride = static_cast<uint32_t>(std::stoul(ops::ut::Trim(items[index++])));
                testCase.expectEps = std::stof(ops::ut::Trim(items[index++]));
                testCase.expectInvD = std::stof(ops::ut::Trim(items[index++]));
                testCase.expectUsedCoreNum = static_cast<uint16_t>(std::stoul(ops::ut::Trim(items[index++])));
            }
            result.cases.emplace_back(std::move(testCase));
        } catch (const std::exception &error) {
            result.errors.emplace_back(ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error));
        }
    }
    if (result.cases.empty()) {
        result.errors.emplace_back("No enabled Ascend950 tiling cases were loaded from: " + csvPath);
    }
    return result;
}

const TilingCsvLoadResult &GetTilingCases()
{
    static const TilingCsvLoadResult result = LoadTilingCases();
    return result;
}

std::string MakeParamName(const testing::TestParamInfo<BlockAttnResUpdateTilingCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

class BlockAttnResUpdateTilingTest : public testing::TestWithParam<BlockAttnResUpdateTilingCase> {};

TEST(BlockAttnResUpdateTilingCsv, LoadsEnabledCases)
{
    const auto &result = GetTilingCases();
    for (const auto &error : result.errors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(result.cases.empty());
}

TEST(BlockAttnResUpdateTilingDirect, RejectsNullContexts)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl(OP_NAME);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    EXPECT_EQ(opImpl->tiling(nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(opImpl->tiling_parse(nullptr), ge::GRAPH_FAILED);
}

TEST(BlockAttnResUpdateTilingDirect, AcceptsMatchingNdPhysicalLayout)
{
    auto para = MakeValidTilingContextPara();
    TilingInfo actual;
    EXPECT_TRUE(ExecuteTiling(para, actual));
}

TEST(BlockAttnResUpdateTilingDirect, RejectsStorageShapeDifferentFromOrigin)
{
    for (size_t inputIndex = 0U; inputIndex < INPUT_COUNT; ++inputIndex) {
        auto para = MakeValidTilingContextPara();
        gert::Shape &storageShape = para.inputTensorDesc_[inputIndex].shape_.MutableStorageShape();
        storageShape.SetDim(0U, storageShape.GetDim(0U) + 1);

        TilingInfo actual;
        EXPECT_FALSE(ExecuteTiling(para, actual)) << "inputIndex=" << inputIndex;
    }

    for (size_t outputIndex = 0U; outputIndex < OUTPUT_COUNT; ++outputIndex) {
        auto para = MakeValidTilingContextPara();
        gert::Shape &storageShape = para.outputTensorDesc_[outputIndex].shape_.MutableStorageShape();
        storageShape.SetDim(0U, storageShape.GetDim(0U) + 1);

        TilingInfo actual;
        EXPECT_FALSE(ExecuteTiling(para, actual)) << "outputIndex=" << outputIndex;
    }
}

TEST(BlockAttnResUpdateTilingDirect, RejectsNonNdInputFormats)
{
    for (size_t inputIndex = 0U; inputIndex < INPUT_COUNT; ++inputIndex) {
        auto para = MakeValidTilingContextPara();
        para.inputTensorDesc_[inputIndex].format_ = ge::FORMAT_NCHW;

        TilingInfo actual;
        EXPECT_FALSE(ExecuteTiling(para, actual)) << "inputIndex=" << inputIndex;
    }
}

TEST_P(BlockAttnResUpdateTilingTest, CsvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(BLOCK_ATTN_RES_UPDATE_ASCEND950, BlockAttnResUpdateTilingTest,
                         testing::ValuesIn(GetTilingCases().cases), MakeParamName);

} // namespace
