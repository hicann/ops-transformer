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
 * \file test_grouped_weight_quant_arch35_tiling.cpp
 * \brief Unit tests for GroupedWeightQuantBatchMatmulTiling
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "test_grouped_matmul_utils.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

class GroupedWeightQuantBatchMatmulTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupedWeightQuantBatchMatmulTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupedWeightQuantBatchMatmulTiling TearDown" << std::endl; }
};

namespace {
using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;
using GMMWeightQuantTilingData = GroupedMatmulTilingData::GMMWeightQuantTilingData;

constexpr int64_t DEFAULT_DYNAMIC_SIZE = -1;

struct MxA8W4SmsTilingParam {
    size_t m = 64;
    size_t k = 128;
    size_t n = 256;
    size_t groupNum = 2;
    int64_t splitItem = 3;
    int64_t groupListType = 1;
    int64_t groupType = 0;
    bool transA = false;
    bool transB = true;
    bool hasBias = false;
    int64_t numX = 1;
    int64_t numWeight = DEFAULT_DYNAMIC_SIZE;
    int64_t numAntiquantScale = DEFAULT_DYNAMIC_SIZE;
    int64_t numBias = DEFAULT_DYNAMIC_SIZE;
    int64_t groupListSize = DEFAULT_DYNAMIC_SIZE;
    ge::DataType xDtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType weightDtype = ge::DT_FLOAT4_E2M1;
    ge::Format weightFormat = ge::FORMAT_FRACTAL_NZ;
    ge::DataType antiquantScaleDtype = ge::DT_FLOAT8_E8M0;
    std::vector<int64_t> antiquantScaleOriginShape = {};
    std::vector<int64_t> antiquantScaleStorageShape = {};
    std::vector<int64_t> perTokenScaleOriginShape = {};
    std::vector<int64_t> perTokenScaleStorageShape = {};
    std::vector<int64_t> biasOriginShape = {};
    std::vector<int64_t> biasStorageShape = {};
};

size_t GetConfiguredSize(int64_t configuredSize, size_t defaultSize)
{
    return configuredSize == DEFAULT_DYNAMIC_SIZE ? defaultSize : static_cast<size_t>(configuredSize);
}

size_t GetNumWeight(const MxA8W4SmsTilingParam &param) { return GetConfiguredSize(param.numWeight, param.groupNum); }

size_t GetNumAntiquantScale(const MxA8W4SmsTilingParam &param)
{
    return GetConfiguredSize(param.numAntiquantScale, GetNumWeight(param));
}

size_t GetNumBias(const MxA8W4SmsTilingParam &param)
{
    return GetConfiguredSize(param.numBias, param.hasBias ? GetNumWeight(param) : 0U);
}

size_t GetGroupListSize(const MxA8W4SmsTilingParam &param)
{
    return GetConfiguredSize(param.groupListSize, GetNumWeight(param));
}

optiling::GMMCompileInfo MakeArch35CompileInfo()
{
    return {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
}

std::vector<OpAttr> MakeMxA8W4SmsAttrs(const MxA8W4SmsTilingParam &param)
{
    return {
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.splitItem)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transB)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transA)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupType)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupListType)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    };
}

TensorDescription MakeTensorDesc(const std::vector<int64_t> &originShape, const std::vector<int64_t> &storageShape,
                                 ge::DataType dtype, ge::Format format)
{
    return {ops::ut::MakeGertStorageShape(originShape, storageShape), dtype, format};
}

TensorDescription MakeTensorDesc(const std::vector<int64_t> &shape, ge::DataType dtype, ge::Format format)
{
    return {ops::ut::MakeGertStorageShape(shape), dtype, format};
}

std::vector<int64_t> MakeDefaultWeightShape(const MxA8W4SmsTilingParam &param)
{
    return {static_cast<int64_t>(param.k / 32), static_cast<int64_t>(param.n / 16), 16, 32};
}

std::vector<int64_t> MakeDefaultAntiquantScaleShape(const MxA8W4SmsTilingParam &param)
{
    return {static_cast<int64_t>(param.n), static_cast<int64_t>(param.k / 64), 2};
}

std::vector<int64_t> MakeDefaultPerTokenScaleShape(const MxA8W4SmsTilingParam &param)
{
    return {static_cast<int64_t>(param.m), static_cast<int64_t>(param.k / 64), 2};
}

std::vector<TensorDescription> MakeMxA8W4SmsInputs(const MxA8W4SmsTilingParam &param)
{
    const size_t numX = GetConfiguredSize(param.numX, 1U);
    const size_t numWeight = GetNumWeight(param);
    const size_t numBias = GetNumBias(param);
    const size_t numAntiquantScale = GetNumAntiquantScale(param);
    std::vector<TensorDescription> inputs;
    inputs.reserve(numX + numWeight + numBias + numAntiquantScale + 2);

    for (size_t i = 0; i < numX; ++i) {
        inputs.emplace_back(MakeTensorDesc({static_cast<int64_t>(param.m), static_cast<int64_t>(param.k)}, param.xDtype,
                                           ge::FORMAT_ND));
    }

    std::vector<int64_t> weightShape = MakeDefaultWeightShape(param);
    for (size_t i = 0; i < numWeight; ++i) {
        inputs.emplace_back(MakeTensorDesc(weightShape, weightShape, param.weightDtype, param.weightFormat));
    }

    std::vector<int64_t> biasOriginShape =
        param.biasOriginShape.empty() ? std::vector<int64_t>{static_cast<int64_t>(param.n)} : param.biasOriginShape;
    std::vector<int64_t> biasStorageShape = param.biasStorageShape.empty() ? biasOriginShape : param.biasStorageShape;
    for (size_t i = 0; i < numBias; ++i) {
        inputs.emplace_back(MakeTensorDesc(biasOriginShape, biasStorageShape, ge::DT_BF16, ge::FORMAT_ND));
    }

    std::vector<int64_t> scaleOriginShape = param.antiquantScaleOriginShape.empty() ?
                                                MakeDefaultAntiquantScaleShape(param) :
                                                param.antiquantScaleOriginShape;
    std::vector<int64_t> scaleStorageShape =
        param.antiquantScaleStorageShape.empty() ? scaleOriginShape : param.antiquantScaleStorageShape;
    for (size_t i = 0; i < numAntiquantScale; ++i) {
        inputs.emplace_back(
            MakeTensorDesc(scaleOriginShape, scaleStorageShape, param.antiquantScaleDtype, ge::FORMAT_ND));
    }

    inputs.emplace_back(MakeTensorDesc({static_cast<int64_t>(GetGroupListSize(param))}, ge::DT_INT64, ge::FORMAT_ND));
    std::vector<int64_t> perTokenScaleOriginShape =
        param.perTokenScaleOriginShape.empty() ? MakeDefaultPerTokenScaleShape(param) : param.perTokenScaleOriginShape;
    std::vector<int64_t> perTokenScaleStorageShape =
        param.perTokenScaleStorageShape.empty() ? perTokenScaleOriginShape : param.perTokenScaleStorageShape;
    inputs.emplace_back(
        MakeTensorDesc(perTokenScaleOriginShape, perTokenScaleStorageShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND));
    return inputs;
}

std::vector<uint32_t> MakeMxA8W4SmsInputInstanceNums(const MxA8W4SmsTilingParam &param)
{
    return {static_cast<uint32_t>(GetConfiguredSize(param.numX, 1U)),
            static_cast<uint32_t>(GetNumWeight(param)),
            static_cast<uint32_t>(GetNumBias(param)),
            0U,
            0U,
            static_cast<uint32_t>(GetNumAntiquantScale(param)),
            0U,
            1U,
            1U};
}

gert::TilingContextPara MakeMxA8W4SmsContext(const MxA8W4SmsTilingParam &param, optiling::GMMCompileInfo *compileInfo)
{
    return gert::TilingContextPara(
        "GroupedMatmul", MakeMxA8W4SmsInputs(param),
        {MakeTensorDesc({static_cast<int64_t>(param.m), static_cast<int64_t>(param.n)}, ge::DT_BF16, ge::FORMAT_ND)},
        MakeMxA8W4SmsAttrs(param), MakeMxA8W4SmsInputInstanceNums(param), {1U}, compileInfo);
}

uint64_t GetTilingKeyIndex(uint64_t value, std::initializer_list<uint64_t> supportedValues)
{
    size_t index = 0;
    for (const auto supportedValue : supportedValues) {
        if (supportedValue == value) {
            return index;
        }
        ++index;
    }
    return 0;
}

uint64_t MakeExpectedMxA8W4TilingKey(bool isSingleMultiSingle)
{
    uint64_t tilingKey = 0;
    uint8_t shift = 0;
    auto append = [&tilingKey, &shift](uint64_t value, uint8_t bitWidth) {
        tilingKey |= value << shift;
        shift += bitWidth;
    };

    append(GetTilingKeyIndex(static_cast<uint64_t>(optiling::WeightFormat::FRACTAL_NZ), {0, 1}), 4);
    append(GetTilingKeyIndex(
               static_cast<uint64_t>(optiling::OptionInputSituation::ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST),
               {0, 1, 2, 3, 4, 5}),
           4);
    append(GetTilingKeyIndex(static_cast<uint64_t>(optiling::QuantType::NONE), {0, 1, 2, 3, 4}), 4);
    append(GetTilingKeyIndex(static_cast<uint64_t>(optiling::QuantType::MX), {0, 1, 2, 3, 4}), 4);
    append(GetTilingKeyIndex(1, {0, 1}), 2); // transB=true
    append(GetTilingKeyIndex(0, {0, 1}), 2); // transA=false
    append(GetTilingKeyIndex(static_cast<uint64_t>(optiling::Mte2Configuration::MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4),
                             {0, 1, 2, 3, 4, 5, 6, 15}),
           4);
    append(static_cast<uint64_t>(isSingleMultiSingle), 1);
    append(GetTilingKeyIndex(static_cast<uint64_t>(optiling::OptimizationAlgorithmSubCategory::N_FIRST_TAIL_RESPLIT),
                             {0, 1, 2, 3}),
           4);
    append(
        GetTilingKeyIndex(static_cast<uint64_t>(optiling::OptimizationAlgorithmCategory::VECTOR_ANTIQUANT), {0, 1, 2}),
        4);
    return tilingKey;
}

const GMMWeightQuantTilingData *GetGMMWeightQuantTilingData(const TilingInfo &tilingInfo)
{
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(GMMWeightQuantTilingData));
    return reinterpret_cast<const GMMWeightQuantTilingData *>(tilingInfo.tilingData.get());
}

bool RunMxA8W4SmsTiling(const MxA8W4SmsTilingParam &param, TilingInfo &tilingInfo)
{
    auto compileInfo = MakeArch35CompileInfo();
    auto tilingContextPara = MakeMxA8W4SmsContext(param, &compileInfo);
    return ExecuteTiling(tilingContextPara, tilingInfo);
}

void ExpectMxA8W4SmsTilingData(const TilingInfo &tilingInfo, const MxA8W4SmsTilingParam &param)
{
    const auto *tilingData = GetGMMWeightQuantTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupNum, static_cast<uint32_t>(GetNumWeight(param)));
    EXPECT_EQ(tilingData->gmmWeightQuantParam.kSize, static_cast<uint64_t>(param.k));
    EXPECT_EQ(tilingData->gmmWeightQuantParam.nSize, static_cast<uint64_t>(param.n));
    EXPECT_EQ(tilingData->gmmWeightQuantParam.singleX, 1);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.singleWeight, 0);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.singleY, 1);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupType, static_cast<int8_t>(param.groupType));
    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupListType, static_cast<uint8_t>(param.groupListType));
    EXPECT_EQ(tilingData->gmmWeightQuantParam.hasBias, static_cast<uint8_t>(GetNumBias(param) > 0 ? 1 : 0));
    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupSize, 32);
    for (size_t i = 0; i < GetNumWeight(param); ++i) {
        EXPECT_EQ(tilingData->gmmArray.mList[i], -1);
        EXPECT_EQ(tilingData->gmmArray.kList[i], static_cast<int32_t>(param.k));
        EXPECT_EQ(tilingData->gmmArray.nList[i], static_cast<int32_t>(param.n));
    }
}

void ExpectMxA8W4SmsTilingSuccess(const MxA8W4SmsTilingParam &param)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunMxA8W4SmsTiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(MakeExpectedMxA8W4TilingKey(true)));
    ExpectMxA8W4SmsTilingData(tilingInfo, param);
}

void ExpectMxA8W4SmsTilingFailure(const MxA8W4SmsTilingParam &param)
{
    TilingInfo tilingInfo;
    EXPECT_FALSE(RunMxA8W4SmsTiling(param, tilingInfo));
}

gert::TilingContextPara MakeMxA8W4SssContext(optiling::GMMCompileInfo *compileInfo)
{
    constexpr size_t m = 64;
    constexpr size_t k = 128;
    constexpr size_t n = 256;
    constexpr size_t groupNum = 2;
    return gert::TilingContextPara(
        "GroupedMatmul",
        {
            MakeTensorDesc({static_cast<int64_t>(m), static_cast<int64_t>(k)}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND),
            MakeTensorDesc(
                {static_cast<int64_t>(groupNum), static_cast<int64_t>(k / 32), static_cast<int64_t>(n / 16), 16, 32},
                {static_cast<int64_t>(groupNum), static_cast<int64_t>(k / 32), static_cast<int64_t>(n / 16), 16, 32},
                ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ),
            MakeTensorDesc({static_cast<int64_t>(groupNum), static_cast<int64_t>(n), static_cast<int64_t>(k / 64), 2},
                           ge::DT_FLOAT8_E8M0, ge::FORMAT_ND),
            MakeTensorDesc({static_cast<int64_t>(groupNum)}, ge::DT_INT64, ge::FORMAT_ND),
            MakeTensorDesc({static_cast<int64_t>(m), static_cast<int64_t>(k / 64), 2}, ge::DT_FLOAT8_E8M0,
                           ge::FORMAT_ND),
        },
        {MakeTensorDesc({static_cast<int64_t>(m), static_cast<int64_t>(n)}, ge::DT_BF16, ge::FORMAT_ND)},
        MakeMxA8W4SmsAttrs(MxA8W4SmsTilingParam{}), {1U, 1U, 0U, 0U, 0U, 1U, 0U, 1U, 1U}, {1U}, compileInfo);
}
} // namespace

// MxA8W4 NZ场景 - 单多单模式, antiquantScale为去掉g轴的3维shape
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_scale3d_no_bias)
{
    MxA8W4SmsTilingParam param;
    ExpectMxA8W4SmsTilingSuccess(param);
}

// MxA8W4 NZ - FLOAT4_E1M2 weight reuses the E2M1 tiling strategy and selects a different kernel decode path.
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_e1m2_sms_scale3d_no_bias)
{
    MxA8W4SmsTilingParam param;
    param.weightDtype = ge::DT_FLOAT4_E1M2;
    ExpectMxA8W4SmsTilingSuccess(param);
}

// MxA8W4 NZ场景 - 单多单模式, 带bias多tensor
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_with_bias)
{
    MxA8W4SmsTilingParam param;
    param.hasBias = true;
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunMxA8W4SmsTiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(MakeExpectedMxA8W4TilingKey(true)));
    ExpectMxA8W4SmsTilingData(tilingInfo, param);
}

// MxA8W4 NZ场景 - splitItem=2时y为单tensor
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_split_item2)
{
    MxA8W4SmsTilingParam param;
    param.splitItem = 2;
    ExpectMxA8W4SmsTilingSuccess(param);
}

// MxA8W4 NZ场景 - groupListType=0累计offset模式
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_group_list_type0)
{
    MxA8W4SmsTilingParam param;
    param.groupListType = 0;
    ExpectMxA8W4SmsTilingSuccess(param);
}

// MxA8W4 NZ场景 - N尾块重切分
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_n_tail_resplit)
{
    MxA8W4SmsTilingParam param;
    param.n = 384;
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunMxA8W4SmsTiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(MakeExpectedMxA8W4TilingKey(true)));
    ExpectMxA8W4SmsTilingData(tilingInfo, param);
    const auto *tilingData = GetGMMWeightQuantTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.mainBlockSize, optiling::BASIC_BLOCK_BASE_N);
    EXPECT_GE(tilingData->gmmWeightQuantParam.firstTailBlockSize, optiling::BASIC_BLOCK_BASE_N_MIN);
    EXPECT_LE(tilingData->gmmWeightQuantParam.firstTailBlockSize, optiling::BASIC_BLOCK_BASE_N);
    if (tilingData->gmmWeightQuantParam.secondTailBlockCount > 0) {
        EXPECT_GE(tilingData->gmmWeightQuantParam.secondTailBlockSize, optiling::BASIC_BLOCK_BASE_N_MIN);
        EXPECT_LE(tilingData->gmmWeightQuantParam.secondTailBlockSize, optiling::BASIC_BLOCK_BASE_N);
    }
}

// MxA8W4 NZ场景 - groupNum=3时perTokenScale仍按单tensor M方向对齐
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_per_token_scale_m_groupnum3)
{
    MxA8W4SmsTilingParam param;
    param.m = 96;
    param.groupNum = 3;
    ExpectMxA8W4SmsTilingSuccess(param);
}

// MxA8W4 NZ场景 - 单单单模式TilingKey中SMS标志位应为0
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sss_tiling_key_sms_flag_false)
{
    auto compileInfo = MakeArch35CompileInfo();
    auto tilingContextPara = MakeMxA8W4SssContext(&compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(MakeExpectedMxA8W4TilingKey(false)));
    EXPECT_NE(tilingInfo.tilingKey, static_cast<int64_t>(MakeExpectedMxA8W4TilingKey(true)));
    const auto *tilingData = GetGMMWeightQuantTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupNum, 2);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.singleX, 1);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.singleWeight, 1);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.singleY, 1);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupSize, 32);
}

// MxA8W4 NZ场景 - 单多单模式, x必须是单tensor
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_multi_x_invalid)
{
    MxA8W4SmsTilingParam param;
    param.numX = 2;
    ExpectMxA8W4SmsTilingFailure(param);
}

// MxA8W4 NZ场景 - 单多单模式, weight不能为空
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_weight_empty_invalid)
{
    MxA8W4SmsTilingParam param;
    param.numWeight = 0;
    param.numAntiquantScale = 0;
    param.groupListSize = 0;
    ExpectMxA8W4SmsTilingFailure(param);
}

// MxA8W4 NZ场景 - 单多单模式, antiquantScale list size必须与weight一致
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_antiquant_scale_list_size_mismatch)
{
    MxA8W4SmsTilingParam param;
    param.numAntiquantScale = 1;
    ExpectMxA8W4SmsTilingFailure(param);
}

// MxA8W4 NZ场景 - 单多单模式, bias list size必须与weight一致
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_bias_list_size_mismatch)
{
    MxA8W4SmsTilingParam param;
    param.hasBias = true;
    param.numBias = 1;
    ExpectMxA8W4SmsTilingFailure(param);
}

// MxA8W4 NZ场景 - 单多单模式, groupList长度必须与weight一致
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_group_list_size_mismatch)
{
    MxA8W4SmsTilingParam param;
    param.groupListSize = 1;
    ExpectMxA8W4SmsTilingFailure(param);
}

// MxA8W4 NZ场景 - 单多单模式, transB必须为true
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_transb_false_invalid)
{
    MxA8W4SmsTilingParam param;
    param.transB = false;
    ExpectMxA8W4SmsTilingFailure(param);
}

// 非MxA8W4组合不能误判为单多单模式
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_like_non_mxa8w4_rejected)
{
    MxA8W4SmsTilingParam param;
    param.xDtype = ge::DT_INT8;
    ExpectMxA8W4SmsTilingFailure(param);
}

// MxA8W4 NZ场景 - perTokenScale的M轴必须与x对齐
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_per_token_scale_m_mismatch)
{
    MxA8W4SmsTilingParam param;
    param.perTokenScaleOriginShape = {static_cast<int64_t>(param.m + 1), static_cast<int64_t>(param.k / 64), 2};
    param.perTokenScaleStorageShape = param.perTokenScaleOriginShape;
    ExpectMxA8W4SmsTilingFailure(param);
}

// MxA8W4 NZ场景 - 单多单模式, antiquantScale单个tensor不能是4维
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_antiquant_scale_dim4_invalid)
{
    MxA8W4SmsTilingParam param;
    param.antiquantScaleOriginShape = {1, static_cast<int64_t>(param.n), static_cast<int64_t>(param.k / 64), 2};
    param.antiquantScaleStorageShape = param.antiquantScaleOriginShape;
    ExpectMxA8W4SmsTilingFailure(param);
}

// MxA8W4 NZ场景 - 单多单模式, antiquantScale单个tensor不能是2维
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_antiquant_scale_dim2_invalid)
{
    MxA8W4SmsTilingParam param;
    param.antiquantScaleOriginShape = {static_cast<int64_t>(param.n), static_cast<int64_t>(param.k / 64)};
    param.antiquantScaleStorageShape = param.antiquantScaleOriginShape;
    ExpectMxA8W4SmsTilingFailure(param);
}

// MxA8W4 NZ场景 - 单多单模式, antiquantScale的N轴必须与同下标weight对齐
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_antiquant_scale_n_mismatch)
{
    MxA8W4SmsTilingParam param;
    param.antiquantScaleOriginShape = {static_cast<int64_t>(param.n + 16), static_cast<int64_t>(param.k / 64), 2};
    param.antiquantScaleStorageShape = param.antiquantScaleOriginShape;
    ExpectMxA8W4SmsTilingFailure(param);
}

// MxA8W4 NZ场景 - 单多单模式, antiquantScale的K分组必须满足MX group size
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_mxa8w4_sms_antiquant_scale_group_size_invalid)
{
    MxA8W4SmsTilingParam param;
    param.antiquantScaleOriginShape = {static_cast<int64_t>(param.n), 1, 2};
    param.antiquantScaleStorageShape = param.antiquantScaleOriginShape;
    ExpectMxA8W4SmsTilingFailure(param);
}

// A16W4 ND场景 - BF16输入, INT4权重, 单单单模式, Split M
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_single_single_single_splitm)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},       // x
            {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},               // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 103817216;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - FP16输入, INT4权重, 单单单模式, Split M
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_fp16_nd_single_single_single_splitm)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // x
            {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // antiquantScale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},            // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 103817216;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - 带antiquantOffset
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_with_offset)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},       // x
            {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantOffset (存在)
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 103817248;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A8W4 NZ场景 - INT8输入, INT4权重, NZ格式
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a8w4_nz_int8_int4)
{
    size_t M = 256;
    size_t K = 512;
    size_t N = 256;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    // NZ格式: (N1, K1, K0, N0)，其中N0=16, K0=32 (对于int4)
    size_t N1 = N / 16;
    size_t K1 = K / 32;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                   // x
            {{{E, N1, K1, 32, 16}, {E, N1, K1, 32, 16}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ}, // weight NZ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // offset
            {{{E, K / 128, N}, {E, K / 128, N}}, ge::DT_UINT64, ge::FORMAT_ND},               // antiquantScale (groupsize=128)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                        // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                                        // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 71311361;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// 多多多模式测试 - A16W4 ND
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_multi_multi_multi)
{
    size_t M = 128;
    size_t K = 512;
    size_t N = 256;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND}, // x
            {{{K, N}, {K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{N}, {N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // offset
            {{{N}, {N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},         // antiquantOffset
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},        // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 103817216;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - 转置weight
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_transb)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},       // x
            {{{E, N, K}, {E, N, K}}, ge::DT_INT4, ge::FORMAT_ND}, // weight (transposed)
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},               // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 70328320;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A8W4 NZ场景 - groupsize=192
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a8w4_nz_groupsize_192)
{
    size_t M = 256;
    size_t K = 384; // 192 * 2
    size_t N = 256;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    size_t N1 = N / 16;
    size_t K1 = K / 32;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                   // x
            {{{E, N1, K1, 32, 16}, {E, N1, K1, 32, 16}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ}, // weight NZ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // offset
            {{{E, K / 192, N}, {E, K / 192, N}}, ge::DT_UINT64, ge::FORMAT_ND},               // antiquantScale (groupsize=192)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                        // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                                        // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 72359937;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - No Split模式
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_no_split)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND}, // x
            {{{K, N}, {K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{N}, {N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // offset
            {{{N}, {N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},         // antiquantOffset
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},        // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 103817216;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A8W4 NZ场景 - FP16输出
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a8w4_nz_fp16_output)
{
    size_t M = 256;
    size_t K = 512;
    size_t N = 256;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    size_t N1 = N / 16;
    size_t K1 = K / 32;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                   // x
            {{{E, N1, K1, 32, 16}, {E, N1, K1, 32, 16}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ}, // weight NZ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // offset
            {{{E, K / 128, N}, {E, K / 128, N}}, ge::DT_UINT64, ge::FORMAT_ND},               // antiquantScale
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                        // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                                        // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 71311361;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - 大N值测试（触发尾块重切分）
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_large_n)
{
    size_t M = 256;
    size_t K = 1024;
    size_t N = 8192; // 大N值
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},       // x
            {{{E, N, K}, {E, N, K}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},               // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, // 转置触发尾块重切分
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 70328320;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// A16W4 ND场景 - K=0但M和N非0时应报错
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_k0_mn_not_zero)
{
    size_t M = 256;
    size_t K = 0; // K=0
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},       // x: K=0, M=256 (非0)
            {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight: K=0, N=512 (非0)
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},               // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    // K=0 但 M 和 N 非 0，应该返回 false
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// A16W4 ND场景 - K=0且M=0时应通过（空tensor场景）
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a16w4_bf16_nd_k0_m0)
{
    size_t M = 0; // M=0
    size_t K = 0; // K=0
    size_t N = 512;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_BF16, ge::FORMAT_ND},       // x: M=0, K=0
            {{{E, K, N}, {E, K, N}}, ge::DT_INT4, ge::FORMAT_ND}, // weight: K=0, N=512
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // offset
            {{{E, N}, {E, N}}, ge::DT_BF16, ge::FORMAT_ND},       // antiquantScale
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},               // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},            // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    // K=0 但 M=0，空tensor场景，CheckEmptyTensor 应通过
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// A8W4 NZ场景 - groupsize=256
TEST_F(GroupedWeightQuantBatchMatmulTilingTest, test_tiling_a8w4_nz_groupsize_256)
{
    size_t M = 256;
    size_t K = 512; // 256 * 2
    size_t N = 256;
    size_t E = 2;
    optiling::GMMCompileInfo compileInfo = {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        196608,                                  // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // ASCEND950
        NpuArch::DAV_3510,
    };
    size_t N1 = N / 16;
    size_t K1 = K / 32;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", // op_name
        {
            // input info
            {{{M, K}, {M, K}}, ge::DT_INT8, ge::FORMAT_ND},                                   // x
            {{{E, N1, K1, 32, 16}, {E, N1, K1, 32, 16}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ}, // weight NZ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // bias
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // scale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // offset
            {{{E, K / 256, N}, {E, K / 256, N}}, ge::DT_UINT64, ge::FORMAT_ND},               // antiquantScale (groupsize=256)
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                        // antiquantOffset
            {{{E}, {E}}, ge::DT_INT64, ge::FORMAT_ND},                                        // groupList
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                          // perTokenScale
        },
        {// output info
         {{{M, N}, {M, N}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // attr
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    int64_t expectTilingKey = 71311361;
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}
