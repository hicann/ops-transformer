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

#include <vector>

#include "../../../op_host/op_tiling/quant_grouped_matmul_inplace_add_tiling.h"
#include "../../../op_kernel/arch35/quant_grouped_matmul_inplace_add_tiling_data.h"
#include "common/include/op_host/op_const_def_transformer.h"
#include "gmm_csv_ge_parse_utils.h"
#include "tiling_case_executor.h"

namespace {
using TilingData = QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams;

struct MxL1Case {
    const char* name;
    ge::DataType aDtype;
    ge::DataType bDtype;
    int32_t baseK;
    int32_t stepK;
    uint32_t scaleFactor;
    uint32_t longKScaleFactorA;
    uint32_t longKScaleFactorB;
    uint64_t baseL1Bytes;
};

const MxL1Case kMxL1Cases[] = {
    {"E5M2", ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, 128, 2, 4, 32, 32, 67584},
    {"E4M3", ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, 128, 2, 4, 32, 32, 67584},
    {"E2M1", ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, 256, 2, 2, 16, 16, 69632},
    {"Hif4", Ops::Transformer::DT_HIFLOAT4, Ops::Transformer::DT_HIFLOAT4, 256, 2, 2, 8, 8, 73728},
    {"E4M3E2M1", ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1, 128, 4, 2, 8, 8, 51200},
    {"E4M3Hif4", ge::DT_FLOAT8_E4M3FN, Ops::Transformer::DT_HIFLOAT4, 128, 4, 2, 6, 5, 52224},
};

bool RunMxTiling(const MxL1Case& param, int64_t k, int64_t groups, uint64_t l1Bytes, TilingInfo& result)
{
    optiling::GMMCompileInfo compileInfo = {32,
                                            32,
                                            262144,
                                            l1Bytes,
                                            134217728,
                                            262144,
                                            65536,
                                            65536,
                                            platform_ascendc::SocVersion::ASCEND950,
                                            NpuArch::DAV_9201,
                                            true};
    constexpr int64_t m = 256;
    constexpr int64_t n = 256;
    const int64_t scaleRows = k / 64 + groups;
    const bool isHif4A = param.aDtype == Ops::Transformer::DT_HIFLOAT4;
    const bool isHif4B = param.bDtype == Ops::Transformer::DT_HIFLOAT4;
    const auto scaleADtype = isHif4A ? Ops::Transformer::DT_HIFLOAT4_SCALE : ge::DT_FLOAT8_E8M0;
    const auto scaleBDtype = isHif4B ? Ops::Transformer::DT_HIFLOAT4_SCALE : ge::DT_FLOAT8_E8M0;
    const auto yShape = ops::ut::MakeGertStorageShape(std::vector<int64_t>{groups, m, n});
    gert::TilingContextPara context(
        "QuantGroupedMatmulInplaceAdd",
        {{ops::ut::MakeGertStorageShape(std::vector<int64_t>{k, m}), param.aDtype, ge::FORMAT_ND},
         {ops::ut::MakeGertStorageShape(std::vector<int64_t>{k, n}), param.bDtype, ge::FORMAT_ND},
         {ops::ut::MakeGertStorageShape(std::vector<int64_t>{scaleRows, n, isHif4B ? 1 : 2}), scaleBDtype,
          ge::FORMAT_ND},
         {ops::ut::MakeGertStorageShape(std::vector<int64_t>{groups}), ge::DT_INT64, ge::FORMAT_ND},
         {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {ops::ut::MakeGertStorageShape(std::vector<int64_t>{scaleRows, m, isHif4A ? 1 : 2}), scaleADtype,
          ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "9201", compileInfo.aicNum, compileInfo.ubSize);
    // ExecuteTiling supplies platform information, which takes precedence over compileInfo for L1 capacity.
    context.socInfoString_ = R"({"hardware_info":{"CORE_NUM":32,"cube_core_cnt":32,"vector_core_cnt":32,)"
                             R"("UB_SIZE":262144,"L2_SIZE":134217728,"L0A_SIZE":65536,"L0B_SIZE":65536,)"
                             R"("L0C_SIZE":262144,"BT_SIZE":0,"load3d_constraints":"1","L1_SIZE":)" +
                             std::to_string(l1Bytes) + "}}";
    return ExecuteTiling(context, result);
}

class TestQGmmInplaceAddL1Tiling : public testing::TestWithParam<MxL1Case> {
protected:
    void CheckTiling(const TilingInfo& result, int64_t k, int64_t groups, uint32_t scaleA, uint32_t scaleB) const
    {
        ASSERT_EQ(result.tilingDataSize, sizeof(TilingData));
        ASSERT_NE(result.tilingData.get(), nullptr);
        EXPECT_EQ(result.tilingKey, 4U);
        EXPECT_EQ(result.blockNum, 32U);
        const auto* data = reinterpret_cast<const TilingData*>(result.tilingData.get());
        EXPECT_EQ(data->quantGmmInplaceAddParams.groupNum, groups);
        EXPECT_EQ(data->quantGmmInplaceAddParams.groupListType, 1);
        const auto& mm = data->mmTilingData;
        EXPECT_EQ(mm.M, 256);
        EXPECT_EQ(mm.N, 256);
        EXPECT_EQ(mm.Ka, k);
        EXPECT_EQ(mm.Kb, k);
        EXPECT_EQ(mm.baseM, 256);
        EXPECT_EQ(mm.baseN, 256);
        EXPECT_EQ(mm.baseK, GetParam().baseK);
        EXPECT_EQ(mm.stepKa, GetParam().stepK);
        EXPECT_EQ(mm.stepKb, GetParam().stepK);
        EXPECT_EQ(mm.depthA1, 2 * GetParam().stepK);
        EXPECT_EQ(mm.depthB1, 2 * GetParam().stepK);
        EXPECT_EQ(mm.mxTypePara & 0xffU, scaleA);
        EXPECT_EQ((mm.mxTypePara >> 8U) & 0xffU, scaleB);
    }
};

TEST_P(TestQGmmInplaceAddL1Tiling, SplitKScaleWindowDoesNotGrowWithGroupCount)
{
    for (int64_t groups : {64, 1024}) {
        SCOPED_TRACE(groups);
        TilingInfo result;
        ASSERT_TRUE(RunMxTiling(GetParam(), 1024, groups, 524288, result));
        CheckTiling(result, 1024, groups, GetParam().scaleFactor, GetParam().scaleFactor);
    }
}

TEST_P(TestQGmmInplaceAddL1Tiling, LongKScaleFactorsRespectEachInputQuantMode)
{
    TilingInfo result;
    ASSERT_TRUE(RunMxTiling(GetParam(), 65536, 1024, 524288, result));
    // E4M3 x HIF4 must budget scales every 32 K on A and every 64 K on B, producing factors 6/5.
    CheckTiling(result, 65536, 1024, GetParam().longKScaleFactorA, GetParam().longKScaleFactorB);
}

TEST_P(TestQGmmInplaceAddL1Tiling, RejectsL1SmallerThanOneDataAndScaleWindow)
{
    TilingInfo result;
    EXPECT_FALSE(RunMxTiling(GetParam(), 1024, 1024, GetParam().baseL1Bytes - 1, result));
}

INSTANTIATE_TEST_SUITE_P(MxSplitK, TestQGmmInplaceAddL1Tiling, testing::ValuesIn(kMxL1Cases),
                         [](const testing::TestParamInfo<MxL1Case>& info) { return info.param.name; });

TEST(QGmmInplaceAddL1Tiling, MixedInputAcceptsUnalignedK)
{
    for (const auto& param : {kMxL1Cases[4], kMxL1Cases[5]}) {
        SCOPED_TRACE(param.name);
        TilingInfo result;
        ASSERT_TRUE(RunMxTiling(param, 129, 64, 524288, result));
        ASSERT_EQ(result.tilingDataSize, sizeof(TilingData));
        const auto* data = reinterpret_cast<const TilingData*>(result.tilingData.get());
        EXPECT_EQ(result.tilingKey, 4U);
        EXPECT_EQ(data->mmTilingData.Ka, 129);
        EXPECT_EQ(data->mmTilingData.Kb, 129);
        EXPECT_EQ(data->mmTilingData.baseN % 64, 0);
        EXPECT_EQ(data->mmTilingData.baseK, 128);
    }
}

TEST(QGmmInplaceAddL1Tiling, RejectsUnsupportedMixedDtypePairs)
{
    auto param = kMxL1Cases[4];
    param.aDtype = ge::DT_FLOAT8_E5M2;
    TilingInfo e5Result;
    EXPECT_FALSE(RunMxTiling(param, 1024, 64, 524288, e5Result));
    param.aDtype = ge::DT_FLOAT8_E4M3FN;
    param.bDtype = ge::DT_FLOAT4_E1M2;
    TilingInfo e1Result;
    EXPECT_FALSE(RunMxTiling(param, 1024, 64, 524288, e1Result));
}
} // namespace
