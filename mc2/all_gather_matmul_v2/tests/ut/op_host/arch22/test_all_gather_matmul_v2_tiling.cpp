/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "mc2/common/tests/ut/formal_error_assertions.h"
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "mc2/common/tests/ut/internal_fallback_assertions.h"
#include "../all_gather_matmul_v2_host_ut_param.h"
#include "mc2_tiling_case_executor.h"
#include "../../../../op_host/op_tiling/arch22/tiling_func.h"

namespace all_gather_matmul_v2_ut {

class AllGatherMatmulV2Arch22TilingTest : public testing::TestWithParam<AllGatherMatmulV2TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulV2 Arch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulV2 Arch22TilingTest TearDown" << std::endl;
    }
};

TEST_P(AllGatherMatmulV2Arch22TilingTest, param)
{
    auto param = GetParam();
    struct AllGatherMatmulV2CompileInfo {};
    AllGatherMatmulV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmulV2",
        {
            param.x1,
            param.x2,
            param.bias,
            param.x1_scale,
            param.x2_scale,
            param.quant_scale,
        },
        {
            param.y,
            param.gather_out,
            param.amax_out,
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gather_index)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rank_size)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_gather_out)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_amax_out)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.comm_mode)}},
        param.inputInstance, param.outputInstance, &compileInfo, param.soc, param.coreNum, param.ubsize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rank_size}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
                       param.expectTilingDataHash, {}, MC2_TILING_DATA_RESERVED_LEN, true);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmulV2, AllGatherMatmulV2Arch22TilingTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    GetCaseInfoString<AllGatherMatmulV2TilingUtParam>);

TEST(AllGatherMatmulV2TilingFuncTest, CeilDevNormal)
{
    EXPECT_EQ(CeilDev(10, 3), 4);
    EXPECT_EQ(CeilDev(9, 3), 3);
    EXPECT_EQ(CeilDev(1, 128), 1);
}

TEST(AllGatherMatmulV2TilingFuncTest, CeilDevZeroDivisor)
{
    EXPECT_EQ(CeilDev(10, 0), 0);
}

TEST(AllGatherMatmulV2OwnerResourceContract, AivAndUbZeroReportSingleRootInternalFallback)
{
    auto cases = GetCasesFromCsv<AllGatherMatmulV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__));
    auto found = std::find_if(cases.begin(), cases.end(),
                              [](const AllGatherMatmulV2TilingUtParam& p) { return p.case_name == "fp16_rank8_910b"; });
    ASSERT_NE(found, cases.end());
    for (int variant : {0, 1}) {
        Mc2ErrorCodeUT::ExcelErrorReportScope reports;
        auto param = *found;
        param.coreNum = variant == 0 ? 0 : 30;
        param.ubsize = variant == 1 ? 256 : 262144; // SDK 910B subtracts 256 bytes of reserved UB.
        struct AllGatherMatmulV2CompileInfo {};
        AllGatherMatmulV2CompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara(
            "AllGatherMatmulV2",
            {
                param.x1,
                param.x2,
                param.bias,
                param.x1_scale,
                param.x2_scale,
                param.quant_scale,
            },
            {
                param.y,
                param.gather_out,
                param.amax_out,
            },
            {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
             {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
             {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
             {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gather_index)},
             {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
             {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rank_size)},
             {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
             {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
             {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_gather_out)},
             {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_amax_out)},
             {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
             {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.comm_mode)}},
            param.inputInstance, param.outputInstance, &compileInfo, param.soc, param.coreNum, param.ubsize);
        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rank_size}};

        tilingContextPara.socInfoString_ =
            std::string(R"({"hardware_info":{"CORE_NUM":30,"cube_core_cnt":30,"vector_core_cnt":)") +
            std::to_string(variant == 0 ? 0 : 60) + ",\"UB_SIZE\":" + std::to_string(param.ubsize) +
            R"(,"L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":131072}})";
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, 0);
        const std::string reason = "Available AIV=" + std::to_string(variant == 0 ? 0 : 60) +
                                   ", UB bytes=" + std::to_string(param.ubsize - 256) +
                                   "; positive AIV and UB resources are required";
        Mc2ErrorCodeUT::ExpectInternalFallbackReport(reason.c_str());
    }
}

} // namespace all_gather_matmul_v2_ut
