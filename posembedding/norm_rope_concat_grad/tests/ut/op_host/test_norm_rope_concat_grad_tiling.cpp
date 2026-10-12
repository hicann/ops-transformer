/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/norm_rope_concat_grad_tiling.h"

using namespace std;

class NormRopeConcatGradTiling : public testing::Test {
public:
    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 48}
                                    })";

protected:
    static void SetUpTestCase()
    {
        std::cout << "NormRopeConcatGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NormRopeConcatGradTiling TearDown" << std::endl;
    }
};

TEST_F(NormRopeConcatGradTiling, NormRopeConcatGrad_bf16_001)
{
    optiling::NormRopeConcatGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "NormRopeConcatGrad",
        {
            // input info
            {{{2, 32, 20, 64}, {2, 32, 20, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 32, 64}, {2, 4, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 16, 32, 64}, {2, 16, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{2, 4, 32, 64}, {2, 4, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 16, 32, 64}, {2, 16, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 200200100;
    string expectTilingData = "200200100 2 4 8 8 16 32 32 20 40 40 8 1 32 32 64 64 32 64 32 4359484440219993516 ";
    std::vector<size_t> expectWorkspaces = {37814272};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NormRopeConcatGradTiling, NormRopeConcatGrad_fp16_001)
{
    optiling::NormRopeConcatGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "NormRopeConcatGrad",
        {
            // input info
            {{{2, 32, 20, 64}, {2, 32, 20, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 32, 64}, {2, 4, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 16, 32, 64}, {2, 16, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{2, 4, 32, 64}, {2, 4, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 16, 32, 64}, {2, 16, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 200200100;
    string expectTilingData = "200200100 2 4 8 8 16 32 32 20 40 40 8 1 32 32 64 64 32 64 32 4359484440219993516 ";
    std::vector<size_t> expectWorkspaces = {37814272};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NormRopeConcatGradTiling, NormRopeConcatGrad_fp32_001)
{
    optiling::NormRopeConcatGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "NormRopeConcatGrad",
        {
            // input info
            {{{2, 32, 20, 64}, {2, 32, 20, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 40, 64}, {2, 32, 40, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 32, 64}, {2, 4, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 64}, {2, 16, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 32, 1}, {2, 4, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 1}, {2, 8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 1}, {2, 16, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 1}, {2, 32, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{2, 4, 32, 64}, {2, 4, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 32, 64}, {2, 8, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16, 32, 64}, {2, 16, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 32, 64}, {2, 32, 32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // attr
            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 200200100;
    string expectTilingData = "200200100 2 4 8 8 16 32 32 20 40 40 8 1 32 32 64 64 32 64 32 4359484440219993516 ";
    std::vector<size_t> expectWorkspaces = {37814272};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
