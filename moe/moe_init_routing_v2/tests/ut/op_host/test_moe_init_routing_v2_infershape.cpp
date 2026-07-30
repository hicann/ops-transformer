/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeInitRoutingV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV2 TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_01)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_02)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_03)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_04)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_05)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}


TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_06)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3, 3}, {3, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_07)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}


TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_08)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 5}, {2, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_09)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_10)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_11)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(99)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}


TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_12)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}


TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_13)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// expanded_row_idx shape = n * k when n >= 0 and k >= 0
TEST_F(MoeInitRoutingV2, infer_shape_expanded_row_idx_n_mul_k_active_num_zero)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 5}, {6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, infer_shape_expanded_row_idx_zero_when_n_zero)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info: n=0, k=3
         {{{0, 5}, {0, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0, 3}, {0, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 5}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, infer_shape_expanded_row_idx_zero_when_k_zero)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info: n=2, k=0
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 0}, {2, 0}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 5}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, infer_shape_expanded_row_idx_n_mul_k_unaffected_by_active_num)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr: active_num limits expanded_x but not expanded_row_idx
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}, {6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, infer_shape_expanded_row_idx_n_mul_k_drop_pad_mode)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {// input info
         {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // attr
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 5}, {6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_datatype_01)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto data_type_func = spaceRegistry->GetOpImpl("MoeInitRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        ge::DataType index_ref = ge::DT_INT32;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 4)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &index_ref})
                                  .OutputDataTypes({&output_ref, &index_ref, &index_ref, &index_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), index_ref);
        EXPECT_EQ(context->GetOutputDataType(2), index_ref);
        EXPECT_EQ(context->GetOutputDataType(3), index_ref);
    }
}

// x dim[0] < -1
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_14)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {{{{-2, 5}, {-2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{8, 3}, {8, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// expertIdx dim[0] < -1
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_15)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {{{{8, 5}, {8, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{-2, 3}, {-2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// x 1D with dim != -2
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {{{{5}, {5}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{8, 3}, {8, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// x 3D+
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_17)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {{{{2, 5, 3}, {2, 5, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// expertIdx 3D+
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_18)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {{{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// expertTokensCountOrCumsumFlag > 2
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_19)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {{{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// dropPadMode=1, expertCapacity > x.dim(0)
TEST_F(MoeInitRoutingV2, moe_init_routing_v2_infer_shape_20)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV2",
        {{{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expert_tokens_count_or_cumsum_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_tokens_before_capacity_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
