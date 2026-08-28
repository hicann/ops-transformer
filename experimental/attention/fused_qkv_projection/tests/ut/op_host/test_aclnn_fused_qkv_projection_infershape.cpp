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
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FusedQkvProjectionTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedQkvProjectionTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedQkvProjectionTest TearDown" << std::endl;
    }
};

// 基础推理形状：batch=2, seq=8, hidden=16, q=16, k=8, v=8, bias
TEST_F(FusedQkvProjectionTest, infershape_basic_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedQkvProjection",
        {
            {{{2, 8, 16}, {2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hidden_states
            {{{16, 32}, {16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},     // weight
            {{{32}, {32}}, ge::DT_FLOAT, ge::FORMAT_ND},             // bias (optional)
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // query
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // key
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // value
        },
        {
            {"q_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"k_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"v_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 8, 16}, {2, 8, 8}, {2, 8, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// float16 推理形状
TEST_F(FusedQkvProjectionTest, infershape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedQkvProjection",
        {
            {{{1, 32, 64}, {1, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // hidden_states
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // weight
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
        },
        {
            {"q_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"k_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"v_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 32, 32}, {1, 32, 16}, {1, 32, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// q=k=v 等分
TEST_F(FusedQkvProjectionTest, infershape_equal_split)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedQkvProjection",
        {
            {{{4, 16, 48}, {4, 16, 48}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hidden_states
            {{{48, 48}, {48, 48}}, ge::DT_FLOAT, ge::FORMAT_ND},       // weight
            {{{48}, {48}}, ge::DT_FLOAT, ge::FORMAT_ND},               // bias
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"q_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"k_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"v_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 16, 16}, {4, 16, 16}, {4, 16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态 shape 推理
TEST_F(FusedQkvProjectionTest, infershape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedQkvProjection",
        {
            {{{-1, -1, 64}, {2, 8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // hidden_states (dynamic batch/seq)
            {{{64, -1}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},       // weight
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"q_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"k_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"v_output_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, 32}, {-1, -1, 16}, {-1, -1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
