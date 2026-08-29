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

class ApplyRotaryPosEmbGradInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRotaryPosEmbGradInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyRotaryPosEmbGradInferShape TearDown" << std::endl;
    }
};

TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_bsnd_dcos0_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 128}, {2, 64, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 64, 4, 128}, {2, 64, 2, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_bsnd_dcos1_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 64, 2, 128}, {2, 64, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 64, 2, 128}, {2, 64, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 64, 4, 128}, {2, 64, 2, 128}, {1, 64, 1, 128}, {1, 64, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_tnd_dcos1_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ApplyRotaryPosEmbGrad",
        {
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 2, 128}, {64, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 2, 128}, {64, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{64, 4, 128}, {64, 2, 128}, {64, 1, 128}, {64, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_dynamic_dim_bsnd_dcos0_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ApplyRotaryPosEmbGrad",
        {
            {{{-1, 64, 4, 128}, {-1, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, 64, 2, 128}, {-1, 64, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 64, 4, 128}, {-1, 64, 2, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_dynamic_dim_bsnd_dcos1_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ApplyRotaryPosEmbGrad",
        {
            {{{-1, 64, 4, 128}, {-1, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, 64, 2, 128}, {-1, 64, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, 64, 4, 128}, {-1, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, 64, 2, 128}, {-1, 64, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 64, 4, 128}, {-1, 64, 2, 128}, {1, 64, 1, 128}, {1, 64, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ApplyRotaryPosEmbGradInferShape, infer_shape_unknown_rank_dcos1_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ApplyRotaryPosEmbGrad",
        {
            {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-2}, {1, 64, 1, 128}, {1, 64, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
