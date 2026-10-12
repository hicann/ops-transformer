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
#include "base/registry/op_impl_space_registry_v2.h"

class ApplyRotaryPosEmbTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRotaryPosEmbTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyRotaryPosEmbTest TearDown" << std::endl;
    }
};

TEST_F(ApplyRotaryPosEmbTest, apply_rotary_pos_emb_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// attr
         {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("half")}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{24, 1, 11, 128}, {24, 1, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ApplyRotaryPosEmbTest, apply_rotary_pos_emb_infer_shape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {// attr
         {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("half")}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{24, 1, 11, 128}, {24, 1, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ApplyRotaryPosEmbTest, apply_rotary_pos_emb_infer_shape_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// attr
         {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("half")}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{24, 1, 11, 128}, {24, 1, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ApplyRotaryPosEmbTest, apply_rotary_pos_emb_infer_shape_fp32_c64)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ApplyRotaryPosEmb",
        {
            // input info
            {{{24, 1, 11, 128}, {24, 1, 11, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{24, 1, 1, 128}, {24, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{24, 1, 1, 64}, {24, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// attr
         {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("half")}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{24, 1, 11, 128}, {24, 1, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
