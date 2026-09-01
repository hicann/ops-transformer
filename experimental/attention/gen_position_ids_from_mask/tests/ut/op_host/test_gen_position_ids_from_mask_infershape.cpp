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

class GenPositionIdsFromMaskInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- GenPositionIdsFromMask InferShape UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- GenPositionIdsFromMask InferShape UT TearDown ---" << std::endl;
    }
};

// 输出 shape 应与输入 mask shape 一致
TEST_F(GenPositionIdsFromMaskInferShapeTest, infershape_same_as_input)
{
    int64_t b = 2, s = 256;

    gert::InfershapeContextPara infershapeContextPara(
        "GenPositionIdsFromMask",
        {// input info
         {{{b, s}, {b, s}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// output info: 占位给空 shape, 由 infershape 从零构造
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
        {// attr
         {"padding_fill_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{b, s}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 另一组 shape, 验证 infershape 跟随输入
TEST_F(GenPositionIdsFromMaskInferShapeTest, infershape_another_shape)
{
    int64_t b = 16, s = 1024;

    gert::InfershapeContextPara infershapeContextPara(
        "GenPositionIdsFromMask",
        {// input info
         {{{b, s}, {b, s}}, ge::DT_INT64, ge::FORMAT_ND}},
        {// output info: 占位给空 shape
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
        {// attr
         {"padding_fill_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{b, s}};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GenPositionIdsFromMaskInferShapeTest, infershape_reject_1d_input)
{
    int64_t s = 256;

    gert::InfershapeContextPara infershapeContextPara(
        "GenPositionIdsFromMask",
        {// input info
         {{{s}, {s}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// output info
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
        {// attr
         {"padding_fill_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(GenPositionIdsFromMaskInferShapeTest, infershape_reject_3d_input)
{
    int64_t b = 2, s = 256;

    gert::InfershapeContextPara infershapeContextPara(
        "GenPositionIdsFromMask",
        {// input info
         {{{b, s, 1}, {b, s, 1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// output info
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
        {// attr
         {"padding_fill_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
