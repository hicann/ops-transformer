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
#include "infer_datatype_context_faker.h"

class NormRopeConcat : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NormRopeConcat SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NormRopeConcat TearDown" << std::endl;
    }
};

TEST_F(NormRopeConcat, NormRopeConcat_infer_shape_00)
{
    gert::InfershapeContextPara infershapeContextPara(
        "NormRopeConcat",
        {
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8, 8, 64}, {1, 8, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 64}, {16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {

            {"norm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"norm_added_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rope_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"concat_order", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
            {"is_training", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 8, 16, 64}, {1, 8, 16, 64}, {1, 8, 16, 64}, {1, 8, 8, 1},
                                                           {1, 8, 8, 1},   {1, 8, 8, 1},   {1, 8, 8, 1},   {1, 8, 8, 1},
                                                           {1, 8, 8, 1},   {1, 8, 8, 1},   {1, 8, 8, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
