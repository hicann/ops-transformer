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

class GroupedMatmulAddInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulAddProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulAddProto TearDown" << std::endl;
    }
};

TEST_F(GroupedMatmulAddInfershape, grouped_matmul_add_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmulAdd",
        {
            // input info
            {{{256, 128}, {256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// output info
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {// attr
         {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 512},
    }; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulAddInfershape, grouped_matmul_add_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmulAdd",
        {
            // input info
            {{{256, 128}, {256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// output info
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {// attr
         {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 512},
    }; // 预期输出shape
    int groupListTypeMode = 2;
    int groupListTypeIndex = 3;
    for (int i = 0; i < groupListTypeMode; i++) {
        infershapeContextPara.attrs_.at(groupListTypeIndex).attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(i);
        ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    }
}
