/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

class MoeGatingTopKSoftmaxV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2 Proto Test TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmaxV2",
                                                      {
                                                        {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                        {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2}, {4, 2}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmaxV2",
                                                      {
                                                        {{{4, 4, 3, 3}, {4, 4, 3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                        {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2}, {4, 2}, {4, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input2)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmaxV2",
                                                      {
                                                        {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
                                                        {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2}, {4, 2}, {4, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input3)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmaxV2",
                                                      {
                                                        {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                        {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                        {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2}, {4, 2}, {4, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2_infershape_diff_test_legal_input4)
{
    gert::InfershapeContextPara infershapeContextPara("MoeGatingTopKSoftmaxV2",
                                                      {
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                        {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 2}, {-1, 2}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
