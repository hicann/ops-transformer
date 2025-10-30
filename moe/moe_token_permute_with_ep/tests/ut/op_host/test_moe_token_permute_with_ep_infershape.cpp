/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeTokenPermuteWithEpInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithEp SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithEp TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteWithEpInferShape, MoeTokenPermuteWithEp_infershape_case_0)
{
    std::vector<int64_t> range({1, 5});
    gert::InfershapeContextPara infershapeContextPara("MoeTokenPermuteWithEp",
                                                      {{{{2, 5}, {2, 5}}, ge::DT_BF16, ge::FORMAT_ND},
                                                       {{{2, 3}, {2, 3}}, ge::DT_BF16, ge::FORMAT_ND},
                                                       {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {
                                                          {{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {{"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
                                                       {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                       {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 5}, {6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
