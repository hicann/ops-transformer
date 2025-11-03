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
#include "base/registry/op_impl_space_registry_v2.h"


class MoeTokenPermuteWithRoutingMapGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenPermuteWithRoutingMapGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenPermuteWithRoutingMapGrad Proto Test TearDown" << std::endl;
  }
};
TEST_F(MoeTokenPermuteWithRoutingMapGrad,infershape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeTokenPermuteWithRoutingMapGrad",
        {// input info
            {{{1024, 7168}, {1024, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 7168}, {1024, 7168}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 512}, {512, 512}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {// output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {// attr
            {"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {512, 7168},
        {512, 2},
    };                                                                            // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}
