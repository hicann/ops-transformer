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
class SwinTransformerLnQKV : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SwinTransformerLnQKV Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SwinTransformerLnQKV Proto Test TearDown" << std::endl;
  }
};

TEST_F(SwinTransformerLnQKV, swin_transformer_ln_qkv_test_1)
{
    gert::InfershapeContextPara infershapeContextPara("SwinTransformerLnQKV",
    { // input info
        {{{2, 2048, 64}, {2, 2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 2048, 64}, {2, 2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    }, 
    { // attr
        {"epsilon",Ops::Transformer::AnyValue::CreateFrom<float>(0.001)},
        {"head_dim",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"head_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"seq_length",Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"shifts",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 4, 64, 32},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}