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
class swin_attention_ffn : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "SwinAttentionFFN Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SwinAttentionFFN Proto Test TearDown" << std::endl;
    }
};

TEST_F(swin_attention_ffn, swin_attention_ffn_test_1)
{
    gert::InfershapeContextPara infershapeContextPara(
    "SwinAttentionFFN",
    { // input info
        {{{4096, 64, 128}, {4096, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{128, 128}, {128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4096, 64, 128}, {4096, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    }, 
    { // output info
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    }, 
    { // attr
        {"shifts",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})}
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4096, 64, 128},}; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}