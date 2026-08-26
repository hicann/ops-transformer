/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_chunk_gated_delta_rule_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class ChunkGatedDeltaRuleTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // std::cout << "ChunkGatedDeltaRuleTest Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        // std::cout << "ChunkGatedDeltaRuleTest Proto TearDown" << std::endl;
    }
};

TEST_F(ChunkGatedDeltaRuleTest, Test0)
{
    // 基础形状参数
    uint32_t bs = 2;
    uint32_t seqLen = 100;
    uint32_t t = bs * seqLen;
    uint32_t nk = 4;
    uint32_t nv = 8;
    uint32_t dk = 128;
    uint32_t dv = 128;

    // 输入 shape（逻辑形状=存储形状）
    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};
    gert::StorageShape gShape = {{t, nv}, {t, nv}};

    gert::InfershapeContextPara infershapeContextPara(
        "ChunkGatedDeltaRule",
        {
            // 输入：query/key/value/beta/initial_state/actual_seq_lengths/g
            {queryShape, ge::DT_BF16, ge::FORMAT_ND},
            {keyShape, ge::DT_BF16, ge::FORMAT_ND},
            {valueShape, ge::DT_BF16, ge::FORMAT_ND},
            {betaShape, ge::DT_BF16, ge::FORMAT_ND},
            {stateShape, ge::DT_BF16, ge::FORMAT_ND},
            {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
            {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出占位：out / final_state
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 常量输入：scale_value
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        });

    // 预期输出 shape：out 与 final_state
    std::vector<std::vector<int64_t>> expectOutputShape = {{t, nv, dv}, {bs, nv, dv, dk}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
