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

namespace {
const std::vector<uint32_t> kInputIrInstanceSbh{1, 1, 1, 1, 1, 1, 0};
const std::vector<uint32_t> kOutputIrInstance{1, 1, 1};

using TensorDesc = gert::InfershapeContextPara::TensorDescription;
} // namespace

class RingAttentionUpdateProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RingAttentionUpdateProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RingAttentionUpdateProto TearDown" << std::endl;
    }
};

// A1: SBH fp32
TEST_F(RingAttentionUpdateProto, RingAttentionUpdate_infershape_A1_sbh_fp32)
{
    gert::InfershapeContextPara para("RingAttentionUpdate",
                                     {
                                         {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {}, kInputIrInstanceSbh, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {1024, 2, 1536},
        {2, 12, 1024, 8},
        {2, 12, 1024, 8},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A2: SBH fp16
TEST_F(RingAttentionUpdateProto, RingAttentionUpdate_infershape_A2_sbh_fp16)
{
    gert::InfershapeContextPara para("RingAttentionUpdate",
                                     {
                                         {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {}, kInputIrInstanceSbh, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {1024, 2, 1536},
        {2, 12, 1024, 8},
        {2, 12, 1024, 8},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A3: SBH bf16
TEST_F(RingAttentionUpdateProto, RingAttentionUpdate_infershape_A3_sbh_bf16)
{
    gert::InfershapeContextPara para("RingAttentionUpdate",
                                     {
                                         {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {}, kInputIrInstanceSbh, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {1024, 2, 1536},
        {2, 12, 1024, 8},
        {2, 12, 1024, 8},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A4: smaller SBH shape
TEST_F(RingAttentionUpdateProto, RingAttentionUpdate_infershape_A4_small_sbh)
{
    gert::InfershapeContextPara para("RingAttentionUpdate",
                                     {
                                         {{{128, 2, 384}, {128, 2, 384}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 3, 128, 8}, {2, 3, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 3, 128, 8}, {2, 3, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{128, 2, 384}, {128, 2, 384}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 3, 128, 8}, {2, 3, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 3, 128, 8}, {2, 3, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {}, kInputIrInstanceSbh, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {128, 2, 384},
        {2, 3, 128, 8},
        {2, 3, 128, 8},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}
