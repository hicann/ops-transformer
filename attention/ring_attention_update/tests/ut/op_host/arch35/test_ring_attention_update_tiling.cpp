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
#include <string>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {
struct RingAttentionUpdateCompileInfo {};

constexpr uint64_t kUtCoreNum = 1U;
constexpr uint64_t kUbSize = 262144ULL;
constexpr uint32_t kTilingDataSize = 8192U;

const std::vector<uint32_t> kInputIrInstanceSbh{1, 1, 1, 1, 1, 1, 0};
const std::vector<uint32_t> kInputIrInstanceTnd{1, 1, 1, 1, 1, 1, 1};
const std::vector<uint32_t> kOutputIrInstance{1, 1, 1};

constexpr uint64_t kTilingKeyRegbaseSbh = 100UL;
constexpr uint64_t kTilingKeyRegbaseSoftmaxTndFp16 = 110UL;
} // namespace

class RingAttentionUpdateRegbaseTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RingAttentionUpdateRegbaseTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RingAttentionUpdateRegbaseTiling TearDown" << std::endl;
    }
};

// A1: regbase SBH fp32
TEST_F(RingAttentionUpdateRegbaseTiling, RingAttentionUpdate_950_tiling_A1_sbh_fp32)
{
    struct RingAttentionUpdateCompileInfo compileInfo {};
    gert::TilingContextPara para(
        "RingAttentionUpdate",
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
            {"input_softmax_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
        },
        kInputIrInstanceSbh, kOutputIrInstance, &compileInfo, "Ascend950", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyRegbaseSbh);
}

// A2: regbase SBH fp16
TEST_F(RingAttentionUpdateRegbaseTiling, RingAttentionUpdate_950_tiling_A2_sbh_fp16)
{
    struct RingAttentionUpdateCompileInfo compileInfo {};
    gert::TilingContextPara para(
        "RingAttentionUpdate",
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
            {"input_softmax_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
        },
        kInputIrInstanceSbh, kOutputIrInstance, &compileInfo, "Ascend950", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyRegbaseSbh);
}

// A3: regbase TND + softmax TND fp16
TEST_F(RingAttentionUpdateRegbaseTiling, RingAttentionUpdate_950_tiling_A3_softmax_tnd_fp16)
{
    struct RingAttentionUpdateCompileInfo compileInfo {};
    int64_t actSeqQlen[] = {500,  1000, 1500, 2100, 2700, 3300, 3900, 4500, 5100,
                            5700, 6300, 6900, 7500, 8100, 8700, 9300, 9900, 12000};
    gert::TilingContextPara para(
        "RingAttentionUpdate",
        {
            {{{12000, 2, 768}, {12000, 2, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12000, 2, 768}, {12000, 2, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{18}, {18}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQlen},
        },
        {
            {{{12000, 2, 768}, {12000, 2, 768}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"input_softmax_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        },
        kInputIrInstanceTnd, kOutputIrInstance, &compileInfo, "Ascend950", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyRegbaseSoftmaxTndFp16);
}

// E1: invalid input_softmax_layout
TEST_F(RingAttentionUpdateRegbaseTiling, RingAttentionUpdate_950_tiling_E1_invalid_softmax_layout)
{
    struct RingAttentionUpdateCompileInfo compileInfo {};
    gert::TilingContextPara para(
        "RingAttentionUpdate",
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
            {"input_softmax_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
        },
        kInputIrInstanceSbh, kOutputIrInstance, &compileInfo, "Ascend950", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
