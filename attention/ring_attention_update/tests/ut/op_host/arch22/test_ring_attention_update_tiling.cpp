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
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint32_t kTilingDataSize = 8192U;

const std::vector<uint32_t> kInputIrInstanceSbh{1, 1, 1, 1, 1, 1, 0};
const std::vector<uint32_t> kInputIrInstanceTnd{1, 1, 1, 1, 1, 1, 1};
const std::vector<uint32_t> kOutputIrInstance{1, 1, 1};

constexpr uint64_t kTilingKeySbhFp32 = 2UL;
constexpr uint64_t kTilingKeySbhFp16 = 0UL;
constexpr uint64_t kTilingKeySbhBf16 = 1UL;
constexpr uint64_t kTilingKeyTndFp16 = 10UL;

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;
} // namespace

class RingAttentionUpdateTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RingAttentionUpdateTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RingAttentionUpdateTiling TearDown" << std::endl;
    }
};

// A1: SBH fp32
TEST_F(RingAttentionUpdateTiling, RingAttentionUpdate_910b_tiling_A1_sbh_fp32)
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
        kInputIrInstanceSbh, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeySbhFp32);
}

// A2: SBH fp16
TEST_F(RingAttentionUpdateTiling, RingAttentionUpdate_910b_tiling_A2_sbh_fp16)
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
        kInputIrInstanceSbh, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeySbhFp16);
}

// A3: SBH bf16
TEST_F(RingAttentionUpdateTiling, RingAttentionUpdate_910b_tiling_A3_sbh_bf16)
{
    struct RingAttentionUpdateCompileInfo compileInfo {};
    gert::TilingContextPara para(
        "RingAttentionUpdate",
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
            {"input_softmax_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
        },
        kInputIrInstanceSbh, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeySbhBf16);
}

// A4: TND fp16 with actual_seq_qlen
TEST_F(RingAttentionUpdateTiling, RingAttentionUpdate_910b_tiling_A4_tnd_fp16)
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
        kInputIrInstanceTnd, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyTndFp16);
}

// E1: TND headDim not aligned to 64
TEST_F(RingAttentionUpdateTiling, RingAttentionUpdate_910b_tiling_E1_tnd_head_dim_misaligned)
{
    struct RingAttentionUpdateCompileInfo compileInfo {};
    int64_t actSeqQlen[] = {500,  1000, 1500, 2100, 2700, 3300, 3900, 4500, 5100,
                            5700, 6300, 6900, 7500, 8100, 8700, 9300, 9900, 12000};
    gert::TilingContextPara para(
        "RingAttentionUpdate",
        {
            {{{12000, 2, 100}, {12000, 2, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12000, 2, 100}, {12000, 2, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{18}, {18}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQlen},
        },
        {
            {{{12000, 2, 100}, {12000, 2, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12000, 2, 8}, {12000, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"input_softmax_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        },
        kInputIrInstanceTnd, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E2: softmax last dim != 8
TEST_F(RingAttentionUpdateTiling, RingAttentionUpdate_910b_tiling_E2_invalid_softmax_tail)
{
    struct RingAttentionUpdateCompileInfo compileInfo {};
    gert::TilingContextPara para(
        "RingAttentionUpdate",
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 4}, {2, 12, 1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 4}, {2, 12, 1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 4}, {2, 12, 1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 4}, {2, 12, 1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 4}, {2, 12, 1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 4}, {2, 12, 1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
            {"input_softmax_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
        },
        kInputIrInstanceSbh, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E3: attn/softmax shape mismatch
TEST_F(RingAttentionUpdateTiling, RingAttentionUpdate_910b_tiling_E3_attn_softmax_mismatch)
{
    struct RingAttentionUpdateCompileInfo compileInfo {};
    gert::TilingContextPara para(
        "RingAttentionUpdate",
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 512, 8}, {2, 12, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
        kInputIrInstanceSbh, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E4: unsupported attn dtype
TEST_F(RingAttentionUpdateTiling, RingAttentionUpdate_910b_tiling_E4_invalid_attn_dtype)
{
    struct RingAttentionUpdateCompileInfo compileInfo {};
    gert::TilingContextPara para(
        "RingAttentionUpdate",
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 2, 1536}, {1024, 2, 1536}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 12, 1024, 8}, {2, 12, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
            {"input_softmax_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
        },
        kInputIrInstanceSbh, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
