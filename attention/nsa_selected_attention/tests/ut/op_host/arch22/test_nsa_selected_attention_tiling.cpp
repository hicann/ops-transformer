/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_nsa_selected_attention_tiling.cpp
 * \brief NsaSelectedAttention tiling-only utest cases.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/nsa_selected_attention_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
using namespace std;

class NsaSelectedAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaSelectedAttentionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaSelectedAttentionTiling TearDown" << std::endl;
    }
};

namespace {
constexpr uint32_t kAivNum = 48U;
constexpr uint32_t kAicNum = 24U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint64_t kL1Size = 524288ULL;
constexpr uint64_t kL0cSize = 131072ULL;
constexpr uint64_t kL2CacheSize = 33554432ULL;

inline optiling::NsaSelectedAttentionCompileInfo MakeCompileInfo()
{
    return optiling::NsaSelectedAttentionCompileInfo{kAivNum, kAicNum, kUbSize, kL1Size, kL0cSize, kL2CacheSize};
}

} // namespace

/* ============================================================
 * Group A1: tk=0 (mode=0) + fp16 + small TND
 * B=2, N2=4, G=2 (N1=8), D=192, D2=128, S1=S2=256
 * ============================================================ */
TEST_F(NsaSelectedAttentionTiling, A1_mode0_TND_fp16)
{
    int64_t actSeqQ[] = {256, 512};
    int64_t actSeqKv[] = {256, 512};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaSelectedAttention",
        {
            {{{512, 8, 192}, {512, 8, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query  T1=512, N1=8, D=192
            {{{512, 4, 192}, {512, 4, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key    T2=512, N2=4, D=192
            {{{512, 4, 128}, {512, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value  T2=512, N2=4, D2=128
            {{{512, 4, 16}, {512, 4, 16}}, ge::DT_INT32, ge::FORMAT_ND},     // topk_indices T1, N2, K
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},      // atten_mask
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},        // actual_seq_qlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqKv},       // actual_seq_kvlen
        },
        {
            {{{512, 8, 8}, {512, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},       // softmax_max
            {{{512, 8, 8}, {512, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},       // softmax_sum
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attention_out
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"selected_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"selected_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        },
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, /*tilingKey*/ 0);
}

/* ============================================================
 * Group A2: tk=0 (mode=0) + bf16
 * ============================================================ */
TEST_F(NsaSelectedAttentionTiling, A2_mode0_TND_bf16)
{
    int64_t actSeqQ[] = {512, 1024};
    int64_t actSeqKv[] = {512, 1024};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaSelectedAttention",
                                 {
                                     {{{1024, 16, 192}, {1024, 16, 192}}, ge::DT_BF16, ge::FORMAT_ND}, // query  N1=16
                                     {{{1024, 8, 192}, {1024, 8, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // key    N2=8
                                     {{{1024, 8, 128}, {1024, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // value
                                     {{{1024, 8, 16}, {1024, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND},    // topk_indices
                                     {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},       // atten_mask
                                     {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},
                                     {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqKv},
                                 },
                                 {
                                     {{{1024, 16, 8}, {1024, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1024, 16, 8}, {1024, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1024, 16, 128}, {1024, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                     {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                     {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"selected_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                     {"selected_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                 },
                                 &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, /*tilingKey*/ 0);
}

/* ============================================================
 * Group A3: tk=1 (mode=2 LEFT_UP_CAUSAL) + bf16
 * ============================================================ */
TEST_F(NsaSelectedAttentionTiling, A3_mode2_TND_bf16)
{
    int64_t actSeqQ[] = {512, 512};
    int64_t actSeqKv[] = {512, 512};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaSelectedAttention",
                                 {
                                     {{{1024, 32, 192}, {1024, 32, 192}}, ge::DT_BF16, ge::FORMAT_ND}, // query  N1=32
                                     {{{1024, 8, 192}, {1024, 8, 192}}, ge::DT_BF16, ge::FORMAT_ND},   // key    N2=8
                                     {{{1024, 8, 128}, {1024, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // value
                                     {{{1024, 8, 16}, {1024, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},
                                     {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqKv},
                                 },
                                 {
                                     {{{1024, 32, 8}, {1024, 32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1024, 32, 8}, {1024, 32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1024, 32, 128}, {1024, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                     {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                     {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"selected_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                     {"selected_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                 },
                                 &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, /*tilingKey*/ 1);
}

/* ============================================================
 * Group E1: unsupported sparseMode=3 -> tiling fail
 * ============================================================ */
TEST_F(NsaSelectedAttentionTiling, E1_mode3_unsupported_fail)
{
    int64_t actSeqQ[] = {512, 512};
    int64_t actSeqKv[] = {512, 512};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaSelectedAttention",
                                 {
                                     {{{1024, 16, 192}, {1024, 16, 192}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{1024, 8, 192}, {1024, 8, 192}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{1024, 8, 128}, {1024, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{1024, 8, 16}, {1024, 8, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},
                                     {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqKv},
                                 },
                                 {
                                     {{{1024, 16, 8}, {1024, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1024, 16, 8}, {1024, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1024, 16, 128}, {1024, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                     {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                     {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}, // unsupported
                                     {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"selected_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                     {"selected_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                 },
                                 &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

/* ============================================================
 * Group H1: selectedBlockSize=8 (not multiple of 16) -> fail
 * ============================================================ */
TEST_F(NsaSelectedAttentionTiling, H1_blkSize_not_aligned_fail)
{
    int64_t actSeqQ[] = {256, 256};
    int64_t actSeqKv[] = {256, 256};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaSelectedAttention",
        {
            {{{512, 8, 192}, {512, 8, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4, 192}, {512, 4, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4, 128}, {512, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4, 16}, {512, 4, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqKv},
        },
        {
            {{{512, 8, 8}, {512, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 8, 8}, {512, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"selected_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}, // not 16-aligned
            {"selected_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        },
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

/* ============================================================
 * Group H2: selectedBlockSize=256 (>128) -> fail
 * ============================================================ */
TEST_F(NsaSelectedAttentionTiling, H2_blkSize_oversize_fail)
{
    int64_t actSeqQ[] = {256, 256};
    int64_t actSeqKv[] = {256, 256};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaSelectedAttention",
        {
            {{{512, 8, 192}, {512, 8, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4, 192}, {512, 4, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4, 128}, {512, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4, 16}, {512, 4, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqKv},
        },
        {
            {{{512, 8, 8}, {512, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 8, 8}, {512, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{512, 8, 128}, {512, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"selected_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, // > 128
            {"selected_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        },
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
