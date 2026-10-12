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
 * \file test_nsa_compress_attention_tiling.cpp
 * \brief NsaCompressAttention tiling-only utest cases.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/nsa_compress_attention_tiling.h"
#include "../../../../op_host/nsa_compress_attention_tiling_common.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
using namespace std;

class NsaCompressAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressAttentionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressAttentionTiling TearDown" << std::endl;
    }
};

namespace {
constexpr uint32_t kAivNum = 48U;
constexpr uint32_t kAicNum = 24U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint64_t kL1Size = 524288ULL;
constexpr uint64_t kL0cSize = 131072ULL;
constexpr uint64_t kL2CacheSize = 33554432ULL;

inline optiling::NsaCompressAttentionCompileInfo MakeCompileInfo()
{
    return optiling::NsaCompressAttentionCompileInfo{kAivNum, kAicNum, kUbSize, kL1Size, kL0cSize, kL2CacheSize};
}

constexpr int64_t kT1 = 65536;
constexpr int64_t kT2 = 4096;
constexpr int64_t kN2 = 4;
constexpr int64_t kG = 16;
constexpr int64_t kHeadNum = kN2 * kG; // 64
constexpr int64_t kD1 = 192;
constexpr int64_t kD2 = 128;
constexpr int64_t kS1 = 65536;
constexpr int64_t kS2 = 4096;
constexpr int64_t kCompressBlockSize = 32;
constexpr int64_t kCompressStride = 16;
constexpr int64_t kSelectedBlockSize = 64;
constexpr int64_t kSelectedBlockCount = 16;

} // namespace

/* ============================================================
 * Group A1: TND + fp16 + atten_mask + topk_mask (sparse=1)
 *   tilingKey == 10000000000000001124UL
 * ============================================================ */
TEST_F(NsaCompressAttentionTiling, A1_TND_fp16_with_masks)
{
    int64_t actSeqQ[] = {kT1};
    int64_t actCmpSeqKv[] = {kT2};
    int64_t actSelSeqKv[] = {kT1 / kSelectedBlockSize};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaCompressAttention",
        {
            // query  (N2=4, T1=65536, G=16, D1=192)
            {{{kN2, kT1, kG, kD1}, {kN2, kT1, kG, kD1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // key    (T2=4096, N2=4, D1=192)
            {{{kT2, kN2, kD1}, {kT2, kN2, kD1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // value  (T2=4096, N2=4, D2=128)
            {{{kT2, kN2, kD2}, {kT2, kN2, kD2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask (S1, S2)
            {{{kS1, kS2}, {kS1, kS2}}, ge::DT_BOOL, ge::FORMAT_ND},
            // actual_seq_qlen (data-dependency, value must be filled)
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},
            // actual_cmp_seq_kvlen
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actCmpSeqKv},
            // actual_sel_seq_kvlen
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actSelSeqKv},
            // topk_mask (S1, S2)
            {{{kS1, kS2}, {kS1, kS2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            // softmax_max     (T1, N2*G, 8)
            {{{kT1, kHeadNum, 8}, {kT1, kHeadNum, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum     (T1, N2*G, 8)
            {{{kT1, kHeadNum, 8}, {kT1, kHeadNum, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_out   (N2, T1, G, D2)
            {{{kN2, kT1, kG, kD2}, {kN2, kT1, kG, kD2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // topk_indices_out (N2, T1, selBlkCnt)
            {{{kN2, kT1, kSelectedBlockCount}, {kN2, kT1, kSelectedBlockCount}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kHeadNum)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressBlockSize)},
            {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressStride)},
            {"select_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockSize)},
            {"select_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockCount)},
        },
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, /*tilingKey*/ 10000000000000001124UL);
}

/* ============================================================
 * Group E1: layout BSH -> CheckParams reject
 * ============================================================ */
TEST_F(NsaCompressAttentionTiling, E1_invalid_layout_bsh_fail)
{
    int64_t actSeqQ[] = {kT1};
    int64_t actCmpSeqKv[] = {kT2};
    int64_t actSelSeqKv[] = {kT1 / kSelectedBlockSize};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaCompressAttention",
        {
            {{{kN2, kT1, kG, kD1}, {kN2, kT1, kG, kD1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kT2, kN2, kD1}, {kT2, kN2, kD1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kT2, kN2, kD2}, {kT2, kN2, kD2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kS1, kS2}, {kS1, kS2}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actCmpSeqKv},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actSelSeqKv},
            {{{kS1, kS2}, {kS1, kS2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{kT1, kHeadNum, 8}, {kT1, kHeadNum, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{kT1, kHeadNum, 8}, {kT1, kHeadNum, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{kN2, kT1, kG, kD2}, {kN2, kT1, kG, kD2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kN2, kT1, kSelectedBlockCount}, {kN2, kT1, kSelectedBlockCount}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kHeadNum)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressBlockSize)},
            {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressStride)},
            {"select_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockSize)},
            {"select_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockCount)},
        },
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

/* ============================================================
 * Group E2: query.dim(3) != key.dim(2)  -> CheckParams reject
 *   (D of query and key must agree, see nsa_compress_attention_tiling.cpp:50)
 * ============================================================ */
TEST_F(NsaCompressAttentionTiling, E2_query_d_mismatch_key_d_fail)
{
    constexpr int64_t kBadD1Query = 192;
    constexpr int64_t kBadD1Key = 128; // mismatch on purpose
    int64_t actSeqQ[] = {kT1};
    int64_t actCmpSeqKv[] = {kT2};
    int64_t actSelSeqKv[] = {kT1 / kSelectedBlockSize};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaCompressAttention",
        {
            {{{kN2, kT1, kG, kBadD1Query}, {kN2, kT1, kG, kBadD1Query}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kT2, kN2, kBadD1Key}, {kT2, kN2, kBadD1Key}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kT2, kN2, kD2}, {kT2, kN2, kD2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kS1, kS2}, {kS1, kS2}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actCmpSeqKv},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actSelSeqKv},
            {{{kS1, kS2}, {kS1, kS2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{kT1, kHeadNum, 8}, {kT1, kHeadNum, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{kT1, kHeadNum, 8}, {kT1, kHeadNum, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{kN2, kT1, kG, kD2}, {kN2, kT1, kG, kD2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kN2, kT1, kSelectedBlockCount}, {kN2, kT1, kSelectedBlockCount}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kHeadNum)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressBlockSize)},
            {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressStride)},
            {"select_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockSize)},
            {"select_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockCount)},
        },
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

/* ============================================================
 * Group E3: key.dim(2) < value.dim(2) -> CheckParams reject
 *   (kD must be >= vD, see nsa_compress_attention_tiling.cpp:54)
 * ============================================================ */
TEST_F(NsaCompressAttentionTiling, E3_key_d_lt_value_d_fail)
{
    constexpr int64_t kSmallKD = 64;
    constexpr int64_t kLargeVD = 128;
    int64_t actSeqQ[] = {kT1};
    int64_t actCmpSeqKv[] = {kT2};
    int64_t actSelSeqKv[] = {kT1 / kSelectedBlockSize};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaCompressAttention",
        {
            {{{kN2, kT1, kG, kSmallKD}, {kN2, kT1, kG, kSmallKD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kT2, kN2, kSmallKD}, {kT2, kN2, kSmallKD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kT2, kN2, kLargeVD}, {kT2, kN2, kLargeVD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kS1, kS2}, {kS1, kS2}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actCmpSeqKv},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actSelSeqKv},
            {{{kS1, kS2}, {kS1, kS2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{kT1, kHeadNum, 8}, {kT1, kHeadNum, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{kT1, kHeadNum, 8}, {kT1, kHeadNum, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{kN2, kT1, kG, kLargeVD}, {kN2, kT1, kG, kLargeVD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kN2, kT1, kSelectedBlockCount}, {kN2, kT1, kSelectedBlockCount}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kHeadNum)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressBlockSize)},
            {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressStride)},
            {"select_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockSize)},
            {"select_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockCount)},
        },
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

/* ============================================================
 * Group E4: atten_mask dtype FP16 -> AnalyzeOptionalInput reject
 *   (only ge::DT_BOOL / ge::DT_UINT8 are accepted for the mask.)
 * ============================================================ */
TEST_F(NsaCompressAttentionTiling, E4_atten_mask_invalid_dtype_fail)
{
    int64_t actSeqQ[] = {kT1};
    int64_t actCmpSeqKv[] = {kT2};
    int64_t actSelSeqKv[] = {kT1 / kSelectedBlockSize};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaCompressAttention",
        {
            {{{kN2, kT1, kG, kD1}, {kN2, kT1, kG, kD1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kT2, kN2, kD1}, {kT2, kN2, kD1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kT2, kN2, kD2}, {kT2, kN2, kD2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask dtype FP16 (invalid)
            {{{kS1, kS2}, {kS1, kS2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqQ},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actCmpSeqKv},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actSelSeqKv},
            {{{kS1, kS2}, {kS1, kS2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{kT1, kHeadNum, 8}, {kT1, kHeadNum, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{kT1, kHeadNum, 8}, {kT1, kHeadNum, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{kN2, kT1, kG, kD2}, {kN2, kT1, kG, kD2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kN2, kT1, kSelectedBlockCount}, {kN2, kT1, kSelectedBlockCount}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kHeadNum)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressBlockSize)},
            {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressStride)},
            {"select_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockSize)},
            {"select_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockCount)},
        },
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
