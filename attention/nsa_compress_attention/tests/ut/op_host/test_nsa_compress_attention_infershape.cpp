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
 * \file test_nsa_compress_attention_infershape.cpp
 * \brief NsaCompressAttention infershape utest cases.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

namespace {
constexpr int64_t kT = 128;      // total token count
constexpr int64_t kN2 = 2;       // kv head_num
constexpr int64_t kG = 2;        // group size (N1 = kN2 * kG)
constexpr int64_t kQueryD = 192; // query / key head dim (D1)
constexpr int64_t kValueD = 128; // value head dim (D2, intentionally != kQueryD)
constexpr int64_t kCompressBlockSize = 32;
constexpr int64_t kCompressStride = 16;
constexpr int64_t kSelectedBlockSize = 64;
constexpr int64_t kSelectedBlockCount = 16;

using TensorDesc = gert::InfershapeContextPara::TensorDescription;
using OpAttr = gert::InfershapeContextPara::OpAttr;

std::vector<OpAttr> MakeAttrs(const std::string& layout)
{
    return {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
        {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kN2 * kG)},
        {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressBlockSize)},
        {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kCompressStride)},
        {"select_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockSize)},
        {"select_block_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSelectedBlockCount)},
    };
}

std::vector<TensorDesc> MakeInputs(ge::DataType dtype)
{
    return {
        // query: (N2, T, G, D1)
        {{{kN2, kT, kG, kQueryD}, {kN2, kT, kG, kQueryD}}, dtype, ge::FORMAT_ND},
        // key:   (T, N2, D1)
        {{{kT, kN2, kQueryD}, {kT, kN2, kQueryD}}, dtype, ge::FORMAT_ND},
        // value: (T, N2, D2)
        {{{kT, kN2, kValueD}, {kT, kN2, kValueD}}, dtype, ge::FORMAT_ND},
        // atten_mask (optional)
        {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        // actual_seq_qlen (optional)
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // actual_cmp_seq_kvlen (optional)
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // actual_sel_seq_kvlen (optional)
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // topk_mask (optional)
        {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
    };
}

std::vector<TensorDesc> MakeOutputs(ge::DataType outDtype)
{
    return {
        // softmax_max
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // softmax_sum
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // attention_out
        {{{}, {}}, outDtype, ge::FORMAT_ND},
        // topk_indices_out
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
    };
}

// Expected output shapes for the standard query (kN2, kT, kG, kQueryD) /
// value (kT, kN2, kValueD) layout.
const std::vector<std::vector<int64_t>> kExpectShapes = {
    {kT, kN2* kG, 8},               // softmax_max
    {kT, kN2* kG, 8},               // softmax_sum
    {kN2, kT, kG, kValueD},         // attention_out: query with dim(3) replaced by value.dim(2)
    {kN2, kT, kSelectedBlockCount}, // topk_indices_out
};
} // namespace

class NsaCompressAttentionProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressAttentionProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressAttentionProto TearDown" << std::endl;
    }
};

TEST_F(NsaCompressAttentionProto, nsa_compress_attention_infershape_tnd_basic_fp16)
{
    gert::InfershapeContextPara para("NsaCompressAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16),
                                     MakeAttrs("TND"));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kExpectShapes);
}

TEST_F(NsaCompressAttentionProto, nsa_compress_attention_infershape_tnd_basic_bf16)
{
    gert::InfershapeContextPara para("NsaCompressAttention", MakeInputs(ge::DT_BF16), MakeOutputs(ge::DT_BF16),
                                     MakeAttrs("TND"));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kExpectShapes);
}

TEST_F(NsaCompressAttentionProto, nsa_compress_attention_infershape_layout_lowercase)
{
    gert::InfershapeContextPara para("NsaCompressAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16),
                                     MakeAttrs("tnd"));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kExpectShapes);
}

TEST_F(NsaCompressAttentionProto, nsa_compress_attention_infershape_layout_mixed_case)
{
    gert::InfershapeContextPara para("NsaCompressAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16),
                                     MakeAttrs("Tnd"));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kExpectShapes);
}

TEST_F(NsaCompressAttentionProto, nsa_compress_attention_infershape_invalid_layout_bsh)
{
    // Anything but "TND" (case-insensitive) must fail.
    gert::InfershapeContextPara para("NsaCompressAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16),
                                     MakeAttrs("BSH"));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(NsaCompressAttentionProto, nsa_compress_attention_infershape_invalid_layout_empty)
{
    gert::InfershapeContextPara para("NsaCompressAttention", MakeInputs(ge::DT_FLOAT16), MakeOutputs(ge::DT_FLOAT16),
                                     MakeAttrs(""));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
