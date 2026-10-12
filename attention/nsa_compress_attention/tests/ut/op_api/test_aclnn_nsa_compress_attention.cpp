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
 * \file test_aclnn_nsa_compress_attention.cpp
 * \brief NsaCompressAttention aclnn op_api UT.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_nsa_compress_attention.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "opdev/op_errno.h"

// The legacy VarLen entry is not declared in the public header.
extern "C" aclnnStatus aclnnNsaCompressAttentionVarLenScoreGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attenMaskOptional,
    const aclTensor* topkMaskOptional, const aclIntArray* actualSeqQLenOptional,
    const aclIntArray* actualCmpSeqKvLenOptional, const aclIntArray* actualSelSeqKvLenOptional, double scaleValue,
    int64_t headNum, char* inputLayout, int64_t sparseMode, int64_t compressBlockSize, int64_t compressStride,
    int64_t selectBlockSize, int64_t selectBlockCount, const aclTensor* softmaxMaxOut, const aclTensor* softmaxSumOut,
    const aclTensor* attentionOutOut, const aclTensor* topkIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

using namespace std;
using namespace op;

namespace {
constexpr int64_t kT = 128;
constexpr int64_t kN1 = 4; // query head_num
constexpr int64_t kN2 = 2; // kv head_num   (G = kN1 / kN2 = 2)
constexpr int64_t kQueryD = 128;
constexpr int64_t kValueD = 128;
constexpr int64_t kCompressBlockSize = 32;
constexpr int64_t kCompressStride = 16;
constexpr int64_t kSelectedBlockSize = 64;
constexpr int64_t kSelectedBlockCount = 4; // (kT / kSelectedBlockSize) == 2 valid values
} // namespace

class nsa_compress_attention_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "nsa_compress_attention_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "nsa_compress_attention_opapi_ut TearDown" << endl;
    }
};

TEST_F(nsa_compress_attention_opapi_ut, nsa_compress_attention_aclnn_0)
{
    // Inputs: required tensors (TND, fp16). All masks omitted.
    auto tensorQ = TensorDesc({kT, kN1, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({kT, kN2, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({kT, kN2, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto actualSeqQLen = IntArrayDesc(vector<int64_t>{kT});
    auto actualCmpSeqKvLen = IntArrayDesc(vector<int64_t>{kT});
    auto actualSelSeqKvLen = IntArrayDesc(vector<int64_t>{kT / kSelectedBlockSize});

    const double scaleValue = 0.088388;
    const int64_t headNum = kN1;
    char layout[] = "TND";
    const int64_t sparseMode = 0;

    // Outputs: shape strictly follows the post-Postprocess layout:
    //   softmax_max / sum : (T, N1, 8)
    //   attention_out     : (T, N1, valueD)
    //   topk_indices_out  : (T, N2, selectBlockCount)
    auto tensorSoftmaxMax = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxSum = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorAttentionOut = TensorDesc({kT, kN1, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorTopkIndicesOut = TensorDesc({kT, kN2, kSelectedBlockCount}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaCompressAttention,
                        INPUT(tensorQ, tensorK, tensorV,
                              nullptr, // attenMaskOptional
                              nullptr, // topkMaskOptional
                              actualSeqQLen, actualCmpSeqKvLen, actualSelSeqKvLen, scaleValue, headNum, layout,
                              sparseMode, kCompressBlockSize, kCompressStride, kSelectedBlockSize, kSelectedBlockCount),
                        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorAttentionOut, tensorTopkIndicesOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(nsa_compress_attention_opapi_ut, nsa_compress_attention_aclnn_bf16)
{
    auto tensorQ = TensorDesc({kT, kN1, kQueryD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({kT, kN2, kQueryD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({kT, kN2, kValueD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto actualSeqQLen = IntArrayDesc(vector<int64_t>{kT});
    auto actualCmpSeqKvLen = IntArrayDesc(vector<int64_t>{kT});
    auto actualSelSeqKvLen = IntArrayDesc(vector<int64_t>{kT / kSelectedBlockSize});

    const double scaleValue = 0.088388;
    const int64_t headNum = kN1;
    char layout[] = "TND";
    const int64_t sparseMode = 0;

    auto tensorSoftmaxMax = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxSum = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorAttentionOut = TensorDesc({kT, kN1, kValueD}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorTopkIndicesOut = TensorDesc({kT, kN2, kSelectedBlockCount}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaCompressAttention,
                        INPUT(tensorQ, tensorK, tensorV, nullptr, nullptr, actualSeqQLen, actualCmpSeqKvLen,
                              actualSelSeqKvLen, scaleValue, headNum, layout, sparseMode, kCompressBlockSize,
                              kCompressStride, kSelectedBlockSize, kSelectedBlockCount),
                        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorAttentionOut, tensorTopkIndicesOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(nsa_compress_attention_opapi_ut, nsa_compress_attention_aclnn_varlen)
{
    // Multi-batch: actualSeqQLen / actualCmpSeqKvLen / actualSelSeqKvLen are prefix-summed.
    // Two equal-length segments of 64 tokens => total T = 128.
    auto tensorQ = TensorDesc({kT, kN1, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({kT, kN2, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({kT, kN2, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto actualSeqQLen = IntArrayDesc(vector<int64_t>{64, 128});
    auto actualCmpSeqKvLen = IntArrayDesc(vector<int64_t>{64, 128});
    auto actualSelSeqKvLen = IntArrayDesc(vector<int64_t>{64 / kSelectedBlockSize, kT / kSelectedBlockSize});

    const double scaleValue = 0.088388;
    const int64_t headNum = kN1;
    char layout[] = "TND";
    const int64_t sparseMode = 0;

    auto tensorSoftmaxMax = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxSum = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorAttentionOut = TensorDesc({kT, kN1, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorTopkIndicesOut = TensorDesc({kT, kN2, kSelectedBlockCount}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaCompressAttention,
                        INPUT(tensorQ, tensorK, tensorV,
                              nullptr, // attenMaskOptional
                              nullptr, // topkMaskOptional
                              actualSeqQLen, actualCmpSeqKvLen, actualSelSeqKvLen, scaleValue, headNum, layout,
                              sparseMode, kCompressBlockSize, kCompressStride, kSelectedBlockSize, kSelectedBlockCount),
                        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorAttentionOut, tensorTopkIndicesOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(nsa_compress_attention_opapi_ut, nsa_compress_attention_aclnn_with_topk_mask)
{
    // topk_mask is an OPTIONAL bool/uint8 tensor of shape (S1, S2). Verify it goes
    auto tensorQ = TensorDesc({kT, kN1, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({kT, kN2, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({kT, kN2, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorTopkMask = TensorDesc({kT, kT}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0});

    auto actualSeqQLen = IntArrayDesc(vector<int64_t>{kT});
    auto actualCmpSeqKvLen = IntArrayDesc(vector<int64_t>{kT});
    auto actualSelSeqKvLen = IntArrayDesc(vector<int64_t>{kT / kSelectedBlockSize});

    const double scaleValue = 0.088388;
    const int64_t headNum = kN1;
    char layout[] = "TND";
    const int64_t sparseMode = 0;

    auto tensorSoftmaxMax = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxSum = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorAttentionOut = TensorDesc({kT, kN1, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorTopkIndicesOut = TensorDesc({kT, kN2, kSelectedBlockCount}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnNsaCompressAttention,
        INPUT(tensorQ, tensorK, tensorV,
              nullptr, // attenMaskOptional
              tensorTopkMask, actualSeqQLen, actualCmpSeqKvLen, actualSelSeqKvLen, scaleValue, headNum, layout,
              sparseMode, kCompressBlockSize, kCompressStride, kSelectedBlockSize, kSelectedBlockCount),
        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorAttentionOut, tensorTopkIndicesOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// any layout != "TND" (returns ACLNN_ERR_PARAM_INVALID at aclnn entry).
TEST_F(nsa_compress_attention_opapi_ut, nsa_compress_attention_aclnn_invalid_layout_bsh_returns_error)
{
    auto tensorQ = TensorDesc({kT, kN1, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({kT, kN2, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({kT, kN2, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto actualSeqQLen = IntArrayDesc(vector<int64_t>{kT});
    auto actualCmpSeqKvLen = IntArrayDesc(vector<int64_t>{kT});
    auto actualSelSeqKvLen = IntArrayDesc(vector<int64_t>{kT / kSelectedBlockSize});

    const double scaleValue = 0.088388;
    const int64_t headNum = kN1;
    char layout[] = "BSH"; // not TND -> aclnn must reject
    const int64_t sparseMode = 0;

    auto tensorSoftmaxMax = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxSum = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorAttentionOut = TensorDesc({kT, kN1, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorTopkIndicesOut = TensorDesc({kT, kN2, kSelectedBlockCount}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaCompressAttention,
                        INPUT(tensorQ, tensorK, tensorV, nullptr, nullptr, actualSeqQLen, actualCmpSeqKvLen,
                              actualSelSeqKvLen, scaleValue, headNum, layout, sparseMode, kCompressBlockSize,
                              kCompressStride, kSelectedBlockSize, kSelectedBlockCount),
                        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorAttentionOut, tensorTopkIndicesOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

// Exercise both entry points with nonempty outputs so shape validation is reached.
TEST_F(nsa_compress_attention_opapi_ut, invalid_input_shapes_return_param_invalid)
{
    const vector<array<vector<int64_t>, 3>> shapes = {
        {{{kT, kN1, kQueryD}, {kT, kN2, kQueryD}, {0, kN2, kValueD}}},
        {{{kT, kN1, kQueryD}, {kT, kN2, kQueryD}, {kT, 0, kValueD}}},
        {{{kT, kN1, kQueryD}, {kT, kN2, 0}, {kT, kN2, kValueD}}},
        {{{kT, kN1, kQueryD}, {kT, kN2, kQueryD}, {kT + 1, kN2, kValueD}}},
        {{{kT, kN1, kQueryD}, {kT, kN2, kQueryD}, {kT, kN2 + 1, kValueD}}},
        {{{kT, kN1, kQueryD}, {kT, kN2, kQueryD + 16}, {kT, kN2, kValueD}}},
        {{{kT, kN1, kQueryD}, {kT, kN2, kQueryD}, {kT, kN2, kQueryD + 16}}},
        {{{kT, 1, kQueryD}, {kT, kN2, kQueryD}, {kT, kN2, kValueD}}},
        {{{kT, 3, kQueryD}, {kT, kN2, kQueryD}, {kT, kN2, kValueD}}},
        {{{kT, kN1, kQueryD}, {kT, 0, kQueryD}, {kT, kN2, kValueD}}},
        {{{kT, kN1}, {kT, kN2, kQueryD}, {kT, kN2, kValueD}}},
        {{{kT, kN1, kQueryD}, {kT}, {kT, kN2, kValueD}}},
        {{{kT, kN1, kQueryD}, {kT, kN2, kQueryD}, {kT, kN2}}},
        {{{kT, kN1, kQueryD, 1}, {kT, kN2, kQueryD}, {kT, kN2, kValueD}}}};
    for (auto entry :
         {aclnnNsaCompressAttentionGetWorkspaceSize, aclnnNsaCompressAttentionVarLenScoreGetWorkspaceSize}) {
        for (size_t i = 0; i < shapes.size(); ++i) {
            SCOPED_TRACE(i);
            auto q = TensorDesc(shapes[i][0], ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
            auto k = TensorDesc(shapes[i][1], ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
            auto v = TensorDesc(shapes[i][2], ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
            auto max = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
            auto sum = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
            auto out = TensorDesc({kT, kN1, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
            auto indices = TensorDesc({kT, kN2, kSelectedBlockCount}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
            char layout[] = "TND";
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            EXPECT_EQ(entry(q.get(), k.get(), v.get(), nullptr, nullptr, nullptr, nullptr, nullptr, 0.088388, kN1,
                            layout, 0, kCompressBlockSize, kCompressStride, kSelectedBlockSize, kSelectedBlockCount,
                            max.get(), sum.get(), out.get(), indices.get(), &workspaceSize, &executor),
                      ACLNN_ERR_PARAM_INVALID);
        }
    }
}

TEST_F(nsa_compress_attention_opapi_ut, required_null_parameters_return_param_nullptr)
{
    auto tensor = TensorDesc({kT, kN1, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    for (auto entry :
         {aclnnNsaCompressAttentionGetWorkspaceSize, aclnnNsaCompressAttentionVarLenScoreGetWorkspaceSize}) {
        for (size_t i = 0; i < 10; ++i) {
            SCOPED_TRACE(i);
            array<const aclTensor*, 7> tensors;
            tensors.fill(tensor.get());
            if (i < tensors.size()) {
                tensors[i] = nullptr;
            }
            char layout[] = "TND";
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            EXPECT_EQ(entry(tensors[0], tensors[1], tensors[2], nullptr, nullptr, nullptr, nullptr, nullptr, 0.088388,
                            kN1, i == 7 ? nullptr : layout, 0, kCompressBlockSize, kCompressStride, kSelectedBlockSize,
                            kSelectedBlockCount, tensors[3], tensors[4], tensors[5], tensors[6],
                            i == 8 ? nullptr : &workspaceSize, i == 9 ? nullptr : &executor),
                      ACLNN_ERR_PARAM_NULLPTR);
        }
    }
}
