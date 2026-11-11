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
 * \file test_aclnn_generic_block_sparse_attention_grad_qsag.cpp
 * \brief ACLNN front end for QSAG (layoutSparsePattern=0): the shared restrictions are enforced
 *        before any executor work is built.
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "acl/acl.h"
#include "opdev/platform.h"

#include "attention/generic_block_sparse_attention_grad/op_host/op_api/aclnn_generic_block_sparse_attention_grad.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kNq = 8;
constexpr int64_t kNk = 2;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kSq = 16;
constexpr int64_t kSk = 32;
constexpr int64_t kTopK = 4;
constexpr int64_t kQcap = kSq;
constexpr double kScale = 0.125;

const vector<int64_t> kBlockShape = {1, 64};
} // namespace

class aclnn_generic_block_sparse_attention_grad_qsag_ut : public testing::Test {};

TEST_F(aclnn_generic_block_sparse_attention_grad_qsag_ut, qsag_tnd_fp16_accepts_null_metadata)
{
    auto query = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dout = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto attentionOut = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse = TensorDesc({kSq, kNq, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kBatch, kNk, kQcap, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kBatch, kNk, kQcap}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSq});
    auto cuSeqLengthsKv = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSk});
    auto blockShape = IntArrayDesc(kBlockShape);
    char layoutQ[] = "TND";
    char layoutKv[] = "TND";
    auto dQuery = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dKey = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dValue = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttentionGrad,
                        INPUT(query, key, value, dout, attentionOut, lse, sparseBlockIdx, sparseBlockCount,
                              nullptr,        // metadataOptional: QSAG requires null
                              nullptr,        // attenMaskOptional: QSAG requires null
                              cuSeqLengthsQ,  // cuSeqLengthsQOptional
                              cuSeqLengthsKv, // cuSeqLengthsKvOptional
                              nullptr,        // sequsedQOptional
                              nullptr,        // sequsedKvOptional
                              blockShape, layoutQ, layoutKv,
                              0,      // layoutSparsePattern: 0 selects QSAG
                              kScale, // scaleValue
                              1,      // maskType
                              0,      // softmaxPrecision
                              -1,     // winLeft
                              -1,     // winRight
                              1,      // residualBlockMode: QSAG requires 1
                              false), // isConsistentTopk
                        OUTPUT(dQuery, dKey, dValue));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(aclnn_generic_block_sparse_attention_grad_qsag_ut, qsag_rejects_non_null_metadata)
{
    auto query = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dout = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto attentionOut = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse = TensorDesc({kSq, kNq, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kBatch, kNk, kQcap, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kBatch, kNk, kQcap}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSq});
    auto cuSeqLengthsKv = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSk});
    auto blockShape = IntArrayDesc(kBlockShape);
    char layoutQ[] = "TND";
    char layoutKv[] = "TND";
    auto dQuery = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dKey = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dValue = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttentionGrad,
                        INPUT(query, key, value, dout, attentionOut, lse, sparseBlockIdx, sparseBlockCount, metadata,
                              nullptr, cuSeqLengthsQ, cuSeqLengthsKv, nullptr, nullptr, blockShape, layoutQ, layoutKv,
                              0, kScale, 1, 0, -1, -1, 1, false),
                        OUTPUT(dQuery, dKey, dValue));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnn_generic_block_sparse_attention_grad_qsag_ut, rejects_non_finite_scale_pattern0)
{
    auto query = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dout = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto attentionOut = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse = TensorDesc({kSq, kNq, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kBatch, kNk, kQcap, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kBatch, kNk, kQcap}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSq});
    auto cuSeqLengthsKv = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSk});
    auto blockShape = IntArrayDesc(kBlockShape);
    char layoutQ[] = "TND";
    char layoutKv[] = "TND";
    auto dQuery = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dKey = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dValue = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    const double nanScale = std::numeric_limits<double>::quiet_NaN();

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttentionGrad,
                        INPUT(query, key, value, dout, attentionOut, lse, sparseBlockIdx, sparseBlockCount, nullptr,
                              nullptr, cuSeqLengthsQ, cuSeqLengthsKv, nullptr, nullptr, blockShape, layoutQ, layoutKv,
                              0, nanScale, 1, 0, -1, -1, 1, false),
                        OUTPUT(dQuery, dKey, dValue));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnn_generic_block_sparse_attention_grad_qsag_ut, rejects_non_finite_scale_pattern1)
{
    // Common finite-scale check runs before path-specific Validate; pattern=1 still returns 161002.
    auto query = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dout = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto attentionOut = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse = TensorDesc({kSq, kNq, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kBatch, kNk, kQcap, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kBatch, kNk, kQcap}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSq});
    auto cuSeqLengthsKv = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSk});
    auto blockShape = IntArrayDesc(kBlockShape);
    char layoutQ[] = "TND";
    char layoutKv[] = "TND";
    auto dQuery = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dKey = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dValue = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    const double infScale = std::numeric_limits<double>::infinity();

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttentionGrad,
                        INPUT(query, key, value, dout, attentionOut, lse, sparseBlockIdx, sparseBlockCount, metadata,
                              nullptr, cuSeqLengthsQ, cuSeqLengthsKv, nullptr, nullptr, blockShape, layoutQ, layoutKv,
                              1, infScale, 1, 0, -1, -1, 0, false),
                        OUTPUT(dQuery, dKey, dValue));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnn_generic_block_sparse_attention_grad_qsag_ut, qsag_accepts_consistent_topk_true)
{
    auto query = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dout = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto attentionOut = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse = TensorDesc({kSq, kNq, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kBatch, kNk, kQcap, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kBatch, kNk, kQcap}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSq});
    auto cuSeqLengthsKv = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSk});
    auto blockShape = IntArrayDesc(kBlockShape);
    char layoutQ[] = "TND";
    char layoutKv[] = "TND";
    auto dQuery = TensorDesc({kSq, kNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dKey = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dValue = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttentionGrad,
                        INPUT(query, key, value, dout, attentionOut, lse, sparseBlockIdx, sparseBlockCount, nullptr,
                              nullptr, cuSeqLengthsQ, cuSeqLengthsKv, nullptr, nullptr, blockShape, layoutQ, layoutKv,
                              0, kScale, 1, 0, -1, -1, 1, true),
                        OUTPUT(dQuery, dKey, dValue));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(aclnn_generic_block_sparse_attention_grad_qsag_ut, qsag_rejects_indivisible_head_counts)
{
    constexpr int64_t kWideNq = 515;
    auto query = TensorDesc({kSq, kWideNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dout = TensorDesc({kSq, kWideNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto attentionOut = TensorDesc({kSq, kWideNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse = TensorDesc({kSq, kWideNq, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kBatch, kNk, kQcap, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kBatch, kNk, kQcap}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSq});
    auto cuSeqLengthsKv = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kSk});
    auto blockShape = IntArrayDesc(vector<int64_t>{1, 32});
    char layoutQ[] = "TND";
    char layoutKv[] = "TND";
    auto dQuery = TensorDesc({kSq, kWideNq, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dKey = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dValue = TensorDesc({kSk, kNk, kHeadDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttentionGrad,
                        INPUT(query, key, value, dout, attentionOut, lse, sparseBlockIdx, sparseBlockCount, nullptr,
                              nullptr, cuSeqLengthsQ, cuSeqLengthsKv, nullptr, nullptr, blockShape, layoutQ, layoutKv,
                              0, kScale, 1, 0, -1, -1, 1, false),
                        OUTPUT(dQuery, dKey, dValue));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
