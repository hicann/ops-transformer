/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "opdev/platform.h"
#include <gtest/gtest.h>
#include "attention/generic_block_sparse_attention/op_host/op_api/aclnn_generic_block_sparse_attention.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;
using namespace op;

class aclnn_generic_block_sparse_attention_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "aclnn_generic_block_sparse_attention_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "aclnn_generic_block_sparse_attention_ut TearDown" << endl;
    }
};

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kS1 = 4;
constexpr int64_t kS2 = 256;
constexpr int64_t kN1 = 4;
constexpr int64_t kN2 = 1;
constexpr int64_t kD = 128;
constexpr int64_t kTopK = 2;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kBlockShapeX = 1;
constexpr int64_t kT = kBatch * kS1;
constexpr int64_t kMaxBlocks = (kS2 + kBlockSize - 1) / kBlockSize;
constexpr int64_t kTotalQBlocks = kT;
constexpr double kScale = 1.0 / sqrt(static_cast<double>(kD));
constexpr int64_t kLayoutSparse = 4; // KVN_TotalQB_KB

constexpr int64_t kA8C4QuantType = 8;

vector<int64_t> MakeDenseShape(const string& layout, int64_t seqLen, int64_t headNum)
{
    if (layout == "TND") {
        return {kBatch * seqLen, headNum, kD};
    }
    if (layout == "BNSD") {
        return {kBatch, headNum, seqLen, kD};
    }
    return {kBatch, seqLen, headNum, kD};
}

aclnnStatus QueryA8C4Workspace(string layoutQ, string layoutKv, int64_t qSeqLen, int64_t kvHeads)
{
    const int64_t qHeads = kN1 * kvHeads;
    const bool tndQ = layoutQ == "TND";
    const bool tndKv = layoutKv == "TND";
    const bool paged = layoutKv == "PA_BBND" || layoutKv == "PA_BNBD";
    const auto qShape = MakeDenseShape(layoutQ, qSeqLen, qHeads);
    auto kvShape = MakeDenseShape(layoutKv, kS2, kvHeads);
    if (layoutKv == "PA_BBND") {
        kvShape = {kMaxBlocks, kBlockSize, kvHeads, kD};
    } else if (layoutKv == "PA_BNBD") {
        kvShape = {kMaxBlocks, kvHeads, kBlockSize, kD};
    }
    auto scaleShape = kvShape;
    scaleShape.back() /= 16;

    auto query = TensorDesc(qShape, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    auto key = TensorDesc(kvShape, ACL_FLOAT4_E2M1, ACL_FORMAT_ND).ToAclType();
    auto kvScale = TensorDesc(scaleShape, ACL_BF16, ACL_FORMAT_ND).ToAclType();
    auto indices = TensorDesc({kvHeads, kBatch * qSeqLen, kTopK}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto counts = TensorDesc({kvHeads, kBatch * qSeqLen}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto cuQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    auto cuKv = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    auto usedQ = TensorDesc({kBatch}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto usedKv = TensorDesc({kBatch}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize}).ToAclType();
    auto output = TensorDesc(qShape, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();

    const auto previousSoc = op::GetCurrentPlatformInfo().GetSocVersion();
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnGenericBlockSparseAttentionGetWorkspaceSize(
        query.get(), key.get(), key.get(), indices.get(), counts.get(), metadata.get(), nullptr, nullptr, kvScale.get(),
        kvScale.get(), nullptr, tndQ ? cuQ.get() : nullptr, tndKv ? cuKv.get() : nullptr, tndQ ? nullptr : usedQ.get(),
        tndKv ? nullptr : usedKv.get(), paged ? blockTable.get() : nullptr, blockShape.get(), layoutQ.data(),
        layoutKv.data(), kLayoutSparse, kScale, 0, kA8C4QuantType, 0.0, 0, -1, -1, 0, 0, false, output.get(), nullptr,
        &workspaceSize, &executor);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    op::SetPlatformSocVersion(previousSoc);
    return status;
}

} // namespace

// ============================================================================
// Case 1: normal TND + PA_BBND smoke
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, normal)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kS1});
    auto sequsedKv = TensorDesc({kBatch}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{kS2});
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              cuSeqLengthsQ,    // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              sequsedKv,        // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(aclnn_generic_block_sparse_attention_ut, int8_dense_tnd_workspace)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    for (int64_t headDim : {128, 256}) {
        auto query = TensorDesc({4, 8, headDim}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-8, 8);
        auto key = TensorDesc({257, 2, headDim}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-8, 8);
        auto sparseBlockIdx = TensorDesc({2, 4, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
        auto sparseBlockCount = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
        auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
        auto qDequantScale = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(.01, .1);
        auto kDequantScale = TensorDesc({2, 257}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(.01, .1);
        auto vDequantScale = TensorDesc({2, headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(.01, .1);
        auto pQuantScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{127.0f});
        auto cuSeqLengthsQ = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 4});
        auto cuSeqLengthsKv = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 257});
        for (int64_t residualBlockMode : {0, 1}) {
            for (int64_t blockShapeY : {4, 8, 16, 32, 64, 128}) {
                auto blockShape = IntArrayDesc(vector<int64_t>{1, blockShapeY});
                auto output = TensorDesc({4, 8, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);
                char layout[] = "TND";
                auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                                    INPUT(query, key, key, sparseBlockIdx, sparseBlockCount, metadata, nullptr,
                                          qDequantScale, kDequantScale, vDequantScale, pQuantScale, cuSeqLengthsQ,
                                          cuSeqLengthsKv, nullptr, nullptr, nullptr, blockShape, layout, layout, 4,
                                          1.0 / 16, 1, 7, 0.0, 0, -1, -1, 0, residualBlockMode, false),
                                    OUTPUT(output, nullptr));
                uint64_t workspaceSize = 0;
                EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
            }
        }
    }
}

TEST_F(aclnn_generic_block_sparse_attention_ut, a8c4_query_sequence_limit)
{
    for (const char* layout : {"BSND", "BNSD"}) {
        SCOPED_TRACE(layout);
        EXPECT_EQ(QueryA8C4Workspace(layout, layout, 16, 1), ACLNN_SUCCESS);
        EXPECT_EQ(QueryA8C4Workspace(layout, layout, 17, 1), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(aclnn_generic_block_sparse_attention_ut, a8c4_kv_head_limit)
{
    for (const char* layoutKv : {"PA_BBND", "BSND", "TND"}) {
        SCOPED_TRACE(layoutKv);
        const char* layoutQ = strcmp(layoutKv, "PA_BBND") == 0 ? "BSND" : layoutKv;
        EXPECT_EQ(QueryA8C4Workspace(layoutQ, layoutKv, kS1, 5), ACLNN_SUCCESS);
        EXPECT_EQ(QueryA8C4Workspace(layoutQ, layoutKv, kS1, 6), ACLNN_ERR_PARAM_INVALID);
    }
    EXPECT_EQ(QueryA8C4Workspace("BNSD", "BNSD", kS1, 6), ACLNN_SUCCESS);
    EXPECT_EQ(QueryA8C4Workspace("BSND", "PA_BNBD", kS1, 6), ACLNN_SUCCESS);
}

// ============================================================================
// Case 2: required query is nullptr
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_query)
{
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(nullptr,          // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              nullptr,          // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              nullptr,          // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 3: null metadata is rejected before the Contiguous path (nullptr -> PARAM_NULLPTR)
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_metadata)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              nullptr,          // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              nullptr,          // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              nullptr,          // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 4: required key is nullptr
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_key)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              nullptr,          // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              nullptr,          // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              nullptr,          // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 5: required attentionOut is nullptr
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_attention_out)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              nullptr,          // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              nullptr,          // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(nullptr, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 6: returnSoftmaxlse=1 with valid LSE output
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, normal_with_lse)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kS1});
    auto sequsedKv = TensorDesc({kBatch}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{kS2});
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto softmaxLse = TensorDesc({kT, kN1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              cuSeqLengthsQ,    // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              sequsedKv,        // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              1,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, softmaxLse));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ============================================================================
// Case 6a: returnSoftmaxlse=1 but softmaxLseOptional is nullptr
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, softmaxLse_flag1_lse_nullptr)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              nullptr,          // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              nullptr,          // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              1,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 6b: returnSoftmaxlse=0 with softmaxLseOptional=nullptr stays legal
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, softmaxLse_flag0_lse_nullptr)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto cuSeqLengthsQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kS1});
    auto sequsedKv = TensorDesc({kBatch}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{kS2});
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query,            // query
                              key,              // key
                              value,            // value
                              sparseBlockIdx,   // sparseBlockIdx
                              sparseBlockCount, // sparseBlockCount
                              metadata,         // metadataOptional
                              nullptr,          // attenMaskOptional
                              nullptr,          // qDequantScaleOptional
                              nullptr,          // kDequantScaleOptional
                              nullptr,          // vDequantScaleOptional
                              nullptr,          // pQuantScaleOptional
                              cuSeqLengthsQ,    // cuSeqLengthsQOptional
                              nullptr,          // cuSeqLengthsKvOptional
                              nullptr,          // sequsedQOptional
                              sequsedKv,        // sequsedKvOptional
                              blockTable,       // blockTableOptional
                              blockShape,       // blockShape
                              layoutQ,          // layoutQ
                              layoutKv,         // layoutKv
                              kLayoutSparse,    // layoutSparsePattern
                              kScale,           // scaleValue
                              1,                // maskType
                              0,                // quantType
                              0.0,              // dstTypeMax
                              0,                // softmaxPrecision
                              -1,               // winLeft
                              -1,               // winRight
                              0,                // returnSoftmaxlse
                              0,                // residualBlockMode
                              false             // isConsistentTopk
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ============================================================================
// Case 7: unsupported layoutQ
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, invalid_layout_q)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "BNSD";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Case 8: null blockShape
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_block_shape)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, nullptr, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 9: invalid blockShape size
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, invalid_block_shape_size)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Case 10: Q/K dtype mismatch
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, dtype_mismatch)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Case 11: invalid softmaxPrecision
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, invalid_softmax_precision)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 2, -1, -1, 0, 0, 0),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Case 12: required value is nullptr
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_value)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, nullptr, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape,
                              layoutQ, layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 13: required sparseBlockIdx is nullptr
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_sparse_block_idx)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, nullptr, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 14: required sparseBlockCount is nullptr
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_sparse_block_count)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, nullptr, metadata, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ, layoutKv,
                              kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 15: unsupported layoutKv
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, invalid_layout_kv)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "TND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Case 16: blockShape element not positive
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, invalid_block_shape_value)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{0, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Case 17: null layoutQ triggers ValidateParams null-pointer branch
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_layout_q)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, nullptr,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 18: null layoutKv triggers ValidateParams null-pointer branch
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, null_layout_kv)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              nullptr, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Case 19: unsupported query dtype
// ============================================================================
TEST_F(aclnn_generic_block_sparse_attention_ut, unsupported_query_dtype)
{
    auto query = TensorDesc({kT, kN1, kD}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({kMaxBlocks, kBlockSize, kN2, kD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto sparseBlockIdx = TensorDesc({kN2, kTotalQBlocks, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto sparseBlockCount = TensorDesc({kN2, kTotalQBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockTable = TensorDesc({kBatch, kMaxBlocks}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
    auto blockShape = IntArrayDesc(vector<int64_t>{kBlockShapeX, kBlockSize});
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    auto attentionOut = TensorDesc({kT, kN1, kD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                        INPUT(query, key, value, sparseBlockIdx, sparseBlockCount, metadata, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, blockTable, blockShape, layoutQ,
                              layoutKv, kLayoutSparse, kScale, 1, 0, 0.0, 0, -1, -1, 0, 0, false),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnn_generic_block_sparse_attention_ut, int8_rejects_fp8_quant_type)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    for (int64_t headDim : {128}) {
        auto query = TensorDesc({4, 8, headDim}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-8, 8);
        auto key = TensorDesc({257, 2, headDim}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-8, 8);
        auto sparseBlockIdx = TensorDesc({2, 4, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
        auto sparseBlockCount = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
        auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});
        auto qDequantScale = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(.01, .1);
        auto kDequantScale = TensorDesc({2, 257}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(.01, .1);
        auto vDequantScale = TensorDesc({2, headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(.01, .1);
        auto pQuantScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{127.0f});
        auto cuSeqLengthsQ = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 4});
        auto cuSeqLengthsKv = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 257});
        for (int64_t residualBlockMode : {0}) {
            for (int64_t blockShapeY : {4}) {
                auto blockShape = IntArrayDesc(vector<int64_t>{1, blockShapeY});
                auto output = TensorDesc({4, 8, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);
                char layout[] = "TND";
                auto ut = OP_API_UT(aclnnGenericBlockSparseAttention,
                                    INPUT(query, key, key, sparseBlockIdx, sparseBlockCount, metadata, nullptr,
                                          qDequantScale, kDequantScale, vDequantScale, pQuantScale, cuSeqLengthsQ,
                                          cuSeqLengthsKv, nullptr, nullptr, nullptr, blockShape, layout, layout, 4,
                                          1.0 / 16, 1, 5, 0.0, 0, -1, -1, 0, residualBlockMode, false),
                                    OUTPUT(output, nullptr));
                uint64_t workspaceSize = 0;
                EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
            }
        }
    }
}

// ============================================================================
// Case 2: required query is nullptr
// ============================================================================

TEST_F(aclnn_generic_block_sparse_attention_ut, mxfp8_precision_and_lse_output_contract)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto query = TensorDesc({4, 24, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    auto key = TensorDesc({256, 2, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    auto idx = TensorDesc({2, 4, 2}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto count = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto cuQ = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 4}).ToAclType();
    auto cuKv = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 256}).ToAclType();
    auto out = TensorDesc({4, 24, 256}, ACL_BF16, ACL_FORMAT_ND).ToAclType();
    auto lse = TensorDesc({4, 24, 1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto badType = TensorDesc({4, 24, 1}, ACL_BF16, ACL_FORMAT_ND).ToAclType();
    auto badShape = TensorDesc({4, 24, 2}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto badRank = TensorDesc({1, 24, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    const vector<int64_t> shape = {1, 4};
    auto blocks = aclCreateIntArray(shape.data(), shape.size());
    char layout[] = "TND";
    auto check = [&](int64_t precision, aclTensor* lseOutput) {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto status = aclnnGenericBlockSparseAttentionGetWorkspaceSize(
            query.get(), key.get(), key.get(), idx.get(), count.get(), metadata.get(), nullptr, nullptr, nullptr,
            nullptr, nullptr, cuQ.get(), cuKv.get(), nullptr, nullptr, nullptr, blocks, layout, layout, 4, 0.0625, 0, 6,
            0.0, precision, -1, -1, 1, 0, false, out.get(), lseOutput, &workspaceSize, &executor);
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
        return status;
    };
    for (int64_t precision : {1, -1, 2}) {
        EXPECT_EQ(check(precision, lse.get()), ACLNN_ERR_PARAM_INVALID);
    }
    EXPECT_EQ(check(0, nullptr), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(check(0, badType.get()), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(check(0, badShape.get()), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(check(0, badRank.get()), ACLNN_ERR_PARAM_INVALID);
    aclDestroyIntArray(blocks);
}
