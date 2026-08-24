/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "opdev/platform.h"
#include <gtest/gtest.h>
#include "attention/block_sparse_attention/op_host/op_api/aclnn_block_sparse_attention_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;
using namespace op;

class aclnn_block_sparse_attention_v2_ut : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "aclnn_block_sparse_attention_v2_ut SetUp" << endl; }

    static void TearDownTestCase() { cout << "aclnn_block_sparse_attention_v2_ut TearDown" << endl; }
};

namespace {
constexpr int64_t batch = 1;
constexpr int64_t qSeqlen = 128;
constexpr int64_t kvSeqlen = 128;
constexpr int64_t numHeads = 2;
constexpr int64_t numKvHeads = 1;
constexpr int64_t headDim = 128;
constexpr int64_t blockShapeX = 128;
constexpr int64_t blockShapeY = 128;
constexpr int64_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;
constexpr int64_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;
constexpr double scaleValue = 1.0 / sqrt(static_cast<double>(headDim));
} // namespace

// ============================================================================
// 用例 1: 正常场景
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, normal)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    auto qDequantScale = TensorDesc({batch, numHeads, qBlockNum, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto kDequantScale = TensorDesc({batch, numKvHeads, kvBlockNum, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto vDequantScale = TensorDesc({batch, numKvHeads, kvBlockNum, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto actualSeqLengths = IntArrayDesc(vector<int64_t>{qSeqlen});
    auto actualSeqLengthsKv = IntArrayDesc(vector<int64_t>{kvSeqlen});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,              // query
                              key,                // key
                              value,              // value
                              blockSparseMask,    // blockSparseMask
                              nullptr,            // attenMask
                              blockShape,         // blockShape
                              actualSeqLengths,   // actualSeqLengths
                              actualSeqLengthsKv, // actualSeqLengthsKv
                              nullptr,            // blockTable
                              qDequantScale,      // qDequantScale
                              kDequantScale,      // kDequantScale
                              vDequantScale,      // vDequantScale
                              qInputLayout,       // qInputLayout
                              kvInputLayout,      // kvInputLayout
                              numKvHeads,         // numKeyValueHeads
                              0,                  // maskType
                              scaleValue,         // scaleValue
                              0,                  // innerPrecise
                              0,                  // blockSize
                              2147483647,         // preTokens
                              2147483647,         // nextTokens
                              0                   // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ============================================================================
// 用例 2: 不支持的query数据类型
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, unsupported_q_dtype)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,           // query
                              key,             // key
                              value,           // value
                              blockSparseMask, // blockSparseMask
                              nullptr,         // attenMask
                              blockShape,      // blockShape
                              nullptr,         // actualSeqLengths
                              nullptr,         // actualSeqLengthsKv
                              nullptr,         // blockTable
                              nullptr,         // qDequantScale
                              nullptr,         // kDequantScale
                              nullptr,         // vDequantScale
                              qInputLayout,    // qInputLayout
                              kvInputLayout,   // kvInputLayout
                              numKvHeads,      // numKeyValueHeads
                              0,               // maskType
                              scaleValue,      // scaleValue
                              0,               // innerPrecise
                              0,               // blockSize
                              2147483647,      // preTokens
                              2147483647,      // nextTokens
                              0                // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 3: 不支持的key/value数据类型
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, unsupported_kv_dtype)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,           // query
                              key,             // key
                              value,           // value
                              blockSparseMask, // blockSparseMask
                              nullptr,         // attenMask
                              blockShape,      // blockShape
                              nullptr,         // actualSeqLengths
                              nullptr,         // actualSeqLengthsKv
                              nullptr,         // blockTable
                              nullptr,         // qDequantScale
                              nullptr,         // kDequantScale
                              nullptr,         // vDequantScale
                              qInputLayout,    // qInputLayout
                              kvInputLayout,   // kvInputLayout
                              numKvHeads,      // numKeyValueHeads
                              0,               // maskType
                              scaleValue,      // scaleValue
                              0,               // innerPrecise
                              0,               // blockSize
                              2147483647,      // preTokens
                              2147483647,      // nextTokens
                              0                // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 4: blockShape维度不为2
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, block_shape_wrong_dim)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY, 1});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,           // query
                              key,             // key
                              value,           // value
                              blockSparseMask, // blockSparseMask
                              nullptr,         // attenMask
                              blockShape,      // blockShape
                              nullptr,         // actualSeqLengths
                              nullptr,         // actualSeqLengthsKv
                              nullptr,         // blockTable
                              nullptr,         // qDequantScale
                              nullptr,         // kDequantScale
                              nullptr,         // vDequantScale
                              qInputLayout,    // qInputLayout
                              kvInputLayout,   // kvInputLayout
                              numKvHeads,      // numKeyValueHeads
                              0,               // maskType
                              scaleValue,      // scaleValue
                              0,               // innerPrecise
                              0,               // blockSize
                              2147483647,      // preTokens
                              2147483647,      // nextTokens
                              0                // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 5: blockShape值存在负数
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, block_shape_data_contains_negative)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{-1, -1});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,           // query
                              key,             // key
                              value,           // value
                              blockSparseMask, // blockSparseMask
                              nullptr,         // attenMask
                              blockShape,      // blockShape
                              nullptr,         // actualSeqLengths
                              nullptr,         // actualSeqLengthsKv
                              nullptr,         // blockTable
                              nullptr,         // qDequantScale
                              nullptr,         // kDequantScale
                              nullptr,         // vDequantScale
                              qInputLayout,    // qInputLayout
                              kvInputLayout,   // kvInputLayout
                              numKvHeads,      // numKeyValueHeads
                              0,               // maskType
                              scaleValue,      // scaleValue
                              0,               // innerPrecise
                              0,               // blockSize
                              2147483647,      // preTokens
                              2147483647,      // nextTokens
                              0                // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 6: qInputLayout或kvInputLayout为空
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, inputLayout_is_null)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,           // query
                              key,             // key
                              value,           // value
                              blockSparseMask, // blockSparseMask
                              nullptr,         // attenMask
                              blockShape,      // blockShape
                              nullptr,         // actualSeqLengths
                              nullptr,         // actualSeqLengthsKv
                              nullptr,         // blockTable
                              nullptr,         // qDequantScale
                              nullptr,         // kDequantScale
                              nullptr,         // vDequantScale
                              nullptr,         // qInputLayout
                              nullptr,         // kvInputLayout
                              numKvHeads,      // numKeyValueHeads
                              0,               // maskType
                              scaleValue,      // scaleValue
                              0,               // innerPrecise
                              0,               // blockSize
                              2147483647,      // preTokens
                              2147483647,      // nextTokens
                              0                // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// 用例 7: 不支持的qInputLayout
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, unsupported_qInputLayout)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    char qInputLayout[] = "BSH";
    char kvInputLayout[] = "BNSD";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,           // query
                              key,             // key
                              value,           // value
                              blockSparseMask, // blockSparseMask
                              nullptr,         // attenMask
                              blockShape,      // blockShape
                              nullptr,         // actualSeqLengths
                              nullptr,         // actualSeqLengthsKv
                              nullptr,         // blockTable
                              nullptr,         // qDequantScale
                              nullptr,         // kDequantScale
                              nullptr,         // vDequantScale
                              qInputLayout,    // qInputLayout
                              kvInputLayout,   // kvInputLayout
                              numKvHeads,      // numKeyValueHeads
                              0,               // maskType
                              scaleValue,      // scaleValue
                              0,               // innerPrecise
                              0,               // blockSize
                              2147483647,      // preTokens
                              2147483647,      // nextTokens
                              0                // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 8: 不支持的kvInputLayout
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, unsupported_kvInputLayout)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BSH";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,           // query
                              key,             // key
                              value,           // value
                              blockSparseMask, // blockSparseMask
                              nullptr,         // attenMask
                              blockShape,      // blockShape
                              nullptr,         // actualSeqLengths
                              nullptr,         // actualSeqLengthsKv
                              nullptr,         // blockTable
                              nullptr,         // qDequantScale
                              nullptr,         // kDequantScale
                              nullptr,         // vDequantScale
                              qInputLayout,    // qInputLayout
                              kvInputLayout,   // kvInputLayout
                              numKvHeads,      // numKeyValueHeads
                              0,               // maskType
                              scaleValue,      // scaleValue
                              0,               // innerPrecise
                              0,               // blockSize
                              2147483647,      // preTokens
                              2147483647,      // nextTokens
                              0                // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 9: qInputLayout和kvInputLayout不一致
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, qInputLayout_kvInputLayout_not_equal)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "TND";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,           // query
                              key,             // key
                              value,           // value
                              blockSparseMask, // blockSparseMask
                              nullptr,         // attenMask
                              blockShape,      // blockShape
                              nullptr,         // actualSeqLengths
                              nullptr,         // actualSeqLengthsKv
                              nullptr,         // blockTable
                              nullptr,         // qDequantScale
                              nullptr,         // kDequantScale
                              nullptr,         // vDequantScale
                              qInputLayout,    // qInputLayout
                              kvInputLayout,   // kvInputLayout
                              numKvHeads,      // numKeyValueHeads
                              0,               // maskType
                              scaleValue,      // scaleValue
                              0,               // innerPrecise
                              0,               // blockSize
                              2147483647,      // preTokens
                              2147483647,      // nextTokens
                              0                // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 10: blockSparseMask维度不为4
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, blockSparseMask_wrong_dim)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum, 1}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,           // query
                              key,             // key
                              value,           // value
                              blockSparseMask, // blockSparseMask
                              nullptr,         // attenMask
                              blockShape,      // blockShape
                              nullptr,         // actualSeqLengths
                              nullptr,         // actualSeqLengthsKv
                              nullptr,         // blockTable
                              nullptr,         // qDequantScale
                              nullptr,         // kDequantScale
                              nullptr,         // vDequantScale
                              qInputLayout,    // qInputLayout
                              kvInputLayout,   // kvInputLayout
                              numKvHeads,      // numKeyValueHeads
                              0,               // maskType
                              scaleValue,      // scaleValue
                              0,               // innerPrecise
                              0,               // blockSize
                              2147483647,      // preTokens
                              2147483647,      // nextTokens
                              0                // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 11: qDequantScale维度不为4
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, qDequantScale_wrong_dim)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    auto qDequantScale = TensorDesc({batch, numHeads, qBlockNum, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto kDequantScale = TensorDesc({batch, numKvHeads, kvBlockNum, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto vDequantScale = TensorDesc({batch, numKvHeads, kvBlockNum, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto actualSeqLengths = IntArrayDesc(vector<int64_t>{qSeqlen});
    auto actualSeqLengthsKv = IntArrayDesc(vector<int64_t>{kvSeqlen});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,              // query
                              key,                // key
                              value,              // value
                              blockSparseMask,    // blockSparseMask
                              nullptr,            // attenMask
                              blockShape,         // blockShape
                              actualSeqLengths,   // actualSeqLengths
                              actualSeqLengthsKv, // actualSeqLengthsKv
                              nullptr,            // blockTable
                              qDequantScale,      // qDequantScale
                              kDequantScale,      // kDequantScale
                              vDequantScale,      // vDequantScale
                              qInputLayout,       // qInputLayout
                              kvInputLayout,      // kvInputLayout
                              numKvHeads,         // numKeyValueHeads
                              0,                  // maskType
                              scaleValue,         // scaleValue
                              0,                  // innerPrecise
                              0,                  // blockSize
                              2147483647,         // preTokens
                              2147483647,         // nextTokens
                              0                   // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 12: kDequantScale维度不为4
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, kDequantScale_wrong_dim)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    auto qDequantScale = TensorDesc({batch, numHeads, qBlockNum, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto kDequantScale = TensorDesc({batch, numKvHeads, kvBlockNum, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto vDequantScale = TensorDesc({batch, numKvHeads, kvBlockNum, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto actualSeqLengths = IntArrayDesc(vector<int64_t>{qSeqlen});
    auto actualSeqLengthsKv = IntArrayDesc(vector<int64_t>{kvSeqlen});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,              // query
                              key,                // key
                              value,              // value
                              blockSparseMask,    // blockSparseMask
                              nullptr,            // attenMask
                              blockShape,         // blockShape
                              actualSeqLengths,   // actualSeqLengths
                              actualSeqLengthsKv, // actualSeqLengthsKv
                              nullptr,            // blockTable
                              qDequantScale,      // qDequantScale
                              kDequantScale,      // kDequantScale
                              vDequantScale,      // vDequantScale
                              qInputLayout,       // qInputLayout
                              kvInputLayout,      // kvInputLayout
                              numKvHeads,         // numKeyValueHeads
                              0,                  // maskType
                              scaleValue,         // scaleValue
                              0,                  // innerPrecise
                              0,                  // blockSize
                              2147483647,         // preTokens
                              2147483647,         // nextTokens
                              0                   // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 13: vDequantScale维度不为4
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, vDequantScale_wrong_dim)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    auto qDequantScale = TensorDesc({batch, numHeads, qBlockNum, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto kDequantScale = TensorDesc({batch, numKvHeads, kvBlockNum, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto vDequantScale = TensorDesc({batch, numKvHeads, kvBlockNum, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto actualSeqLengths = IntArrayDesc(vector<int64_t>{qSeqlen});
    auto actualSeqLengthsKv = IntArrayDesc(vector<int64_t>{kvSeqlen});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,              // query
                              key,                // key
                              value,              // value
                              blockSparseMask,    // blockSparseMask
                              nullptr,            // attenMask
                              blockShape,         // blockShape
                              actualSeqLengths,   // actualSeqLengths
                              actualSeqLengthsKv, // actualSeqLengthsKv
                              nullptr,            // blockTable
                              qDequantScale,      // qDequantScale
                              kDequantScale,      // kDequantScale
                              vDequantScale,      // vDequantScale
                              qInputLayout,       // qInputLayout
                              kvInputLayout,      // kvInputLayout
                              numKvHeads,         // numKeyValueHeads
                              0,                  // maskType
                              scaleValue,         // scaleValue
                              0,                  // innerPrecise
                              0,                  // blockSize
                              2147483647,         // preTokens
                              2147483647,         // nextTokens
                              0                   // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// 用例 14: 不支持的innerPrecise
// ============================================================================
TEST_F(aclnn_block_sparse_attention_v2_ut, unsupported_innerPrecise)
{
    auto query = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto key = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto value = TensorDesc({batch, numKvHeads, kvSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto blockSparseMask =
        TensorDesc({batch, numHeads, qBlockNum, kvBlockNum}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto blockShape = IntArrayDesc(vector<int64_t>{blockShapeX, blockShapeY});
    char qInputLayout[] = "BNSD";
    char kvInputLayout[] = "BNSD";
    int64_t innerPrecise = 2;
    auto attentionOut = TensorDesc({batch, numHeads, qSeqlen, headDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBlockSparseAttentionV2,
                        INPUT(query,           // query
                              key,             // key
                              value,           // value
                              blockSparseMask, // blockSparseMask
                              nullptr,         // attenMask
                              blockShape,      // blockShape
                              nullptr,         // actualSeqLengths
                              nullptr,         // actualSeqLengthsKv
                              nullptr,         // blockTable
                              nullptr,         // qDequantScale
                              nullptr,         // kDequantScale
                              nullptr,         // vDequantScale
                              qInputLayout,    // qInputLayout
                              kvInputLayout,   // kvInputLayout
                              numKvHeads,      // numKeyValueHeads
                              0,               // maskType
                              scaleValue,      // scaleValue
                              innerPrecise,    // innerPrecise
                              0,               // blockSize
                              2147483647,      // preTokens
                              2147483647,      // nextTokens
                              0                // softmaxLseFlag
                              ),
                        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
