/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_und_gen_qkv_rms_norm_rope_cache.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;
using namespace op;

namespace {
constexpr int64_t UND_LEN = 5;
constexpr int64_t GEN_LEN = 3;
constexpr int64_t TOTAL = UND_LEN + GEN_LEN;
constexpr int64_t NUM_HEAD_Q = 8;
constexpr int64_t NUM_HEAD_K = 1;
constexpr int64_t NUM_HEAD_V = 1;
constexpr int64_t NUM_HEAD = NUM_HEAD_Q + NUM_HEAD_K + NUM_HEAD_V;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t BLOCK_NUM = 2;
constexpr int64_t MAX_POS = 4096;
constexpr double NORM_EPS = 1e-6;

// 连续 BBND cache：[Bn, Bs, H, D]
TensorDesc ContiguousCache(int64_t numHead)
{
    return TensorDesc({BLOCK_NUM, BLOCK_SIZE, numHead, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
}

// 非连续 cache：view 仍是 [Bn, Bs, H, D]，但首轴 stride 是连续值的 2 倍
// （底层 storage 有 2*Bn 个 block，view 每隔一个 block 取一个），
// 等价于调用方传了一个 slice 出来的 cache 视图。
TensorDesc NonContiguousCache(int64_t numHead)
{
    const int64_t contiguousDim0Stride = BLOCK_SIZE * numHead * HEAD_DIM;
    const vector<int64_t> strides = {contiguousDim0Stride * 2, numHead * HEAD_DIM, HEAD_DIM, 1};
    const vector<int64_t> storageDims = {BLOCK_NUM * 2, BLOCK_SIZE, numHead, HEAD_DIM};
    return TensorDesc({BLOCK_NUM, BLOCK_SIZE, numHead, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND, strides, 0, storageDims);
}
} // namespace

class und_gen_qkv_rms_norm_rope_cache_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "und_gen_qkv_rms_norm_rope_cache_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "und_gen_qkv_rms_norm_rope_cache_opapi_ut TearDown" << endl;
    }
};

// k_cache/v_cache 均连续：连续性校验不应拦截（基线，用于证明后面两个用例失败的原因是非连续本身）
TEST_F(und_gen_qkv_rms_norm_rope_cache_opapi_ut, aclnn_contiguous_kv_cache_passes_check)
{
    auto undQkv = TensorDesc({UND_LEN, NUM_HEAD, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto undWeightsQ = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto undWeightsK = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto cosSinCache = TensorDesc({MAX_POS, HEAD_DIM}, ACL_FLOAT, ACL_FORMAT_ND);
    auto kCacheRef = ContiguousCache(NUM_HEAD_K);
    auto vCacheRef = ContiguousCache(NUM_HEAD_V);
    auto slotMapping = TensorDesc({TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto positions = TensorDesc({3, TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto genQkv = TensorDesc({GEN_LEN, NUM_HEAD, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto genWeightsQ = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto genWeightsK = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto catIndices = TensorDesc({TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto mropeSection = IntArrayDesc(vector<int64_t>{16, 16, 16});
    auto qOut = TensorDesc({TOTAL, NUM_HEAD_Q, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnUndGenQkvRmsNormRopeCache,
        INPUT(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCacheRef, vCacheRef, slotMapping, positions, genQkv,
              genWeightsQ, genWeightsK, catIndices, NUM_HEAD_Q, NUM_HEAD_K, NUM_HEAD_V, NORM_EPS, mropeSection),
        OUTPUT(qOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// k_cache 非连续：原地写会落到 Contiguous 副本上，必须在 L2 直接拒掉
TEST_F(und_gen_qkv_rms_norm_rope_cache_opapi_ut, aclnn_non_contiguous_k_cache_rejected)
{
    auto undQkv = TensorDesc({UND_LEN, NUM_HEAD, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto undWeightsQ = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto undWeightsK = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto cosSinCache = TensorDesc({MAX_POS, HEAD_DIM}, ACL_FLOAT, ACL_FORMAT_ND);
    auto kCacheRef = NonContiguousCache(NUM_HEAD_K);
    auto vCacheRef = ContiguousCache(NUM_HEAD_V);
    auto slotMapping = TensorDesc({TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto positions = TensorDesc({3, TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto genQkv = TensorDesc({GEN_LEN, NUM_HEAD, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto genWeightsQ = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto genWeightsK = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto catIndices = TensorDesc({TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto mropeSection = IntArrayDesc(vector<int64_t>{16, 16, 16});
    auto qOut = TensorDesc({TOTAL, NUM_HEAD_Q, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnUndGenQkvRmsNormRopeCache,
        INPUT(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCacheRef, vCacheRef, slotMapping, positions, genQkv,
              genWeightsQ, genWeightsK, catIndices, NUM_HEAD_Q, NUM_HEAD_K, NUM_HEAD_V, NORM_EPS, mropeSection),
        OUTPUT(qOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// v_cache 非连续：同上
TEST_F(und_gen_qkv_rms_norm_rope_cache_opapi_ut, aclnn_non_contiguous_v_cache_rejected)
{
    auto undQkv = TensorDesc({UND_LEN, NUM_HEAD, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto undWeightsQ = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto undWeightsK = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto cosSinCache = TensorDesc({MAX_POS, HEAD_DIM}, ACL_FLOAT, ACL_FORMAT_ND);
    auto kCacheRef = ContiguousCache(NUM_HEAD_K);
    auto vCacheRef = NonContiguousCache(NUM_HEAD_V);
    auto slotMapping = TensorDesc({TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto positions = TensorDesc({3, TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto genQkv = TensorDesc({GEN_LEN, NUM_HEAD, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto genWeightsQ = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto genWeightsK = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto catIndices = TensorDesc({TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto mropeSection = IntArrayDesc(vector<int64_t>{16, 16, 16});
    auto qOut = TensorDesc({TOTAL, NUM_HEAD_Q, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnUndGenQkvRmsNormRopeCache,
        INPUT(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCacheRef, vCacheRef, slotMapping, positions, genQkv,
              genWeightsQ, genWeightsK, catIndices, NUM_HEAD_Q, NUM_HEAD_K, NUM_HEAD_V, NORM_EPS, mropeSection),
        OUTPUT(qOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// k_cache 为空指针：必须先被空指针校验拦住（连续性校验会解引用，顺序不能反）
TEST_F(und_gen_qkv_rms_norm_rope_cache_opapi_ut, aclnn_null_k_cache_rejected_before_contiguous_check)
{
    auto undQkv = TensorDesc({UND_LEN, NUM_HEAD, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto undWeightsQ = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto undWeightsK = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto cosSinCache = TensorDesc({MAX_POS, HEAD_DIM}, ACL_FLOAT, ACL_FORMAT_ND);
    auto vCacheRef = ContiguousCache(NUM_HEAD_V);
    auto slotMapping = TensorDesc({TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto positions = TensorDesc({3, TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto genQkv = TensorDesc({GEN_LEN, NUM_HEAD, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto genWeightsQ = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto genWeightsK = TensorDesc({HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);
    auto catIndices = TensorDesc({TOTAL}, ACL_INT64, ACL_FORMAT_ND);
    auto mropeSection = IntArrayDesc(vector<int64_t>{16, 16, 16});
    auto qOut = TensorDesc({TOTAL, NUM_HEAD_Q, HEAD_DIM}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnUndGenQkvRmsNormRopeCache,
        INPUT(undQkv, undWeightsQ, undWeightsK, cosSinCache, (aclTensor*)nullptr, vCacheRef, slotMapping, positions,
              genQkv, genWeightsQ, genWeightsK, catIndices, NUM_HEAD_Q, NUM_HEAD_K, NUM_HEAD_V, NORM_EPS,
              mropeSection),
        OUTPUT(qOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
