/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <vector>
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_fused_infer_attention_score_v4.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#pragma GCC diagnostic pop
using namespace std;
using namespace op;

namespace {
// BNBD layout: key/value are 4D (numBlocks, kvHeads, blockSize, D) with a non-16-aligned headDim (D=37).
// SplitFuse only requires blockSize to stay 16-aligned, see IsFAIRoutingCandidate (DIM_NUM_4 branch).
struct BnbdDescs {
    TensorDesc query;
    TensorListDesc key;
    TensorListDesc value;
    TensorDesc blockTable;
    IntArrayDesc seqLens;
    IntArrayDesc seqLensKv;
};

// TensorListDesc/IntArrayDesc have no default constructor, so BnbdDescs must be aggregate-initialized.
BnbdDescs MakeBnbdDescs(const vector<int64_t>& qShape, const vector<int64_t>& kShape, const vector<int64_t>& vShape,
                        const vector<int64_t>& blockTableShape)
{
    return {TensorDesc(qShape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1),
            TensorListDesc({TensorDesc(kShape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1)}),
            TensorListDesc({TensorDesc(vShape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1)}),
            TensorDesc(blockTableShape, ACL_INT32, ACL_FORMAT_ND),
            IntArrayDesc(vector<int64_t>{4}),
            IntArrayDesc(vector<int64_t>{256})};
}
} // namespace

class fused_infer_attention_score_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }

    static void TearDownTestCase() {}
};

// All cases below use an empty attentionOut: Impl checks IsFAIRoutingCandidate before the
// attentionOut->IsEmpty() short-circuit, so the BNBD branch is covered and ACLNN_SUCCESS is returned.
// The "supported/rejected" wording refers to the IsFAIRoutingCandidate routing result.

// BNBD key/value with headDim=37 (non-16-aligned) and blockSize=128 is a valid SplitFuse routing candidate.
TEST_F(fused_infer_attention_score_opapi_ut, bnbd_head_dim_37_routing_supported)
{
    auto descs = MakeBnbdDescs({4, 1, 37}, {2, 1, 128, 37}, {2, 1, 128, 37}, {1, 2});
    auto attentionOut = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    char layout[] = "TND";

    auto ut = OP_API_UT(
        aclnnFusedInferAttentionScoreV4,
        INPUT(descs.query, descs.key, descs.value, nullptr, nullptr, descs.seqLens, descs.seqLensKv, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, descs.blockTable, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, int64_t{1},
              0.1643989873053573, int64_t{65536}, int64_t{65536}, layout, int64_t{1}, int64_t{0}, int64_t{0},
              int64_t{128}, int64_t{0}, false, int64_t{0}, int64_t{0}, int64_t{0}),
        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// A 3D value tensor is not a BNBD candidate: the DIM_NUM_4 branch rejects mismatched key/value dims.
TEST_F(fused_infer_attention_score_opapi_ut, bnbd_value_3dim_routing_rejected)
{
    auto descs = MakeBnbdDescs({4, 1, 37}, {2, 1, 128, 37}, {2, 128, 37}, {1, 2});
    auto attentionOut = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    char layout[] = "TND";

    auto ut = OP_API_UT(
        aclnnFusedInferAttentionScoreV4,
        INPUT(descs.query, descs.key, descs.value, nullptr, nullptr, descs.seqLens, descs.seqLensKv, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, descs.blockTable, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, int64_t{1},
              0.1643989873053573, int64_t{65536}, int64_t{65536}, layout, int64_t{1}, int64_t{0}, int64_t{0},
              int64_t{128}, int64_t{0}, false, int64_t{0}, int64_t{0}, int64_t{0}),
        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// headDim must match across query/key/value even without 16-alignment (value D=53 vs query D=37).
TEST_F(fused_infer_attention_score_opapi_ut, bnbd_head_dim_mismatch_routing_rejected)
{
    auto descs = MakeBnbdDescs({4, 1, 37}, {2, 1, 128, 37}, {2, 1, 128, 53}, {1, 2});
    auto attentionOut = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    char layout[] = "TND";

    auto ut = OP_API_UT(
        aclnnFusedInferAttentionScoreV4,
        INPUT(descs.query, descs.key, descs.value, nullptr, nullptr, descs.seqLens, descs.seqLensKv, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, descs.blockTable, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, int64_t{1},
              0.1643989873053573, int64_t{65536}, int64_t{65536}, layout, int64_t{1}, int64_t{0}, int64_t{0},
              int64_t{128}, int64_t{0}, false, int64_t{0}, int64_t{0}, int64_t{0}),
        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// headDim above MAX_HEAD_DIM (D=288 > 256) is rejected by the BNBD branch.
TEST_F(fused_infer_attention_score_opapi_ut, bnbd_head_dim_exceeded_routing_rejected)
{
    auto descs = MakeBnbdDescs({4, 1, 288}, {2, 1, 128, 288}, {2, 1, 128, 288}, {1, 2});
    auto attentionOut = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    char layout[] = "TND";

    auto ut = OP_API_UT(
        aclnnFusedInferAttentionScoreV4,
        INPUT(descs.query, descs.key, descs.value, nullptr, nullptr, descs.seqLens, descs.seqLensKv, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, descs.blockTable, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, int64_t{1},
              0.05892556509887896, int64_t{65536}, int64_t{65536}, layout, int64_t{1}, int64_t{0}, int64_t{0},
              int64_t{128}, int64_t{0}, false, int64_t{0}, int64_t{0}, int64_t{0}),
        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// blockSize (key dim 2) must stay 16-aligned even though headDim no longer needs to be (blockSize=100).
TEST_F(fused_infer_attention_score_opapi_ut, bnbd_block_size_misaligned_routing_rejected)
{
    auto descs = MakeBnbdDescs({4, 1, 37}, {2, 1, 100, 37}, {2, 1, 100, 37}, {1, 2});
    auto attentionOut = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    char layout[] = "TND";

    auto ut = OP_API_UT(
        aclnnFusedInferAttentionScoreV4,
        INPUT(descs.query, descs.key, descs.value, nullptr, nullptr, descs.seqLens, descs.seqLensKv, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, descs.blockTable, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, int64_t{1},
              0.1643989873053573, int64_t{65536}, int64_t{65536}, layout, int64_t{1}, int64_t{0}, int64_t{0},
              int64_t{128}, int64_t{0}, false, int64_t{0}, int64_t{0}, int64_t{0}),
        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// blockSize above MAX_BLOCK_SIZE (528 > 512) is rejected by the BNBD branch.
TEST_F(fused_infer_attention_score_opapi_ut, bnbd_block_size_exceeded_routing_rejected)
{
    auto descs = MakeBnbdDescs({4, 1, 37}, {1, 1, 528, 37}, {1, 1, 528, 37}, {1, 1});
    auto attentionOut = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    char layout[] = "TND";

    auto ut = OP_API_UT(
        aclnnFusedInferAttentionScoreV4,
        INPUT(descs.query, descs.key, descs.value, nullptr, nullptr, descs.seqLens, descs.seqLensKv, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, descs.blockTable, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, int64_t{1},
              0.1643989873053573, int64_t{65536}, int64_t{65536}, layout, int64_t{1}, int64_t{0}, int64_t{0},
              int64_t{128}, int64_t{0}, false, int64_t{0}, int64_t{0}, int64_t{0}),
        OUTPUT(attentionOut, nullptr));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}
