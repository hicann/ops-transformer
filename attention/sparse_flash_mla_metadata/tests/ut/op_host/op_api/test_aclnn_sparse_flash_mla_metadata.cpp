/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>

#include "gtest/gtest.h"
#include "../../../../op_host/sparse_flash_mla_metadata_check.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {
constexpr int64_t NUM_HEADS_Q = 64;
constexpr int64_t NUM_HEADS_KV = 1;
constexpr int64_t HEAD_DIM = 512;
constexpr int64_t BATCH_SIZE = 2;
constexpr int64_t MAX_SEQLEN_Q = 16;
constexpr int64_t MAX_SEQLEN_ORI_KV = 16;
constexpr int64_t MAX_SEQLEN_CMP_KV = 4;
constexpr int64_t ORI_TOPK = 0;
constexpr int64_t CMP_TOPK = 512;
constexpr int64_t CMP_RATIO = 4;
constexpr int64_t ORI_MASK_MODE = 4;
constexpr int64_t CMP_MASK_MODE = 3;
constexpr int64_t ORI_WIN_LEFT = 127;
constexpr int64_t ORI_WIN_RIGHT = 0;
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;

struct TndC4aParams {
    std::unique_ptr<aclTensor, void (*)(aclTensor*)> cuSeqlensQ = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    std::unique_ptr<aclTensor, void (*)(aclTensor*)> cuSeqlensOriKv =
        TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    std::unique_ptr<aclTensor, void (*)(aclTensor*)> cuSeqlensCmpKv =
        TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    std::unique_ptr<aclTensor, void (*)(aclTensor*)> cmpResidualKv =
        TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    std::unique_ptr<aclTensor, void (*)(aclTensor*)> metadata =
        TensorDesc({optiling::SMLA_METADATA_TOTAL_SIZE}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
};

aclnnStatus RunParamsCheck(const TndC4aParams& params, const aclTensor* cmpResidualKv)
{
    return ParamsCheck(params.cuSeqlensQ.get(), params.cuSeqlensOriKv.get(), params.cuSeqlensCmpKv.get(), nullptr,
                       nullptr, nullptr, cmpResidualKv, nullptr, nullptr, NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM,
                       BATCH_SIZE, MAX_SEQLEN_Q, MAX_SEQLEN_ORI_KV, MAX_SEQLEN_CMP_KV, ORI_TOPK, CMP_TOPK, CMP_RATIO,
                       ORI_MASK_MODE, CMP_MASK_MODE, ORI_WIN_LEFT, ORI_WIN_RIGHT, "TND", "TND", true, true,
                       AIC_CORE_NUM, AIV_CORE_NUM, "Ascend910B", params.metadata.get());
}
} // namespace

class SparseFlashMlaMetadataApiTest : public testing::Test {};

TEST_F(SparseFlashMlaMetadataApiTest, tnd_c4a_requires_cmp_residual_kv)
{
    TndC4aParams params;

    EXPECT_EQ(RunParamsCheck(params, nullptr), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SparseFlashMlaMetadataApiTest, tnd_c4a_accepts_cmp_residual_kv)
{
    TndC4aParams params;

    EXPECT_EQ(RunParamsCheck(params, params.cmpResidualKv.get()), ACLNN_SUCCESS);
}

TEST_F(SparseFlashMlaMetadataApiTest, cmp_residual_kv_batch_must_match)
{
    TndC4aParams params;
    auto wrongBatchResidual = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();

    EXPECT_EQ(RunParamsCheck(params, wrongBatchResidual.get()), ACLNN_ERR_PARAM_INVALID);
}
