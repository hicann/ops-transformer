/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_lightning_indexer_kl_loss.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace {
void DestroyAclTensor(aclTensor *tensor) { Release(tensor); }

using AclTensorPtr = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>;

AclTensorPtr MakeTensor(const vector<int64_t> &shape, aclDataType dtype, aclFormat format = ACL_FORMAT_ND)
{
    return AclTensorPtr(TensorDesc(shape, dtype, format).ToAclTypeRawPtr(), DestroyAclTensor);
}
} // namespace

class LightningIndexerKLLossOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "LightningIndexerKLLossOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase() { cout << "LightningIndexerKLLossOpapiUt TearDown" << endl; }
};

TEST_F(LightningIndexerKLLossOpapiUt, A1_BSK_fp16_logits_success)
{
    auto targetScore = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, A2_TK_bf16_logits_success)
{
    auto targetScore = MakeTensor({64, 128}, ACL_BF16);
    auto indexProbs = MakeTensor({64, 128}, ACL_BF16);
    auto loss = MakeTensor({1}, ACL_BF16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, A3_BSK_fp32_logits_success)
{
    auto targetScore = MakeTensor({2, 8, 64}, ACL_FLOAT);
    auto indexProbs = MakeTensor({2, 8, 64}, ACL_FLOAT);
    auto loss = MakeTensor({1}, ACL_FLOAT);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, A4_BSK_fp16_probs_success)
{
    auto targetScore = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "probs";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E1_null_targetScore_failed)
{
    auto indexProbs = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(nullptr, indexProbs.get(), eps, weightType,
                                                                     loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E2_null_indexProbs_failed)
{
    auto targetScore = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), nullptr, eps, weightType,
                                                                     loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E3_null_loss_failed)
{
    auto targetScore = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({2, 8, 64}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, nullptr, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E4_dtype_mismatch_failed)
{
    auto targetScore = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({2, 8, 64}, ACL_BF16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E5_shape_mismatch_failed)
{
    auto targetScore = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({2, 8, 128}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E6_null_weightType_failed)
{
    auto targetScore = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps, nullptr,
                                                                     loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E7_invalid_weightType_failed)
{
    auto targetScore = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "invalid";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E8_last_dim_zero_failed)
{
    auto targetScore = MakeTensor({2, 8, 0}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({2, 8, 0}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, A5_last_dim_8192_success)
{
    auto targetScore = MakeTensor({2, 8, 8192}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({2, 8, 8192}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, A6_TK_8192_success)
{
    auto targetScore = MakeTensor({1024, 8192}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({1024, 8192}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E9_last_dim_8193_failed)
{
    auto targetScore = MakeTensor({2, 8, 8193}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({2, 8, 8193}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// 3D 输入 B 维边界测试
TEST_F(LightningIndexerKLLossOpapiUt, A7_batch_512_success)
{
    auto targetScore = MakeTensor({512, 8, 64}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({512, 8, 64}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E10_batch_zero_failed)
{
    auto targetScore = MakeTensor({0, 8, 64}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({0, 8, 64}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerKLLossOpapiUt, E11_batch_513_failed)
{
    auto targetScore = MakeTensor({513, 8, 64}, ACL_FLOAT16);
    auto indexProbs = MakeTensor({513, 8, 64}, ACL_FLOAT16);
    auto loss = MakeTensor({1}, ACL_FLOAT16);

    double eps = 1e-12;
    char weightType[] = "logits";

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerKLLossGetWorkspaceSize(targetScore.get(), indexProbs.get(), eps,
                                                                     weightType, loss.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}
