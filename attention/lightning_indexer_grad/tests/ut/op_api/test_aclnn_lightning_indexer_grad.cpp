/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
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

#include "../../../op_host/op_api/aclnn_lightning_indexer_grad.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace {
void DestroyAclTensor(aclTensor *tensor)
{
    Release(tensor);
}

using AclTensorPtr = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>;

AclTensorPtr MakeTensor(const vector<int64_t> &shape, aclDataType dtype, aclFormat format = ACL_FORMAT_ND)
{
    return AclTensorPtr(TensorDesc(shape, dtype, format).ToAclTypeRawPtr(), DestroyAclTensor);
}
} // namespace

class LightningIndexerGradOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "LightningIndexerGradOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "LightningIndexerGradOpapiUt TearDown" << endl;
    }
};

TEST_F(LightningIndexerGradOpapiUt, A1_bsnd_fp16_success)
{
    auto query = MakeTensor({2, 8, 64, 128}, ACL_FLOAT16);
    auto key = MakeTensor({2, 16, 1, 128}, ACL_FLOAT16);
    auto dy = MakeTensor({2, 8, 64, 128}, ACL_FLOAT16);
    auto sparseIndices = MakeTensor({2, 8, 2048}, ACL_INT32);
    auto weights = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto dqOut = MakeTensor({2, 8, 64, 128}, ACL_FLOAT16);
    auto dkOut = MakeTensor({2, 16, 1, 128}, ACL_FLOAT16);
    auto dweightsOut = MakeTensor({2, 8, 64}, ACL_FLOAT16);

    char inputLayout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerGradGetWorkspaceSize(
        query.get(), key.get(), dy.get(), sparseIndices.get(), weights.get(),
        nullptr, nullptr, 64, inputLayout, 3, 65536, 65536, false,
        dqOut.get(), dkOut.get(), dweightsOut.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(LightningIndexerGradOpapiUt, A2_tnd_bf16_success)
{
    auto query = MakeTensor({64, 64, 128}, ACL_BF16);
    auto key = MakeTensor({128, 1, 128}, ACL_BF16);
    auto dy = MakeTensor({64, 64, 128}, ACL_BF16);
    auto sparseIndices = MakeTensor({64, 2048}, ACL_INT32);
    auto weights = MakeTensor({64, 64}, ACL_BF16);
    auto actualSeqQ = MakeTensor({1}, ACL_INT32);
    auto actualSeqK = MakeTensor({1}, ACL_INT32);
    auto dqOut = MakeTensor({64, 64, 128}, ACL_BF16);
    auto dkOut = MakeTensor({128, 1, 128}, ACL_BF16);
    auto dweightsOut = MakeTensor({64, 64}, ACL_BF16);

    char inputLayout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerGradGetWorkspaceSize(
        query.get(), key.get(), dy.get(), sparseIndices.get(), weights.get(),
        actualSeqQ.get(), actualSeqK.get(), 64, inputLayout, 3, 65536, 65536, false,
        dqOut.get(), dkOut.get(), dweightsOut.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

TEST_F(LightningIndexerGradOpapiUt, E1_null_query_failed)
{
    auto key = MakeTensor({2, 16, 1, 128}, ACL_FLOAT16);
    auto dy = MakeTensor({2, 8, 64, 128}, ACL_FLOAT16);
    auto sparseIndices = MakeTensor({2, 8, 2048}, ACL_INT32);
    auto weights = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto dqOut = MakeTensor({2, 8, 64, 128}, ACL_FLOAT16);
    auto dkOut = MakeTensor({2, 16, 1, 128}, ACL_FLOAT16);
    auto dweightsOut = MakeTensor({2, 8, 64}, ACL_FLOAT16);

    char inputLayout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerGradGetWorkspaceSize(
        nullptr, key.get(), dy.get(), sparseIndices.get(), weights.get(),
        nullptr, nullptr, 64, inputLayout, 3, 65536, 65536, false,
        dqOut.get(), dkOut.get(), dweightsOut.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerGradOpapiUt, E2_dtype_mismatch_failed)
{
    auto query = MakeTensor({2, 8, 64, 128}, ACL_FLOAT16);
    auto key = MakeTensor({2, 16, 1, 128}, ACL_BF16);
    auto dy = MakeTensor({2, 8, 64, 128}, ACL_BF16);
    auto sparseIndices = MakeTensor({2, 8, 2048}, ACL_INT32);
    auto weights = MakeTensor({2, 8, 64}, ACL_BF16);
    auto dqOut = MakeTensor({2, 8, 64, 128}, ACL_FLOAT16);
    auto dkOut = MakeTensor({2, 16, 1, 128}, ACL_FLOAT16);
    auto dweightsOut = MakeTensor({2, 8, 64}, ACL_FLOAT16);

    char inputLayout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerGradGetWorkspaceSize(
        query.get(), key.get(), dy.get(), sparseIndices.get(), weights.get(),
        nullptr, nullptr, 64, inputLayout, 3, 65536, 65536, false,
        dqOut.get(), dkOut.get(), dweightsOut.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(LightningIndexerGradOpapiUt, E3_null_dq_output_failed)
{
    auto query = MakeTensor({2, 8, 64, 128}, ACL_FLOAT16);
    auto key = MakeTensor({2, 16, 1, 128}, ACL_FLOAT16);
    auto dy = MakeTensor({2, 8, 64, 128}, ACL_FLOAT16);
    auto sparseIndices = MakeTensor({2, 8, 2048}, ACL_INT32);
    auto weights = MakeTensor({2, 8, 64}, ACL_FLOAT16);
    auto dkOut = MakeTensor({2, 16, 1, 128}, ACL_FLOAT16);
    auto dweightsOut = MakeTensor({2, 8, 64}, ACL_FLOAT16);

    char inputLayout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerGradGetWorkspaceSize(
        query.get(), key.get(), dy.get(), sparseIndices.get(), weights.get(),
        nullptr, nullptr, 64, inputLayout, 3, 65536, 65536, false,
        nullptr, dkOut.get(), dweightsOut.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}
