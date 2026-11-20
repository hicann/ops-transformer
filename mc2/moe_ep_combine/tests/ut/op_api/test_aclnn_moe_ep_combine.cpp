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
 * \file test_aclnn_moe_ep_combine.cpp
 * \brief MoeEpCombine aclnn UT
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "aclnn/aclnn_base.h"

extern "C" {
aclnnStatus aclnnMoeEpCombineGetWorkspaceSize(const aclTensor* context, const aclTensor* x, const aclTensor* topkIdx,
                                              const aclTensor* recvSrcMetadata, const aclTensor* numRecvTokensPerExpert,
                                              const aclTensor* topkWeights, int64_t epWorldSize, int64_t epRankId,
                                              int64_t numExperts, int64_t numMaxTokensPerRank, int64_t cclBufferSize,
                                              int64_t topoType, int64_t rankNumPerServer, uint64_t* workspaceSize,
                                              aclOpExecutor** executor);
aclnnStatus aclnnMoeEpCombine(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
}

using namespace op;

class AclnnMoeEpCombineTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
    }
};

TEST_F(AclnnMoeEpCombineTest, nullptr_context)
{
    auto x = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto recvSrcMetadata = TensorDesc({768}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND);
    auto topkWeights = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnMoeEpCombine,
        INPUT(nullptr, x, topkIdx, recvSrcMetadata, numRecvTokensPerExpert, topkWeights, 8, 0, 32, 8, 12000, 0, 8),
        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMoeEpCombineTest, invalid_epWorldSize)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({8, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto recvSrcMetadata = TensorDesc({768}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND);
    auto topkWeights = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnMoeEpCombine,
        INPUT(context, x, topkIdx, recvSrcMetadata, numRecvTokensPerExpert, topkWeights, 1, 0, 32, 8, 12000, 0, 1),
        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMoeEpCombineTest, execute_entry)
{
    EXPECT_EQ(aclnnMoeEpCombine(nullptr, 0, nullptr, nullptr), ACLNN_SUCCESS);
}
