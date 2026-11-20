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
 * \file test_aclnn_moe_ep_dispatch_epilogue.cpp
 * \brief MoeEpDispatchEpilogue aclnn UT
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "aclnn/aclnn_base.h"

extern "C" {
aclnnStatus aclnnMoeEpDispatchEpilogueGetWorkspaceSize(
    const aclTensor* context, const aclTensor* x, const aclTensor* topkIdx, const aclTensor* numRecvTokensPerRank,
    const aclTensor* numRecvTokensPerExpert, const aclTensor* cachedRecvSrcMetadata, int64_t epWorldSize,
    int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank, int64_t cclBufferSize, bool hasTopkWeights,
    int64_t topoType, int64_t rankNumPerServer, aclTensor* recvX, aclTensor* recvSrcMetadata,
    aclTensor* recvTopkWeights, aclTensor* recvScales, uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnMoeEpDispatchEpilogue(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);
}

using namespace op;

class AclnnMoeEpDispatchEpilogueTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
    }
};

TEST_F(AclnnMoeEpDispatchEpilogueTest, nullptr_context)
{
    auto x = TensorDesc({4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto recvX = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto recvSrcMetadata = TensorDesc({768}, ACL_INT32, ACL_FORMAT_ND);
    auto recvTopkWeights = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto recvScales = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(nullptr, x, topkIdx, numRecvTokensPerRank, numRecvTokensPerExpert, nullptr, 2, 0, 4, 8,
                              12000, true, 0, 1, recvX, recvSrcMetadata, recvTopkWeights, recvScales),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMoeEpDispatchEpilogueTest, nullptr_recvScales)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto recvX = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto recvSrcMetadata = TensorDesc({768}, ACL_INT32, ACL_FORMAT_ND);
    auto recvTopkWeights = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(context, x, topkIdx, numRecvTokensPerRank, numRecvTokensPerExpert, nullptr, 2, 0, 4, 8,
                              12000, true, 0, 1, recvX, recvSrcMetadata, recvTopkWeights, nullptr),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMoeEpDispatchEpilogueTest, nullptr_x)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto recvX = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto recvSrcMetadata = TensorDesc({768}, ACL_INT32, ACL_FORMAT_ND);
    auto recvTopkWeights = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto recvScales = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(context, nullptr, topkIdx, numRecvTokensPerRank, numRecvTokensPerExpert, nullptr, 2, 0, 4,
                              8, 12000, true, 0, 1, recvX, recvSrcMetadata, recvTopkWeights, recvScales),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMoeEpDispatchEpilogueTest, invalid_epWorldSize)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto recvX = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto recvSrcMetadata = TensorDesc({768}, ACL_INT32, ACL_FORMAT_ND);
    auto recvTopkWeights = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto recvScales = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(context, x, topkIdx, numRecvTokensPerRank, numRecvTokensPerExpert, nullptr, 1, 0, 4, 8,
                              12000, true, 0, 1, recvX, recvSrcMetadata, recvTopkWeights, recvScales),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMoeEpDispatchEpilogueTest, invalid_epRankId)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto recvX = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto recvSrcMetadata = TensorDesc({768}, ACL_INT32, ACL_FORMAT_ND);
    auto recvTopkWeights = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto recvScales = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(context, x, topkIdx, numRecvTokensPerRank, numRecvTokensPerExpert, nullptr, 2, 2, 4, 8,
                              12000, true, 0, 1, recvX, recvSrcMetadata, recvTopkWeights, recvScales),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMoeEpDispatchEpilogueTest, invalid_topo)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto recvX = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto recvSrcMetadata = TensorDesc({768}, ACL_INT32, ACL_FORMAT_ND);
    auto recvTopkWeights = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto recvScales = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(context, x, topkIdx, numRecvTokensPerRank, numRecvTokensPerExpert, nullptr, 2, 0, 4, 8,
                              12000, true, 3, 1, recvX, recvSrcMetadata, recvTopkWeights, recvScales),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMoeEpDispatchEpilogueTest, invalid_rank_per_server)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto recvX = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto recvSrcMetadata = TensorDesc({768}, ACL_INT32, ACL_FORMAT_ND);
    auto recvTopkWeights = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto recvScales = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatchEpilogue,
                        INPUT(context, x, topkIdx, numRecvTokensPerRank, numRecvTokensPerExpert, nullptr, 2, 0, 4, 8,
                              12000, true, 0, 3, recvX, recvSrcMetadata, recvTopkWeights, recvScales),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMoeEpDispatchEpilogueTest, execute_entry)
{
    EXPECT_EQ(aclnnMoeEpDispatchEpilogue(nullptr, 0, nullptr, nullptr), ACLNN_SUCCESS);
}
