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
 * \file test_aclnn_engram_fetch.cpp
 * \brief engram_fetch 算子 op_api 侧 aclnn 接口 UT
 *
 * 测试 aclnnEngramFetchGetWorkspaceSize 的参数校验:
 * - 推理场景: 可选 input/output 为 nullptr, with_grad=0
 * - 训练场景: 可选 input/output 非空, with_grad=1
 * - nullptr 场景: 必选输入/输出为 nullptr
 *
 * aclnnEngramFetch 接口签名:
 *   input:  commContext, indices, localStorageAddr(optional)
 *   output: fetched, permOut(opt), sendCountsOut(opt), recvCountsOut(opt),
 *           recvLocalEntryOut(opt), numRecvOut(opt)
 *   attr:   hidden_size, num_entries_per_rank, num_max_tokens_per_rank(opt),
 *           comm_buffer_size(opt), with_grad(opt)
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "aclnn/aclnn_base.h"

extern "C" {
aclnnStatus aclnnEngramFetchGetWorkspaceSize(const aclTensor *commContext, const aclTensor *indices,
                                             const aclTensor *localStorageAddr, aclTensor *fetched, aclTensor *permOut,
                                             aclTensor *sendCountsOut, aclTensor *recvCountsOut,
                                             aclTensor *recvLocalEntryOut, aclTensor *numRecvOut, int32_t hiddenSize,
                                             int64_t numEntriesPerRank, int64_t numMaxTokensPerRank,
                                             int64_t commBufferSize, int64_t withGrad, uint64_t *workspaceSize,
                                             aclOpExecutor **executor);

aclnnStatus aclnnEngramFetch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
}

using namespace op;
using namespace std;

class AclnnEngramFetchTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "EngramFetch AclnnEngramFetchTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "EngramFetch AclnnEngramFetchTest TearDown" << std::endl;
    }
};

TEST_F(AclnnEngramFetchTest, ascend950_inference_success)
{
    auto commContext_desc = TensorDesc({2048}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto fetched_desc = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);

    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t zero = 0;

    auto ut = OP_API_UT(aclnnEngramFetch,
                        INPUT(commContext_desc, indices_desc, nullptr, fetched_desc, nullptr, nullptr, nullptr, nullptr,
                              nullptr, hiddenSize, numEntriesPerRank, zero, zero, zero),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnEngramFetchTest, ascend950_training_success)
{
    auto commContext_desc = TensorDesc({6146}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto localStorageAddr_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto fetched_desc = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    auto permOut_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto sendCountsOut_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto recvCountsOut_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 16);
    auto recvLocalEntryOut_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto numRecvOut_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t numMaxTokensPerRank = 8;
    int64_t commBufferSize = 4194304;
    int64_t withGrad = 1;

    auto ut = OP_API_UT(aclnnEngramFetch,
                        INPUT(commContext_desc, indices_desc, localStorageAddr_desc, fetched_desc, permOut_desc,
                              sendCountsOut_desc, recvCountsOut_desc, recvLocalEntryOut_desc, numRecvOut_desc,
                              hiddenSize, numEntriesPerRank, numMaxTokensPerRank, commBufferSize, withGrad),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnEngramFetchTest, ascend950_nullptr_commContext)
{
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto fetched_desc = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);

    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t zero = 0;

    auto ut = OP_API_UT(aclnnEngramFetch,
                        INPUT(nullptr, indices_desc, nullptr, fetched_desc, nullptr, nullptr, nullptr, nullptr, nullptr,
                              hiddenSize, numEntriesPerRank, zero, zero, zero),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnEngramFetchTest, ascend950_nullptr_indices)
{
    auto commContext_desc = TensorDesc({2048}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto fetched_desc = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);

    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t zero = 0;

    auto ut = OP_API_UT(aclnnEngramFetch,
                        INPUT(commContext_desc, nullptr, nullptr, fetched_desc, nullptr, nullptr, nullptr, nullptr,
                              nullptr, hiddenSize, numEntriesPerRank, zero, zero, zero),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnEngramFetchTest, ascend950_nullptr_fetched)
{
    auto commContext_desc = TensorDesc({2048}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);

    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t zero = 0;

    auto ut = OP_API_UT(aclnnEngramFetch,
                        INPUT(commContext_desc, indices_desc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, hiddenSize, numEntriesPerRank, zero, zero, zero),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnEngramFetchTest, ascend950_execute_entry)
{
    aclnnStatus ret = aclnnEngramFetch(nullptr, 0, nullptr, nullptr);
    EXPECT_THAT(ret, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                    testing::Eq(ACLNN_ERR_PARAM_INVALID), testing::Eq(ACLNN_ERR_INNER)));
}
