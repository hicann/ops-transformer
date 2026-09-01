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
 * \file test_aclnn_engram_fetch_grad.cpp
 * \brief EngramFetchGrad 算子 op_api 侧 aclnn 接口 UT
 *
 * 测试 aclnnEngramFetchGradGetWorkspaceSize 的参数校验:
 * - 正常场景: 合法输入输出
 * - nullptr 场景: 各输入/输出为 nullptr
 *
 * aclnnEngramFetchGrad 接口签名:
 *   input:  commContext, gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv
 *   output: gradUniqueOut, uniqueLocalEntryOut, numUniqueOut
 *   attr:   numEntriesPerRank, commBufferSize
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "aclnn/aclnn_base.h"

extern "C" {
aclnnStatus aclnnEngramFetchGradGetWorkspaceSize(const aclTensor *commContext, const aclTensor *gradFetched,
                                                 const aclTensor *perm, const aclTensor *sendCounts,
                                                 const aclTensor *recvCounts, const aclTensor *recvLocalEntry,
                                                 const aclTensor *numRecv, aclTensor *gradUniqueOut,
                                                 aclTensor *uniqueLocalEntryOut, aclTensor *numUniqueOut,
                                                 int64_t numEntriesPerRank, int64_t commBufferSize,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnEngramFetchGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
}

using namespace op;
using namespace std;

class AclnnEngramFetchGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "EngramFetchGrad AclnnTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        std::cout << "EngramFetchGrad AclnnTest TearDown" << std::endl;
    }
};

TEST_F(AclnnEngramFetchGradTest, ascend950_success)
{
    auto commContext_desc = TensorDesc({6146}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto gradFetched_desc = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    auto perm_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto sendCounts_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto recvCounts_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 16);
    auto recvLocalEntry_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto numRecv_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto gradUnique_desc = TensorDesc({16, 512}, ACL_BF16, ACL_FORMAT_ND);
    auto uniqueLocalEntry_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto numUnique_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    int64_t numEntriesPerRank = 4;
    int64_t commBufferSize = 4194304;

    auto ut = OP_API_UT(
        aclnnEngramFetchGrad,
        INPUT(commContext_desc, gradFetched_desc, perm_desc, sendCounts_desc, recvCounts_desc, recvLocalEntry_desc,
              numRecv_desc, gradUnique_desc, uniqueLocalEntry_desc, numUnique_desc, numEntriesPerRank, commBufferSize),
        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnEngramFetchGradTest, ascend950_nullptr_commContext)
{
    auto gradFetched_desc = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    auto perm_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto sendCounts_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto recvCounts_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto recvLocalEntry_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecv_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto gradUnique_desc = TensorDesc({16, 512}, ACL_BF16, ACL_FORMAT_ND);
    auto uniqueLocalEntry_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto numUnique_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    int64_t numEntriesPerRank = 4;
    int64_t commBufferSize = 4194304;

    auto ut = OP_API_UT(
        aclnnEngramFetchGrad,
        INPUT(nullptr, gradFetched_desc, perm_desc, sendCounts_desc, recvCounts_desc, recvLocalEntry_desc, numRecv_desc,
              gradUnique_desc, uniqueLocalEntry_desc, numUnique_desc, numEntriesPerRank, commBufferSize),
        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnEngramFetchGradTest, ascend950_nullptr_gradFetched)
{
    auto commContext_desc = TensorDesc({6146}, ACL_INT32, ACL_FORMAT_ND);
    auto perm_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto sendCounts_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto recvCounts_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto recvLocalEntry_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecv_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto gradUnique_desc = TensorDesc({16, 512}, ACL_BF16, ACL_FORMAT_ND);
    auto uniqueLocalEntry_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto numUnique_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    int64_t numEntriesPerRank = 4;
    int64_t commBufferSize = 4194304;

    auto ut = OP_API_UT(
        aclnnEngramFetchGrad,
        INPUT(commContext_desc, nullptr, perm_desc, sendCounts_desc, recvCounts_desc, recvLocalEntry_desc, numRecv_desc,
              gradUnique_desc, uniqueLocalEntry_desc, numUnique_desc, numEntriesPerRank, commBufferSize),
        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnEngramFetchGradTest, ascend950_nullptr_gradUniqueOut)
{
    auto commContext_desc = TensorDesc({6146}, ACL_INT32, ACL_FORMAT_ND);
    auto gradFetched_desc = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    auto perm_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto sendCounts_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto recvCounts_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto recvLocalEntry_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecv_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto uniqueLocalEntry_desc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    auto numUnique_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    int64_t numEntriesPerRank = 4;
    int64_t commBufferSize = 4194304;

    auto ut = OP_API_UT(
        aclnnEngramFetchGrad,
        INPUT(commContext_desc, gradFetched_desc, perm_desc, sendCounts_desc, recvCounts_desc, recvLocalEntry_desc,
              numRecv_desc, nullptr, uniqueLocalEntry_desc, numUnique_desc, numEntriesPerRank, commBufferSize),
        OUTPUT());

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnEngramFetchGradTest, ascend950_execute_entry)
{
    aclnnStatus ret = aclnnEngramFetchGrad(nullptr, 0, nullptr, nullptr);
    EXPECT_THAT(ret, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                    testing::Eq(ACLNN_ERR_PARAM_INVALID), testing::Eq(ACLNN_ERR_INNER)));
}
