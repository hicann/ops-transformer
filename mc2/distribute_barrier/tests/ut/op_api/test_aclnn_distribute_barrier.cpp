/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_distribute_barrier.h"
#include "../../../op_api/aclnn_distribute_barrier_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class L2AclnnDistributeBarrierTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
        cout << "L2AclnnDistributeBarrierTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AclnnDistributeBarrierTest TearDown" << endl;
    }
};

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierFirstApi)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrier, INPUT(xRef, "test_distribute_barrier", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierFirstApiNullptrGroup)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrier, INPUT(xRef, nullptr, worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierFirstApiGroupMin)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrier, INPUT(xRef, "", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2FirstApi)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, nullptr, nullptr, "test_distribute_barrier", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2FirstApiNullptrGroup)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, nullptr, nullptr, nullptr, worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2FirstApiGroupMin)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, nullptr, nullptr, "", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2FirstApiTimeOut)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc timeOut = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, timeOut, nullptr, "test_distribute_barrier", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2FirstApiElasticInfo)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc elasticInfo = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND);

    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                        INPUT(xRef, nullptr, elasticInfo, "test_distribute_barrier", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierSecondApi)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrier, INPUT(xRef, "test_distribute_barrier", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    if (aclRet == ACLNN_SUCCESS && executor != nullptr) {
        aclRet = aclnnDistributeBarrier(nullptr, workspaceSize, executor, nullptr);
    }
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2SecondApi)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, nullptr, nullptr, "test_distribute_barrier", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    if (aclRet == ACLNN_SUCCESS && executor != nullptr) {
        aclRet = aclnnDistributeBarrierV2(nullptr, workspaceSize, executor, nullptr);
    }
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierSecondApiTimeOut)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc timeOut = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2, INPUT(xRef, timeOut, nullptr, "test_distribute_barrier", worldSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    if (aclRet == ACLNN_SUCCESS && executor != nullptr) {
        aclRet = aclnnDistributeBarrierV2(nullptr, workspaceSize, executor, nullptr);
    }
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierSecondApiElasticInfo)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc elasticInfo = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND);
    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                        INPUT(xRef, nullptr, elasticInfo, "test_distribute_barrier", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    if (aclRet == ACLNN_SUCCESS && executor != nullptr) {
        aclRet = aclnnDistributeBarrierV2(nullptr, workspaceSize, executor, nullptr);
    }
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierSecondApiTimeOutAndElasticInfo)
{
    TensorDesc xRef = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc timeOut = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc elasticInfo = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND);
    int64_t worldSize = 16;

    auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                        INPUT(xRef, timeOut, elasticInfo, "test_distribute_barrier", worldSize), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    if (aclRet == ACLNN_SUCCESS && executor != nullptr) {
        aclRet = aclnnDistributeBarrierV2(nullptr, workspaceSize, executor, nullptr);
    }
}

// ========== 新增测试用例：直接调用第二段接口覆盖未覆盖代码 ==========

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierSecondApiDirect)
{
    // 直接调用第二段接口，覆盖 aclnnDistributeBarrier 和 aclnnDistributeBarrierBase 函数体
    aclnnStatus aclRet = aclnnDistributeBarrier(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2SecondApiDirect)
{
    // 直接调用第二段接口，覆盖 aclnnDistributeBarrierV2 和 aclnnDistributeBarrierBase 函数体
    aclnnStatus aclRet = aclnnDistributeBarrierV2(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// ========== 参数校验测试用例 ==========

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierNullptrXRef)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = aclnnDistributeBarrierGetWorkspaceSize(nullptr, "test_group", 16, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierV2NullptrXRef)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = aclnnDistributeBarrierV2GetWorkspaceSize(nullptr, nullptr, nullptr, "test_group", 16,
                                                                  &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ========== dtype 测试用例 ==========

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierDtypeFloat32)
{
    TensorDesc xRefDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xRefPtr = xRefDesc.ToAclType();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet =
        aclnnDistributeBarrierGetWorkspaceSize(xRefPtr.get(), "test_distribute_barrier", 16, &workspaceSize, &executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierDtypeBfloat16)
{
    TensorDesc xRefDesc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto xRefPtr = xRefDesc.ToAclType();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet =
        aclnnDistributeBarrierGetWorkspaceSize(xRefPtr.get(), "test_distribute_barrier", 16, &workspaceSize, &executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnDistributeBarrierTest, TestAclnnDistributeBarrierDtypeInt32)
{
    TensorDesc xRefDesc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto xRefPtr = xRefDesc.ToAclType();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet =
        aclnnDistributeBarrierGetWorkspaceSize(xRefPtr.get(), "test_distribute_barrier", 16, &workspaceSize, &executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}
