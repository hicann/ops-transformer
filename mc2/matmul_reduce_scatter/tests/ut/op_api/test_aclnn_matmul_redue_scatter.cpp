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
#include "../../../op_api/aclnn_matmul_reduce_scatter.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MatmulReduceScatter {
class L2AclnnMatmulReduceScatterTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AclnnMatmulReduceScatterTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AclnnMatmulReduceScatterTest TearDown" << endl;
    }
};

TEST_F(L2AclnnMatmulReduceScatterTest, TestAclnnMatmulReduceScatterFirstApi)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMatmulReduceScatterTest, TestAclnnMatmulReduceScatterFirstApi2)
{
    TensorDesc x1Desc = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMatmulReduceScatterTest, TestAclnnMatmulReduceScatterFirstApi3)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMatmulReduceScatterTest, TestSixApiA5)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// 1. streamMode 非法值测试（覆盖 CheckAttr 的错误分支）
TEST_F(L2AclnnMatmulReduceScatterTest, TestStreamModeInvalid)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 0), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 2. 第二阶段函数 - 空 workspace（覆盖 aclnnMatmulReduceScatter 的空 workspace 路径）
TEST_F(L2AclnnMatmulReduceScatterTest, TestSecondApiEmptyWorkspace)
{
    aclnnStatus ret = aclnnMatmulReduceScatter(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

// 3. 第二阶段函数 - 带非空 workspace（覆盖 aclnnMatmulReduceScatter 的内部调用路径）
TEST_F(L2AclnnMatmulReduceScatterTest, TestSecondApiWithWorkspace)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    ASSERT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    uint64_t fakeWorkspaceSize = std::max(workspaceSize, (uint64_t)1024);
    void* fakeWorkspace = malloc(fakeWorkspaceSize);
    ASSERT_NE(fakeWorkspace, nullptr);
    aclnnStatus ret = aclnnMatmulReduceScatter(fakeWorkspace, fakeWorkspaceSize, executor, nullptr);
    // UT 模式下内部函数可能返回非零错误码
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    free(fakeWorkspace);
}

// 4. null x1 测试（覆盖 CheckNotNull）
TEST_F(L2AclnnMatmulReduceScatterTest, TestNullX1)
{
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(nullptr, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 5. null x2 测试
TEST_F(L2AclnnMatmulReduceScatterTest, TestNullX2)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, nullptr, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 6. null output 测试（覆盖 CheckNotNull 的 output 检查）
TEST_F(L2AclnnMatmulReduceScatterTest, TestNullOutput)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 7. BF16 dtype 测试
TEST_F(L2AclnnMatmulReduceScatterTest, TestBf16Dtype)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 8. dtype 不匹配测试（覆盖 CheckDtypeValid）
TEST_F(L2AclnnMatmulReduceScatterTest, TestDtypeMismatch)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 9. K=0 空 tensor 测试（覆盖 CheckNotEmptyTensor）
TEST_F(L2AclnnMatmulReduceScatterTest, TestEmptyKDim)
{
    TensorDesc x1Desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 10. 无 bias 测试（覆盖 bias=nullptr 路径）
TEST_F(L2AclnnMatmulReduceScatterTest, TestNoBias)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, nullptr, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 11. K 维度不匹配测试（覆盖 CheckShape 中 K 不匹配分支）
TEST_F(L2AclnnMatmulReduceScatterTest, TestKDimMismatch)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({200, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 12. K 超范围测试（K=100 < 256，覆盖 K < 256 分支）
TEST_F(L2AclnnMatmulReduceScatterTest, TestKBelowMin)
{
    TensorDesc x1Desc = TensorDesc({16, 100}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({100, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({100}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 13. N 维度不匹配测试
TEST_F(L2AclnnMatmulReduceScatterTest, TestNDimMismatch)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 14. K 超上限测试（K=65536 >= 65535）
TEST_F(L2AclnnMatmulReduceScatterTest, TestKAboveMax)
{
    TensorDesc x1Desc = TensorDesc({16, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({65536, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMatmulReduceScatterTest, TestDebugEnvVar)
{
    const char* envValue = getenv("ASCEND_MC2_DEBUG_MODE");
    std::string originalStr = (envValue != nullptr) ? std::string(envValue) : "";
    setenv("ASCEND_MC2_DEBUG_MODE", "0", 1);
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    if (originalStr.empty()) {
        unsetenv("ASCEND_MC2_DEBUG_MODE");
    } else {
        setenv("ASCEND_MC2_DEBUG_MODE", originalStr.c_str(), 1);
    }
}

TEST_F(L2AclnnMatmulReduceScatterTest, TestDebugEnvVarMode4)
{
    const char* envValue = getenv("ASCEND_MC2_DEBUG_MODE");
    std::string originalStr = (envValue != nullptr) ? std::string(envValue) : "";
    setenv("ASCEND_MC2_DEBUG_MODE", "4", 1);
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    if (originalStr.empty()) {
        unsetenv("ASCEND_MC2_DEBUG_MODE");
    } else {
        setenv("ASCEND_MC2_DEBUG_MODE", originalStr.c_str(), 1);
    }
}

TEST_F(L2AclnnMatmulReduceScatterTest, TestKMinBoundary)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // namespace MatmulReduceScatter
