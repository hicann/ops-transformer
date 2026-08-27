/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "../../../op_api/aclnn_all_gather_matmul_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"

using namespace op;
using namespace std;

namespace {

class AllGatherMatmulV2AclnnTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "AllGatherMatmulV2AclnnTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "AllGatherMatmulV2AclnnTest TearDown" << endl;
    }
};

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFirstApi1)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFirstApi2)
{
    TensorDesc x1 = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherGatherOutFalse)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFourthApi)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFifthApi)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherSixthApi)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckHif8BaisInvaild)
{
    TensorDesc x1 = TensorDesc({8, 1}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({1, 1}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckScale)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quant_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, quant_scale, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckAmaxout)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckX1scaleNotNullptr)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckX2scaleNotNullptr)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckX2scaleNotScalar)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckQuantscaleNotScalar)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quant_quant_scale = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, x1Scale, x2Scale, quant_quant_scale, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
        OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherMatmulTransPerblock)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, {1, 256});
    TensorDesc x1Scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 549764202624, "ai_cpu"),
        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestStreamModeInvalid)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 0, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestDtypeMismatchX1X2)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestDtypeMismatchX1Output)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestBiasDtypeMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestKAxisTooSmall)
{
    TensorDesc x1 = TensorDesc({8, 100}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({100, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 100}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestKAxisTooLarge)
{
    TensorDesc x1 = TensorDesc({8, 70000}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({70000, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 70000}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestKAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestNAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestFP16EmptyX1WithValidK)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestFP16KZeroEmpty)
{
    TensorDesc x1 = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutWithFP16)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutWrongDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutNotScalar)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestQuantScaleWrongDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quantScale = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, nullptr, x1Scale, x2Scale, quantScale, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestHif8DtypeMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestUnsupportedOutputDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestUnsupportedBiasDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutNot1D)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

class AllGatherMatmulV2AclnnAIVTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // 不依赖 tests/ 框架里 SetPlatformNpuArch(DAV_2201) 映射；直接设 SocVersion 使 GetCurNpuArch()==DAV_2201
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "AllGatherMatmulV2AclnnAIVTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "AllGatherMatmulV2AclnnAIVTest TearDown" << endl;
    }
};

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVFP16Basic)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVBF16Basic)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVINT8WithScale)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVINT4WithScale)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVInvalidDtypeFP8)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVStreamModeInvalid)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 0, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVKAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVGatherOutKAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVNAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVNonContiguousX2WithoutTranspose)
{
    TensorDesc x1 = TensorDesc({32, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND, {128 + 10, 1}, 0, {256, 128 + 10});
    TensorDesc output = TensorDesc({32, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({32, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVX1TransposedRejected)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 8});
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestGetWorkspaceSizeUnsupportedNpuArch)
{
    op::SetPlatformNpuArch(NpuArch::DAV_RESV);
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestSecondStageSkipsWhenWorkspaceEmpty)
{
    aclnnStatus aclRet = aclnnAllGatherMatmulV2(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutEmptyTensorTreatedAsNoAmax)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(AllGatherMatmulV2AclnnTest, TestFP8EmptyX1ExercisesDealEmptyTensor)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    (void)ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(AllGatherMatmulV2AclnnTest, TestFP8EmptyX2ExercisesDealEmptyTensor)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    (void)ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(AllGatherMatmulV2AclnnTest, TestX2ScaleTransposeBranch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {2, 2});
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(AllGatherMatmulV2AclnnTest, TestSecondStageWithWorkspaceInvokesInnerAndHcclStub)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus wsRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (wsRet != ACLNN_SUCCESS || executor == nullptr || workspaceSize == 0) {
        if (executor != nullptr) {
            delete executor;
        }
        GTEST_SKIP() << "GetWorkspaceSize did not yield runnable second stage";
    }
    std::vector<uint8_t> workspace(workspaceSize);
    aclnnStatus runRet = aclnnAllGatherMatmulV2(workspace.data(), workspaceSize, executor, nullptr);
    EXPECT_EQ(runRet, ACLNN_SUCCESS);
    delete executor;
}

TEST_F(AllGatherMatmulV2AclnnTest, TestCommModeCcu)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ccu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestCommModeEmptyAutoDetect)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, ""),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestCommModeInvalid)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "bad_mode"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestX1TransposedRejected)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 8});
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestCommModeNullptr)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = aclnnAllGatherMatmulV2GetWorkspaceSize(
        x1.ToAclTypeRawPtr(), x2.ToAclTypeRawPtr(), nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0,
        8, 1, 0, nullptr, output.ToAclTypeRawPtr(), gatherOut.ToAclTypeRawPtr(), nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestSecondStageExecutorNull)
{
    std::vector<uint8_t> workspace(64, 0);
    aclnnStatus aclRet = aclnnAllGatherMatmulV2(workspace.data(), workspace.size(), nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// ====== mxfp4 (DT_FLOAT4_E2M1) 用例 ======
// 合法组合：x1=x2=FP4，x1Scale/x2Scale=E8M0(shape [1])，bias=FLOAT，
// K>=256 且为 32 倍数。x2 view shape 为 {K, N}。覆盖 3 种 y dtype。
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4ValidYFp16)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4ValidYBf16)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4ValidYFloat)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 失败：x1=FP4 / x2=FP8（dtype 不一致）→ CheckDataTypeFp8Fp4Valid 触发 OP_CHECK_DTYPE_NOT_SAME
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2Fp8DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 失败：FP4 输入但 x1Scale=nullptr → CheckScale 返回 ACLNN_ERR_PARAM_INVALID
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1ScaleNullptrFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ====== mxfp4 dtype mismatch：x1=FP4, x2 为其他量化 dtype ======
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2Fp16DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2Bf16DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2Fp8E5m2DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2Hif8DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ====== mxfp4 dtype mismatch：x1 为其他 dtype, x2=FP4 ======
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1Fp16X2Fp4DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1Bf16X2Fp4DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1Hif8X2Fp4DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1Fp8E4m3X2Fp4DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1Fp8E5m2X2Fp4DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// 失败：FP4 输入但 x2Scale=nullptr → CheckScale 返回 ACLNN_ERR_PARAM_INVALID
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2ScaleNullptrFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, nullptr, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 失败：FP4 输入但 x1 转置 → CheckShape 返回 ACLNN_ERR_PARAM_INVALID
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1TransposedFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, {1, 8});
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ====== mxfp4 空 tensor：M=0, N=0 ======
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4MZeroEmpty)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4NZeroEmpty)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4MNZeroEmpty)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVX1Nullptr)
{
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnAllGatherMatmulV2GetWorkspaceSize(nullptr, // x1：本用例只让它为空
                                                                x2.ToAclTypeRawPtr(),
                                                                nullptr, // bias：A2 本来就允许/要求为空
                                                                nullptr, // x1Scale
                                                                nullptr, // x2Scale
                                                                nullptr, // quantScale
                                                                0,       // blockSize
                                                                "test_all_gather_group",
                                                                0,     // gatherIndex
                                                                8,     // commTurn
                                                                1,     // streamMode，必须使用合法值
                                                                0,     // groupSize
                                                                "aiv", // 保证进入 A2/AIV 分支
                                                                output.ToAclTypeRawPtr(), gatherOut.ToAclTypeRawPtr(),
                                                                nullptr, // amaxOut
                                                                &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVRequiredTensorsAllNullptr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus ret = aclnnAllGatherMatmulV2GetWorkspaceSize(nullptr,                 // x1
                                                             nullptr,                 // x2
                                                             nullptr,                 // bias，可选
                                                             nullptr,                 // x1Scale，可选
                                                             nullptr,                 // x2Scale，可选
                                                             nullptr,                 // quantScale，可选
                                                             0,                       // blockSize
                                                             "test_all_gather_group", // 不能是nullptr
                                                             0,                       // gatherIndex
                                                             8,                       // 沿用已有UT合法值
                                                             1,                       // streamMode不能为0
                                                             0,                       // groupSize
                                                             "aiv",          // 不能是nullptr，否则不会进入A2
                                                             nullptr,        // output
                                                             nullptr,        // gatherOut，可选
                                                             nullptr,        // amaxOut，可选
                                                             &workspaceSize, // 必须是有效地址
                                                             &executor);     // 必须是有效地址

    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

} // namespace
