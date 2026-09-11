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
#include "../../../op_api/aclnn_matmul_reduce_scatter_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"

using namespace op;
using namespace std;

namespace {

class MatmulReduceScatterV2AclnnTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "MatmulReduceScatterV2AclnnTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "MatmulReduceScatterV2AclnnTest TearDown" << endl;
    }
};

TEST_F(MatmulReduceScatterV2AclnnTest, Basic)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc quantScale = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, x1Scale, x2Scale, quantScale, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Basic2)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc quantScale = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, x1Scale, x2Scale, quantScale, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, 3Scale)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({32, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc quantScale = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, x1Scale, x2Scale, quantScale, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, EmptyTensor)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Fp16)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, 16Bit)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, 16BitTrans)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_BF16, ACL_FORMAT_ND, {1, 256});
    TensorDesc bias = TensorDesc({256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 256}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, E4m3fnNotSupport)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quantScale = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMatmulReduceScatterV2,
        INPUT(x1, x2, nullptr, x1Scale, x2Scale, quantScale, 0, "test_group", "sum", 8, 1, 549764202624, "ai_cpu"),
        OUTPUT(output, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// =================== CCU 模式测试 (DAV_3510 / ASCEND950) ===================

class MatmulReduceScatterV2CcuModeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
        cout << "MatmulReduceScatterV2CcuModeTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "MatmulReduceScatterV2CcuModeTest TearDown" << endl;
    }
};

TEST_F(MatmulReduceScatterV2CcuModeTest, NullX1)
{
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(nullptr, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, NullX2)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(x1, nullptr, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, NullOutput)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(nullptr, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, InvalidX1Dtype)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, InvalidOutputDtype)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, InvalidBiasDtype)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, InvalidStreamMode)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyFp16WithBias)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyBf16NoBias)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyFp16NoBias)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyX1X2DtypeMismatch)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyX1OutputDtypeMismatch)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyBiasDtypeMismatch)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, EmptyX1)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, EmptyX2)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyE4m3fnPerTensor)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyE4m3fnPerBlock)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({16, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({128, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyE5m2PerTensor)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyHifloat8PerTensor)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyHifloat8DtypeMismatch)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyNullX1Scale)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyNullX2Scale)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyInvalidScaleDim)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracy3DScaleE8M0)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1, 1, 1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracy3DScaleNonE8M0)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, AmaxOutNotNull)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, Non2DX1)
{
    TensorDesc x1 = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, KMismatch)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, KTooSmall)
{
    TensorDesc x1 = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, KTooLarge)
{
    TensorDesc x1 = TensorDesc({16, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({65536, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, TransposeX1Error)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 256});
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, TransposeX2Ccu)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 256});
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, MixedDtypeCase)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyTransX2Scale)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({16, 256}, ACL_FLOAT, ACL_FORMAT_ND, {1, 16});
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// =================== AIV 模式补充测试 (DAV_2201 / ASCEND910B) ===================

TEST_F(MatmulReduceScatterV2AclnnTest, NullX1Aiv)
{
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(nullptr, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NullX2Aiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(x1, nullptr, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NullOutputAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(nullptr, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Int8Fp16OutputAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Int8Bf16OutputAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, InvalidOutputDtypeAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, InvalidX1DtypeAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, InvalidStreamModeAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NonContiguousX2Aiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {32, 1});
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Non2DX1Aiv)
{
    TensorDesc x1 = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, KMismatchAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, KTooSmallAiv)
{
    TensorDesc x1 = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, KTooLargeAiv)
{
    TensorDesc x1 = TensorDesc({16, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({65536, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, TransposeX1Aiv)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 256});
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NullX1Test)
{
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(nullptr, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NullX2Test)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(x1, nullptr, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NullOutputTest)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(nullptr, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, InvalidStreamMode)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, TransAX1Test)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 256});
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, KValueOutOfRange)
{
    TensorDesc x1 = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, MismatchKValue)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Bf16EmptyTensor)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, BiasNotNullFp16)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, WrongBiasDtype)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// =================== CCU 模式补充测试 - 覆盖空 Scale/Bias ===================

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyEmptyX1Scale)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyEmptyBias)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyEmptyX2Scale)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// =================== AIV 模式补充测试 - NZ 格式 ===================

/*
TEST_F(MatmulReduceScatterV2AclnnTest, NzFormatX2Aiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}*/

// =================== 第二段接口测试 ===================

TEST_F(MatmulReduceScatterV2AclnnTest, ExecuteNullWorkspace)
{
    aclnnStatus ret = aclnnMatmulReduceScatterV2(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

// =================== CCU 模式 ai_cpu 通信模式测试 ===================

TEST_F(MatmulReduceScatterV2CcuModeTest, ai_cpuCommModeEnv)
{
    setenv("ENV_MC2_COMM_MODE_ai_cpu", "1", 1);
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    unsetenv("ENV_MC2_COMM_MODE_ai_cpu");
}

} // namespace
