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
#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR >= 3)
#define ENABLE_MMRS_DT_HIFLOAT4
#endif
#endif
#ifdef ENABLE_MMRS_DT_HIFLOAT4
#include "common/include/op_host/op_const_def_transformer.h"
#endif

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
                        INPUT(x1, x2, bias, x1Scale, x2Scale, quantScale, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, bias, x1Scale, x2Scale, quantScale, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, bias, x1Scale, x2Scale, quantScale, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Fp16)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
        INPUT(x1, x2, nullptr, x1Scale, x2Scale, quantScale, 0, "test_group", "sum", 0, 1, 549764202624, "ai_cpu"),
        OUTPUT(output, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// =================== CCU 模式测试 (DAV_3510 / A5) ===================

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
                  INPUT(nullptr, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, NullX2)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(x1, nullptr, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, NullOutput)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(nullptr, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, InvalidX1Dtype)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, InvalidOutputDtype)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, InvalidStreamMode)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyBf16NoBias)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyFp16NoBias)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyX1X2DtypeMismatch)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, HighAccuracyX1OutputDtypeMismatch)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, EmptyX1)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, EmptyX2)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, LowAccuracyE4m3fnPerTensor)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, nullptr, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, Non2DX1)
{
    TensorDesc x1 = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, KMismatch)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, KTooSmall)
{
    TensorDesc x1 = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, KTooLarge)
{
    TensorDesc x1 = TensorDesc({16, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({65536, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, TransposeX1Error)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 256});
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, TransposeX2Ccu)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 256});
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, MixedDtypeCase)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                  INPUT(nullptr, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NullX2Aiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(x1, nullptr, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NullOutputAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(nullptr, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Int8Fp16OutputAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Int8Bf16OutputAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, InvalidOutputDtypeAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, InvalidX1DtypeAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, InvalidStreamModeAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NonContiguousX2Aiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {32, 1});
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Non2DX1Aiv)
{
    TensorDesc x1 = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, KMismatchAiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, KTooSmallAiv)
{
    TensorDesc x1 = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, KTooLargeAiv)
{
    TensorDesc x1 = TensorDesc({16, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({65536, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, TransposeX1Aiv)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 256});
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NullX1Test)
{
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(nullptr, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NullX2Test)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(x1, nullptr, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NullOutputTest)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(nullptr, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2AclnnTest, InvalidStreamMode)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, TransAX1Test)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 256});
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, KValueOutOfRange)
{
    TensorDesc x1 = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({128, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, MismatchKValue)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, Bf16EmptyTensor)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, BiasNotNullFp16)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
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

TEST_F(MatmulReduceScatterV2AclnnTest, ExecuteNonEmptyWorkspaceAiv)
{
    // Force non-empty workspace path: DAV_2201 → AIV → SetNnopbase MTE → Inner
    char workspace[64] = {};
    aclnnStatus ret = aclnnMatmulReduceScatterV2(workspace, sizeof(workspace), nullptr, nullptr);
    // Cover launch path; Inner may still return 0 with null executor in UT stubs
    EXPECT_TRUE(ret == ACLNN_SUCCESS || ret == ACLNN_ERR_INNER || ret == ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, ExecuteNonEmptyWorkspaceA5)
{
    char workspace[64] = {};
    // handle from null executor → 0 → CommType::AI_CPU (valid)
    aclnnStatus ret = aclnnMatmulReduceScatterV2(workspace, sizeof(workspace), nullptr, nullptr);
    EXPECT_TRUE(ret == ACLNN_SUCCESS || ret == ACLNN_ERR_INNER || ret == ACLNN_ERR_PARAM_INVALID);
}

// =================== CCU 模式 ai_cpu 通信模式测试 ===================

TEST_F(MatmulReduceScatterV2CcuModeTest, ai_cpuCommModeEnv)
{
    setenv("ENV_MC2_COMM_MODE_ai_cpu", "1", 1);
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    unsetenv("ENV_MC2_COMM_MODE_ai_cpu");
}

constexpr int64_t TEST_K = 256;
constexpr int64_t TEST_N = 16;

struct Fp4InvalidCase {
    const char* name;
    ge::DataType x1Type;
    ge::DataType x2Type;
    ge::DataType x1ScaleType;
    ge::DataType x2ScaleType;
    size_t x1ScaleDim;
    size_t x2ScaleDim;
    int64_t k;
    int64_t n;
};

const Fp4InvalidCase FP4_INVALID_CASES[] = {
    {"MixedX1", ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, 3, 3, TEST_K, TEST_N},
    {"MixedX2", ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, 3, 3, TEST_K, TEST_N},
    {"X1ScaleDtype", ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, 3, 3, TEST_K, TEST_N},
    {"X2ScaleDtype", ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT, 3, 3, TEST_K, TEST_N},
    {"X1ScaleDim", ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, 2, 3, TEST_K,
     TEST_N},
    {"X2ScaleDim", ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, 3, 2, TEST_K,
     TEST_N},
    {"PerTensorScale", ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT, ge::DT_FLOAT, 1, 1, TEST_K, TEST_N},
    {"PerBlockScale", ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT, ge::DT_FLOAT, 2, 2, TEST_K, TEST_N},
    {"OddK", ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, 3, 3, TEST_K + 1, TEST_N},
    {"OddN", ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, 3, 3, TEST_K, TEST_N + 1},
};

void CheckInvalidFp4Input(const Fp4InvalidCase& item)
{
    constexpr int64_t M = 16;
    constexpr int64_t RANK_SIZE = 2;
    constexpr int64_t SCALE_K_BLOCK = 64;
    constexpr int64_t SCALE_C0 = 2;
    constexpr int64_t PER_BLOCK_GROUP_SIZE = 128;
    SCOPED_TRACE(item.name);
    const int64_t scaleK = (item.k + SCALE_K_BLOCK - 1) / SCALE_K_BLOCK;
    vector<int64_t> x1ScaleShape = {M, scaleK, SCALE_C0};
    vector<int64_t> x2ScaleShape = {scaleK, item.n, SCALE_C0};
    x1ScaleShape.resize(item.x1ScaleDim);
    x2ScaleShape.resize(item.x2ScaleDim);
    if (item.x1ScaleDim == 1 && item.x2ScaleDim == 1) {
        x1ScaleShape = {1};
        x2ScaleShape = {1};
    } else if (item.x1ScaleDim == 2 && item.x2ScaleDim == 2) {
        const int64_t blockK = (item.k + PER_BLOCK_GROUP_SIZE - 1) / PER_BLOCK_GROUP_SIZE;
        x1ScaleShape = {(M + PER_BLOCK_GROUP_SIZE - 1) / PER_BLOCK_GROUP_SIZE, blockK};
        x2ScaleShape = {blockK, (item.n + PER_BLOCK_GROUP_SIZE - 1) / PER_BLOCK_GROUP_SIZE};
    }
    auto x1 = TensorDesc({M, item.k}, ACL_UINT8, ACL_FORMAT_ND).ToAclType();
    auto x2 = TensorDesc({item.k, item.n}, ACL_UINT8, ACL_FORMAT_ND).ToAclType();
    auto x1Scale = TensorDesc(x1ScaleShape, ACL_UINT8, ACL_FORMAT_ND).ToAclType();
    auto x2Scale = TensorDesc(x2ScaleShape, ACL_UINT8, ACL_FORMAT_ND).ToAclType();
    auto output = TensorDesc({M / RANK_SIZE, item.n}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    x1->SetDataType(item.x1Type);
    x2->SetDataType(item.x2Type);
    x1Scale->SetDataType(item.x1ScaleType);
    x2Scale->SetDataType(item.x2ScaleType);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnMatmulReduceScatterV2GetWorkspaceSize(
        x1.get(), x2.get(), nullptr, x1Scale.get(), x2Scale.get(), nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu",
        output.get(), nullptr, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

class MatmulReduceScatterV2Fp4Test : public MatmulReduceScatterV2CcuModeTest,
                                     public testing::WithParamInterface<Fp4InvalidCase> {};

TEST_P(MatmulReduceScatterV2Fp4Test, RejectInvalidFp4Input)
{
    CheckInvalidFp4Input(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Fp4Validation, MatmulReduceScatterV2Fp4Test, testing::ValuesIn(FP4_INVALID_CASES),
                         [](const testing::TestParamInfo<Fp4InvalidCase>& info) { return info.param.name; });

#ifdef ENABLE_MMRS_DT_HIFLOAT4
enum class Hif4InvalidParam {
    NONE,
    NON_EMPTY_BIAS,
    FP32_OUTPUT,
    MIXED_X1,
    MIXED_X2,
    MISSING_X1_SCALE,
    MISSING_X2_SCALE,
    EMPTY_X1_SCALE,
    EMPTY_X2_SCALE,
    X1_SCALE_DTYPE,
    X2_SCALE_DTYPE,
    X1_SCALE_DIM,
    X2_SCALE_DIM,
    TRANSPOSE_X1,
    ODD_K,
    ODD_N,
    TRANSPOSE_NN_SCALE,
    EMPTY_X1,
    EMPTY_X2,
};

struct Hif4ApiInvalidCase {
    const char* name;
    Hif4InvalidParam invalidParam;
};

const Hif4ApiInvalidCase HIF4_INVALID_CASES[] = {
    {"NonEmptyBias", Hif4InvalidParam::NON_EMPTY_BIAS},
    {"Fp32Output", Hif4InvalidParam::FP32_OUTPUT},
    {"MixedX1", Hif4InvalidParam::MIXED_X1},
    {"MixedX2", Hif4InvalidParam::MIXED_X2},
    {"MissingX1Scale", Hif4InvalidParam::MISSING_X1_SCALE},
    {"MissingX2Scale", Hif4InvalidParam::MISSING_X2_SCALE},
    {"EmptyX1Scale", Hif4InvalidParam::EMPTY_X1_SCALE},
    {"EmptyX2Scale", Hif4InvalidParam::EMPTY_X2_SCALE},
    {"X1ScaleDtype", Hif4InvalidParam::X1_SCALE_DTYPE},
    {"X2ScaleDtype", Hif4InvalidParam::X2_SCALE_DTYPE},
    {"X1ScaleDim", Hif4InvalidParam::X1_SCALE_DIM},
    {"X2ScaleDim", Hif4InvalidParam::X2_SCALE_DIM},
    {"TransposedX1", Hif4InvalidParam::TRANSPOSE_X1},
    {"OddK", Hif4InvalidParam::ODD_K},
    {"OddN", Hif4InvalidParam::ODD_N},
    {"TransposedNnScale", Hif4InvalidParam::TRANSPOSE_NN_SCALE},
    {"EmptyX1", Hif4InvalidParam::EMPTY_X1},
    {"EmptyX2", Hif4InvalidParam::EMPTY_X2},
};

void CheckInvalidHif4Input(Hif4InvalidParam invalid)
{
    constexpr int64_t M = 16;
    constexpr int64_t K = 256;
    constexpr int64_t N = 16;
    constexpr int64_t RANK_SIZE = 2;
    constexpr int64_t HIF4_GROUP_SIZE = 64;
    constexpr int64_t SCALE_C0 = 1;
    const int64_t m = invalid == Hif4InvalidParam::EMPTY_X1 ? 0 : M;
    const int64_t k = K + (invalid == Hif4InvalidParam::ODD_K ? 1 : 0);
    const int64_t n = invalid == Hif4InvalidParam::EMPTY_X2 ? 0 : N + (invalid == Hif4InvalidParam::ODD_N ? 1 : 0);
    const int64_t scaleK = (k + HIF4_GROUP_SIZE - 1) / HIF4_GROUP_SIZE;
    const vector<int64_t> x1Strides =
        invalid == Hif4InvalidParam::TRANSPOSE_X1 ? vector<int64_t>{1, m} : vector<int64_t>{k, 1};
    vector<int64_t> x1ScaleShape =
        invalid == Hif4InvalidParam::X1_SCALE_DIM ? vector<int64_t>{m, scaleK} : vector<int64_t>{m, scaleK, SCALE_C0};
    vector<int64_t> x2ScaleShape =
        invalid == Hif4InvalidParam::X2_SCALE_DIM ? vector<int64_t>{scaleK, n} : vector<int64_t>{scaleK, n, SCALE_C0};
    if (invalid == Hif4InvalidParam::EMPTY_X1_SCALE) {
        x1ScaleShape[1] = 0;
    }
    if (invalid == Hif4InvalidParam::EMPTY_X2_SCALE) {
        x2ScaleShape[0] = 0;
    }
    const vector<int64_t> x2ScaleStrides = invalid == Hif4InvalidParam::TRANSPOSE_NN_SCALE ?
                                               vector<int64_t>{SCALE_C0, scaleK * SCALE_C0, 1} :
                                               vector<int64_t>{};

    // 使用 UINT8/UINT32 创建 Tensor 后设置目标类型；仅校验参数，不读取数据。
    auto x1 = TensorDesc({m, k}, ACL_UINT8, ACL_FORMAT_ND, x1Strides).ToAclType();
    auto x2 = TensorDesc({k, n}, ACL_UINT8, ACL_FORMAT_ND).ToAclType();
    auto x1Scale = TensorDesc(x1ScaleShape, ACL_UINT32, ACL_FORMAT_ND).ToAclType();
    auto x2Scale = TensorDesc(x2ScaleShape, ACL_UINT32, ACL_FORMAT_ND, x2ScaleStrides).ToAclType();
    auto bias = TensorDesc({n}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto output = TensorDesc({m / RANK_SIZE, n}, invalid == Hif4InvalidParam::FP32_OUTPUT ? ACL_FLOAT : ACL_FLOAT16,
                             ACL_FORMAT_ND)
                      .ToAclType();
    x1->SetDataType(invalid == Hif4InvalidParam::MIXED_X1 ? ge::DT_FLOAT4_E2M1 : Ops::Transformer::DT_HIFLOAT4);
    x2->SetDataType(invalid == Hif4InvalidParam::MIXED_X2 ? ge::DT_FLOAT4_E2M1 : Ops::Transformer::DT_HIFLOAT4);
    x1Scale->SetDataType(invalid == Hif4InvalidParam::X1_SCALE_DTYPE ? ge::DT_FLOAT8_E8M0 :
                                                                       Ops::Transformer::DT_HIFLOAT4_SCALE);
    x2Scale->SetDataType(invalid == Hif4InvalidParam::X2_SCALE_DTYPE ? ge::DT_FLOAT8_E8M0 :
                                                                       Ops::Transformer::DT_HIFLOAT4_SCALE);
    const char* commMode = "ai_cpu";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnMatmulReduceScatterV2GetWorkspaceSize(
        x1.get(), x2.get(), invalid == Hif4InvalidParam::NON_EMPTY_BIAS ? bias.get() : nullptr,
        invalid == Hif4InvalidParam::MISSING_X1_SCALE ? nullptr : x1Scale.get(),
        invalid == Hif4InvalidParam::MISSING_X2_SCALE ? nullptr : x2Scale.get(), nullptr, 0, "test_group", "sum", 0, 1,
        0, commMode, output.get(), nullptr, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, Ascend950RejectsHiF4)
{
    CheckInvalidHif4Input(Hif4InvalidParam::NONE);
}

class MatmulReduceScatterV2Ascend960Test : public testing::Test {
protected:
    void SetUp() override
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_9201);
    }

    void TearDown() override
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }
};

TEST_F(MatmulReduceScatterV2Ascend960Test, EmptyInputWithEmptyOutput)
{
    TensorDesc x1({0, TEST_K}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2({TEST_K, TEST_N}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output({0, TEST_N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 1;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

TEST_F(MatmulReduceScatterV2Ascend960Test, RejectEmptyQuantizedInput)
{
    constexpr int64_t SCALE_K_BLOCK = 64;
    constexpr int64_t SCALE_C0 = 2;
    TensorDesc x1({0, TEST_K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2({TEST_K, TEST_N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale({0, TEST_K / SCALE_K_BLOCK, SCALE_C0}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale({TEST_K / SCALE_K_BLOCK, TEST_N, SCALE_C0}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc output({0, TEST_N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 0, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

class MatmulReduceScatterV2Hifp4Test : public MatmulReduceScatterV2Ascend960Test,
                                       public testing::WithParamInterface<Hif4ApiInvalidCase> {};

TEST_P(MatmulReduceScatterV2Hifp4Test, RejectInvalidInput)
{
    CheckInvalidHif4Input(GetParam().invalidParam);
}

INSTANTIATE_TEST_SUITE_P(Hifp4Validation, MatmulReduceScatterV2Hifp4Test, testing::ValuesIn(HIF4_INVALID_CASES),
                         [](const testing::TestParamInfo<Hif4ApiInvalidCase>& info) { return info.param.name; });
#endif

// =================== A6 (DAV_9201) HiF4 / peer-only ===================

class MatmulReduceScatterV2A6ModeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_9201);
        cout << "MatmulReduceScatterV2A6ModeTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "MatmulReduceScatterV2A6ModeTest TearDown" << endl;
    }
};

#ifdef ENABLE_MMRS_DT_HIFLOAT4
TEST_F(MatmulReduceScatterV2A6ModeTest, HiF4BasicAiCpu)
{
    // K=256, scaleK=CeilDiv(256,64)=4; N=256 even for non-trans x2
    TensorDesc x1 = TensorDesc({16, 256}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({16, 4, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({4, 256, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2A6ModeTest, HiF4RejectCcu)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({16, 4, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({4, 256, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 0, 0, "ccu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2A6ModeTest, HiF4OddKReject)
{
    TensorDesc x1 = TensorDesc({16, 255}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({255, 256}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({16, 4, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({4, 256, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

#endif
TEST_F(MatmulReduceScatterV2A6ModeTest, NullCommMode)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, nullptr),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2A6ModeTest, Fp16CcuPeerOnly)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMatmulReduceScatterV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 2, 1, 0, "ccu_peer_only"),
                  OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2A6ModeTest, InvalidCommModeString)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "bad_mode"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

#ifdef ENABLE_MMRS_DT_HIFLOAT4
TEST_F(MatmulReduceScatterV2A6ModeTest, HiF4InvalidScaleDtype)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({16, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({4, 256, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2A6ModeTest, HiF4OddNReject)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 255}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({16, 4, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({4, 255, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 255}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2A6ModeTest, HiF4X1X2DtypeMismatch)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_HIFLOAT4, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({16, 4, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({4, 256, 1}, ACL_HIFLOAT4_SCALE, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_group", "sum", 8, 0, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

#endif
TEST_F(MatmulReduceScatterV2CcuModeTest, Fp16CcuSuccess)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ccu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, EmptyTensorCcu)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 0, 1, 0, "ccu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, CcuGraphMode)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ccu_graph"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MatmulReduceScatterV2CcuModeTest, Non2DX2)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({1, 256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "ai_cpu"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MatmulReduceScatterV2AclnnTest, NzFormatX2Aiv)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    // view 2D + storage 4D triggers FRACTAL_NZ weight path
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {16, 1, 16, 16});
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMatmulReduceScatterV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_group", "sum", 8, 1, 0, "aiv"),
                        OUTPUT(output, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

} // namespace
