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
#include "../../../op_api/aclnn_all_to_all_all_gather_batch_matmul.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace AlltoAllAllGatherBatchMatmul {
class L2AllToAllAllGatherBatchMatmulTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
        cout << "L2AllToAllAllGatherBatchMatmulTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AllToAllAllGatherBatchMatmulTest TearDown" << endl;
    }
};

// E = 4, C = 2, H = 6, ep = 2, tp = 2, M = 4
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulXshardtype0)
{
    TensorDesc x = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 0, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// xShardType = 1, E = 4, C = 2, H = 6, ep = 2, tp = 2, M = 4
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulXshardtype1)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// xShardType 不是 0 或 1 , E = 4, C = 2, H = 6, ep = 2, tp = 2, M = 4
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulXshardtypeInvalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 5, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// y2_out 不输出
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulY2OutFalse)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 4),
                        OUTPUT(y1OutDesc, nullptr, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// y3_out 因actType不输出
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulY3OutFalse)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 0),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ep name 非法
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulEpGroupInvalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                  INPUT(x, weight, biasOptional, "", "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                  OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// tp name 为空
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulTpGroupNull)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAlltoAllAllGatherBatchMatMul,
        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group", nullptr, 2, 2, 1, 1),
        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// weight 三维判断
TEST_F(L2AllToAllAllGatherBatchMatmulTest, testAllToAllAllGatherBatchMatmulWeightDimInvalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// H != H/tp * tp
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulHInvalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// E != E/ep * ep
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulEInvalid)
{
    TensorDesc x = TensorDesc({3, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// K = 0
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulK0)
{
    TensorDesc x = TensorDesc({4, 1, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 0, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// actType invalid
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulActtypeInvalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 8),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ep world size invalid
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulEpworldsizeInvalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 3, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// tp world size invalid
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulTpworldsizeInvalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 7, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// bias shape invalid
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulBiasShapeInvalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// transpose
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulTranspose)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {12, 1, 6});
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// M/tp > 65535
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulMTp65536)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// C = 0
TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulC0)
{
    TensorDesc x = TensorDesc({4, 0, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 0, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 0, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 0, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulY1And0Invalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({3, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulY1And2Invalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulY1And1Invalid)
{
    TensorDesc x = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 5, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 1, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulXshardtype0W1Invalid)
{
    TensorDesc x = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 5, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 0, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AllToAllAllGatherBatchMatmulTest, TestAllToAllAllGatherBatchMatmulXshardtype0Y1Invalid)
{
    TensorDesc x = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({2, 6, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc biasOptional = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y1OutDesc = TensorDesc({2, 5, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y2OutOptionalDesc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y3OutOptionalDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllAllGatherBatchMatMul,
                        INPUT(x, weight, biasOptional, "test_all_to_all_all_gather_batch_matmul_ep_group",
                              "test_all_to_all_all_gather_batch_matmul_tp_group", 2, 2, 0, 1),
                        OUTPUT(y1OutDesc, y2OutOptionalDesc, y3OutOptionalDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
} // namespace AlltoAllAllGatherBatchMatmul
