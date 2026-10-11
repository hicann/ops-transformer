/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cfloat>

#include <array>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_matmul_all_reduce_add_rms_norm.h"
#include "../../../op_api/aclnn_quant_matmul_all_reduce_add_rms_norm.h"
#include "../../../op_api/aclnn_weight_quant_matmul_all_reduce_add_rms_norm.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MatmulAllReduceAddRmsNormUT {

class L2MatmulAllReduceAddRmsNormTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MatmulAllReduceAddRmsNormTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MatmulAllReduceAddRmsNormTest TearDown" << endl;
    }
};

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormFirstApi)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNorm_empty_M)
{
    TensorDesc x1Desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormEmptyK)
{
    TensorDesc x1Desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormEmptyN)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormWrongK)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({30, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormWrongN)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormFirstApi)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNorm_empty_M)
{
    TensorDesc x1Desc = TensorDesc({0, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormEmptyK)
{
    TensorDesc x1Desc = TensorDesc({16, 0}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({0, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormEmptyN)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 0}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({0}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNormFirstApi)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, antiquantScale, antiquantOffset, residualDesc, gammaDesc, 0.000001,
                              "test_group", "sum", 8, 1, 0),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNorm_empty_M)
{
    TensorDesc x1Desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, antiquantScale, antiquantOffset, residualDesc, gammaDesc, 0.000001,
                              "test_group", "sum", 8, 1, 0),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNormEmptyK)
{
    TensorDesc x1Desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({0, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, antiquantScale, antiquantOffset, residualDesc, gammaDesc, 0.000001,
                              "test_group", "sum", 8, 1, 0),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNormEmptyN)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 0}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, antiquantScale, antiquantOffset, residualDesc, gammaDesc, 0.000001,
                              "test_group", "sum", 8, 1, 0),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNormWithoutOffset)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, antiquantScale, nullptr, residualDesc, gammaDesc, 0.000001,
                              "test_group", "sum", 8, 1, 0),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormWrongStreamMode)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 2),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormWrongStreamMode)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 2),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNormWrongStreamMode)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, antiquantScale, antiquantOffset, residualDesc, gammaDesc, 0.000001,
                              "test_group", "sum", 8, 2, 0),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormBfloat16)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormBfloat16)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormInt64Scale)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNormBfloat16)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, antiquantScale, antiquantOffset, residualDesc, gammaDesc, 0.000001,
                              "test_group", "sum", 8, 1, 0),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNorm3DInput)
{
    TensorDesc x1Desc = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNorm3DInput)
{
    TensorDesc x1Desc = TensorDesc({1, 16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNorm3DInput)
{
    TensorDesc x1Desc = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, antiquantScale, antiquantOffset, residualDesc, gammaDesc, 0.000001,
                              "test_group", "sum", 8, 1, 0),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormWithoutBias)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, nullptr, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNormWithoutBias)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, nullptr, antiquantScale, antiquantOffset, residualDesc, gammaDesc,
                              0.000001, "test_group", "sum", 8, 1, 0),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 第二阶段函数覆盖 ====================

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMatmulAllReduceAddRmsNormSecondApiEmptyWorkspace)
{
    aclnnStatus ret = aclnnMatmulAllReduceAddRmsNorm(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormSecondApiEmptyWorkspace)
{
    aclnnStatus ret = aclnnQuantMatmulAllReduceAddRmsNorm(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNormSecondApiEmptyWorkspace)
{
    aclnnStatus ret = aclnnWeightQuantMatmulAllReduceAddRmsNorm(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMatmulAllReduceAddRmsNormSecondApiWithExecutor)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    uint64_t fakeWorkspaceSize = (workspaceSize > 0) ? workspaceSize : 1024;
    void* fakeWorkspace = malloc(fakeWorkspaceSize);
    ASSERT_NE(fakeWorkspace, nullptr);
    aclnnStatus ret = aclnnMatmulAllReduceAddRmsNorm(fakeWorkspace, fakeWorkspaceSize, executor, nullptr);
    free(fakeWorkspace);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormSecondApiWithExecutor)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    uint64_t fakeWorkspaceSize = (workspaceSize > 0) ? workspaceSize : 1024;
    void* fakeWorkspace = malloc(fakeWorkspaceSize);
    ASSERT_NE(fakeWorkspace, nullptr);
    aclnnStatus ret = aclnnQuantMatmulAllReduceAddRmsNorm(fakeWorkspace, fakeWorkspaceSize, executor, nullptr);
    free(fakeWorkspace);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNormSecondApiWithExecutor)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, antiquantScale, antiquantOffset, residualDesc, gammaDesc, 0.000001,
                              "test_group", "sum", 8, 1, 0),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    uint64_t fakeWorkspaceSize = (workspaceSize > 0) ? workspaceSize : 1024;
    void* fakeWorkspace = malloc(fakeWorkspaceSize);
    ASSERT_NE(fakeWorkspace, nullptr);
    aclnnStatus ret = aclnnWeightQuantMatmulAllReduceAddRmsNorm(fakeWorkspace, fakeWorkspaceSize, executor, nullptr);
    free(fakeWorkspace);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormScalarInput)
{
    TensorDesc x1Desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, nullptr, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 更多边界场景覆盖 ====================

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormNullX1)
{
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(nullptr, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormNullX2)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, nullptr, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormNullResidual)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, nullptr, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormNullGamma)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, nullptr, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormNullDequantScale)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnQuantMatmulAllReduceAddRmsNorm,
                  INPUT(x1Desc, x2Desc, bias, nullptr, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                  OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestWeightQuantMatmulAllReduceAddRmsNormNullAntiquantScale)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnWeightQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, nullptr, nullptr, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1, 0),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestMmAllReduceAddRmsNormKMismatch2D)
{
    TensorDesc x1Desc = TensorDesc({16, 31}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormDequantScaleBfloat16)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MatmulAllReduceAddRmsNormTest, TestQuantMatmulAllReduceAddRmsNormDequantScaleFloat)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, bias, dequantScale, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(yDesc, normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // namespace MatmulAllReduceAddRmsNormUT
