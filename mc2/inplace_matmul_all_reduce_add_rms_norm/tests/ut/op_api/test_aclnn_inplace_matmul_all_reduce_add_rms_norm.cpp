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
#include "../../../op_api/aclnn_inplace_matmul_all_reduce_add_rms_norm.h"
#include "../../../op_api/aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.h"
#include "../../../op_api/aclnn_inplace_weight_quant_matmul_all_reduce_add_rms_norm.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

// IFA aclnn ut for 910b has error in UT environment. Deleted.
class L2InplaceMatmulAllReduceAddRmsNormTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2InplaceMatmulAllReduceAddRmsNormTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2InplaceMatmulAllReduceAddRmsNormTest TearDown" << endl;
    }
};

TEST_F(L2InplaceMatmulAllReduceAddRmsNormTest, TestInplaceMmAllReduceAddRmsNormFirstApi)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
                        OUTPUT(normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2InplaceMatmulAllReduceAddRmsNormTest, TestInplaceMmAllReduceAddRmsNormWrongStreamMode)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, bias, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 0),
                        OUTPUT(normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2InplaceMatmulAllReduceAddRmsNormTest, TestInplaceQuantMmAllReduceAddRmsNormFirstApi)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc biasDesc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScaleDesc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnInplaceQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, biasDesc, dequantScaleDesc, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 1),
        OUTPUT(normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2InplaceMatmulAllReduceAddRmsNormTest, TestInplaceQuantMmAllReduceAddRmsNormWrongStreamMode)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc biasDesc = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc dequantScaleDesc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnInplaceQuantMatmulAllReduceAddRmsNorm,
        INPUT(x1Desc, x2Desc, biasDesc, dequantScaleDesc, residualDesc, gammaDesc, 0.000001, "test_group", "sum", 8, 0),
        OUTPUT(normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2InplaceMatmulAllReduceAddRmsNormTest, TestInplaceWeightQuantMmAllReduceAddRmsNormFirstApi)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc biasDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScaleDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffsetDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, biasDesc, antiquantScaleDesc, antiquantOffsetDesc, residualDesc,
                              gammaDesc, 0.000001, "test_group", "sum", 8, 1, 0),
                        OUTPUT(normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2InplaceMatmulAllReduceAddRmsNormTest, TestInplaceWeightQuantMmAllReduceAddRmsNormWrongStreamMode)
{
    TensorDesc x1Desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc biasDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScaleDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffsetDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc residualDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc normOutDesc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm,
                        INPUT(x1Desc, x2Desc, biasDesc, antiquantScaleDesc, antiquantOffsetDesc, residualDesc,
                              gammaDesc, 0.000001, "test_group", "sum", 8, 0, 0),
                        OUTPUT(normOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
