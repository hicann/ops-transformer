/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <gtest/gtest.h>
#include "../../../../op_host/op_api/aclnn_mhc_post_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class MhcPostBackwardOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MhcPostBackwardOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "MhcPostBackwardOpapiUt TearDown" << endl;
    }
};

TEST_F(MhcPostBackwardOpapiUt, basic_4d_fp16)
{
    auto tensorGradOutput = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcPostBackwardOpapiUt, basic_4d_bf16)
{
    auto tensorGradOutput = TensorDesc({2, 64, 4, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorX = TensorDesc({2, 64, 4, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHRes = TensorDesc({2, 64, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHOut = TensorDesc({2, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHPost = TensorDesc({2, 64, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorGradX = TensorDesc({2, 64, 4, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorGradHRes = TensorDesc({2, 64, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorGradHOut = TensorDesc({2, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorGradHPost = TensorDesc({2, 64, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcPostBackwardOpapiUt, basic_3d_fp16)
{
    auto tensorGradOutput = TensorDesc({512, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({512, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({512, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({512, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({512, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({512, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({512, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({512, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({512, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcPostBackwardOpapiUt, basic_3d_bf16)
{
    auto tensorGradOutput = TensorDesc({256, 4, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorX = TensorDesc({256, 4, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHRes = TensorDesc({256, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHOut = TensorDesc({256, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHPost = TensorDesc({256, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorGradX = TensorDesc({256, 4, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorGradHRes = TensorDesc({256, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorGradHOut = TensorDesc({256, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorGradHPost = TensorDesc({256, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcPostBackwardOpapiUt, null_gradOutput)
{
    auto tensorGradOutput = nullptr;
    auto tensorX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MhcPostBackwardOpapiUt, null_x)
{
    auto tensorGradOutput = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = nullptr;
    auto tensorHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MhcPostBackwardOpapiUt, null_gradX)
{
    auto tensorGradOutput = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = nullptr;
    auto tensorGradHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MhcPostBackwardOpapiUt, empty_tensor_4d_dim0)
{
    auto tensorGradOutput = TensorDesc({0, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({0, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({0, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({0, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({0, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({0, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({0, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({0, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({0, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostBackwardOpapiUt, empty_tensor_3d_dim0)
{
    auto tensorGradOutput = TensorDesc({0, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({0, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({0, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({0, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({0, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({0, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({0, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostBackwardOpapiUt, invalid_dtype_gradOutput)
{
    auto tensorGradOutput = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostBackwardOpapiUt, invalid_dtype_hRes)
{
    auto tensorGradOutput = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostBackwardOpapiUt, invalid_shape_gradOutput_dim0)
{
    auto tensorGradOutput = TensorDesc({2, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostBackwardOpapiUt, invalid_shape_hRes_nxn)
{
    auto tensorGradOutput = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostBackwardOpapiUt, invalid_dim_2d)
{
    auto tensorGradOutput = TensorDesc({128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostBackwardOpapiUt, invalid_dim_5d)
{
    auto tensorGradOutput = TensorDesc({1, 128, 4, 1024, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorX = TensorDesc({1, 128, 4, 1024, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 4, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({1, 128, 4, 1024, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({1, 128, 4, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostBackwardOpapiUt, invalid_format_gradOutput)
{
    auto tensorGradOutput = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    auto tensorX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradX = TensorDesc({1, 128, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHRes = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorGradHPost = TensorDesc({1, 128, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPostBackward, INPUT(tensorGradOutput, tensorX, tensorHRes, tensorHOut, tensorHPost),
                        OUTPUT(tensorGradX, tensorGradHRes, tensorGradHOut, tensorGradHPost));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
