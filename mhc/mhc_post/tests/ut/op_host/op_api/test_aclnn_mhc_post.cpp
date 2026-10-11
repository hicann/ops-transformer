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
#include "../../../../op_host/op_api/aclnn_mhc_post.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class MhcPostOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MhcPostOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "MhcPostOpapiUt TearDown" << endl;
    }
};

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_basic_4d_fp16)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_basic_4d_bf16)
{
    auto tensorX = TensorDesc({2, 64, 8, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHRes = TensorDesc({2, 64, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHOut = TensorDesc({2, 64, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHPost = TensorDesc({2, 64, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorOut = TensorDesc({2, 64, 8, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_basic_3d_fp16)
{
    auto tensorX = TensorDesc({512, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({512, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({512, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({512, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({512, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_basic_3d_bf16)
{
    auto tensorX = TensorDesc({256, 8, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHRes = TensorDesc({256, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHOut = TensorDesc({256, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorHPost = TensorDesc({256, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorOut = TensorDesc({256, 8, 512}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_empty_tensor_4d_dim0)
{
    auto tensorX = TensorDesc({0, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({0, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({0, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({0, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({0, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_empty_tensor_4d_dim1)
{
    auto tensorX = TensorDesc({1, 0, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 0, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 0, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 0, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 0, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_empty_tensor_3d_dim0)
{
    auto tensorX = TensorDesc({0, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({0, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({0, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({0, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({0, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_empty_null_x)
{
    auto tensorX = nullptr;
    auto tensorHRes = TensorDesc({1, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_empty_null_hRes)
{
    auto tensorX = TensorDesc({1, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = nullptr;
    auto tensorHOut = TensorDesc({1, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_empty_null_out)
{
    auto tensorX = TensorDesc({1, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = nullptr;

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_invalid_dtype_hRes)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_invalid_dtype_hOut)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_invalid_dtype_out)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_invalid_dtype_x)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_invalid_shape_hRes_dim0)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({2, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_invalid_shape_hRes_nxn)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_invalid_shape_hOut_dim2)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_invalid_dim_2d)
{
    auto tensorX = TensorDesc({128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_invalid_dim_5d)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPostOpapiUt, aclnn_mhc_post_invalid_format_x)
{
    auto tensorX = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    auto tensorHRes = TensorDesc({1, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHOut = TensorDesc({1, 128, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorHPost = TensorDesc({1, 128, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOut = TensorDesc({1, 128, 16, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnMhcPost, INPUT(tensorX, tensorHRes, tensorHOut, tensorHPost), OUTPUT(tensorOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
