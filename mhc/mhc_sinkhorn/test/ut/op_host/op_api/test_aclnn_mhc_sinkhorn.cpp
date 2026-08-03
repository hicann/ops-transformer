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
#include "../../../../op_host/op_api/aclnn_mhc_sinkhorn.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class MhcSinkhornOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MhcSinkhornOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "MhcSinkhornOpapiUt TearDown" << endl;
    }
};

TEST_F(MhcSinkhornOpapiUt, aclnn_mhc_sinkhorn_basic_4d_fp32)
{
    auto x = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto outPut = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({40, 4, 8, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto SumOut = TensorDesc({40, 8, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float eps = 1e-6;
    int64_t num_iters = 20;

    auto ut = OP_API_UT(
        aclnnMhcSinkhorn,
        INPUT(x, eps, num_iters),
        OUTPUT(outPut, normOut, SumOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcSinkhornOpapiUt, aclnn_mhc_sinkhorn_basic_3d_fp32)
{
    auto x = TensorDesc({1024, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto outPut = TensorDesc({1024, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({30, 6, 8, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto SumOut = TensorDesc({30, 8, 1024}, ACL_FLOAT, ACL_FORMAT_ND);

    float eps = 1e-7;
    int64_t num_iters = 15;

    auto ut = OP_API_UT(
        aclnnMhcSinkhorn,
        INPUT(x, eps, num_iters),
        OUTPUT(outPut, normOut, SumOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcSinkhornOpapiUt, aclnn_mhc_sinkhorn_empty_tensor_4d)
{
    auto x = TensorDesc({1024, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto outPut = TensorDesc({1, 0, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({1, 0, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto SumOut = TensorDesc({1, 0, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    float eps = 1e-8;
    int64_t num_iters = 12;

    auto ut = OP_API_UT(
        aclnnMhcSinkhorn,
        INPUT(x, eps, num_iters),
        OUTPUT(outPut, normOut, SumOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(MhcSinkhornOpapiUt, aclnn_mhc_sinkhorn_invalid_dtype)
{
    auto x = TensorDesc({12, 64, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto outPut = TensorDesc({12, 64, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto normOut = TensorDesc({12, 64, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto SumOut = TensorDesc({12, 64, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);

    float eps = 1e-6;
    int64_t num_iters = 5;

    auto ut = OP_API_UT(
        aclnnMhcSinkhorn,
        INPUT(x, eps, num_iters),
        OUTPUT(outPut, normOut, SumOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcSinkhornOpapiUt, aclnn_mhc_sinkhorn_invalid_shape)
{
    auto x = TensorDesc({1, 32, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto outPut = TensorDesc({1, 32, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({1, 32, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto SumOut = TensorDesc({1, 32, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    float eps = 1e-6;
    int64_t num_iters = 5;

    auto ut = OP_API_UT(
        aclnnMhcSinkhorn,
        INPUT(x, eps, num_iters),
        OUTPUT(outPut, normOut, SumOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcSinkhornOpapiUt, aclnn_mhc_sinkhorn_invalid_attr_num_iters)
{
    auto x = TensorDesc({2, 16, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto outPut = TensorDesc({2, 16, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({2, 16, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto SumOut = TensorDesc({2, 16, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    float eps = 1e-6;
    int64_t num_iters = 101;

    auto ut = OP_API_UT(
        aclnnMhcSinkhorn,
        INPUT(x, eps, num_iters),
        OUTPUT(outPut, normOut, SumOut));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
