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
#include "../../../../op_host/op_api/aclnn_mhc_sinkhorn_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class MhcSinkhornBackwardOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MhcSinkhornBackwardOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "MhcSinkhornBackwardOpapiUt TearDown" << endl;
    }
};

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_basic_4d_fp32)
{
    auto gradOutput = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({81920}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({20480}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_basic_3d_fp32)
{
    auto gradOutput = TensorDesc({1024, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({1474560}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({245760}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1024, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_empty_tensor)
{
    auto gradOutput = TensorDesc({0, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({0, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_invalid_dtype_grad_output)
{
    auto gradOutput = TensorDesc({12, 64, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto normOut = TensorDesc({1966080}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({327680}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({12, 64, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_invalid_dtype_norm_out)
{
    auto gradOutput = TensorDesc({12, 64, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({1966080}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({327680}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({12, 64, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_invalid_dtype_sum_out)
{
    auto gradOutput = TensorDesc({12, 64, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({1966080}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({327680}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({12, 64, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_invalid_dtype_out)
{
    auto gradOutput = TensorDesc({12, 64, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({1966080}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({327680}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({12, 64, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_output_shape_mismatch)
{
    auto gradOutput = TensorDesc({2, 16, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({737280}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({122880}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_n8_4d)
{
    auto gradOutput = TensorDesc({4, 256, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({1310720}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({163840}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 256, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_n6_3d)
{
    auto gradOutput = TensorDesc({512, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({1474560}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({245760}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({512, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_n4_4d)
{
    auto gradOutput = TensorDesc({2, 256, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({327680}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({81920}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 256, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_n4_3d)
{
    auto gradOutput = TensorDesc({512, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({327680}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({81920}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({512, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_large_batch)
{
    auto gradOutput = TensorDesc({4, 256, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({655360}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({163840}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({4, 256, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(MhcSinkhornBackwardOpapiUt, aclnn_mhc_sinkhorn_backward_large_t)
{
    auto gradOutput = TensorDesc({4096, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({13107200}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({1638400}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({4096, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcSinkhornBackward, INPUT(gradOutput, normOut, sumOut), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
