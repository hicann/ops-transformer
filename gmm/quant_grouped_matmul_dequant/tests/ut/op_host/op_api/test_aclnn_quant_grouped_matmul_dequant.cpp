/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_quant_grouped_matmul_dequant.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
namespace {
class l2_quant_grouped_matmul_dequant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND310P);
        cout << "l2_quant_grouped_matmul_dequant_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_quant_grouped_matmul_dequant_test TearDown" << endl;
    }
};

TEST_F(l2_quant_grouped_matmul_dequant_test, ascend310P_normal_1)
{
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({4, 512, 256}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({4, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto group_list_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{7, 29, 31, 64});
    aclTensor* bias_desc = nullptr;
    auto x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* x_offset_desc = nullptr;
    bool weightTrans = true;
    auto smooth_scale_desc = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";

    auto ut = OP_API_UT(aclnnQuantGroupedMatmulDequant,
                        INPUT(x_desc, weight_desc, weight_scale_desc, group_list_desc, bias_desc, x_scale_desc,
                              x_offset_desc, smooth_scale_desc, quantMode, weightTrans),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_quant_grouped_matmul_dequant_test, ascend310P_normal_2)
{
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({4, 8, 32, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto weight_scale_desc = TensorDesc({4, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto group_list_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{7, 29, 31, 64});
    aclTensor* bias_desc = nullptr;
    auto x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* x_offset_desc = nullptr;
    auto smooth_scale_desc = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    bool weightTrans = true;

    auto ut = OP_API_UT(aclnnQuantGroupedMatmulDequant,
                        INPUT(x_desc, weight_desc, weight_scale_desc, group_list_desc, bias_desc, x_scale_desc,
                              x_offset_desc, smooth_scale_desc, quantMode, weightTrans),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_quant_grouped_matmul_dequant_test, ascend310P_normal_3)
{
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({4, 8, 32, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto weight_scale_desc = TensorDesc({4, 512}, ACL_INT64, ACL_FORMAT_ND);
    auto group_list_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{7, 29, 31, 64});
    aclTensor* bias_desc = nullptr;
    auto x_scale_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* x_offset_desc = nullptr;
    auto smooth_scale_desc = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool weightTrans = true;
    char* quantMode = "pertoken";

    auto ut = OP_API_UT(aclnnQuantGroupedMatmulDequant,
                        INPUT(x_desc, weight_desc, weight_scale_desc, group_list_desc, bias_desc, x_scale_desc,
                              x_offset_desc, smooth_scale_desc, quantMode, weightTrans),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_quant_grouped_matmul_dequant_test, ascend310P_normal_4)
{
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({4, 8, 32, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto weight_scale_desc = TensorDesc({4, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto group_list_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{7, 29, 31, 64});
    aclTensor* bias_desc = nullptr;
    auto x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* x_offset_desc = nullptr;
    auto smooth_scale_desc = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    bool weightTrans = true;
    char* quantMode = "pertoken";

    auto ut = OP_API_UT(aclnnQuantGroupedMatmulDequant,
                        INPUT(x_desc, weight_desc, weight_scale_desc, group_list_desc, bias_desc, x_scale_desc,
                              x_offset_desc, smooth_scale_desc, quantMode, weightTrans),
                        OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
} // namespace
