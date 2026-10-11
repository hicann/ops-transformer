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
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_weight_nz.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_GroupedMatmulFinalizeRoutingWeightNz_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_GroupedMatmulFinalizeRoutingWeightNz_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_GroupedMatmulFinalizeRoutingWeightNz_test TearDown" << endl;
    }
};

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_add_value)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_shared_input_null)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, nullptr, nullptr, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_bias_not_null)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_x2_not_nz)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_x2_null)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, nullptr, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_dtype_not_0)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 1, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_scale_noteq_e)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e + 1, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_perTokenScale_noteq_m)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_shared_input_more_than_bs)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bs + 1, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_group_list_type_not_1)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bs + 1, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_view_shape_0)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({0, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bs + 1, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_x12_viewdims_wrong)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_shareinput_outputBS_noeq)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32});
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc shared_input_desc = TensorDesc({bsdp + 1, n + 1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_logit_1stdim_noeq_xMdim)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m + 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_rowindex1stdim_noeq_xMdim)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m + 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_groupListdim_wrong)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e + 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_weightview0)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({0}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_normal_case_nvalid)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7000;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({0}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_not_support_k_shape_case)
{
    int64_t m = 192;
    int64_t k = 4098;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_x1_type_not_same_with_x2)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_scale_shape_invalid_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e + 1, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRoutingWeightNz_test, ascend910B2_test_k_n_not_support_w8a8_case)
{
    int64_t m = 192;
    int64_t k = 2048 - 1;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({e, k, n / 8}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32})
                             .ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, k / quantGroupSize, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNz,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
