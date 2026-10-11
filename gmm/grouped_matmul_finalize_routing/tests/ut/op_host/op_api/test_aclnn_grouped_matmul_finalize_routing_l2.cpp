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
#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_GroupedMatmulFinalizeRouting_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_GroupedMatmulFinalizeRouting_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_GroupedMatmulFinalizeRouting_test TearDown" << endl;
    }
};

static const int NORMAL_K_VAL = 2048;
static const int NORMAL_N_VAL = 7168;

TEST_F(l2_GroupedMatmulFinalizeRouting_test, ascend910B2_test_opapi_w4a8_normal_case)
{
    int64_t m = 192;
    int64_t k = NORMAL_K_VAL;
    int64_t n = NORMAL_N_VAL;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, k / quantGroupSize, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRouting,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulFinalizeRouting_test, ascend910B2_test_opapi_w4a8_bias_not_null_case)
{
    int64_t m = 192;
    int64_t k = NORMAL_K_VAL;
    int64_t n = NORMAL_N_VAL;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, k / quantGroupSize, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);

    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRouting,
                        INPUT(x1_desc, x2_desc, scale_desc, nullptr, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_GroupedMatmulFinalizeRouting_test, ascend910B2_test_opapi_nd_not_support_w8a8_case)
{
    int64_t m = 192;
    int64_t k = NORMAL_K_VAL;
    int64_t n = NORMAL_N_VAL;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, k / quantGroupSize, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRouting,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRouting_test, ascend910B2_test_opapi_w4a8_scale_not_int64_case)
{
    int64_t m = 192;
    int64_t k = NORMAL_K_VAL;
    int64_t n = NORMAL_N_VAL;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, k / quantGroupSize, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRouting,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRouting_test, ascend910B2_test_not_support_k_n_case)
{
    int64_t m = 192;
    int64_t k = 256;
    int64_t n = NORMAL_N_VAL;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, k / quantGroupSize, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRouting,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRouting_test, ascend910B2_test_scale_dim_not_3_case)
{
    int64_t m = 192;
    int64_t k = NORMAL_K_VAL;
    int64_t n = NORMAL_N_VAL;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRouting,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRouting_test, ascend910B2_test_scale_shape_invalid_case)
{
    int64_t m = 192;
    int64_t k = NORMAL_N_VAL;
    int64_t n = NORMAL_K_VAL;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e + 1, k / quantGroupSize, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRouting,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_GroupedMatmulFinalizeRouting_test, ascend910B2_test_null_ptr_case)
{
    int64_t m = 192;
    int64_t k = NORMAL_K_VAL;
    int64_t n = NORMAL_N_VAL;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, k / quantGroupSize, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRouting,
                        INPUT(nullptr, x2_desc, scale_desc, bias_desc, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, false, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_GroupedMatmulFinalizeRouting_test, ascend910B2_test_transpose_true_not_support_case)
{
    int64_t m = 192;
    int64_t k = NORMAL_K_VAL;
    int64_t n = NORMAL_N_VAL;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc =
        TensorDesc({e, k, n / 8}, ACL_INT32, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({e, k / quantGroupSize, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc perTokenScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc bias_desc = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc shared_input_desc = TensorDesc({bsdp, n}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc logits_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc row_index_desc = TensorDesc({m}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc({bs, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRouting,
                        INPUT(x1_desc, x2_desc, scale_desc, bias_desc, perTokenScale_desc, groupList_desc,
                              shared_input_desc, logits_desc, row_index_desc, 0, 1.0, 0, true, false, 1),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
