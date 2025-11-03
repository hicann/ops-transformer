/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../../op_api/aclnn_batch_matmul_reduce_scatter_all_to_all.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace BatchMatmulReduceScatterAlltoAll {
// IFA aclnn ut for 910b has error in UT environment. Deleted.
class l2_batch_matmul_reduce_scatter_all_to_all_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_batch_matmul_reduce_scatter_all_to_all_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_batch_matmul_reduce_scatter_all_to_all_test TearDown" << endl; }
};

// E = 4, C = 2, H = 6, ep = 2, tp = 2, M = 4
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_yshardtype_0) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 0),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// yShardType = 1, E = 4, C = 2, H = 6, ep = 2, tp = 2, M = 4
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_yshardtype_1) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// yShardType = 1, E = 4, C = 2, H = 6, ep = 2, tp = 2, M = 4, trans & no autocontinguous
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_yshardtype_1_trans_1) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND,{12,1,2});
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// E > 2048
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_E_2050) {
    TensorDesc x = TensorDesc({1025, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({1025, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({1025, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({2050, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// K = 0
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_K_0) {
  TensorDesc x = TensorDesc({2, 4, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 0, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ep group name empty
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_ep_name_empty) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// tp group name null
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_tp_name_null) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      nullptr, 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// x dim invalid
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_x_dim_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// E != E/ep * ep
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_E_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({5, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// H != H/tp * tp
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_H_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// bias shape invalid
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_bias_shape_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 0, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// epWorldSize invalid
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_epworldsize_invalid)
{
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 10, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// tpWorldSize invalid
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_tpworldsize_invalid)
{
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 3, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// H > 65535
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_H_65536)
{
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// C = 0
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_C_0)
{
  TensorDesc x = TensorDesc({2, 0, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 0, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// E/ep = 0
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_E_ep_0)
{
  TensorDesc x = TensorDesc({0, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({0, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({0, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({0, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_w_0_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({3, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_w_1_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 3, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_y_2_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_y_1_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_yshardtype_1_y_1_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// E = 4, C = 2, H = 6, ep = 2, tp = 2, M = 4
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_yshardtype_2_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 2),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// E = 4, C = 2, H = 6, ep = 2, tp = 2, M = 4
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_yshardtype_0_X1_invalid) {
  TensorDesc x = TensorDesc({2, 7, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 0),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// E = 4, C = 2, H = 7, ep = 2, tp = 2, M = 4
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_yshardtype_0_H_invalid) {
  TensorDesc x = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 7}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 0),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// E = 4, C = 2, H = 6, ep = 2, tp = 2, M = 4
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_yshardtype_1_X1_invalid) {
  TensorDesc x = TensorDesc({2, 7, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 1),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// E = 4, C = 2, H = 6, ep = 2, tp = 2, M = 4
TEST_F(l2_batch_matmul_reduce_scatter_all_to_all_test, test_batch_matmul_reduce_scatter_all_to_all_yshardtype_0_bias_invalid) {
  TensorDesc x = TensorDesc({2, 7, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc weight = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc biasOptional = TensorDesc({2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({4, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBatchMatMulReduceScatterAlltoAll, INPUT(x, weight, biasOptional,
                      "test_batch_matmul_reduce_scatter_all_to_all_ep_group",
                      "test_batch_matmul_reduce_scatter_all_to_all_tp_group", 2, 2, 0),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
} // BatchMatmulReduceScatterAlltoAll