/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cfloat>

#include <array>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../../op_api/aclnn_matmul_all_reduce.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MatmulAllReduceUT {

class l2_matmul_all_reduce_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_matmul_all_reduce_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_matmul_all_reduce_test TearDown" << endl; }
};

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_first_api) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_empty_K) {
  TensorDesc x1_desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_empty_M) {
  TensorDesc x1_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_null_x1) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT((aclTensor*)nullptr, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_null_x2) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, (aclTensor*)nullptr, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_null_output) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT((aclTensor*)nullptr));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_reduce_op) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "max", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_stream_mode) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 0), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_shape_x1) {
  TensorDesc x1_desc = TensorDesc({16, 32, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_shape_x2) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_shape_x1_x2) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_shape_x1_output) {
  TensorDesc x1_desc = TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_shape_x2_output) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_shape_bias_output) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_dtype_x1_x2) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_dtype_x1_x2_2) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_dtype_x1_output) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_wrong_dtype_bias_x1) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_success) {
  TensorDesc x1_desc = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_all_reduce_test, test_mm_all_reduce_success_no_cube) {
  TensorDesc x1_desc = TensorDesc({128, 8192}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({8192, 11296}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({11296}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({128,11296}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

} // MatmulAllReduceUT