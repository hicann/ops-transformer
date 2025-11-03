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
#include "../../../../op_api/aclnn_matmul_all_reduce_add_rms_norm.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MatmulAllReduceAddRmsNormUT {

class l2_matmul_all_reduce_add_rms_norm_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_matmul_all_reduce_add_rms_norm_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_matmul_all_reduce_add_rms_norm_test TearDown" << endl; }
};

TEST_F(l2_matmul_all_reduce_add_rms_norm_test, test_mm_all_reduce_add_rms_norm_first_api) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc residual_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gamma_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc normOut_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1_desc, x2_desc, bias, residual_desc, gamma_desc, 0.000001,
                      "test_group", "sum", 8, 1), OUTPUT(y_desc, normOut_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_all_reduce_add_rms_norm_test, test_mm_all_reduce_add_rms_norm_empty_M) {
  TensorDesc x1_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc residual_desc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gamma_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc normOut_desc = TensorDesc({1, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1_desc, x2_desc, bias, residual_desc, gamma_desc, 0.000001,
                      "test_group", "sum", 8, 1), OUTPUT(y_desc, normOut_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_all_reduce_add_rms_norm_test, test_mm_all_reduce_add_rms_norm_empty_K) {
  TensorDesc x1_desc = TensorDesc({16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc residual_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gamma_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc normOut_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1_desc, x2_desc, bias, residual_desc, gamma_desc, 0.000001,
                      "test_group", "sum", 8, 1), OUTPUT(y_desc, normOut_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_add_rms_norm_test, test_mm_all_reduce_add_rms_norm_empty_N) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc residual_desc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gamma_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc normOut_desc = TensorDesc({1, 16, 0}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1_desc, x2_desc, bias, residual_desc, gamma_desc, 0.000001,
                      "test_group", "sum", 8, 1), OUTPUT(y_desc, normOut_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_all_reduce_add_rms_norm_test, test_mm_all_reduce_add_rms_norm_wrong_k) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({30, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc residual_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gamma_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc normOut_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1_desc, x2_desc, bias, residual_desc, gamma_desc, 0.000001,
                      "test_group", "sum", 8, 1), OUTPUT(y_desc, normOut_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_all_reduce_add_rms_norm_test, test_mm_all_reduce_add_rms_norm_wrong_n) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc residual_desc = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gamma_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc normOut_desc = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulAllReduceAddRmsNorm, INPUT(x1_desc, x2_desc, bias, residual_desc, gamma_desc, 0.000001,
                      "test_group", "sum", 8, 1), OUTPUT(y_desc, normOut_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // MatmulAllReduceAddRmsNormUT