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
#include "../../../../op_api/aclnn_all_gather_matmul.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"

using namespace op;
using namespace std;

namespace AllGatherMatmulUT {

class l2_all_gather_matmul_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_all_gather_matmul_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_all_gather_matmul_test TearDown" << endl; }
};

TEST_F(l2_all_gather_matmul_test, test_all_gather_first_api) {
  TensorDesc x1_desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gather_out_desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1_desc, x2_desc, bias, "test_all_gather_group", 0, 8, 1),
                      OUTPUT(out_desc, gather_out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_all_gather_matmul_test, test_all_gather_first_api_2) {
  TensorDesc x1_desc = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gather_out_desc = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1_desc, x2_desc, bias, "test_all_gather_group", 0, 8, 1),
                      OUTPUT(out_desc, gather_out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_all_gather_matmul_test, test_all_gather_first_api_3) {
  TensorDesc x1_desc = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gather_out_desc = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1_desc, x2_desc, bias, "test_all_gather_group", 0, 8, 1),
                      OUTPUT(out_desc, gather_out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_all_gather_matmul_test, test_all_gather_first_api_4) {
  TensorDesc x1_desc = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gather_out_desc = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1_desc, x2_desc, bias, "test_all_gather_group", 0, 8, 1),
                      OUTPUT(out_desc, gather_out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_all_gather_matmul_test, test_all_gather_first_api_5) {
  TensorDesc x1_desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gather_out_desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1_desc, x2_desc, bias, "test_all_gather_group", 0, 8, 1),
                      OUTPUT(out_desc, gather_out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_all_gather_matmul_test, test_all_gather_first_api_input_false) {
  TensorDesc x1_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gather_out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1_desc, x2_desc, bias, "test_all_gather_group", 0, 8, 1),
                      OUTPUT(out_desc, gather_out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_all_gather_matmul_test, test_all_gather_first_api_gather_out_false) {
  TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc gather_out_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1_desc, x2_desc, bias, "test_all_gather_group", 0, 8, 1),
                      OUTPUT(out_desc, gather_out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // AllGatherMatmulUT