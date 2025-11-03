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
#include "../../../../op_api/aclnn_distribute_barrier.h"
#include "../../../../op_api/aclnn_distribute_barrier_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_aclnn_distribute_barrier_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_aclnn_distribute_barrier_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_aclnn_distribute_barrier_test TearDown" << endl; }
};

TEST_F(l2_aclnn_distribute_barrier_test, test_aclnn_distribute_barrier_first_api) {
  TensorDesc x_ref = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  int64_t worldSize = 16;

  auto ut = OP_API_UT(aclnnDistributeBarrier,
                      INPUT(x_ref, "test_distribute_barrier", worldSize),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_distribute_barrier_test, test_aclnn_distribute_barrier_first_api_nullptr_group) {
  TensorDesc x_ref = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  int64_t worldSize = 16;

  auto ut = OP_API_UT(aclnnDistributeBarrier,
                      INPUT(x_ref, nullptr, worldSize),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_aclnn_distribute_barrier_test, test_aclnn_distribute_barrier_first_api_group_min) {
  TensorDesc x_ref = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  int64_t worldSize = 16;

  auto ut = OP_API_UT(aclnnDistributeBarrier,
                      INPUT(x_ref, "", worldSize),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_distribute_barrier_test, test_aclnn_distribute_barrier_v2_first_api) {
  TensorDesc x_ref = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  int64_t worldSize = 16;

  auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                      INPUT(x_ref, nullptr, nullptr, "test_distribute_barrier", worldSize),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_distribute_barrier_test, test_aclnn_distribute_barrier_v2_first_api_nullptr_group) {
  TensorDesc x_ref = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  int64_t worldSize = 16;

  auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                      INPUT(x_ref, nullptr, nullptr, nullptr, worldSize),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_aclnn_distribute_barrier_test, test_aclnn_distribute_barrier_v2_first_api_group_min) {
  TensorDesc x_ref = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  int64_t worldSize = 16;

  auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                      INPUT(x_ref, nullptr, nullptr, "", worldSize),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_distribute_barrier_test, test_aclnn_distribute_barrier_v2_first_api_time_out) {
  TensorDesc x_ref = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc time_out = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

  int64_t worldSize = 16;

  auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                      INPUT(x_ref, time_out, nullptr, "test_distribute_barrier", worldSize),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_distribute_barrier_test, test_aclnn_distribute_barrier_v2_first_api_elastic_info) {
  TensorDesc x_ref = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc elastic_info = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND);

  int64_t worldSize = 16;

  auto ut = OP_API_UT(aclnnDistributeBarrierV2,
                      INPUT(x_ref, nullptr, elastic_info, "test_distribute_barrier", worldSize),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}