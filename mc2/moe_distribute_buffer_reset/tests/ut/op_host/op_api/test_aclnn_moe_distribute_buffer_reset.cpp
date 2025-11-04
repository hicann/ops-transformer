/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_moe_distribute_buffer_reset.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../../op_api/aclnn_moe_distribute_buffer_reset.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_aclnn_moe_distribute_buffer_reset_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_aclnn_moe_distribute_buffer_reset_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_aclnn_moe_distribute_buffer_reset_test TearDown" << endl; }
};

TEST_F(l2_aclnn_moe_distribute_buffer_reset_test, test_aclnn_moe_distribute_buffer_reset_api) {
  TensorDesc elastic_info = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
  int32_t world_size = 16;
  int32_t need_sync = 0;

  auto ut = OP_API_UT(aclnnMoeDistributeBufferReset,
                      INPUT(elastic_info, "test_moe_distribute_buffer_reset", world_size, need_sync),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_moe_distribute_buffer_reset_test, test_aclnn_moe_distribute_buffer_reset_nullptr) {
  TensorDesc elastic_info = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
  int32_t world_size = 16;
  int32_t need_sync = 1;

  auto ut = OP_API_UT(aclnnMoeDistributeBufferReset,
                      INPUT(elastic_info, nullptr, world_size, need_sync),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_aclnn_moe_distribute_buffer_reset_test, test_aclnn_moe_distribute_buffer_reset_empty) {
  TensorDesc elastic_info = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
  int32_t world_size = 16;
  int32_t need_sync = 0;

  auto ut = OP_API_UT(aclnnMoeDistributeBufferReset,
                      INPUT(elastic_info, "", world_size, need_sync),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}