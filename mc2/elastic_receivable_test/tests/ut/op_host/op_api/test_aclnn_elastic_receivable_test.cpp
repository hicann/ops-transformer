/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../../op_host/op_api/aclnn_elastic_receivable_test.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_aclnn_elastic_receivable_test_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_aclnn_elastic_receivable_test_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_aclnn_elastic_receivable_test_test TearDown" << endl; }
};

TEST_F(l2_aclnn_elastic_receivable_test_test, test_aclnn_elastic_receivable_test_api) {
  TensorDesc dst_rank = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);

  int64_t world_size = 128;
  int64_t rank_num = 16;

  auto ut = OP_API_UT(aclnnElasticReceivableTest,
                      INPUT(dst_rank, "test_elastic_receivable_test", world_size, rank_num),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_elastic_receivable_test_test, test_aclnn_elastic_receivable_test_nullptr) {
  TensorDesc dst_rank = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);

  int64_t world_size = 128;
  int64_t rank_num = 16;

  auto ut = OP_API_UT(aclnnElasticReceivableTest,
                      INPUT(dst_rank, nullptr, world_size, rank_num),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_aclnn_elastic_receivable_test_test, test_aclnn_elastic_receivable_test_empty) {
  TensorDesc dst_rank = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);

  int64_t world_size = 128;
  int64_t rank_num = 16;

  auto ut = OP_API_UT(aclnnElasticReceivableTest,
                      INPUT(dst_rank, "", world_size, rank_num),
                      OUTPUT());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}