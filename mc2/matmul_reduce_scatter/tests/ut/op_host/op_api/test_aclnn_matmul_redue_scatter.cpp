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
#include "../../../../op_api/aclnn_matmul_reduce_scatter.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MatmulReduceScatter {
class l2_aclnn_matmul_reduce_scatter_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_aclnn_matmul_reduce_scatter_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_aclnn_matmul_reduce_scatter_test TearDown" << endl; }
};

TEST_F(l2_aclnn_matmul_reduce_scatter_test, test_aclnn_matmul_reduce_scatter_first_api) {
  TensorDesc x1_desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_matmul_reduce_scatter_test, test_aclnn_matmul_reduce_scatter_first_api_2) {
  TensorDesc x1_desc = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_matmul_reduce_scatter_test, test_aclnn_matmul_reduce_scatter_first_api_3) {
  TensorDesc x1_desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_matmul_reduce_scatter_test, test_six_api_910_95) {
  fe::PlatformInfoManager::Instance().InitRuntimePlatformInfos("ASCEND910_95");
  TensorDesc x1_desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2_desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1_desc, x2_desc, bias, "test_group", "sum", 8, 1), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
} // MatmulReduceScatter