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
#include "../../../../op_api/aclnn_moe_distribute_combine.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
using namespace op;
using namespace std;

namespace MowDistributeCombine {
class l2_moe_distribute_combine_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_moe_distribute_combine_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_moe_distribute_combine_test TearDown" << endl; }
};

TEST_F(l2_moe_distribute_combine_test, test_moe_distribute_combine_first_api) {
  TensorDesc expandX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc expandIdx = TensorDesc({32*8}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc epSendCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc tpSendCounts = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc xActiveMask = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
  TensorDesc activationScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc weightScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc groupList = TensorDesc({288}, ACL_INT64, ACL_FORMAT_ND);
  TensorDesc expandScales = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);

  int64_t epWorldSize = 288;
  int64_t tpWorldSize = 2;
  int64_t epRankId = 0;
  int64_t tpRankId = 0;
  int64_t expertShardType = 0;
  int64_t sharedExpertNum = 1;
  int64_t sharedExpertRankNum = 32;
  int64_t moeExpertNum = 256;
  int64_t globalBs = 0;
  int64_t outDtype = 0;
  int64_t commQuantMode = 0;
  int64_t groupList_type = 0;

  TensorDesc x = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMoeDistributeCombine, INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts,
                      xActiveMask, activationScale, weightScale, groupList, expandScales,
                      "test_moe_distribute_combine_ep", epWorldSize, epRankId, moeExpertNum,
                      "test_moe_distribute_combine_tp", tpWorldSize, tpRankId,
                      expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupList_type),
                                        OUTPUT(x));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_moe_distribute_combine_test, ascend910B2_test_moe_distribute_combine_tp_not_empty) {
  TensorDesc expandX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc expandIdx = TensorDesc({32*8}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc epSendCounts = TensorDesc({256}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc tpSendCounts = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc xActiveMask = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);
  TensorDesc activationScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc weightScale = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc groupList = TensorDesc({256}, ACL_INT64, ACL_FORMAT_ND);
  TensorDesc expandScales = TensorDesc({16384}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);

  int64_t epWorldSize = 256;
  int64_t tpWorldSize = 0;
  int64_t epRankId = 0;
  int64_t tpRankId = 0;
  int64_t expertShardType = 0;
  int64_t sharedExpertNum = 0;
  int64_t sharedExpertRankNum = 0;
  int64_t moeExpertNum = 256;
  int64_t globalBs = 0;
  int64_t outDtype = 0;
  int64_t commQuantMode = 0;
  int64_t groupList_type = 0;

  TensorDesc x = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMoeDistributeCombine, INPUT(expandX, expertIds, expandIdx, epSendCounts, expertScales, tpSendCounts,
                      xActiveMask, activationScale, weightScale, groupList, expandScales,
                      "", epWorldSize, epRankId, moeExpertNum,
                      "test_moe_distribute_combine_tp", tpWorldSize, tpRankId,
                      expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupList_type),
                                        OUTPUT(x));
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
} // MowDistributeCombine