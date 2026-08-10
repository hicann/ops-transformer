/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_init_routing_v4.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_init_routing_v4_test : public testing::Test
{
protected:
};

// basic: no scale, no offset, no active_num, no topk_weight
TEST_F(l2_moe_init_routing_v4_test, Ascend950_moe_init_routing_v4_basic)
{
    auto x = TensorDesc({5, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expertIdx = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);

    auto expandedXOut = TensorDesc({20, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedRowIdxOut = TensorDesc({20}, ACL_INT32, ACL_FORMAT_ND);
    auto expertTokensCountOrCumsumOut = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({20}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedTopkWeightOut = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    std::vector<int64_t> activeExpertRangeArray = {0, 8};
    aclIntArray *activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());

    int64_t expertCapacity = -1;
    int64_t expertNum = 32;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    int64_t expertTokensNumFlag = true;
    int64_t quantMode = -1;
    int64_t rowIdxType = 1;

    auto ut = OP_API_UT(aclnnMoeInitRoutingV4, INPUT(x, expertIdx, nullptr, nullptr, nullptr, nullptr,
                                                      expertCapacity, expertNum, dropPadMode,
                                                      expertTokensNumType, expertTokensNumFlag, quantMode,
                                                      activeExpertRange, rowIdxType),
                        OUTPUT(expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut,
                               expandedScaleOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// with scale and offset
TEST_F(l2_moe_init_routing_v4_test, Ascend950_moe_init_routing_v4_with_scale_offset)
{
    auto x = TensorDesc({5, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expertIdx = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto scale = TensorDesc({5, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto offset = TensorDesc({5, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto expandedXOut = TensorDesc({20, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedRowIdxOut = TensorDesc({20}, ACL_INT32, ACL_FORMAT_ND);
    auto expertTokensCountOrCumsumOut = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({20}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedTopkWeightOut = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    std::vector<int64_t> activeExpertRangeArray = {0, 8};
    aclIntArray *activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());

    int64_t expertCapacity = -1;
    int64_t expertNum = 32;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    int64_t expertTokensNumFlag = true;
    int64_t quantMode = -1;
    int64_t rowIdxType = 1;

    auto ut = OP_API_UT(aclnnMoeInitRoutingV4, INPUT(x, expertIdx, scale, offset, nullptr, nullptr,
                                                      expertCapacity, expertNum, dropPadMode,
                                                      expertTokensNumType, expertTokensNumFlag, quantMode,
                                                      activeExpertRange, rowIdxType),
                        OUTPUT(expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut,
                               expandedScaleOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// with topk_weight input and expanded_topk_weight output
TEST_F(l2_moe_init_routing_v4_test, Ascend950_moe_init_routing_v4_with_topkweight)
{
    auto x = TensorDesc({5, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expertIdx = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto topkWeight = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    auto expandedXOut = TensorDesc({20, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedRowIdxOut = TensorDesc({20}, ACL_INT32, ACL_FORMAT_ND);
    auto expertTokensCountOrCumsumOut = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({20}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedTopkWeightOut = TensorDesc({20}, ACL_FLOAT, ACL_FORMAT_ND);
    std::vector<int64_t> activeExpertRangeArray = {0, 8};
    aclIntArray *activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());

    int64_t expertCapacity = -1;
    int64_t expertNum = 32;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    int64_t expertTokensNumFlag = true;
    int64_t quantMode = -1;
    int64_t rowIdxType = 1;

    auto ut = OP_API_UT(aclnnMoeInitRoutingV4, INPUT(x, expertIdx, nullptr, nullptr, nullptr, topkWeight,
                                                      expertCapacity, expertNum, dropPadMode,
                                                      expertTokensNumType, expertTokensNumFlag, quantMode,
                                                      activeExpertRange, rowIdxType),
                        OUTPUT(expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut,
                               expandedScaleOut, expandedTopkWeightOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// with active_num input (scalar int64 tensor)
TEST_F(l2_moe_init_routing_v4_test, Ascend950_moe_init_routing_v4_with_active_num)
{
    auto x = TensorDesc({5, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expertIdx = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto activeNumTensor = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto expandedXOut = TensorDesc({20, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedRowIdxOut = TensorDesc({20}, ACL_INT32, ACL_FORMAT_ND);
    auto expertTokensCountOrCumsumOut = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({20}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedTopkWeightOut = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    std::vector<int64_t> activeExpertRangeArray = {0, 8};
    aclIntArray *activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());

    int64_t expertCapacity = -1;
    int64_t expertNum = 32;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    int64_t expertTokensNumFlag = true;
    int64_t quantMode = -1;
    int64_t rowIdxType = 1;

    auto ut = OP_API_UT(aclnnMoeInitRoutingV4, INPUT(x, expertIdx, nullptr, nullptr, activeNumTensor, nullptr,
                                                      expertCapacity, expertNum, dropPadMode,
                                                      expertTokensNumType, expertTokensNumFlag, quantMode,
                                                      activeExpertRange, rowIdxType),
                        OUTPUT(expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut,
                               expandedScaleOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// with both active_num and topk_weight
TEST_F(l2_moe_init_routing_v4_test, Ascend950_moe_init_routing_v4_with_active_num_and_topkweight)
{
    auto x = TensorDesc({5, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expertIdx = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto scale = TensorDesc({5, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto activeNumTensor = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto topkWeight = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    auto expandedXOut = TensorDesc({20, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedRowIdxOut = TensorDesc({20}, ACL_INT32, ACL_FORMAT_ND);
    auto expertTokensCountOrCumsumOut = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({20}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expandedTopkWeightOut = TensorDesc({20}, ACL_FLOAT, ACL_FORMAT_ND);
    std::vector<int64_t> activeExpertRangeArray = {0, 8};
    aclIntArray *activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());

    int64_t expertCapacity = -1;
    int64_t expertNum = 32;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    int64_t expertTokensNumFlag = true;
    int64_t quantMode = -1;
    int64_t rowIdxType = 1;

    auto ut = OP_API_UT(aclnnMoeInitRoutingV4, INPUT(x, expertIdx, scale, nullptr, activeNumTensor, topkWeight,
                                                      expertCapacity, expertNum, dropPadMode,
                                                      expertTokensNumType, expertTokensNumFlag, quantMode,
                                                      activeExpertRange, rowIdxType),
                        OUTPUT(expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut,
                               expandedScaleOut, expandedTopkWeightOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// phase2: null executor path
TEST(l2_moe_init_routing_v4_phase2_test, phase2_null_executor_path)
{
    aclnnStatus ret = aclnnMoeInitRoutingV4(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(ret, ACLNN_SUCCESS);
}
