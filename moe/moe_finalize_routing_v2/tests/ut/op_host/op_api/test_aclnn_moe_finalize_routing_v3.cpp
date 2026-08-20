/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_finalize_routing_v3.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_finalize_routing_v3_test : public testing::Test {};

TEST_F(l2_moe_finalize_routing_v3_test, Ascend910B2_moe_finalize_routing_v3_fp32)
{
    auto expandedX = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expandedRowIdx = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 15);
    auto x1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto x2 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto bias = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scales = TensorDesc({16, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto expertIdx = TensorDesc({16, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dropPadMode = 0;
    std::vector<int64_t> defaultRange = {-1, -1};
    aclIntArray *zeroExpertRange = aclCreateIntArray(defaultRange.data(), defaultRange.size());
    aclIntArray *copyExpertRange = aclCreateIntArray(defaultRange.data(), defaultRange.size());
    aclIntArray *constantExpertRange = aclCreateIntArray(defaultRange.data(), defaultRange.size());

    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV3,
                        INPUT(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, (aclTensor *)nullptr,
                              (aclTensor *)nullptr, (aclTensor *)nullptr, (aclTensor *)nullptr, dropPadMode,
                              zeroExpertRange, copyExpertRange, constantExpertRange),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_finalize_routing_v3_test, Ascend910B2_moe_finalize_routing_v3_constant_expert)
{
    auto expandedX = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expandedRowIdx = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 15);
    auto x1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scales = TensorDesc({16, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto expertIdx = TensorDesc({16, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto x = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto alpha1 = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto alpha2 = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto v = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dropPadMode = 0;
    std::vector<int64_t> zeroRange = {0, 2};
    std::vector<int64_t> copyRange = {2, 4};
    std::vector<int64_t> constRange = {4, 6};
    aclIntArray *zeroExpertRange = aclCreateIntArray(zeroRange.data(), zeroRange.size());
    aclIntArray *copyExpertRange = aclCreateIntArray(copyRange.data(), copyRange.size());
    aclIntArray *constantExpertRange = aclCreateIntArray(constRange.data(), constRange.size());

    auto ut =
        OP_API_UT(aclnnMoeFinalizeRoutingV3,
                  INPUT(expandedX, expandedRowIdx, x1, (aclTensor *)nullptr, (aclTensor *)nullptr, scales, expertIdx, x,
                        alpha1, alpha2, v, dropPadMode, zeroExpertRange, copyExpertRange, constantExpertRange),
                  OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_finalize_routing_v3_test, Ascend910B2_moe_finalize_routing_v3_droppad_ncl_bf16)
{
    auto expandedX = TensorDesc({4, 8, 16}, ACL_BF16, ACL_FORMAT_NCL).ValueRange(-10, 10);
    auto expandedRowIdx = TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 31);
    auto x1 = TensorDesc({2, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scales = TensorDesc({2, 8}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto expertIdx = TensorDesc({2, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto out = TensorDesc({2, 16}, ACL_BF16, ACL_FORMAT_ND);
    int64_t dropPadMode = 1;
    std::vector<int64_t> defaultRange = {-1, -1};
    aclIntArray *zeroExpertRange = aclCreateIntArray(defaultRange.data(), defaultRange.size());
    aclIntArray *copyExpertRange = aclCreateIntArray(defaultRange.data(), defaultRange.size());
    aclIntArray *constantExpertRange = aclCreateIntArray(defaultRange.data(), defaultRange.size());

    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV3,
                        INPUT(expandedX, expandedRowIdx, x1, (aclTensor *)nullptr, (aclTensor *)nullptr, scales,
                              expertIdx, (aclTensor *)nullptr, (aclTensor *)nullptr, (aclTensor *)nullptr,
                              (aclTensor *)nullptr, dropPadMode, zeroExpertRange, copyExpertRange, constantExpertRange),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_finalize_routing_v3_test, Ascend910B2_moe_finalize_routing_v3_droppad_bf16)
{
    auto expandedX = TensorDesc({4, 8, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expandedRowIdx = TensorDesc({32}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 31);
    auto x1 = TensorDesc({2, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto scales = TensorDesc({2, 8}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto expertIdx = TensorDesc({2, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto out = TensorDesc({2, 16}, ACL_BF16, ACL_FORMAT_ND);
    int64_t dropPadMode = 1;
    std::vector<int64_t> defaultRange = {-1, -1};
    aclIntArray *zeroExpertRange = aclCreateIntArray(defaultRange.data(), defaultRange.size());
    aclIntArray *copyExpertRange = aclCreateIntArray(defaultRange.data(), defaultRange.size());
    aclIntArray *constantExpertRange = aclCreateIntArray(defaultRange.data(), defaultRange.size());

    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV3,
                        INPUT(expandedX, expandedRowIdx, x1, (aclTensor *)nullptr, (aclTensor *)nullptr, scales,
                              expertIdx, (aclTensor *)nullptr, (aclTensor *)nullptr, (aclTensor *)nullptr,
                              (aclTensor *)nullptr, dropPadMode, zeroExpertRange, copyExpertRange, constantExpertRange),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
