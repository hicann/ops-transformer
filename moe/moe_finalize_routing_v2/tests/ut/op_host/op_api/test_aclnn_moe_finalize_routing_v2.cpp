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
#include "../../../../op_host/op_api/aclnn_moe_finalize_routing_v2.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_finalize_routing_v2_test : public testing::Test {};

TEST_F(l2_moe_finalize_routing_v2_test, Ascend910B2_moe_finalize_routing_v2_fp32)
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

    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2,
                        INPUT(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, dropPadMode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_finalize_routing_v2_test, Ascend910B2_moe_finalize_routing_v2_bf16)
{
    auto expandedX = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expandedRowIdx = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 15);
    auto x1 = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    int64_t dropPadMode = 0;

    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2,
                        INPUT(expandedX, expandedRowIdx, x1, (aclTensor *)nullptr, (aclTensor *)nullptr,
                              (aclTensor *)nullptr, (aclTensor *)nullptr, dropPadMode),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_finalize_routing_v2_test, Ascend310P_moe_finalize_routing_v2_fp32)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND310P);
    auto expandedX = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expandedRowIdx = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 15);
    auto x1 = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dropPadMode = 0;

    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2,
                        INPUT(expandedX, expandedRowIdx, x1, (aclTensor *)nullptr, (aclTensor *)nullptr,
                              (aclTensor *)nullptr, (aclTensor *)nullptr, dropPadMode),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_moe_finalize_routing_v2_test, Ascend910B2_moe_finalize_routing_v2_fp16)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto expandedX = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expandedRowIdx = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 15);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dropPadMode = 0;

    auto ut = OP_API_UT(aclnnMoeFinalizeRoutingV2,
                        INPUT(expandedX, expandedRowIdx, (aclTensor *)nullptr, (aclTensor *)nullptr, (aclTensor *)nullptr,
                              (aclTensor *)nullptr, (aclTensor *)nullptr, dropPadMode),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
