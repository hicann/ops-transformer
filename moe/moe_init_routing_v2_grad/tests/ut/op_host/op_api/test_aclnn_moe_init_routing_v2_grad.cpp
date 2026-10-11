/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_init_routing_v2_grad.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_moe_init_routing_v2_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_moe_init_routing_v2_grad_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_moe_init_routing_v2_grad_test TearDown" << endl;
    }
};

// dtype fp32：一阶段 GetWorkspaceSize + 二阶段 aclnnMoeInitRoutingV2Grad
TEST_F(l2_moe_init_routing_v2_grad_test, Ascend910B2_moe_init_routing_v2_grad_fp32)
{
    auto gradExpandedXDesc = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expandedRowIdxDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto outDesc = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* gradExpandedX = DescToAclContainer(gradExpandedXDesc);
    aclTensor* expandedRowIdx = DescToAclContainer(expandedRowIdxDesc);
    aclTensor* out = DescToAclContainer(outDesc);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = aclnnMoeInitRoutingV2GradGetWorkspaceSize(gradExpandedX, expandedRowIdx, 1, 0, 0,
                                                                               out, &workspaceSize, &executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ASSERT_NE(executor, nullptr);

    std::vector<uint8_t> workspace(workspaceSize > 0 ? workspaceSize : 1);
    aclnnStatus runResult = aclnnMoeInitRoutingV2Grad(workspace.data(), workspaceSize, executor, nullptr);
    EXPECT_EQ(runResult, ACLNN_SUCCESS);
}
