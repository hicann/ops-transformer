/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

class l2_moe_init_routing_v2_grad_test : public testing::Test
{
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

// dtype fp32
TEST_F(l2_moe_init_routing_v2_grad_test, Ascend910B2_moe_init_routing_v2_grad_fp32)
{
    auto gradExpandedX = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto expandedRowIdx = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0);
    auto out = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeInitRoutingV2Grad, INPUT(gradExpandedX, expandedRowIdx, 1, 0, 0), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}