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
#include "../../../../op_api/aclnn_moe_update_expert.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MoeUpdateExpert {
class l2_aclnn_moe_update_expert_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_aclnn_moe_update_expert_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_aclnn_moe_update_expert_test TearDown" << endl; }
};

TEST_F(l2_aclnn_moe_update_expert_test, test_moe_update_expert_no_tailor) {
    TensorDesc expertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc eplbTable = TensorDesc({256, 5}, ACL_INT32, ACL_FORMAT_ND);

    int64_t localRankId = 0;
    int64_t worldSize = 8;
    int64_t balanceMode = 0;

    TensorDesc balancedExpertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc balancedActiveMask = TensorDesc({50, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeUpdateExpert,
                        INPUT(expertIds, eplbTable, nullptr, nullptr, nullptr, localRankId, worldSize, balanceMode),
                        OUTPUT(balancedExpertIds, balancedActiveMask));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_aclnn_moe_update_expert_test, test_moe_update_expert_expert_tailor) {
    TensorDesc expertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc eplbTable = TensorDesc({256, 5}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({50, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc pruningThreshold = TensorDesc({4,}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc activeMask = TensorDesc({50,}, ACL_BOOL, ACL_FORMAT_ND);

    int64_t localRankId = 0;
    int64_t worldSize = 8;
    int64_t balanceMode = 0;

    TensorDesc balancedExpertIds = TensorDesc({50, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc balancedActiveMask = TensorDesc({50, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMoeUpdateExpert,
                        INPUT(expertIds, eplbTable, expertScales, pruningThreshold, activeMask,
                              localRankId, worldSize, balanceMode),
                        OUTPUT(balancedExpertIds, balancedActiveMask));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
} // MoeUpdateExpert