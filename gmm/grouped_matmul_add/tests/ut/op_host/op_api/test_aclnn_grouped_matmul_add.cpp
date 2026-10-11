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
#include "../../../../op_api/aclnn_grouped_matmul_add.h"
#include "../../../../op_api/aclnn_grouped_matmul_add_v2.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_grouped_matmul_add_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_grouped_matmul_add_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_grouped_matmul_add_test TearDown" << endl;
    }
};

TEST_F(l2_grouped_matmul_add_test, Ascend910B2_grouped_matmul_add_fp16)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorDesc({K, M}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto groupList = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto yRef = TensorDesc({M, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t groupType = 2;
    bool transposeX = true;
    bool transposeWeight = false;

    auto ut = OP_API_UT(aclnnGroupedMatmulAdd,
                        INPUT(x, weight, groupList, yRef, transposeX, transposeWeight, groupType), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_grouped_matmul_add_test, Ascend910B2_grouped_matmul_add_fp16_v2)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorDesc({K, M}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto groupList = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto yRef = TensorDesc({M, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t groupType = 2;
    bool transposeX = true;
    bool transposeWeight = false;

    uint64_t workspaceSize = 0;
    int groupListTypeMode = 2;
    for (int i = 0; i < groupListTypeMode; i++) {
        // groupListType = 0 or 1.
        auto ut = OP_API_UT(aclnnGroupedMatmulAddV2,
                            INPUT(x, weight, groupList, yRef, transposeX, transposeWeight, groupType, i), OUTPUT());

        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    }
}
