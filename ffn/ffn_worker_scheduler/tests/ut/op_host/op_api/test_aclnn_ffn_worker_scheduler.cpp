/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <cstdio>
#include <vector>
#include <fstream>
#include "gtest/gtest.h"

#include "aclnn_ffn_worker_scheduler.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/inner/rts_interface.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_ffn_worker_scheduler_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_ffn_worker_scheduler_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_ffn_worker_scheduler_test TearDown" << endl;
    }
};

TEST_F(l2_ffn_worker_scheduler_test, case_norm_int8)
{
    TensorDesc scheduleContextDesc = TensorDesc({1024}, ACL_INT8, ACL_FORMAT_ND);
    int32_t syncGroupSize = 1;
    int32_t executeMode = 0;
    auto ut =
        OP_API_UT(aclnnInplaceFfnWorkerScheduler, INPUT(scheduleContextDesc, syncGroupSize, executeMode), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_ffn_worker_scheduler_test, case_norm_empty_tensor)
{
    TensorDesc scheduleContextDesc = TensorDesc({0}, ACL_INT8, ACL_FORMAT_ND);
    int32_t syncGroupSize = 1;
    int32_t executeMode = 0;
    auto ut =
        OP_API_UT(aclnnInplaceFfnWorkerScheduler, INPUT(scheduleContextDesc, syncGroupSize, executeMode), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_ffn_worker_scheduler_test, invalid_input_dim)
{
    TensorDesc scheduleContextDesc = TensorDesc({1024, 2}, ACL_INT8, ACL_FORMAT_ND);
    int32_t syncGroupSize = 1;
    int32_t executeMode = 0;
    auto ut =
        OP_API_UT(aclnnInplaceFfnWorkerScheduler, INPUT(scheduleContextDesc, syncGroupSize, executeMode), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_ffn_worker_scheduler_test, invalid_input_excute_mode)
{
    TensorDesc scheduleContextDesc = TensorDesc({1024}, ACL_INT8, ACL_FORMAT_ND);
    int32_t syncGroupSize = 1;
    int32_t executeMode = 1;
    auto ut =
        OP_API_UT(aclnnInplaceFfnWorkerScheduler, INPUT(scheduleContextDesc, syncGroupSize, executeMode), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}
