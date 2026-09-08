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

#include "aclnn_attention_worker_scheduler.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/inner/rts_interface.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_attention_worker_scheduler_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "tensor_attention_worker_scheduler_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "tensor_attention_worker_scheduler_test TearDown" << endl;
    }
};

TEST_F(l2_attention_worker_scheduler_test, normal_input_success)
{
    auto scheduleContextDesc = TensorDesc({1024}, ACL_INT8, ACL_FORMAT_ND);
    auto scheduleContextRef = DescToAclContainer(scheduleContextDesc);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet =
        aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(scheduleContextRef, &workspace_size, &executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    unique_ptr<void, decltype(&FreeDeviceMemory)> workspace_ptr(MallocDeviceMemory(workspace_size), FreeDeviceMemory);
    aclRet = aclnnInplaceAttentionWorkerScheduler(workspace_ptr.get(), workspace_size, executor, nullptr);
    SynchronizeStream();
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(l2_attention_worker_scheduler_test, empty_tensor_success)
{
    auto scheduleContextDesc = TensorDesc({0}, ACL_INT8, ACL_FORMAT_ND);
    auto scheduleContextRef = DescToAclContainer(scheduleContextDesc);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet =
        aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(scheduleContextRef, &workspace_size, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(l2_attention_worker_scheduler_test, invalid_input_dim)
{
    auto scheduleContextDesc = TensorDesc({1024, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto scheduleContextRef = DescToAclContainer(scheduleContextDesc);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet =
        aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(scheduleContextRef, &workspace_size, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    if (executor != nullptr) {
        delete executor;
    }
}
