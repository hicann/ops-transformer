/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "attention_worker_scheduler.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>

#include "opdev/make_op_executor.h"

using namespace op;
using namespace std;

const int64_t DATA_SIZE = 1024;

class AttentionWorkerScheduler : public ::testing::Test {
public:
    AttentionWorkerScheduler()
        : exe(nullptr)
    {}

    aclTensor *CreateAclTensor(std::vector<int64_t> shape, aclDataType dtype)
    {
        return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                               data);
    }

    void Clear()
    {
        exe->kernelLaunchObjList_.clear();
    }

    void SetUp() override
    {
        auto executor = &exe;
        auto unique_executor = CREATE_EXECUTOR();
        unique_executor.ReleaseTo(executor);
    }

    void TearDown() override
    {
        delete exe;
    }

public:
    aclOpExecutor *exe;
    int64_t data[DATA_SIZE] = {0};
};

TEST_F(AttentionWorkerScheduler, AttentionWorkerScheduler_SUCC)
{
    auto scheduleContext = CreateAclTensor({1024}, ACL_INT8);
    auto out = CreateAclTensor({1024}, ACL_INT8);
    auto res = l0op::AttentionWorkerScheduler(scheduleContext, out, exe);
    EXPECT_NE(res, nullptr);
}
