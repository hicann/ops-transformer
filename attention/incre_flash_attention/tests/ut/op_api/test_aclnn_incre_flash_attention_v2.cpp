/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "../../../op_api/aclnn_incre_flash_attention_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#pragma GCC diagnostic pop
using namespace std;
using namespace op;

class incre_flash_attention_v2_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }

    static void TearDownTestCase() {}
};

// dav5102 reject
TEST_F(incre_flash_attention_v2_opapi_ut, incre_flash_attention_v2_aclnn_5102_unsupported)
{
    op::ScopedPlatformNpuArch guard(NpuArch::DAV_5102);
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV2GetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8, 1.0,
        inputLayout, 8, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
    EXPECT_EQ(executor, nullptr);

    aclRet = aclnnIncreFlashAttentionV2(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
}
