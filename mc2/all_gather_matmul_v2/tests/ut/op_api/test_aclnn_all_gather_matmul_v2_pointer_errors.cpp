/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../op_api/aclnn_all_gather_matmul_v2.h"
#include "../../../../common/tests/ut/api_pointer_error_assertions.h"

TEST(AllGatherMatmulV2PointerErrors, MissingCommModeReportsOnce)
{
    Mc2ErrorCodeUT::ExcelErrorReportScope errors;
    uint64_t size = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnAllGatherMatmulV2GetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, "group",
                                                     0, 1, 0, 8, nullptr, nullptr, nullptr, nullptr, &size, &executor),
              ACLNN_ERR_INNER_NULLPTR);
    Mc2ErrorCodeUT::ExpectRequiredApiPointerError("aclnnAllGatherMatmulV2GetWorkspaceSize", "commMode");
}

TEST(AllGatherMatmulV2PointerErrors, NonemptyLaunchMissingExecutorReportsOnce)
{
    Mc2ErrorCodeUT::ExcelErrorReportScope errors;
    uint64_t workspace = 0;
    EXPECT_EQ(aclnnAllGatherMatmulV2(&workspace, sizeof(workspace), nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
    Mc2ErrorCodeUT::ExpectRequiredApiPointerError("aclnnAllGatherMatmulV2", "executor");
}

TEST(AllGatherMatmulV2PointerErrors, EmptyLaunchKeepsSuccessWithoutFormalError)
{
    Mc2ErrorCodeUT::ExcelErrorReportScope errors;
    EXPECT_EQ(aclnnAllGatherMatmulV2(nullptr, 0, nullptr, nullptr), ACLNN_SUCCESS);
    EXPECT_TRUE(ErrorManager::GetInstance().GetRawErrorMessages().empty());
}
