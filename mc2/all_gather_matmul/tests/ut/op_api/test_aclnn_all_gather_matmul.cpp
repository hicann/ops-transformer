/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "all_gather_matmul_api_ut_param.h"
#include "op_api_ut_common/op_api_ut.h"
#include "../../../op_api/aclnn_all_gather_matmul.h"

namespace AllGatherMatmulUT {

class AclnnAllGatherMatmulTest : public testing::TestWithParam<AllGatherMatmulApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmul AclnnAllGatherMatmulTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        std::cout << "AllGatherMatmul AclnnAllGatherMatmulTest TearDown" << std::endl;
    }
};

TEST_P(AclnnAllGatherMatmulTest, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmul,
        INPUT(param.x1, param.x2, param.bias, param.group.c_str(), param.gatherIndex, param.commTurn, param.streamMode),
        OUTPUT(param.out, param.gatherOut));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            (void)aclnnAllGatherMatmul(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmul, AclnnAllGatherMatmulTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AllGatherMatmulApiUtParam>);

class AclnnAllGatherMatmulLaunchTest : public testing::Test {};

TEST_F(AclnnAllGatherMatmulLaunchTest, LaunchApiCoverage)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    (void)aclnnAllGatherMatmul(nullptr, 0, nullptr, nullptr);
}

TEST_F(AclnnAllGatherMatmulLaunchTest, Ascend950LaunchCcuCommMode)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    (void)aclnnAllGatherMatmul(nullptr, 0, nullptr, nullptr);
    op::SetPlatformNpuArch(NpuArch::DAV_3103);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

} // namespace AllGatherMatmulUT
