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
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_csv_case_loader.h"
#include "../../../op_api/aclnn_all_gather_matmul.h"

namespace AllGatherMatmulUT {

class AclnnAllGatherMatmulNullptrTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmul AclnnAllGatherMatmulNullptrTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        std::cout << "AllGatherMatmul AclnnAllGatherMatmulNullptrTest TearDown" << std::endl;
    }
};

TEST_F(AclnnAllGatherMatmulNullptrTest, TestAllGatherFirstApiX2NonContiguousWithoutTranspose)
{
    // 测试场景：非转置的连续x2输入，应返回校验失败
    TensorDesc x1Desc = TensorDesc({128, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    // 设置x2非转置非连续步长，假设每行之间间隔10个元素[64 + 10, 1]，实际内存存储storageShape为[512, 64 + 10]
    TensorDesc x2Desc = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND, {512 + 10, 1});
    TensorDesc outDesc = TensorDesc({128, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({128, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1Desc, x2Desc, nullptr, "test_all_gather_group", 0, 8, 1),
                        OUTPUT(outDesc, gatherOutDesc));

    // 预期：由于 x2 是非连续的（非转置），应该返回 ACLNN_ERR_PARAM_INVALID
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnAllGatherMatmulNullptrTest, TestStreamModeInvalid)
{
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1Desc, x2Desc, nullptr, "test_all_gather_group", 0, 8, 0),
                        OUTPUT(outDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnAllGatherMatmulNullptrTest, TestAscend910A5Platform)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmul, INPUT(x1Desc, x2Desc, nullptr, "test_all_gather_group", 0, 8, 1),
                        OUTPUT(outDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
    op::SetPlatformNpuArch(NpuArch::DAV_3103);
}

} // namespace AllGatherMatmulUT
