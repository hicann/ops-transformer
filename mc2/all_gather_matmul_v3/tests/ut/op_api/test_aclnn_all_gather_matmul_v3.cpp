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
#include "test_all_gather_matmul_v3_api_ut_param.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

extern "C" aclnnStatus aclnnAllGatherQuantMatmulV3GetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional, const aclTensor *x2ScaleOptional, const char *group, int64_t rankSize,
    int64_t hcclBufferSize, int64_t groupSize, const char *commMode, const aclTensor *output,
    const aclTensor *gatherOut, uint64_t *workspaceSize, aclOpExecutor **executor);
extern "C" aclnnStatus aclnnAllGatherQuantMatmulV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream);

namespace AllGatherMatmulV3UT {

class TestAclnnAllGatherMatmulV3 : public testing::TestWithParam<AllGatherMatmulV3ApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "TestAclnnAllGatherMatmulV3 SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "TestAclnnAllGatherMatmulV3 TearDown" << endl;
    }
};

TEST_P(TestAclnnAllGatherMatmulV3, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    aclTensor *context = param.context.GetViewDims().empty() ? nullptr : param.context.ToAclTypeRawPtr();
    aclTensor *x1 = param.x1.GetViewDims().empty() ? nullptr : param.x1.ToAclTypeRawPtr();
    aclTensor *x2 = param.x2.GetViewDims().empty() ? nullptr : param.x2.ToAclTypeRawPtr();
    aclTensor *bias = param.bias.GetViewDims().empty() ? nullptr : param.bias.ToAclTypeRawPtr();
    aclTensor *x1Scale = param.x1Scale.GetViewDims().empty() ? nullptr : param.x1Scale.ToAclTypeRawPtr();
    aclTensor *x2Scale = param.x2Scale.GetViewDims().empty() ? nullptr : param.x2Scale.ToAclTypeRawPtr();
    aclTensor *output = param.output.GetViewDims().empty() ? nullptr : param.output.ToAclTypeRawPtr();
    aclTensor *gatherOut = param.gatherOut.GetViewDims().empty() ? nullptr : param.gatherOut.ToAclTypeRawPtr();

    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(context, x1, x2, bias, x1Scale, x2Scale, param.group.c_str(), param.rankSize,
                              param.hcclBufferSize, param.groupSize, param.commMode.c_str()),
                        OUTPUT(output, gatherOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, param.expectResult);
    }
}

TEST_F(TestAclnnAllGatherMatmulV3, SecondPhaseNullExecutorTest)
{
    aclnnStatus aclRet = aclnnAllGatherQuantMatmulV3(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(TestAclnnAllGatherMatmulV3, NullGroupTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherQuantMatmulV3,
                        INPUT(contextDesc, x1Desc, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc, (const char *)nullptr, 4,
                              200, 4295032864L, "urma"),
                        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnAllGatherMatmulV3, EmptyGroupTest)
{
    TensorDesc contextDesc = TensorDesc({100}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x1Desc = TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Desc = TensorDesc({2560, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1ScaleDesc = TensorDesc({2048, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2ScaleDesc = TensorDesc({2560, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc outputDesc = TensorDesc({8192, 2560}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOutDesc = TensorDesc({8192, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherQuantMatmulV3,
        INPUT(contextDesc, x1Desc, x2Desc, nullptr, x1ScaleDesc, x2ScaleDesc, "", 4, 200, 4295032864L, "urma"),
        OUTPUT(outputDesc, gatherOutDesc));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmulV3, TestAclnnAllGatherMatmulV3,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulV3ApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AllGatherMatmulV3ApiUtParam>);

} // namespace AllGatherMatmulV3UT
