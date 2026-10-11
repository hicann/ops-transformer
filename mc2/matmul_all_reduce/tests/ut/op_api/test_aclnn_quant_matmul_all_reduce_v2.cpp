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
#include "matmul_all_reduce_api_ut_param.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "../../../op_api/aclnn_quant_matmul_all_reduce_v2.h"

namespace MatmulAllReduceUT {

namespace {

void RunQuantMatmulAllReduceV2Ut(op::SocVersion soc, const TensorDesc& x1, const TensorDesc& x2, const aclTensor* bias,
                                 const aclTensor* x3, const TensorDesc& dequantScale, const aclTensor* pertokenScale,
                                 aclnnStatus expectResult)
{
    op::SetPlatformSocVersion(soc);
    TensorDesc output = {{x1.GetViewDims()[0], x2.GetViewDims()[1]}, ACL_FLOAT16, ACL_FORMAT_ND};
    if (x1.GetViewDims().size() == 3) {
        output = {{x1.GetViewDims()[0], x1.GetViewDims()[1], x2.GetViewDims()[1]}, ACL_FLOAT16, ACL_FORMAT_ND};
    }
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ut = OP_API_UT(aclnnQuantMatmulAllReduceV2,
                        INPUT(x1, x2, bias, x3, dequantScale, pertokenScale, group, reduceOp, commTurn, streamMode),
                        OUTPUT(output));
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            aclnnQuantMatmulAllReduceV2(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(expectResult, aclnnRet);
    }
}

} // namespace

class AclnnQuantMatmulAllReduceV2Test : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV2Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV2Test TearDown" << std::endl;
    }
};

TEST_P(AclnnQuantMatmulAllReduceV2Test, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut = OP_API_UT(aclnnQuantMatmulAllReduceV2,
                        INPUT(param.x1, param.x2, param.bias, param.x3, param.dequantScale, param.pertokenScale,
                              param.group.c_str(), param.reduceOp.c_str(), param.commTurn, param.streamMode),
                        OUTPUT(param.output));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            aclnnQuantMatmulAllReduceV2(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce, AclnnQuantMatmulAllReduceV2Test,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceApiUtParam>);

class AclnnQuantMatmulAllReduceV2ExtraTest : public testing::Test {};

TEST_F(AclnnQuantMatmulAllReduceV2ExtraTest, NzFormatWeightX2)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 64, 128}};
    TensorDesc dequantScale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, dequantScale, nullptr,
                                ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV2ExtraTest, Ascend310PPreTransposedWeight)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {1, 1, 64, 128}};
    TensorDesc dequantScale = {{128}, ACL_INT64, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND310P, x1, x2, nullptr, nullptr, dequantScale, nullptr,
                                ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV2ExtraTest, InvalidNonContiguousX2)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND, {128, 2}};
    TensorDesc dequantScale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, dequantScale, nullptr,
                                ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnQuantMatmulAllReduceV2ExtraTest, Int64DequantScale)
{
    TensorDesc x1 = {{16, 32}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{64}, ACL_INT64, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, dequantScale, nullptr,
                                ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV2ExtraTest, Dav3510GetWorkspaceSize)
{
    TensorDesc x1 = {{16, 32}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{64}, ACL_FLOAT, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND950, x1, x2, nullptr, nullptr, dequantScale, nullptr,
                                ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV2ExtraTest, LaunchApiCoverage)
{
    (void)aclnnQuantMatmulAllReduceV2(nullptr, 0, nullptr, nullptr);
}

TEST_F(AclnnQuantMatmulAllReduceV2ExtraTest, Dav3510LaunchCcuCommMode)
{
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);
    (void)aclnnQuantMatmulAllReduceV2(nullptr, 0, nullptr, nullptr);
}

} // namespace MatmulAllReduceUT
