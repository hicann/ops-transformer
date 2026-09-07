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
#include "../../../op_api/aclnn_quant_matmul_all_reduce_v3.h"

namespace MatmulAllReduceUT {

namespace {

void RunQuantMatmulAllReduceV3Ut(op::SocVersion soc, const TensorDesc &x1, const TensorDesc &x2, const aclTensor *bias,
                                 const aclTensor *x3, const TensorDesc &dequantScale, const aclTensor *pertokenScale,
                                 const aclTensor *commQuantScale1, const aclTensor *commQuantScale2,
                                 const TensorDesc &output, aclnnStatus expectResult)
{
    op::SetPlatformSocVersion(soc);
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto ut = OP_API_UT(aclnnQuantMatmulAllReduceV3,
                        INPUT(x1, x2, bias, x3, dequantScale, pertokenScale, commQuantScale1, commQuantScale2, group,
                              reduceOp, commTurn, streamMode),
                        OUTPUT(output));
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            aclnnQuantMatmulAllReduceV3(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(expectResult, aclnnRet);
    }
}

} // namespace

class AclnnQuantMatmulAllReduceV3Test : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV3Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV3Test TearDown" << std::endl;
    }
};

TEST_P(AclnnQuantMatmulAllReduceV3Test, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceV3,
        INPUT(param.x1, param.x2, param.bias, param.x3, param.dequantScale, param.pertokenScale, param.commQuantScale1,
              param.commQuantScale2, param.group.c_str(), param.reduceOp.c_str(), param.commTurn, param.streamMode),
        OUTPUT(param.output));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            aclnnQuantMatmulAllReduceV3(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce, AclnnQuantMatmulAllReduceV3Test,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceApiUtParam>);

class AclnnQuantMatmulAllReduceV3ExtraTest : public testing::Test {};

TEST_F(AclnnQuantMatmulAllReduceV3ExtraTest, NzFormatWeightX2)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 64, 128}};
    TensorDesc dequantScale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc pertokenScale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV3Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, dequantScale,
                                pertokenScale.ToAclTypeRawPtr(), nullptr, nullptr, output, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV3ExtraTest, Ascend310PPreTransposedWeight)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {1, 1, 64, 128}};
    TensorDesc dequantScale = {{128}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV3Ut(op::SocVersion::ASCEND310P, x1, x2, nullptr, nullptr, dequantScale, nullptr, nullptr,
                                nullptr, output, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV3ExtraTest, Ascend310PWithPertokenDtypeMismatch)
{
    TensorDesc x1 = {{16, 32}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{64}, ACL_UINT64, ACL_FORMAT_ND};
    TensorDesc pertokenScale = {{16}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{16, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV3Ut(op::SocVersion::ASCEND310P, x1, x2, nullptr, nullptr, dequantScale,
                                pertokenScale.ToAclTypeRawPtr(), nullptr, nullptr, output, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnQuantMatmulAllReduceV3ExtraTest, TransposedX2Int8)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 64}};
    TensorDesc dequantScale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc pertokenScale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV3Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, dequantScale,
                                pertokenScale.ToAclTypeRawPtr(), nullptr, nullptr, output, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV3ExtraTest, Int64DequantScale)
{
    TensorDesc x1 = {{16, 32}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{64}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc output = {{16, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV3Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, dequantScale, nullptr, nullptr,
                                nullptr, output, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV3ExtraTest, Ascend950GetWorkspaceSize)
{
    TensorDesc x1 = {{16, 32}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{64}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{16, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV3Ut(op::SocVersion::ASCEND950, x1, x2, nullptr, nullptr, dequantScale, nullptr, nullptr,
                                nullptr, output, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV3ExtraTest, InvalidNonContiguousX2)
{
    TensorDesc x1 = {{16, 32}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND, {64, 2}};
    TensorDesc dequantScale = {{64}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{16, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV3Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, dequantScale, nullptr, nullptr,
                                nullptr, output, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnQuantMatmulAllReduceV3ExtraTest, LaunchApiCoverage)
{
    (void)aclnnQuantMatmulAllReduceV3(nullptr, 0, nullptr, nullptr);
}

TEST_F(AclnnQuantMatmulAllReduceV3ExtraTest, Ascend950LaunchCcuCommMode)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    (void)aclnnQuantMatmulAllReduceV3(nullptr, 0, nullptr, nullptr);
}

} // namespace MatmulAllReduceUT
