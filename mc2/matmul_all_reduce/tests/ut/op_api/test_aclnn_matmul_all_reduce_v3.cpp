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
#include "matmul_all_reduce_api_ut_param.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "../../../op_api/aclnn_matmul_all_reduce_v3.h"

namespace MatmulAllReduceUT {

namespace {

void RunMatmulAllReduceV3Ut(op::SocVersion soc, const TensorDesc& x1, const TensorDesc& x2, const aclTensor* bias,
                            const aclTensor* x3, const TensorDesc& output, const char* commMode,
                            aclnnStatus expectResult)
{
    op::SetPlatformSocVersion(soc);
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ut = OP_API_UT(aclnnMatmulAllReduceV3,
                        INPUT(x1, x2, bias, x3, group, reduceOp, commMode, commTurn, streamMode), OUTPUT(output));
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            aclnnMatmulAllReduceV3(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(expectResult, aclnnRet);
    }
}

} // namespace

class AclnnMatmulAllReduceV3Test : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnMatmulAllReduceV3Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce AclnnMatmulAllReduceV3Test TearDown" << std::endl;
    }
};

TEST_P(AclnnMatmulAllReduceV3Test, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut = OP_API_UT(aclnnMatmulAllReduceV3,
                        INPUT(param.x1, param.x2, param.bias, param.x3, param.group.c_str(), param.reduceOp.c_str(),
                              param.commMode.c_str(), param.commTurn, param.streamMode),
                        OUTPUT(param.output));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            aclnnMatmulAllReduceV3(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce, AclnnMatmulAllReduceV3Test,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceApiUtParam>);

class AclnnMatmulAllReduceV3ExtraTest : public testing::Test {};

TEST_F(AclnnMatmulAllReduceV3ExtraTest, Ascend310PWithoutX3)
{
    TensorDesc x1 = {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc bias = {{16}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunMatmulAllReduceV3Ut(op::SocVersion::ASCEND310P, x1, x2, bias.ToAclTypeRawPtr(), nullptr, output, "ai_cpu",
                           ACLNN_SUCCESS);
}

TEST_F(AclnnMatmulAllReduceV3ExtraTest, TransposedX2)
{
    TensorDesc x1 = {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 32}};
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunMatmulAllReduceV3Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, output, "ai_cpu", ACLNN_SUCCESS);
}

TEST_F(AclnnMatmulAllReduceV3ExtraTest, LaunchApiNullExecutor)
{
    EXPECT_EQ(ACLNN_ERR_INNER, aclnnMatmulAllReduceV3(nullptr, 0, nullptr, nullptr));
}

TEST_F(AclnnMatmulAllReduceV3ExtraTest, InvalidNonContiguousX2)
{
    TensorDesc x1 = {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {16, 2}};
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunMatmulAllReduceV3Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, output, "", ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMatmulAllReduceV3ExtraTest, Dav3510NullCommMode)
{
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);
    TensorDesc x1 = {{16, 32}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ut =
        OP_API_UT(aclnnMatmulAllReduceV3,
                  INPUT(x1, x2, nullptr, nullptr, group, reduceOp, nullptr, commTurn, streamMode), OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID, ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

} // namespace MatmulAllReduceUT
