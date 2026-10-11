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
#include "../../../op_api/aclnn_quant_matmul_all_reduce_v5.h"

namespace MatmulAllReduceUT {

namespace {

void RunQuantMatmulAllReduceV5Ut(op::SocVersion soc, const TensorDesc& x1, const TensorDesc& x2, const aclTensor* bias,
                                 const aclTensor* x3, const aclTensor* x1Scale, const TensorDesc& x2Scale,
                                 const aclTensor* commQuantScale1, const aclTensor* commQuantScale2,
                                 const TensorDesc& output, const char* commMode, int64_t commQuantMode,
                                 aclnnStatus expectResult)
{
    op::SetPlatformSocVersion(soc);
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ut = OP_API_UT(aclnnQuantMatmulAllReduceV5,
                        INPUT(x1, x2, bias, x3, x1Scale, x2Scale, commQuantScale1, commQuantScale2, group, reduceOp,
                              commMode, commTurn, streamMode, groupSize, commQuantMode),
                        OUTPUT(output));
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            aclnnQuantMatmulAllReduceV5(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(expectResult, aclnnRet);
    }
}

} // namespace

class AclnnQuantMatmulAllReduceV5Test : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV5Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV5Test TearDown" << std::endl;
    }
};

TEST_P(AclnnQuantMatmulAllReduceV5Test, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut =
        OP_API_UT(aclnnQuantMatmulAllReduceV5,
                  INPUT(param.x1, param.x2, param.bias, param.x3, param.x1Scale, param.x2Scale, param.commQuantScale1,
                        param.commQuantScale2, param.group.c_str(), param.reduceOp.c_str(), param.commMode.c_str(),
                        param.commTurn, param.streamMode, param.groupSize, param.commQuantMode),
                  OUTPUT(param.output));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            aclnnQuantMatmulAllReduceV5(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce, AclnnQuantMatmulAllReduceV5Test,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceApiUtParam>);

class AclnnQuantMatmulAllReduceV5ExtraTest : public testing::Test {};

TEST_F(AclnnQuantMatmulAllReduceV5ExtraTest, NzFormatWeightX2)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 64, 128}};
    TensorDesc x2Scale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV5Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, nullptr, x2Scale, nullptr,
                                nullptr, output, "ai_cpu", 0, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV5ExtraTest, Dav3510LaunchCcuCommMode)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2Scale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV5Ut(op::SocVersion::ASCEND950, x1, x2, nullptr, nullptr, nullptr, x2Scale, nullptr, nullptr,
                                output, "ccu", 0, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV5ExtraTest, Fp8Float32Output)
{
    TensorDesc x1 = {{32, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND};
    TensorDesc bias = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc x3 = {{32, 128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc x1Scale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc x2Scale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV5Ut(op::SocVersion::ASCEND910B, x1, x2, bias.ToAclTypeRawPtr(), x3.ToAclTypeRawPtr(),
                                x1Scale.ToAclTypeRawPtr(), x2Scale, nullptr, nullptr, output, "ai_cpu", 0,
                                ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV5ExtraTest, TransposedX2Int8)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 64}};
    TensorDesc x2Scale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char* group = "group";
    const char* reduceOp = "sum";
    const char* commMode = "ai_cpu";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    int64_t commQuantMode = 0;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ut = OP_API_UT(aclnnQuantMatmulAllReduceV5,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, x2Scale, nullptr, nullptr, group, reduceOp, commMode,
                              commTurn, streamMode, groupSize, commQuantMode),
                        OUTPUT(output));
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
}

TEST_F(AclnnQuantMatmulAllReduceV5ExtraTest, ReduceOpNullptr)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2Scale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char* group = "group";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    int64_t commQuantMode = 0;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ut = OP_API_UT(aclnnQuantMatmulAllReduceV5,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, x2Scale, nullptr, nullptr, group, nullptr, "",
                              commTurn, streamMode, groupSize, commQuantMode),
                        OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnQuantMatmulAllReduceV5ExtraTest, TransposedX2With2DScale)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 64}};
    TensorDesc x2Scale = {{2, 128}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV5Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, nullptr, x2Scale, nullptr,
                                nullptr, output, "ai_cpu", 0, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV5ExtraTest, TransposedFloat4WithMxScale)
{
    TensorDesc x1 = {{32, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, {1, 64}};
    TensorDesc x1Scale = {{2, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND};
    TensorDesc x2Scale = {{128, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, {2, 256, 1}};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV5Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, nullptr, x1Scale.ToAclTypeRawPtr(),
                                x2Scale, nullptr, nullptr, output, "ai_cpu", 0, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV5ExtraTest, Ascend310PPreTransposedWeight)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {1, 1, 64, 128}};
    TensorDesc x2Scale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV5Ut(op::SocVersion::ASCEND310P, x1, x2, nullptr, nullptr, nullptr, x2Scale, nullptr,
                                nullptr, output, "ai_cpu", 0, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV5ExtraTest, Dav3510LaunchAicpuCommMode)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2Scale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunQuantMatmulAllReduceV5Ut(op::SocVersion::ASCEND950, x1, x2, nullptr, nullptr, nullptr, x2Scale, nullptr, nullptr,
                                output, "ai_cpu", 0, ACLNN_SUCCESS);
}

TEST_F(AclnnQuantMatmulAllReduceV5ExtraTest, LaunchApiNullExecutor)
{
    EXPECT_EQ(ACLNN_ERR_INNER, aclnnQuantMatmulAllReduceV5(nullptr, 0, nullptr, nullptr));
}

} // namespace MatmulAllReduceUT
