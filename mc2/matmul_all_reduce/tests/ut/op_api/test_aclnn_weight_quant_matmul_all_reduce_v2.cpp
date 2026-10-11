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
#include "../../../op_api/aclnn_weight_quant_matmul_all_reduce_v2.h"

namespace MatmulAllReduceUT {

namespace {

void RunWeightQuantMatmulAllReduceV2Ut(op::SocVersion soc, const TensorDesc& x1, const TensorDesc& x2,
                                       const TensorDesc* bias, const TensorDesc& antiquantScale,
                                       const TensorDesc* antiquantOffset, const TensorDesc* x3,
                                       const TensorDesc& output, const char* commMode, int64_t groupSize,
                                       aclnnStatus expectResult)
{
    op::SetPlatformSocVersion(soc);
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclnnRet = ACLNN_SUCCESS;

    if (bias != nullptr && antiquantOffset != nullptr) {
        if (x3 != nullptr) {
            auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceV2,
                                INPUT(x1, x2, *bias, antiquantScale, *antiquantOffset, *x3, group, reduceOp, commMode,
                                      commTurn, streamMode, groupSize),
                                OUTPUT(output));
            aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
        } else {
            auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceV2,
                                INPUT(x1, x2, *bias, antiquantScale, *antiquantOffset, nullptr, group, reduceOp,
                                      commMode, commTurn, streamMode, groupSize),
                                OUTPUT(output));
            aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
        }
    } else {
        auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceV2,
                            INPUT(x1, x2, nullptr, antiquantScale, nullptr, nullptr, group, reduceOp, commMode,
                                  commTurn, streamMode, groupSize),
                            OUTPUT(output));
        aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    }

    if (expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            aclnnWeightQuantMatmulAllReduceV2(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(expectResult, aclnnRet);
    }
}

} // namespace

class AclnnWeightQuantMatmulAllReduceV2Test : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnWeightQuantMatmulAllReduceV2Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce AclnnWeightQuantMatmulAllReduceV2Test TearDown" << std::endl;
    }
};

TEST_P(AclnnWeightQuantMatmulAllReduceV2Test, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut = OP_API_UT(aclnnWeightQuantMatmulAllReduceV2,
                        INPUT(param.x1, param.x2, param.bias, param.antiquantScale, param.antiquantOffset, param.x3,
                              param.group.c_str(), param.reduceOp.c_str(), param.commMode.c_str(), param.commTurn,
                              param.streamMode, param.groupSize),
                        OUTPUT(param.output));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
        if (aclnnRet != ACLNN_ERR_PARAM_NULLPTR && executor != nullptr) {
            aclnnWeightQuantMatmulAllReduceV2(nullptr, workspace_size, executor, nullptr);
        }
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce, AclnnWeightQuantMatmulAllReduceV2Test,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceApiUtParam>);

class AclnnWeightQuantMatmulAllReduceV2ExtraTest : public testing::Test {};

TEST_F(AclnnWeightQuantMatmulAllReduceV2ExtraTest, NzFormatWeightX2)
{
    TensorDesc x1 = {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 64, 128}};
    TensorDesc bias = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc antiquantScale = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc antiquantOffset = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunWeightQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND910B, x1, x2, &bias, antiquantScale, &antiquantOffset,
                                      nullptr, output, "ai_cpu", 0, ACLNN_SUCCESS);
}

TEST_F(AclnnWeightQuantMatmulAllReduceV2ExtraTest, Ascend310PPreTransposedWeight)
{
    TensorDesc x1 = {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {1, 1, 64, 128}};
    TensorDesc antiquantScale = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunWeightQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND310P, x1, x2, nullptr, antiquantScale, nullptr, nullptr,
                                      output, "ai_cpu", 0, ACLNN_SUCCESS);
}

TEST_F(AclnnWeightQuantMatmulAllReduceV2ExtraTest, Ascend310PInvalidBf16)
{
    TensorDesc x1 = {{32, 64}, ACL_BF16, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc antiquantScale = {{128}, ACL_BF16, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_BF16, ACL_FORMAT_ND};
    RunWeightQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND310P, x1, x2, nullptr, antiquantScale, nullptr, nullptr,
                                      output, "", 0, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnWeightQuantMatmulAllReduceV2ExtraTest, Dav3510LaunchCcuCommMode)
{
    TensorDesc x1 = {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc antiquantScale = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunWeightQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND950, x1, x2, nullptr, antiquantScale, nullptr, nullptr,
                                      output, "ccu", 0, ACLNN_SUCCESS);
}

TEST_F(AclnnWeightQuantMatmulAllReduceV2ExtraTest, InvalidScaleContiguousWithTransposedX2)
{
    TensorDesc x1 = {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND, {1, 64}};
    TensorDesc antiquantScale = {{2, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunWeightQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND910B, x1, x2, nullptr, antiquantScale, nullptr, nullptr,
                                      output, "", 32, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnWeightQuantMatmulAllReduceV2ExtraTest, Dav3510LaunchAicpuCommMode)
{
    TensorDesc x1 = {{32, 64}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc antiquantScale = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    RunWeightQuantMatmulAllReduceV2Ut(op::SocVersion::ASCEND950, x1, x2, nullptr, antiquantScale, nullptr, nullptr,
                                      output, "ai_cpu", 0, ACLNN_SUCCESS);
}

} // namespace MatmulAllReduceUT
