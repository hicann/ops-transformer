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
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "../../../op_api/aclnn_matmul_all_reduce.h"
#include "../../../op_api/aclnn_matmul_all_reduce_v2.h"
#include "../../../op_api/aclnn_quant_matmul_all_reduce.h"
#include "../../../op_api/aclnn_quant_matmul_all_reduce_v2.h"
#include "../../../op_api/aclnn_quant_matmul_all_reduce_v3.h"
#include "../../../op_api/aclnn_quant_matmul_all_reduce_v4.h"
#include "../../../op_api/aclnn_weight_quant_matmul_all_reduce.h"

namespace MatmulAllReduceUT {

class MatmulAllReduceAclnnInputNullPtrTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnInputNullPtrTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        std::cout << "MatmulAllReduce AclnnInputNullPtrTest TearDown" << std::endl;
    }
};

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnMatmulAllReduce)
{
    TensorDesc x1 = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 8;
    int64_t streamMode = 1;
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut_null_x1 = OP_API_UT(aclnnMatmulAllReduce, INPUT(nullptr, x2, bias, group, reduceOp, commTurn, streamMode),
                                OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 = OP_API_UT(aclnnMatmulAllReduce, INPUT(x1, nullptr, bias, group, reduceOp, commTurn, streamMode),
                                OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_bias =
        OP_API_UT(aclnnMatmulAllReduce, INPUT(x1, x2, nullptr, group, reduceOp, commTurn, streamMode), OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output =
        OP_API_UT(aclnnMatmulAllReduce, INPUT(x1, x2, bias, group, reduceOp, commTurn, streamMode), OUTPUT(nullptr));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnMatmulAllReduceV2)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc bias = {{128}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc x3 = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc pertokenScale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut_null_x1 =
        OP_API_UT(aclnnQuantMatmulAllReduceV2,
                  INPUT(nullptr, x2, bias, x3, dequantScale, pertokenScale, group, reduceOp, commTurn, streamMode),
                  OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 =
        OP_API_UT(aclnnQuantMatmulAllReduceV2,
                  INPUT(x1, nullptr, bias, x3, dequantScale, pertokenScale, group, reduceOp, commTurn, streamMode),
                  OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_dScale = OP_API_UT(
        aclnnQuantMatmulAllReduceV2,
        INPUT(x1, x2, bias, x3, nullptr, pertokenScale, group, reduceOp, commTurn, streamMode), OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_dScale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output = OP_API_UT(
        aclnnQuantMatmulAllReduceV2,
        INPUT(x1, x2, bias, x3, dequantScale, pertokenScale, group, reduceOp, commTurn, streamMode), OUTPUT(nullptr));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduce)
{
    TensorDesc x1 = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x3 = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc dequantScale = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut_null_x1 =
        OP_API_UT(aclnnQuantMatmulAllReduce,
                  INPUT(nullptr, x2, bias, x3, dequantScale, group, reduceOp, commTurn, streamMode), OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 =
        OP_API_UT(aclnnQuantMatmulAllReduce,
                  INPUT(x1, nullptr, bias, x3, dequantScale, group, reduceOp, commTurn, streamMode), OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_bias =
        OP_API_UT(aclnnQuantMatmulAllReduce,
                  INPUT(x1, x2, nullptr, x3, dequantScale, group, reduceOp, commTurn, streamMode), OUTPUT(output));
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, ut_null_bias.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x3 =
        OP_API_UT(aclnnQuantMatmulAllReduce,
                  INPUT(x1, x2, bias, nullptr, dequantScale, group, reduceOp, commTurn, streamMode), OUTPUT(output));
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, ut_null_x3.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_dequant =
        OP_API_UT(aclnnQuantMatmulAllReduce, INPUT(x1, x2, bias, x3, nullptr, group, reduceOp, commTurn, streamMode),
                  OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR,
              ut_null_dequant.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output =
        OP_API_UT(aclnnQuantMatmulAllReduce,
                  INPUT(x1, x2, bias, x3, dequantScale, group, reduceOp, commTurn, streamMode), OUTPUT(nullptr));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduceV2)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc bias = {{128}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc x3 = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc pertokenScale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut_null_x1 =
        OP_API_UT(aclnnQuantMatmulAllReduceV2,
                  INPUT(nullptr, x2, bias, x3, dequantScale, pertokenScale, group, reduceOp, commTurn, streamMode),
                  OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 =
        OP_API_UT(aclnnQuantMatmulAllReduceV2,
                  INPUT(x1, nullptr, bias, x3, dequantScale, pertokenScale, group, reduceOp, commTurn, streamMode),
                  OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_dScale = OP_API_UT(
        aclnnQuantMatmulAllReduceV2,
        INPUT(x1, x2, bias, x3, nullptr, pertokenScale, group, reduceOp, commTurn, streamMode), OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_dScale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output = OP_API_UT(
        aclnnQuantMatmulAllReduceV2,
        INPUT(x1, x2, bias, x3, dequantScale, pertokenScale, group, reduceOp, commTurn, streamMode), OUTPUT(nullptr));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduceV3)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc bias = {{128}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc x3 = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc pertokenScale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc commQuantScale1 = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc commQuantScale2 = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut_null_x1 = OP_API_UT(aclnnQuantMatmulAllReduceV3,
                                INPUT(nullptr, x2, bias, x3, dequantScale, pertokenScale, commQuantScale1,
                                      commQuantScale2, group, reduceOp, commTurn, streamMode),
                                OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 = OP_API_UT(aclnnQuantMatmulAllReduceV3,
                                INPUT(x1, nullptr, bias, x3, dequantScale, pertokenScale, commQuantScale1,
                                      commQuantScale2, group, reduceOp, commTurn, streamMode),
                                OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_dScale = OP_API_UT(aclnnQuantMatmulAllReduceV3,
                                    INPUT(x1, x2, bias, x3, nullptr, pertokenScale, commQuantScale1, commQuantScale2,
                                          group, reduceOp, commTurn, streamMode),
                                    OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_dScale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output = OP_API_UT(aclnnQuantMatmulAllReduceV3,
                                    INPUT(x1, x2, bias, x3, dequantScale, pertokenScale, commQuantScale1,
                                          commQuantScale2, group, reduceOp, commTurn, streamMode),
                                    OUTPUT(nullptr));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduceV4)
{
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc bias = {{128}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc x3 = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x1Scale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc x2Scale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc commQuantScale1 = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc commQuantScale2 = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    int64_t commQuantMode = 0;
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut_null_x1 = OP_API_UT(aclnnQuantMatmulAllReduceV4,
                                INPUT(nullptr, x2, bias, x3, x1Scale, x2Scale, commQuantScale1, commQuantScale2, group,
                                      reduceOp, commTurn, streamMode, groupSize, commQuantMode),
                                OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 = OP_API_UT(aclnnQuantMatmulAllReduceV4,
                                INPUT(x1, nullptr, bias, x3, x1Scale, x2Scale, commQuantScale1, commQuantScale2, group,
                                      reduceOp, commTurn, streamMode, groupSize, commQuantMode),
                                OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2Scale = OP_API_UT(aclnnQuantMatmulAllReduceV4,
                                     INPUT(x1, x2, bias, x3, x1Scale, nullptr, commQuantScale1, commQuantScale2, group,
                                           reduceOp, commTurn, streamMode, groupSize, commQuantMode),
                                     OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR,
              ut_null_x2Scale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output = OP_API_UT(aclnnQuantMatmulAllReduceV4,
                                    INPUT(x1, x2, bias, x3, x1Scale, x2Scale, commQuantScale1, commQuantScale2, group,
                                          reduceOp, commTurn, streamMode, groupSize, commQuantMode),
                                    OUTPUT(nullptr));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_reduceOp_v4 = OP_API_UT(aclnnQuantMatmulAllReduceV4,
                                         INPUT(x1, x2, bias, x3, x1Scale, x2Scale, commQuantScale1, commQuantScale2,
                                               group, nullptr, commTurn, streamMode, groupSize, commQuantMode),
                                         OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR,
              ut_null_reduceOp_v4.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, AclnnWeightQuantMatmulAllReduce)
{
    TensorDesc x1 = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({64, 128}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3 = TensorDesc({32, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    TensorDesc output = TensorDesc({32, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut_null_x1 = OP_API_UT(
        aclnnWeightQuantMatmulAllReduce,
        INPUT(nullptr, x2, bias, antiquantScale, antiquantOffset, x3, group, reduceOp, commTurn, streamMode, groupSize),
        OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x1.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_x2 = OP_API_UT(
        aclnnWeightQuantMatmulAllReduce,
        INPUT(x1, nullptr, bias, antiquantScale, antiquantOffset, x3, group, reduceOp, commTurn, streamMode, groupSize),
        OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_x2.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_scale =
        OP_API_UT(aclnnWeightQuantMatmulAllReduce,
                  INPUT(x1, x2, bias, nullptr, antiquantOffset, x3, group, reduceOp, commTurn, streamMode, groupSize),
                  OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_scale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));

    auto ut_null_output = OP_API_UT(
        aclnnWeightQuantMatmulAllReduce,
        INPUT(x1, x2, bias, antiquantScale, antiquantOffset, x3, group, reduceOp, commTurn, streamMode, groupSize),
        OUTPUT(nullptr));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR, ut_null_output.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduceV2_Int64Dequant)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{16, 32}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 16}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{16}, ACL_INT64, ACL_FORMAT_ND};
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantMatmulAllReduceV2,
                        INPUT(x1, x2, nullptr, nullptr, dequantScale, nullptr, group, reduceOp, commTurn, streamMode),
                        OUTPUT(output));
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, ret);
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduceV3_OnlyCommQuantScale1)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc bias = {{128}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{128}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc pertokenScale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc commQuantScale1 = {{128}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantMatmulAllReduceV3,
                        INPUT(x1, x2, bias, nullptr, dequantScale, pertokenScale, commQuantScale1, nullptr, group,
                              reduceOp, commTurn, streamMode),
                        OUTPUT(output));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID, ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduceV3_Int64Dequant)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{16, 32}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 16}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{16}, ACL_INT64, ACL_FORMAT_ND};
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceV3,
        INPUT(x1, x2, nullptr, nullptr, dequantScale, nullptr, nullptr, nullptr, group, reduceOp, commTurn, streamMode),
        OUTPUT(output));
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, ret);
}

/*TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduceV3_NzFormatX2)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{16, 16}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{16, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16}};
    TensorDesc dequantScale = {{16}, ACL_FLOAT, ACL_FORMAT_ND};
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceV3,
        INPUT(x1, x2, nullptr, nullptr, dequantScale, nullptr,
              nullptr, nullptr, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    // NZ format should pass shape check and enter CopyTensor/NZ path
    EXPECT_NE(ACLNN_ERR_PARAM_NULLPTR, ret);
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduce_NzFormatX2)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{16, 16}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{16, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16}};
    TensorDesc bias = {{16}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc x3 = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{16}, ACL_UINT64, ACL_FORMAT_ND};
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduce,
        INPUT(x1, x2, bias, x3, dequantScale, group, reduceOp, commTurn, streamMode),
        OUTPUT(output)
    );
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_NULLPTR, ret);
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, AclnnWeightQuantMatmulAllReduce_NzFormatX2)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x2 = {{16, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16}};
    TensorDesc bias = {{16}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc antiquantScale = {{16}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc antiquantOffset = {{16}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ut = OP_API_UT(
        aclnnWeightQuantMatmulAllReduce,
        INPUT(x1, x2, bias, antiquantScale, antiquantOffset, nullptr, group, reduceOp, commTurn, streamMode, groupSize),
        OUTPUT(output)
    );
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_NULLPTR, ret);
}*/

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduce_Int64Dequant)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{16, 32}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{32, 16}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc bias = {{16}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc x3 = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc dequantScale = {{16}, ACL_INT64, ACL_FORMAT_ND};
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantMatmulAllReduce,
                        INPUT(x1, x2, bias, x3, dequantScale, group, reduceOp, commTurn, streamMode), OUTPUT(output));
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, ret);
}

TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduceV4_Int64X2Scale)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{32, 64}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{64, 128}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc bias = {{128}, ACL_INT32, ACL_FORMAT_ND};
    TensorDesc x3 = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc x1Scale = {{32}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc x2Scale = {{128}, ACL_INT64, ACL_FORMAT_ND};
    const char *group = "group";
    const char *reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    int64_t commQuantMode = 0;
    TensorDesc output = {{32, 128}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantMatmulAllReduceV4,
                        INPUT(x1, x2, bias, x3, x1Scale, x2Scale, nullptr, nullptr, group, reduceOp, commTurn,
                              streamMode, groupSize, commQuantMode),
                        OUTPUT(output));
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, ret);
}

/*
TEST_F(MatmulAllReduceAclnnInputNullPtrTest, aclnnQuantMatmulAllReduceV4_NzFormatX2)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    TensorDesc x1 = {{16, 16}, ACL_INT8, ACL_FORMAT_ND};
    TensorDesc x2 = {{16, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16}};
    TensorDesc x2Scale = {{16}, ACL_FLOAT, ACL_FORMAT_ND};
    const char* group = "group";
    const char* reduceOp = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    int64_t commQuantMode = 0;
    TensorDesc output = {{16, 16}, ACL_FLOAT16, ACL_FORMAT_ND};
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceV4,
        INPUT(x1, x2, nullptr, nullptr, nullptr, x2Scale, nullptr, nullptr,
              group, reduceOp, commTurn, streamMode, groupSize, commQuantMode),
        OUTPUT(output)
    );
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    // NZ format should pass shape check and enter CopyTensor/NZ path
    EXPECT_NE(ACLNN_ERR_PARAM_NULLPTR, ret);
}*/

} // namespace MatmulAllReduceUT
