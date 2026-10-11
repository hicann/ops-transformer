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
#include "../../../op_api/aclnn_allto_allv_quant_grouped_mat_mul.h"

namespace AlltoAllvQuantGroupedMatMulUT {

class AclnnAlltoAllvQuantGroupedMatMulNullptrTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllvQuantGroupedMatMul AclnnAlltoAllvQuantGroupedMatMulNullptrTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvQuantGroupedMatMul AclnnAlltoAllvQuantGroupedMatMulNullptrTest TearDown" << std::endl;
    }
};

TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullGmmX)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc sendCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc recvCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    std::vector<int64_t> sendCountsList(8, 1024);
    std::vector<int64_t> recvCountsList(8, 1024);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_null_gmm_x =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(nullptr, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight,
                        groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID, ut_null_gmm_x.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullGmmWeight)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc sendCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc recvCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    std::vector<int64_t> sendCountsList(8, 1024);
    std::vector<int64_t> recvCountsList(8, 1024);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_null_gmm_weight =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(gmmX, nullptr, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight,
                        groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR,
              ut_null_gmm_weight.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullGmmXScale)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc sendCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc recvCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    std::vector<int64_t> sendCountsList(8, 1024);
    std::vector<int64_t> recvCountsList(8, 1024);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_null_gmm_x_scale =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(gmmX, gmmWeight, nullptr, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight,
                        groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_null_gmm_x_scale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullGmmWeightScale)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc sendCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc recvCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    std::vector<int64_t> sendCountsList(8, 1024);
    std::vector<int64_t> recvCountsList(8, 1024);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_null_gmm_weight_scale = OP_API_UT(
        aclnnAlltoAllvQuantGroupedMatMul,
        INPUT(gmmX, gmmWeight, gmmXScale, nullptr, nullptr, nullptr, mmXOptional, mmWeightOptional, mmXScaleOptional,
              mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, group,
              epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, groupSize, permuteOutFlag),
        OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_null_gmm_weight_scale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNotNullSendTensor)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc sendCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc recvCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    std::vector<int64_t> sendCountsList(8, 1024);
    std::vector<int64_t> recvCountsList(8, 1024);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_not_null_send_tensor =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, sendCountsTensorOptional, nullptr, mmXOptional,
                        mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode,
                        mmXQuantMode, mmWeightQuantMode, group, epWorldSize, sendCounts, recvCounts, transGmmWeight,
                        transMmWeight, groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_not_null_send_tensor.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNotNullRecvTensor)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc sendCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc recvCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    std::vector<int64_t> sendCountsList(8, 1024);
    std::vector<int64_t> recvCountsList(8, 1024);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_not_null_recv_tensor =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, recvCountsTensorOptional, mmXOptional,
                        mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode,
                        mmXQuantMode, mmWeightQuantMode, group, epWorldSize, sendCounts, recvCounts, transGmmWeight,
                        transMmWeight, groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_not_null_recv_tensor.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNotNullPermuteoutFalse)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc sendCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc recvCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    std::vector<int64_t> sendCountsList(8, 1024);
    std::vector<int64_t> recvCountsList(8, 1024);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_not_null_permuteout_false =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight,
                        groupSize, 0),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_not_null_permuteout_false.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullPermuteoutTrue)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc sendCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc recvCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    std::vector<int64_t> sendCountsList(8, 1024);
    std::vector<int64_t> recvCountsList(8, 1024);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_null_permuteout_true =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight,
                        groupSize, permuteOutOptional),
                  OUTPUT(gmmY, mmYOptional, nullptr));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_null_permuteout_true.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulQuantModeNotVaild)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc sendCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc recvCountsTensorOptional = {{8}, ACL_INT64, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    std::vector<int64_t> sendCountsList(8, 1024);
    std::vector<int64_t> recvCountsList(8, 1024);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_quant_mode_not_vaild = OP_API_UT(
        aclnnAlltoAllvQuantGroupedMatMul,
        INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
              mmXScaleOptional, mmWeightScaleOptional, 0, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, group,
              epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, groupSize, permuteOutOptional),
        OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_quant_mode_not_vaild.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

// Test: gmmY is null - should reach CheckNotNull gmmY null check
TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullGmmY)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    int listSize = 8;
    int listValue = 1024;
    std::vector<int64_t> sendCountsList(listSize, listValue);
    std::vector<int64_t> recvCountsList(listSize, listValue);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto ut_null_gmm_y =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight,
                        groupSize, permuteOutFlag),
                  OUTPUT(nullptr, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID, ut_null_gmm_y.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

// Test: notContiguous gmmWeight + transGmmWeight=true → error
TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulTransGmmWeightNotContiguous)
{
    // Create non-contiguous gmmWeight with swapped strides (transposed last two dims)
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    // shape [4, 7168, 4096] with strides [7168*4096, 1, 7168] → last two dims transposed
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND, {29360128, 1, 7168}};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    int listSize = 8;
    int listValue = 1024;
    std::vector<int64_t> sendCountsList(listSize, listValue);
    std::vector<int64_t> recvCountsList(listSize, listValue);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 1; // transGmmWeight=true with non-contiguous → error
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto ut_trans_not_contiguous =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight,
                        groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_trans_not_contiguous.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

// Test: notContiguous mmWeight + transMmWeight=true → error
TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulTransMmWeightNotContiguous)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    // Non-contiguous mmWeight: shape [7168, 4096] with strides [1, 7168] → transposed
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND, {1, 7168}};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    int listSize = 8;
    int listValue = 1024;
    std::vector<int64_t> sendCountsList(listSize, listValue);
    std::vector<int64_t> recvCountsList(listSize, listValue);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 1; // transMmWeight=true with non-contiguous → error
    int32_t permuteOutFlag = 1;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc permuteOutOptional = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto ut_mm_trans_not_contiguous =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight,
                        groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_mm_trans_not_contiguous.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

// Non-contiguous gmmWeight on A5 (arch35/DAV_3510) → triggers TransGmmWeightTensor +
// TransGmmWeightScaleTensor(pertensor early return)
TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNonContiguousGmmWeightArch35)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND, {29360128, 1, 7168}};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    int listSize = 8;
    int listValue = 1024;
    std::vector<int64_t> sendCountsList(listSize, listValue);
    std::vector<int64_t> recvCountsList(listSize, listValue);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 0;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut = OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                        INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional,
                              mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode,
                              gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, group, epWorldSize, sendCounts,
                              recvCounts, transGmmWeight, transMmWeight, groupSize, permuteOutFlag),
                        OUTPUT(gmmY, mmYOptional, nullptr));
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, ret);
}

// Non-contiguous mmWeight on A5 (arch35/DAV_3510) → triggers TransMmWeightOptionalTensor
TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulNonContiguousMmWeightArch35)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND, {1, 7168}};
    TensorDesc mmXScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWeightQuantMode = 1;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    int listSize = 8;
    int listValue = 1024;
    std::vector<int64_t> sendCountsList(listSize, listValue);
    std::vector<int64_t> recvCountsList(listSize, listValue);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 0;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut = OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                        INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional,
                              mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode,
                              gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, group, epWorldSize, sendCounts,
                              recvCounts, transGmmWeight, transMmWeight, groupSize, permuteOutFlag),
                        OUTPUT(gmmY, mmYOptional, nullptr));
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, ret);
}

// MX quant with non-contiguous gmmWeight + gmmWeightScale (4D) on A5 → triggers TransGmmWeightScaleTensor MX
// path
TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulMxQuantNonContiguousArch35)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND, {29360128, 1, 7168}};
    TensorDesc gmmXScale = {{8192, 112, 2}, ACL_FLOAT, ACL_FORMAT_ND};
    // 4D gmmWeightScale with non-contiguous strides (dims 1,2 swapped) → triggers TransGmmWeightScaleTensor MX path
    TensorDesc gmmWeightScale = {{4, 112, 4096, 2}, ACL_FLOAT, ACL_FORMAT_ND, {917504, 2, 8192, 1}};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc mmXScaleOptional = {{4096, 112, 2}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmWeightScaleOptional = {{112, 4096, 2}, ACL_FLOAT, ACL_FORMAT_ND};
    int64_t gmmXQuantMode = 6;
    int64_t gmmWeightQuantMode = 6;
    int64_t mmXQuantMode = 6;
    int64_t mmWeightQuantMode = 6;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    int listSize = 8;
    int listValue = 1024;
    std::vector<int64_t> sendCountsList(listSize, listValue);
    std::vector<int64_t> recvCountsList(listSize, listValue);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 0;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut = OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                        INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional,
                              mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode,
                              gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, group, epWorldSize, sendCounts,
                              recvCounts, transGmmWeight, transMmWeight, groupSize, permuteOutFlag),
                        OUTPUT(gmmY, mmYOptional, nullptr));
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, ret);
}

// MX quant mm: non-contiguous mmWeight + 3D non-contiguous mmWeightScale on A5 → triggers
// TransMmWeightScaleTensor
TEST_F(AclnnAlltoAllvQuantGroupedMatMulNullptrTest, aclnnAlltoAllvQuantGroupedMatMulMxQuantMmNonContiguousArch35)
{
    TensorDesc gmmX = {{8192, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmWeight = {{4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    TensorDesc gmmXScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc gmmWeightScale = {{1}, ACL_FLOAT, ACL_FORMAT_ND};
    TensorDesc mmXOptional = {{4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND};
    // Non-contiguous mmWeight: shape [7168, 4096] strides [1, 7168] (transposed)
    TensorDesc mmWeightOptional = {{7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND, {1, 7168}};
    TensorDesc mmXScaleOptional = {{4096, 112, 2}, ACL_FLOAT, ACL_FORMAT_ND};
    // 3D non-contiguous mmWeightScale: shape [112, 4096, 2] strides [2, 8192, 1] (dims 0,1 swapped)
    TensorDesc mmWeightScaleOptional = {{112, 4096, 2}, ACL_FLOAT, ACL_FORMAT_ND, {2, 8192, 1}};
    int64_t gmmXQuantMode = 1;
    int64_t gmmWeightQuantMode = 1;
    int64_t mmXQuantMode = 6;
    int64_t mmWeightQuantMode = 6;
    int64_t epWorldSize = 0;
    int64_t groupSize = 0;
    int listSize = 8;
    int listValue = 1024;
    std::vector<int64_t> sendCountsList(listSize, listValue);
    std::vector<int64_t> recvCountsList(listSize, listValue);
    aclIntArray* sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray* recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    int32_t transGmmWeight = 0;
    int32_t transMmWeight = 0;
    int32_t permuteOutFlag = 0;
    TensorDesc gmmY = {{8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    TensorDesc mmYOptional = {{4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND};
    const char* group = "group";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut = OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                        INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional,
                              mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode,
                              gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, group, epWorldSize, sendCounts,
                              recvCounts, transGmmWeight, transMmWeight, groupSize, permuteOutFlag),
                        OUTPUT(gmmY, mmYOptional, nullptr));
    auto ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(ACLNN_ERR_PARAM_INVALID, ret);
}

} // namespace AlltoAllvQuantGroupedMatMulUT
