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
#include "../../../op_api/aclnn_allto_allv_quant_grouped_mat_mul_v2.h"

namespace AlltoAllvQuantGroupedMatMulV2UT {

class AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllvQuantGroupedMatMul AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvQuantGroupedMatMul AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest TearDown" << std::endl;
    }
};

TEST_F(AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullGmmX)
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
    const char* commMode = "ccu";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_null_gmm_x =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMulV2,
                  INPUT(nullptr, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, commMode, epWorldSize, sendCounts, recvCounts, transGmmWeight,
                        transMmWeight, groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID, ut_null_gmm_x.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullGmmWeight)
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
    const char* commMode = "ccu";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_null_gmm_weight =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMulV2,
                  INPUT(gmmX, nullptr, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, commMode, epWorldSize, sendCounts, recvCounts, transGmmWeight,
                        transMmWeight, groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_NULLPTR,
              ut_null_gmm_weight.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullGmmXScale)
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
    const char* commMode = "ccu";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_null_gmm_x_scale =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, nullptr, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, commMode, epWorldSize, sendCounts, recvCounts, transGmmWeight,
                        transMmWeight, groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_null_gmm_x_scale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullGmmWeightScale)
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
    const char* commMode = "ccu";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_null_gmm_weight_scale = OP_API_UT(
        aclnnAlltoAllvQuantGroupedMatMulV2,
        INPUT(gmmX, gmmWeight, gmmXScale, nullptr, nullptr, nullptr, mmXOptional, mmWeightOptional, mmXScaleOptional,
              mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, group,
              commMode, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, groupSize, permuteOutFlag),
        OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_null_gmm_weight_scale.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest, aclnnAlltoAllvQuantGroupedMatMulNotNullSendTensor)
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
    const char* commMode = "ccu";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_not_null_send_tensor =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, sendCountsTensorOptional, nullptr, mmXOptional,
                        mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode,
                        mmXQuantMode, mmWeightQuantMode, group, commMode, epWorldSize, sendCounts, recvCounts,
                        transGmmWeight, transMmWeight, groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_not_null_send_tensor.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest, aclnnAlltoAllvQuantGroupedMatMulNotNullRecvTensor)
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
    const char* commMode = "ccu";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_not_null_recv_tensor =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, recvCountsTensorOptional, mmXOptional,
                        mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode,
                        mmXQuantMode, mmWeightQuantMode, group, commMode, epWorldSize, sendCounts, recvCounts,
                        transGmmWeight, transMmWeight, groupSize, permuteOutFlag),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_not_null_recv_tensor.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest, aclnnAlltoAllvQuantGroupedMatMulNotNullPermuteoutFalse)
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
    const char* commMode = "ccu";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_not_null_permuteout_false =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, commMode, epWorldSize, sendCounts, recvCounts, transGmmWeight,
                        transMmWeight, groupSize, 0),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_not_null_permuteout_false.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest, aclnnAlltoAllvQuantGroupedMatMulNullPermuteoutTrue)
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
    const char* commMode = "ccu";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_null_permuteout_true =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode,
                        mmWeightQuantMode, group, commMode, epWorldSize, sendCounts, recvCounts, transGmmWeight,
                        transMmWeight, groupSize, permuteOutOptional),
                  OUTPUT(gmmY, mmYOptional, nullptr));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_null_permuteout_true.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

TEST_F(AclnnAlltoAllvQuantGroupedMatMulV2NullptrTest, aclnnAlltoAllvQuantGroupedMatMulQuantModeNotVaild)
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
    const char* commMode = "ccu";
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    auto ut_quant_mode_not_vaild =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, gmmXScale, gmmWeightScale, nullptr, nullptr, mmXOptional, mmWeightOptional,
                        mmXScaleOptional, mmWeightScaleOptional, 0, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode,
                        group, commMode, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, groupSize,
                        permuteOutOptional),
                  OUTPUT(gmmY, mmYOptional, permuteOutOptional));
    EXPECT_EQ(ACLNN_ERR_PARAM_INVALID,
              ut_quant_mode_not_vaild.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor));
}

} // namespace AlltoAllvQuantGroupedMatMulV2UT
