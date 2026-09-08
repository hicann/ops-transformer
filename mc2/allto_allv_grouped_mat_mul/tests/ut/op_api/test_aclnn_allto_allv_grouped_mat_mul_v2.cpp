/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../op_api/aclnn_allto_allv_grouped_mat_mul_v2.h"

#include <array>
#include <vector>

#include <gmock/gmock.h>
#include "gtest/gtest.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace AlltoAllvGroupedMatMulV2UT {
class L2AlltoAllvGroupedMatMulV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
        cout << "L2AlltoAllvGroupedMatMulV2Test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AlltoAllvGroupedMatMulV2Test TearDown" << endl;
    }
};

static aclnnStatus CallGetWorkspaceSize(const char *commMode, aclDataType dtype, aclFormat weightFormat = ACL_FORMAT_ND,
                                        op::SocVersion socVersion = op::SocVersion::ASCEND910_93)
{
    op::SetPlatformSocVersion(socVersion);
    TensorDesc gmmX = TensorDesc({4096, 7168}, dtype, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, dtype, weightFormat);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    TensorDesc gmmY = TensorDesc({4096, 4096}, dtype, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        commMode, epWorldSize, sendCounts, recvCounts, false, false, false),
                  OUTPUT(gmmY, nullptr, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    return ret;
}

TEST_F(L2AlltoAllvGroupedMatMulV2Test, Test)
{
    // A valid call reaches NNopbase initialization. The API UT has no bound device
    // context, so assert that the V2 parameter checker accepts the request.
    EXPECT_NE(CallGetWorkspaceSize("aiv", ACL_BF16), ACLNN_ERR_PARAM_INVALID);
    EXPECT_NE(CallGetWorkspaceSize("ai_cpu", ACL_BF16), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CallGetWorkspaceSize("ccu", ACL_BF16), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CallGetWorkspaceSize("", ACL_BF16), ACLNN_ERR_PARAM_INVALID);
    EXPECT_NE(CallGetWorkspaceSize("aiv", ACL_FLOAT16), ACLNN_ERR_PARAM_INVALID);
    EXPECT_NE(CallGetWorkspaceSize("aiv", ACL_BF16, ACL_FORMAT_NCL), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CallGetWorkspaceSize("aiv", ACL_BF16, ACL_FORMAT_FRACTAL_NZ), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestAivSupportedOnA2AndA3)
{
    EXPECT_NE(CallGetWorkspaceSize("aiv", ACL_BF16, ACL_FORMAT_ND, op::SocVersion::ASCEND910B),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_NE(CallGetWorkspaceSize("aiv", ACL_BF16, ACL_FORMAT_ND, op::SocVersion::ASCEND910_93),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_NE(CallGetWorkspaceSize("ai_cpu", ACL_BF16, ACL_FORMAT_ND, op::SocVersion::ASCEND910B),
              ACLNN_ERR_PARAM_INVALID);
}

// sendCounts null
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestSendCountsNull)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        "ccu", epWorldSize, nullptr, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// recvCounts null
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestRecvCountsNull)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        "ccu", epWorldSize, sendCounts, nullptr, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmx null
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestGmmxNull)
{
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAlltoAllvGroupedMatMulV2,
        INPUT(nullptr, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group", "ccu",
              epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// gmmWeight null
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestGmmWeightNull)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                  INPUT(gmmX, nullptr, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        "ccu", epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// gmmY null
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestGmmYNull)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    auto ut =
        OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        "ccu", epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(nullptr, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// group ep null
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestGroupEpNull)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, nullptr, "ccu", epWorldSize,
                              sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// group ep invalid
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestGroupEpInvalid)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr,
                        "test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group_"
                        "test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group_"
                        "test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group_"
                        "test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group",
                        "ccu", epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmx not_null mmweight null mmy null
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestMmXInvalid)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmX = TensorDesc({1024, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, mmX, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        "ccu", epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestPermuteOutFlagInvalid)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = true;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        "ccu", epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
// sendCountsTensorOptional not nullptr
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestSendCountsTensorOptional)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sendCountsTensor = TensorDesc({32}, ACL_INT64, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    std::vector<int64_t> sendCountsList(32, 256);
    std::vector<int64_t> recvCountsList(32, 256);
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                        INPUT(gmmX, gmmWeight, sendCountsTensor, nullptr, nullptr, nullptr, "test_ep_group", "ccu",
                              epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// recvCountsTensorOptional not nullptr
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestRecvCountsTensorOptional)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc recvCountsTensor = TensorDesc({32}, ACL_INT64, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    std::vector<int64_t> sendCountsList(32, 256);
    std::vector<int64_t> recvCountsList(32, 256);
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                        INPUT(gmmX, gmmWeight, nullptr, recvCountsTensor, nullptr, nullptr, "test_ep_group", "ccu",
                              epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// both count tensors not nullptr
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestBothCountsTensorOptional)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sendCountsTensor = TensorDesc({32}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc recvCountsTensor = TensorDesc({32}, ACL_INT64, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    std::vector<int64_t> sendCountsList(32, 256);
    std::vector<int64_t> recvCountsList(32, 256);
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                  INPUT(gmmX, gmmWeight, sendCountsTensor, recvCountsTensor, nullptr, nullptr, "test_ep_group", "ccu",
                        epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// recvCounts empty array
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestRecvCountsEmpty)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    std::vector<int64_t> sendCountsList(32, 256);
    std::vector<int64_t> recvCountsList; // empty
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_ep_group", "ccu", epWorldSize,
                              sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// sendCounts empty array
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestSendCountsEmpty)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    std::vector<int64_t> sendCountsList; // empty
    std::vector<int64_t> recvCountsList(32, 256);
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_ep_group", "ccu", epWorldSize,
                              sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// second API direct call - cover aclnnAlltoAllvGroupedMatMulV2 function body
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestSecondApiDirect)
{
    aclnnStatus aclRet = aclnnAlltoAllvGroupedMatMulV2(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmWeight not null but mmX null - triggers mm consistency check
TEST_F(L2AlltoAllvGroupedMatMulV2Test, TestMmWeightInvalid)
{
    TensorDesc gmmX = TensorDesc({4096, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeight = TensorDesc({4, 7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeight = TensorDesc({7168, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    constexpr int64_t epWorldSize = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t e = 4;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    bool transGmmWeight = false;
    bool transMmWeight = false;
    bool permuteOutFlag = false;
    TensorDesc gmmY_desc = TensorDesc({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMulV2,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, mmWeight, "test_ep_group", "ccu", epWorldSize,
                              sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
} // namespace AlltoAllvGroupedMatMulV2UT
