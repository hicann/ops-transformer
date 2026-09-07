/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../op_api/aclnn_allto_allv_grouped_mat_mul.h"

#include <array>
#include <vector>

#include <gmock/gmock.h>
#include "gtest/gtest.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace AlltoAllvGroupedMatMulUT {
class L2AlltoAllvGroupedMatMulTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
        cout << "L2AlltoAllvGroupedMatMulTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AlltoAllvGroupedMatMulTest TearDown" << endl;
    }
};

TEST_F(L2AlltoAllvGroupedMatMulTest, Test)
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
        OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// sendCounts null
TEST_F(L2AlltoAllvGroupedMatMulTest, TestSendCountsNull)
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
        OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        epWorldSize, nullptr, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// recvCounts null
TEST_F(L2AlltoAllvGroupedMatMulTest, TestRecvCountsNull)
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
        OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        epWorldSize, sendCounts, nullptr, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmx null
TEST_F(L2AlltoAllvGroupedMatMulTest, TestGmmxNull)
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
        aclnnAlltoAllvGroupedMatMul,
        INPUT(nullptr, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
              epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// gmmWeight null
TEST_F(L2AlltoAllvGroupedMatMulTest, TestGmmWeightNull)
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
        OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                  INPUT(gmmX, nullptr, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// gmmY null
TEST_F(L2AlltoAllvGroupedMatMulTest, TestGmmYNull)
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
        OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(nullptr, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// group ep null
TEST_F(L2AlltoAllvGroupedMatMulTest, TestGroupEpNull)
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
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, nullptr, epWorldSize, sendCounts,
                              recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// group ep invalid
TEST_F(L2AlltoAllvGroupedMatMulTest, TestGroupEpInvalid)
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
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr,
                              "test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group_"
                              "test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group_"
                              "test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group_"
                              "test_allto_allv_grouped_mat_mul_ep_group_test_allto_allv_grouped_mat_mul_ep_group",
                              epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmx not_null mmweight null mmy null
TEST_F(L2AlltoAllvGroupedMatMulTest, TestMmXInvalid)
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
        OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, mmX, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2AlltoAllvGroupedMatMulTest, TestPermuteOutFlagInvalid)
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
        OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                  INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_allto_allv_grouped_mat_mul_ep_group",
                        epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                  OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
// sendCountsTensorOptional not nullptr
TEST_F(L2AlltoAllvGroupedMatMulTest, TestSendCountsTensorOptional)
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
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                        INPUT(gmmX, gmmWeight, sendCountsTensor, nullptr, nullptr, nullptr, "test_ep_group",
                              epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// recvCounts empty array
TEST_F(L2AlltoAllvGroupedMatMulTest, TestRecvCountsEmpty)
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
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_ep_group", epWorldSize,
                              sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// sendCounts empty array
TEST_F(L2AlltoAllvGroupedMatMulTest, TestSendCountsEmpty)
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
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, nullptr, "test_ep_group", epWorldSize,
                              sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// second API direct call - cover aclnnAlltoAllvGroupedMatMul function body
TEST_F(L2AlltoAllvGroupedMatMulTest, TestSecondApiDirect)
{
    aclnnStatus aclRet = aclnnAlltoAllvGroupedMatMul(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// mmWeight not null but mmX null - triggers mm consistency check
TEST_F(L2AlltoAllvGroupedMatMulTest, TestMmWeightInvalid)
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
    auto ut = OP_API_UT(aclnnAlltoAllvGroupedMatMul,
                        INPUT(gmmX, gmmWeight, nullptr, nullptr, nullptr, mmWeight, "test_ep_group", epWorldSize,
                              sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag),
                        OUTPUT(gmmY_desc, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
} // namespace AlltoAllvGroupedMatMulUT
