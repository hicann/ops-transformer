/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_quant_grouped_mat_mul_allto_allv_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class test_aclnn_quant_grouped_mat_mul_allto_all_v2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "test_aclnn_quant_grouped_mat_mul_allto_all_v2 SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "test_aclnn_quant_grouped_mat_mul_allto_all_v2 TearDown" << endl;
    }
};

// ============================================================================
// Group 1: QuantMode/Scale 一致性 (CheckQuantMode)
// ============================================================================

// gmmXQuantMode=0 但 gmmXScale 非空 → PARAM_INVALID
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_gmmX_quantmode0_with_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, nullptr, // gmmXScale 非空, gmmWeightScale 空
                              nullptr, nullptr,                       // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,      // mm scale
                              nullptr,                                // commQuantScale
                              0, 0, 0, 0, 0,                          // gmmXQM=0, 其余=0
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmXQuantMode=1 但 gmmXScale=nullptr → PARAM_INVALID
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_gmmX_quantmode1_without_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // gmmXScale=nullptr
                              nullptr, nullptr,                    // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,   // mm scale
                              nullptr,                             // commQuantScale
                              1, 0, 0, 0, 0,                       // gmmXQM=1, 其余=0
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmWeightQuantMode=0 但 gmmWeightScale 非空 → PARAM_INVALID
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_gmmWeight_quantmode0_with_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, gmmWeightScale_, // gmmWeightScale 非空
                              nullptr, nullptr,                            // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,           // mm scale
                              nullptr,                                     // commQuantScale
                              0, 0, 0, 0, 0,                               // gmmWQM=0
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gmmWeightQuantMode=1 但 gmmWeightScale=nullptr → PARAM_INVALID
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_gmmWeight_quantmode1_without_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // gmmWeightScale=nullptr
                              nullptr, nullptr,                    // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,   // mm scale
                              nullptr,                             // commQuantScale
                              0, 1, 0, 0, 0,                       // gmmWQM=1
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmXQuantMode=0 但 mmXScale 非空（mm组全非空） → PARAM_INVALID
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mmX_quantmode0_with_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // gmm scale
                              nullptr, nullptr,                    // counts tensor
                              mmX_, mmWeight_, mmXScale_, nullptr, // mmXScale 非空
                              nullptr,                             // commQuantScale
                              0, 0, 0, 0, 0,                       // mmXQM=0
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmXQuantMode=1 但 mmXScale=nullptr（mm组全非空） → PARAM_INVALID
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mmX_quantmode1_without_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // gmm scale
                              nullptr, nullptr,                    // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,   // mmXScale=nullptr
                              nullptr,                             // commQuantScale
                              0, 0, 1, 0, 0,                       // mmXQM=1
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmWeightQuantMode=0 但 mmWeightScale 非空 → PARAM_INVALID
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mmWeight_quantmode0_with_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr,      // gmm scale
                              nullptr, nullptr,                         // counts tensor
                              mmX_, mmWeight_, nullptr, mmWeightScale_, // mmWeightScale 非空
                              nullptr,                                  // commQuantScale
                              0, 0, 0, 0, 0,                            // mmWQM=0
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mmWeightQuantMode=1 但 mmWeightScale=nullptr → PARAM_INVALID
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mmWeight_quantmode1_without_scale)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // gmm scale
                              nullptr, nullptr,                    // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,   // mmWeightScale=nullptr
                              nullptr,                             // commQuantScale
                              0, 0, 0, 1, 0,                       // mmWQM=1
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Group 3: CountsTensor 不支持 (CheckNullStatus)
// ============================================================================

// sendCountsTensorOptional 非空 → PARAM_NULLPTR
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_sendCountsTensor_not_null)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sendCountsTensor_({32}, ACL_INT64, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // scale
                              sendCountsTensor_, nullptr,          // sendCountsTensor 非空
                              mmX_, mmWeight_, nullptr, nullptr,   // mm scale
                              nullptr,                             // commQuantScale
                              0, 0, 0, 0, 0, -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
                              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// recvCountsTensorOptional 非空 → PARAM_NULLPTR
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_recvCountsTensor_not_null)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc recvCountsTensor_({32}, ACL_INT64, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // scale
                              nullptr, recvCountsTensor_,          // recvCountsTensor 非空
                              mmX_, mmWeight_, nullptr, nullptr,   // mm scale
                              nullptr,                             // commQuantScale
                              0, 0, 0, 0, 0, -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
                              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Group 4: SendCounts/RecvCounts 边界 (CheckSendAndRecv)
// ============================================================================

// sendCounts size=0 → PARAM_INVALID
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_sendCounts_empty_array)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    aclIntArray *sendCounts = aclCreateIntArray(nullptr, 0);
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // scale
                              nullptr, nullptr,                    // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,   // mm scale
                              nullptr,                             // commQuantScale
                              0, 0, 0, 0, 0, -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
                              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// recvCounts size=0 → PARAM_INVALID
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_recvCounts_empty_array)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(nullptr, 0);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // scale
                              nullptr, nullptr,                    // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,   // mm scale
                              nullptr,                             // commQuantScale
                              0, 0, 0, 0, 0, -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
                              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// ============================================================================
// Group 5: Group nullptr
// ============================================================================

// group=nullptr → PARAM_NULLPTR
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_group_nullptr)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // scale
                              nullptr, nullptr,                    // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,   // mm scale
                              nullptr,                             // commQuantScale
                              0, 0, 0, 0, 0, -1, 0,
                              (const char *)nullptr, // group=nullptr
                              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ============================================================================
// Group 6: 功能正确性
// ============================================================================

// commQuantMode=1 → PARAM_INVALID（API 层新增校验，仅支持 0）
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_commQuantMode_nonzero)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, // scale
                              nullptr, nullptr,                    // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,   // mm scale
                              nullptr,                             // commQuantScale
                              0, 0, 0, 0, 1,                       // commQuantMode=1
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TT 量化异常：gmmXQM=1, gmmWQM=1, mmXQM=0, mmWQM=0
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_tt_quant_normal)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, // TT: 双 scale 非空
                              nullptr, nullptr,                               // counts tensor
                              mmX_, mmWeight_, nullptr, nullptr,              // mm scale
                              nullptr,                                        // commQuantScale
                              1, 1, 0, 0, 0,                                  // gmmXQM=1, gmmWQM=1
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// commQuantMode=1 → PARAM_INVALID（新增校验专用用例）
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_commQuantMode_reject)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, nullptr, nullptr, mmX_, mmWeight_, nullptr, nullptr,
                              nullptr, 0, 0, 0, 0, 1, // commQuantMode=1
                              -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ============================================================================
// Group: MX quant UT
// ============================================================================

// 异常场景:mmX_, mmWeight_非空时，mmXScale为空
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_gmmx_dtype_invalid)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 4096, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc mmX_({4096, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mmWeight_({4096, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc mmY_({4096, 4096}, ACL_BF16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_, nullptr, nullptr,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, true, true),
        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// MX异常：gmmX为空
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_gmmX_nullptr)
{
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc mmX_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(nullptr, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_,
                              mmXScale_, mmWeightScale_, nullptr, 6, 6, 0, 0, 0, -1, groupSize,
                              (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// MX异常：gmmWeight为空
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_gmmWeight_nullptr)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc mmX_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, nullptr, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_, mmXScale_, mmWeightScale_,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// MX异常：y为空
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_y_nullptr)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc mmX_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_,
                              mmXScale_, mmWeightScale_, nullptr, 6, 6, 0, 0, 0, -1, groupSize,
                              (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(nullptr, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// MX异常：gmmXScale为空
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_gmmXScale_null)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc mmX_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, nullptr, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_, mmXScale_, mmWeightScale_,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// MX异常：gmmWeightScale为空
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_gmmWeightScale_null)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 4096, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc mmX_({}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_BF16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, nullptr, nullptr, nullptr, mmX_, mmWeight_, mmXScale_, mmWeightScale_,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, true, false),
        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// MX异常：mm存在但量化模式为(6,6,0,0)，非法
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_mm_mode_noquant_invalid)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc mmX_({4096, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_,
                              mmXScale_, mmWeightScale_, nullptr, 6, 6, 0, 0, 0, -1, groupSize,
                              (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// MX异常：mm存在但量化模式与gmm不一致，例如(6,6,1,1)
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_mm_mode_mismatch_invalid)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 4096, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc mmX_({4096, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mmWeight_({4096, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);

    TensorDesc y_({8192, 4096}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc mmY_({4096, 4096}, ACL_BF16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_,
                              mmXScale_, mmWeightScale_, nullptr, 6, 6, 1, 1, 0, -1, groupSize,
                              (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, true, true),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// MX异常：mm存在但mmXScale为空
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_mmXScale_null)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 4096, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc mmX_({4096, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mmWeight_({4096, 7168}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc y_({8192, 4096}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc mmY_({4096, 4096}, ACL_BF16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_,
                              nullptr, mmWeightScale_, nullptr, 6, 6, 6, 6, 0, -1, groupSize,
                              (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, true, true),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// MX异常：mm存在但mmWeightScale为空
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_mmWeightScale_null)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc mmX_({4096, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_, mmXScale_, nullptr,
              nullptr, 6, 6, 6, 6, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

namespace {
constexpr size_t kHcclGroupNameMax = 128U;
}

// MX + 无 mm：sendCounts=nullptr → CheckIntArrayNotEmpty
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_sendCounts_nullptr_valid_mx)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = nullptr;
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_recvCounts_nullptr_valid_mx)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = nullptr;
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mm 三者未同时为空或非空 → CheckMmOptionalConsistency（CheckNullStatus 失败码为 PARAM_NULLPTR）
TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mm_optional_partial_mmY_nullptr)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmX_({4096, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_,
                              mmXScale_, mmWeightScale_, nullptr, 6, 6, 6, 6, 0, -1, groupSize,
                              (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_quant_mode_perchannel_unsupported)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, nullptr, nullptr, mmX_, mmWeight_, nullptr, nullptr,
                              nullptr, 2, 2, 0, 0, 0, -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
                              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_quant_mode_unknown_enum)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    constexpr int64_t kBadQuantMode = 99;
    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, nullptr, nullptr, mmX_, mmWeight_, nullptr, nullptr, nullptr,
              kBadQuantMode, kBadQuantMode, 0, 0, 0, -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_group_empty_string)
{
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmX_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, nullptr, nullptr, nullptr, nullptr, mmX_, mmWeight_, nullptr, nullptr, nullptr, 0, 0,
              0, 0, 0, -1, 0, (const char *)"", (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_group_name_length_ge_max)
{
    static const std::string longGroup(kHcclGroupNameMax, 'a');

    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, 6, 6, 0, 0, 0, -1, groupSize, longGroup.c_str(),
                              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_gmm_weight_scale_rank_invalid)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_gmm_weight_scale_last_dim_invalid)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_mm_weight_scale_last_dim_invalid)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmX_({4096, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({112, 4096, 4}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_,
                              mmXScale_, mmWeightScale_, nullptr, 6, 6, 6, 6, 0, -1, groupSize,
                              (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_sendCounts_empty_array_after_mx_ok)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(nullptr, 0);
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_recvCounts_empty_array_after_mx_ok)
{
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(nullptr, 0);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_gmm_weight_transpose_conflict_trans_true)
{
    std::vector<int64_t> gmmWs = {7168LL * 4096LL, 1LL, 7168LL};
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, gmmWs);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, true, false),
        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_gmm_weight_transpose_auto_swap_reaches_inner)
{
    std::vector<int64_t> gmmWs = {7168LL * 4096LL, 1LL, 7168LL};
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, gmmWs);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 6, 6, 0, 0, 0, -1, groupSize, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_mm_weight_transpose_conflict_trans_true)
{
    std::vector<int64_t> gmmWs = {7168LL * 4096LL, 1LL, 7168LL};
    std::vector<int64_t> mmWs = {1LL, 7168LL};
    std::vector<int64_t> mmSs = {8192LL, 1LL, 4096LL};
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, gmmWs);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmX_({4096, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, mmWs);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, mmSs);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_,
                              mmXScale_, mmWeightScale_, nullptr, 6, 6, 6, 6, 0, -1, groupSize,
                              (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, true),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_mx_mm_weight_transpose_auto_swap_reaches_inner)
{
    std::vector<int64_t> gmmWs = {7168LL * 4096LL, 1LL, 7168LL};
    std::vector<int64_t> mmWs = {1LL, 7168LL};
    std::vector<int64_t> mmSs = {8192LL, 1LL, 4096LL};
    TensorDesc gmmX_({8192, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, gmmWs);
    TensorDesc gmmXScale_({8192, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({4, 112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmX_({4096, 7168}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, mmWs);
    TensorDesc mmXScale_({4096, 112, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({112, 4096, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, mmSs);
    TensorDesc y_({8192, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 2;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t groupSize = 32;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(aclnnQuantGroupedMatMulAlltoAllvV2,
                        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_,
                              mmXScale_, mmWeightScale_, nullptr, 6, 6, 6, 6, 0, -1, groupSize,
                              (const char *)"test_grouped_mat_mul_allto_allv_ep_group", (const char *)"ccu",
                              epWorldSize, sendCounts, recvCounts, false, false),
                        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_tt_gmm_weight_transpose_conflict_trans_true)
{
    std::vector<int64_t> gmmWs = {7168LL * 4096LL, 1LL, 7168LL};
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND, gmmWs);
    TensorDesc gmmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 1, 1, 0, 0, 0, -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, true, false),
        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_tt_gmm_weight_transpose_auto_swap_reaches_inner)
{
    std::vector<int64_t> gmmWs = {7168LL * 4096LL, 1LL, 7168LL};
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND, gmmWs);
    TensorDesc gmmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, 1, 1, 0, 0, 0, -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_tt_mm_weight_transpose_conflict_trans_true)
{
    std::vector<int64_t> gmmWs = {7168LL * 4096LL, 1LL, 7168LL};
    std::vector<int64_t> mmWs = {1LL, 7168LL};
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND, gmmWs);
    TensorDesc gmmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_HIFLOAT8, ACL_FORMAT_ND, mmWs);
    TensorDesc mmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_, mmXScale_,
              mmWeightScale_, nullptr, 1, 1, 1, 1, 0, -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, true),
        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(test_aclnn_quant_grouped_mat_mul_allto_all_v2, test_tt_mm_weight_transpose_auto_swap_reaches_inner)
{
    std::vector<int64_t> gmmWs = {7168LL * 4096LL, 1LL, 7168LL};
    std::vector<int64_t> mmWs = {1LL, 7168LL};
    TensorDesc gmmX_({4096, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc gmmWeight_({4, 7168, 4096}, ACL_HIFLOAT8, ACL_FORMAT_ND, gmmWs);
    TensorDesc gmmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gmmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc mmX_({1024, 7168}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc mmWeight_({7168, 1024}, ACL_HIFLOAT8, ACL_FORMAT_ND, mmWs);
    TensorDesc mmXScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc mmWeightScale_({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_({4096, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc mmY_({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);

    constexpr int64_t epWorldSize = 8;
    constexpr int64_t e = 4;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    std::vector<int64_t> sendCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    std::vector<int64_t> recvCountsList(epWorldSize * e, BS * K / (epWorldSize * e));
    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    auto ut = OP_API_UT(
        aclnnQuantGroupedMatMulAlltoAllvV2,
        INPUT(gmmX_, gmmWeight_, gmmXScale_, gmmWeightScale_, nullptr, nullptr, mmX_, mmWeight_, mmXScale_,
              mmWeightScale_, nullptr, 1, 1, 1, 1, 0, -1, 0, (const char *)"test_grouped_mat_mul_allto_allv_ep_group",
              (const char *)"ccu", epWorldSize, sendCounts, recvCounts, false, false),
        OUTPUT(y_, mmY_));
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}
