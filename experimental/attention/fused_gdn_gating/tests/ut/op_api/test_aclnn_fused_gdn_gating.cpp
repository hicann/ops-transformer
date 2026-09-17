/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_fused_gdn_gating.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class aclnnFusedGdnGating_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "aclnnFusedGdnGating_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "aclnnFusedGdnGating_test TearDown" << endl;
    }
};

class aclnnFusedGdnGating_test_case {
public:
    uint32_t batch = 4;
    uint32_t numHeads = 8;

    TensorDesc aLog;
    TensorDesc a;
    TensorDesc b;
    TensorDesc dtBias;
    TensorDesc g;
    TensorDesc betaOutput;

    void FggTestCase(int validIdx, int nullIdx)
    {
        aLog = TensorDesc({numHeads}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        a = TensorDesc({batch, numHeads}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        b = TensorDesc({batch, numHeads}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        dtBias = TensorDesc({numHeads}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        g = TensorDesc({1, batch, numHeads}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        betaOutput = TensorDesc({1, batch, numHeads}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);

        if (validIdx == 1) {
            aLog = TensorDesc({numHeads}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 2) {
            a = TensorDesc({batch, numHeads}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 3) {
            b = TensorDesc({batch, numHeads}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 4) {
            dtBias = TensorDesc({numHeads}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 5) {
            g = TensorDesc({1, batch, numHeads}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 6) {
            betaOutput = TensorDesc({1, batch, numHeads}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        }
        aclnnStatus aclRet = utTest(nullIdx);
        bool pass = false;
        if (nullIdx == 0 && validIdx == 0) {
            pass = (ACLNN_SUCCESS == aclRet);
        } else {
            pass = (ACLNN_ERR_PARAM_INVALID == aclRet);
        }
        EXPECT_EQ(pass, true);
    }

    aclnnStatus utTest(int nullIdx)
    {
        uint64_t workspace_size = 0;
        aclnnStatus aclRet;
        if (nullIdx == 0) {
            auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, dtBias, 1.0f, 20.0f), OUTPUT(g, betaOutput));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 1) {
            auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(nullptr, a, b, dtBias, 1.0f, 20.0f), OUTPUT(g, betaOutput));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 2) {
            auto ut =
                OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, nullptr, b, dtBias, 1.0f, 20.0f), OUTPUT(g, betaOutput));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 3) {
            auto ut =
                OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, nullptr, dtBias, 1.0f, 20.0f), OUTPUT(g, betaOutput));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 4) {
            auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, nullptr, 1.0f, 20.0f), OUTPUT(g, betaOutput));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 5) {
            auto ut =
                OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, dtBias, 1.0f, 20.0f), OUTPUT(nullptr, betaOutput));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 6) {
            auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, dtBias, 1.0f, 20.0f), OUTPUT(g, nullptr));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else {
            auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, dtBias, 1.0f, 20.0f), OUTPUT(nullptr, nullptr));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        }
        return aclRet;
    }
};

aclnnFusedGdnGating_test_case fgg_test;
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case0)
{
    fgg_test.FggTestCase(0, 0);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case1)
{
    fgg_test.FggTestCase(1, 0);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case2)
{
    fgg_test.FggTestCase(2, 0);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case3)
{
    fgg_test.FggTestCase(3, 0);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case4)
{
    fgg_test.FggTestCase(4, 0);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case5)
{
    fgg_test.FggTestCase(5, 0);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case6)
{
    fgg_test.FggTestCase(6, 0);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case7)
{
    fgg_test.FggTestCase(0, 1);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case8)
{
    fgg_test.FggTestCase(0, 2);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case9)
{
    fgg_test.FggTestCase(0, 3);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case10)
{
    fgg_test.FggTestCase(0, 4);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case11)
{
    fgg_test.FggTestCase(0, 5);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case12)
{
    fgg_test.FggTestCase(0, 6);
}
TEST_F(aclnnFusedGdnGating_test, ascend910B2_test_opapi_case13)
{
    fgg_test.FggTestCase(0, 7);
}

class aclnnFusedGdnGating_310P_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND310P);
        cout << "aclnnFusedGdnGating_310P_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "aclnnFusedGdnGating_310P_test TearDown" << endl;
    }
};

TEST_F(aclnnFusedGdnGating_310P_test, ascend310P_test_opapi_case0)
{
    uint32_t batch = 4;
    uint32_t numHeads = 8;

    auto aLog = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto a = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto b = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto dtBias = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto g = TensorDesc({1, batch, numHeads}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto betaOutput = TensorDesc({1, batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, dtBias, 1.0f, 20.0f), OUTPUT(g, betaOutput));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(aclnnFusedGdnGating_310P_test, ascend310P_test_opapi_case1)
{
    uint32_t batch = 1;
    uint32_t numHeads = 16;

    auto aLog = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto a = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto b = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto dtBias = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto g = TensorDesc({1, batch, numHeads}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto betaOutput = TensorDesc({1, batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, dtBias, 1.0f, 20.0f), OUTPUT(g, betaOutput));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(aclnnFusedGdnGating_310P_test, ascend310P_test_opapi_case2)
{
    uint32_t batch = 16;
    uint32_t numHeads = 32;

    auto aLog = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto a = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto b = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto dtBias = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto g = TensorDesc({1, batch, numHeads}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto betaOutput = TensorDesc({1, batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, dtBias, 1.0f, 20.0f), OUTPUT(g, betaOutput));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(aclnnFusedGdnGating_310P_test, ascend310P_test_opapi_case3)
{
    uint32_t batch = 4;
    uint32_t numHeads = 8;

    auto aLog = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto a = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto b = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto dtBias = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto g = TensorDesc({1, batch, numHeads}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto betaOutput = TensorDesc({1, batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, dtBias, 0.5f, 10.0f), OUTPUT(g, betaOutput));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(aclnnFusedGdnGating_310P_test, ascend310P_test_opapi_case4)
{
    uint32_t batch = 4;
    uint32_t numHeads = 8;

    auto aLog = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto a = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto b = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto dtBias = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto g = TensorDesc({1, batch, numHeads}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto betaOutput = TensorDesc({1, batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(nullptr, a, b, dtBias, 1.0f, 20.0f), OUTPUT(g, betaOutput));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnFusedGdnGating_310P_test, ascend310P_test_opapi_case5)
{
    uint32_t batch = 4;
    uint32_t numHeads = 8;

    auto aLog = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto a = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto b = TensorDesc({batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto dtBias = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto g = TensorDesc({1, batch, numHeads}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto betaOutput = TensorDesc({1, batch, numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, dtBias, 1.0f, 20.0f), OUTPUT(nullptr, betaOutput));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnFusedGdnGating_310P_test, ascend310P_test_opapi_case6)
{
    uint32_t batch = 4;
    uint32_t numHeads = 8;

    auto aLog = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto a = TensorDesc({batch, numHeads}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto b = TensorDesc({batch, numHeads}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto dtBias = TensorDesc({numHeads}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto g = TensorDesc({1, batch, numHeads}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto betaOutput = TensorDesc({1, batch, numHeads}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnFusedGdnGating, INPUT(aLog, a, b, dtBias, 1.0f, 20.0f), OUTPUT(g, betaOutput));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
