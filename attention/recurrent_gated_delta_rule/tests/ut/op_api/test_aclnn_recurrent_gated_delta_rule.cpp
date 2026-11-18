/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_api/aclnn_recurrent_gated_delta_rule.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class aclnnRecurrentGatedDeltaRule_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // cout << "aclnnRecurrentGatedDeltaRule_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        // cout << "aclnnRecurrentGatedDeltaRule_test TearDown" << endl;
    }
};

class aclnnRecurrentGatedDeltaRule_test_case {
public:
    uint32_t b = 4;
    uint32_t mtp = 2;
    uint32_t t = b * mtp; // 8
    uint32_t nk = 4;
    uint32_t nv = 4;
    uint32_t dk = 128;
    uint32_t dv = 128;

    TensorDesc state;
    TensorDesc query;
    TensorDesc key;
    TensorDesc value;
    TensorDesc beta;
    TensorDesc gama;
    TensorDesc gamaK;
    TensorDesc actSeqLen;
    TensorDesc ssmStaId;
    TensorDesc numAccTok;
    TensorDesc out;
    void RGDRTestCase(int validIdx, int nullIdx)
    {
        state = TensorDesc({t, nv, dv, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        query = TensorDesc({t, nk, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        key = TensorDesc({t, nk, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        value = TensorDesc({t, nv, dv}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        beta = TensorDesc({t, nv}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        gama = TensorDesc({t, nv}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        gamaK = TensorDesc({t, nv, dk}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        actSeqLen = TensorDesc({b}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
        ssmStaId = TensorDesc({t}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
        numAccTok = TensorDesc({b}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
        out = TensorDesc({t, nv, dv}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);

        if (validIdx == 1) {
            state = TensorDesc({t, nv, dv, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 2) {
            query = TensorDesc({t, nk, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 3) {
            key = TensorDesc({t, nk, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 4) {
            value = TensorDesc({t, nv, dv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 5) {
            beta = TensorDesc({t, nv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 6) {
            gama = TensorDesc({t, nv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 7) {
            actSeqLen = TensorDesc({b}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 8) {
            ssmStaId = TensorDesc({t}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 9) {
            numAccTok = TensorDesc({b}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 10) {
            out = TensorDesc({t, nv, dv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 11) {
            query = TensorDesc({1, t, nk, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 12) {
            key = TensorDesc({t + 1, nk, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 13) {
            value = TensorDesc({t, nv, dv + 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 14) {
            beta = TensorDesc({t, nv, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 15) {
            state = TensorDesc({t - 1, nv, dv, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 16) {
            actSeqLen = TensorDesc({b + 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 17) {
            ssmStaId = TensorDesc({t + 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 18) {
            out = TensorDesc({t, nv, dv + 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 19) {
            gama = TensorDesc({t + 1, nv}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        } else if (validIdx == 20) {
            numAccTok = TensorDesc({b + 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
        }
        aclnnStatus aclRet = utTest(nullIdx);
        bool pass = false;
        if ((nullIdx == 0 || nullIdx == 8 || nullIdx == 9 || nullIdx == 10) && validIdx == 0) {
            pass = (ACLNN_SUCCESS == aclRet);
        } else if ((nullIdx >= 1 && nullIdx <= 7) || nullIdx == 11) {
            pass = (ACLNN_ERR_PARAM_NULLPTR == aclRet);
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
            auto ut = OP_API_UT(
                aclnnRecurrentGatedDeltaRule,
                INPUT(query, key, value, beta, state, actSeqLen, ssmStaId, gama, nullptr, numAccTok, 1.0), OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 1) {
            auto ut =
                OP_API_UT(aclnnRecurrentGatedDeltaRule,
                          INPUT(nullptr, key, value, beta, state, actSeqLen, ssmStaId, gama, nullptr, numAccTok, 1.0),
                          OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 2) {
            auto ut =
                OP_API_UT(aclnnRecurrentGatedDeltaRule,
                          INPUT(query, nullptr, value, beta, state, actSeqLen, ssmStaId, gama, nullptr, numAccTok, 1.0),
                          OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 3) {
            auto ut =
                OP_API_UT(aclnnRecurrentGatedDeltaRule,
                          INPUT(query, key, nullptr, beta, state, actSeqLen, ssmStaId, gama, nullptr, numAccTok, 1.0),
                          OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 4) {
            auto ut =
                OP_API_UT(aclnnRecurrentGatedDeltaRule,
                          INPUT(query, key, value, nullptr, state, actSeqLen, ssmStaId, gama, nullptr, numAccTok, 1.0),
                          OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 5) {
            auto ut =
                OP_API_UT(aclnnRecurrentGatedDeltaRule,
                          INPUT(query, key, value, beta, nullptr, actSeqLen, ssmStaId, gama, nullptr, numAccTok, 1.0),
                          OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 6) {
            auto ut = OP_API_UT(aclnnRecurrentGatedDeltaRule,
                                INPUT(query, key, value, beta, state, nullptr, ssmStaId, gama, nullptr, numAccTok, 1.0),
                                OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 7) {
            auto ut = OP_API_UT(
                aclnnRecurrentGatedDeltaRule,
                INPUT(query, key, value, beta, state, actSeqLen, nullptr, gama, nullptr, numAccTok, 1.0), OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 8) {
            auto ut =
                OP_API_UT(aclnnRecurrentGatedDeltaRule,
                          INPUT(query, key, value, beta, state, actSeqLen, ssmStaId, nullptr, nullptr, numAccTok, 1.0),
                          OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 9) {
            auto ut = OP_API_UT(aclnnRecurrentGatedDeltaRule,
                                INPUT(query, key, value, beta, state, actSeqLen, ssmStaId, gama, gamaK, numAccTok, 1.0),
                                OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (nullIdx == 10) {
            auto ut = OP_API_UT(aclnnRecurrentGatedDeltaRule,
                                INPUT(query, key, value, beta, state, actSeqLen, ssmStaId, gama, nullptr, nullptr, 1.0),
                                OUTPUT(out));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else {
            auto ut = OP_API_UT(aclnnRecurrentGatedDeltaRule,
                                INPUT(query, key, value, beta, state, actSeqLen, ssmStaId, gama, nullptr, nullptr, 1.0),
                                OUTPUT(nullptr));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        }
        return aclRet;
    }
};

aclnnRecurrentGatedDeltaRule_test_case rgdr_test;
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case0)
{
    rgdr_test.RGDRTestCase(0, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case1)
{
    rgdr_test.RGDRTestCase(1, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case2)
{
    rgdr_test.RGDRTestCase(2, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case3)
{
    rgdr_test.RGDRTestCase(3, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case4)
{
    rgdr_test.RGDRTestCase(4, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case5)
{
    rgdr_test.RGDRTestCase(5, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case6)
{
    rgdr_test.RGDRTestCase(6, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case7)
{
    rgdr_test.RGDRTestCase(7, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case8)
{
    rgdr_test.RGDRTestCase(8, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case9)
{
    rgdr_test.RGDRTestCase(9, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case10)
{
    rgdr_test.RGDRTestCase(10, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case11)
{
    rgdr_test.RGDRTestCase(0, 1);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case12)
{
    rgdr_test.RGDRTestCase(0, 2);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case13)
{
    rgdr_test.RGDRTestCase(0, 3);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case14)
{
    rgdr_test.RGDRTestCase(0, 4);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case15)
{
    rgdr_test.RGDRTestCase(0, 5);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case16)
{
    rgdr_test.RGDRTestCase(0, 6);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case17)
{
    rgdr_test.RGDRTestCase(0, 7);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case18)
{
    rgdr_test.RGDRTestCase(0, 8);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case19)
{
    rgdr_test.RGDRTestCase(0, 9);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case20)
{
    rgdr_test.RGDRTestCase(0, 10);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case21)
{
    rgdr_test.RGDRTestCase(0, 11);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case22)
{
    rgdr_test.RGDRTestCase(11, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case23)
{
    rgdr_test.RGDRTestCase(12, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case24)
{
    rgdr_test.RGDRTestCase(13, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case25)
{
    rgdr_test.RGDRTestCase(14, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case26)
{
    rgdr_test.RGDRTestCase(15, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case27)
{
    rgdr_test.RGDRTestCase(16, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case28)
{
    rgdr_test.RGDRTestCase(17, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case29)
{
    rgdr_test.RGDRTestCase(18, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case30)
{
    rgdr_test.RGDRTestCase(19, 0);
}
TEST_F(aclnnRecurrentGatedDeltaRule_test, ascend910B2_test_opapi_case31)
{
    rgdr_test.RGDRTestCase(20, 0);
}
