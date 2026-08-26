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
#include <functional>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_chunk_gated_delta_rule.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class aclnnChunkGatedDeltaRule_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // cout << "aclnnChunkGatedDeltaRule_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        // cout << "aclnnChunkGatedDeltaRule_test TearDown" << endl;
    }
};

class aclnnChunkGatedDeltaRule_test_case {
public:
    uint32_t bs = 2;
    uint32_t seqLen = 100;
    uint32_t t = bs * seqLen;
    uint32_t nk = 4;
    uint32_t nv = 8;
    uint32_t dk = 128;
    uint32_t dv = 128;

    TensorDesc initialState;
    TensorDesc query;
    TensorDesc key;
    TensorDesc value;
    TensorDesc beta;
    TensorDesc actualSeqLengths;
    TensorDesc gOptional;
    TensorDesc out;
    TensorDesc finalState;

    void ApplyInvalidDtype(int validIdx, int32_t low, int32_t high)
    {
        const std::array<std::function<void()>, 10> cases = {
            []() {},
            [this, low, high]() {
                initialState = TensorDesc({bs, nv, dv, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(low, high);
            },
            [this, low, high]() { query = TensorDesc({t, nk, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(low, high); },
            [this, low, high]() { key = TensorDesc({t, nk, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(low, high); },
            [this, low, high]() { value = TensorDesc({t, nv, dv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(low, high); },
            [this, low, high]() { beta = TensorDesc({t, nv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(low, high); },
            [this, low, high]() { actualSeqLengths = TensorDesc({bs}, ACL_INT8, ACL_FORMAT_ND).ValueRange(low, high); },
            [this, low, high]() { gOptional = TensorDesc({t, nv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(low, high); },
            [this, low, high]() { out = TensorDesc({t, nv, dv}, ACL_INT8, ACL_FORMAT_ND).ValueRange(low, high); },
            [this, low, high]() {
                finalState = TensorDesc({bs, nv, dv, dk}, ACL_INT8, ACL_FORMAT_ND).ValueRange(low, high);
            },
        };
        if (validIdx >= 0 && validIdx < static_cast<int>(cases.size())) {
            cases[validIdx]();
        }
    }

    void ChunkGatedDeltaRuleTestCase(int validIdx, int nullIdx)
    {
        constexpr int32_t kValueLow = 0;
        constexpr int32_t kValueHigh = 1;
        // actualSeqLengths: [seqLen, 2 * seqLen, ..., bs * seqLen] 的等差序列
        vector<int32_t> actualSeqLens(bs, 0);
        for (uint32_t i = 0; i < bs; ++i) {
            actualSeqLens[i] = static_cast<int32_t>((i + 1) * seqLen);
        }

        initialState = TensorDesc({bs, nv, dv, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        query = TensorDesc({t, nk, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        key = TensorDesc({t, nk, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        value = TensorDesc({t, nv, dv}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        beta = TensorDesc({t, nv}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        actualSeqLengths = TensorDesc({bs}, ACL_INT32, ACL_FORMAT_ND).Value(actualSeqLens);
        gOptional = TensorDesc({t, nv}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        out = TensorDesc({t, nv, dv}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        finalState = TensorDesc({bs, nv, dv, dk}, ACL_BF16, ACL_FORMAT_ND).ValueRange(kValueLow, kValueHigh);
        ApplyInvalidDtype(validIdx, kValueLow, kValueHigh);
        aclnnStatus aclRet = utTest(nullIdx);
        // 仅 gOptional 允许为空（nullIdx == 7）时仍视为合法
        aclnnStatus expected =
            ((nullIdx == 0 || nullIdx == 7) && validIdx == 0) ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID;
        EXPECT_EQ(aclRet, expected);
    }

    aclnnStatus RunNullCase(int idx, uint64_t &ws)
    {
        constexpr float scale = 1.0f;
        switch (idx) {
            case 0:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(query, key, value, beta, initialState, actualSeqLengths, gOptional, scale),
                                 OUTPUT(out, finalState))
                    .TestGetWorkspaceSize(&ws);
            case 1:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(nullptr, key, value, beta, initialState, actualSeqLengths, gOptional, scale),
                                 OUTPUT(out, finalState))
                    .TestGetWorkspaceSize(&ws);
            case 2:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(query, nullptr, value, beta, initialState, actualSeqLengths, gOptional, scale),
                                 OUTPUT(out, finalState))
                    .TestGetWorkspaceSize(&ws);
            case 3:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(query, key, nullptr, beta, initialState, actualSeqLengths, gOptional, scale),
                                 OUTPUT(out, finalState))
                    .TestGetWorkspaceSize(&ws);
            case 4:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(query, key, value, nullptr, initialState, actualSeqLengths, gOptional, scale),
                                 OUTPUT(out, finalState))
                    .TestGetWorkspaceSize(&ws);
            case 5:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(query, key, value, beta, nullptr, actualSeqLengths, gOptional, scale),
                                 OUTPUT(out, finalState))
                    .TestGetWorkspaceSize(&ws);
            case 6:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(query, key, value, beta, initialState, nullptr, gOptional, scale),
                                 OUTPUT(out, finalState))
                    .TestGetWorkspaceSize(&ws);
            case 7:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(query, key, value, beta, initialState, actualSeqLengths, nullptr, scale),
                                 OUTPUT(out, finalState))
                    .TestGetWorkspaceSize(&ws);
            case 8:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(query, key, value, beta, initialState, actualSeqLengths, gOptional, scale),
                                 OUTPUT(nullptr, finalState))
                    .TestGetWorkspaceSize(&ws);
            case 9:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(query, key, value, beta, initialState, actualSeqLengths, gOptional, scale),
                                 OUTPUT(out, nullptr))
                    .TestGetWorkspaceSize(&ws);
            default:
                return OP_API_UT(aclnnChunkGatedDeltaRule,
                                 INPUT(query, key, value, beta, initialState, actualSeqLengths, gOptional, scale),
                                 OUTPUT(nullptr, nullptr))
                    .TestGetWorkspaceSize(&ws);
        }
    }

    aclnnStatus utTest(int nullIdx)
    {
        uint64_t ws = 0;
        return RunNullCase((nullIdx >= 0 && nullIdx <= 10) ? nullIdx : 10, ws);
    }
};

aclnnChunkGatedDeltaRule_test_case cgdr_test;
// case 0-9: validIdx=0~9, nullIdx=0（覆盖各输入/输出 dtype 非法）
// case 10-19: validIdx=0，nullIdx=1~10（覆盖各输入/输出为 nullptr）
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case0)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case1)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(1, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case2)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(2, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case3)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(3, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case4)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(4, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case5)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(5, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case6)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(6, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case7)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(7, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case8)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(8, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case9)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(9, 0);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case10)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 1);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case11)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 2);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case12)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 3);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case13)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 4);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case14)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 5);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case15)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 6);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case16)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 7);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case17)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 8);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case18)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 9);
}
TEST_F(aclnnChunkGatedDeltaRule_test, ascend910B2_test_opapi_case19)
{
    cgdr_test.ChunkGatedDeltaRuleTestCase(0, 10);
}
