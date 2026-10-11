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
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MhcPreProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreInferShape TearDown" << std::endl;
    }
};

/*
 * 测试用例描述：TND场景正常Shape推导 - B1 S1024 n4 D5120
 * 输入：正常shape
 * 预期结果：成功
 */
TEST_F(MhcPreProto, Ut_Check_mHCPreProto_TND_B1_S1024_n4_D5120)
{
    uint32_t T = 1024;
    uint32_t n = 4;
    uint32_t D = 5120;
    uint32_t nD = n * D;              // 20480
    uint32_t phiRows = n * n + 2 * n; // 24

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPre",
        {
            {{{T, n, D}, {T, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // x
            {{{phiRows, nD}, {phiRows, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // alpha
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},         // bias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                    // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // hin
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_post
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_res
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // inv_rms (optional)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_mix (optional)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // h_pre (optional)
        },
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {T, D},       // hin: [T, D]
        {T, n},       // h_post: [T, n]
        {T, n, n},    // h_res: [T, n, n]
        {T},          // inv_rms: [T]
        {T, phiRows}, // h_mix: [T, phiRows]
        {T, n}        // h_pre: [T, n]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 测试用例描述：TND场景正常Shape推导 - B1 S2048 n4 D2560
 * 输入：正常shape
 * 预期结果：成功
 */
TEST_F(MhcPreProto, Ut_Check_mHCPreProto_TND_B1_S2048_n4_D2560)
{
    uint32_t T = 2048;
    uint32_t n = 4;
    uint32_t D = 2560;
    uint32_t nD = n * D;              // 10240
    uint32_t phiRows = n * n + 2 * n; // 24

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPre",
        {
            {{{T, n, D}, {T, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // x
            {{{phiRows, nD}, {phiRows, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // alpha
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},         // bias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                    // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // hin
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_post
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_res
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // inv_rms (optional)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_mix (optional)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // h_pre (optional)
        },
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {T, D},       // hin: [T, D]
        {T, n},       // h_post: [T, n]
        {T, n, n},    // h_res: [T, n, n]
        {T},          // inv_rms: [T]
        {T, phiRows}, // h_mix: [T, phiRows]
        {T, n}        // h_pre: [T, n]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 测试用例描述：TND场景正常Shape推导 - B1 S4096 n4 D2560
 * 输入：正常shape
 * 预期结果：成功
 */
TEST_F(MhcPreProto, Ut_Check_mHCPreProto_TND_B1_S4096_n4_D2560)
{
    uint32_t T = 4096;
    uint32_t n = 4;
    uint32_t D = 2560;
    uint32_t nD = n * D;              // 10240
    uint32_t phiRows = n * n + 2 * n; // 24

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPre",
        {
            {{{T, n, D}, {T, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // x
            {{{phiRows, nD}, {phiRows, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // alpha
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},         // bias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                    // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // hin
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_post
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_res
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // inv_rms (optional)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_mix (optional)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // h_pre (optional)
        },
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {T, D},       // hin: [T, D]
        {T, n},       // h_post: [T, n]
        {T, n, n},    // h_res: [T, n, n]
        {T},          // inv_rms: [T]
        {T, phiRows}, // h_mix: [T, phiRows]
        {T, n}        // h_pre: [T, n]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 测试用例描述：BSND场景正常Shape推导
 * 输入：BSND格式 [1, 1024, 4, 5120]
 * 预期结果：成功
 */
TEST_F(MhcPreProto, Ut_Check_mHCPreProto_BSND_Normal)
{
    uint32_t B = 1;
    uint32_t S = 1024;
    uint32_t n = 4;
    uint32_t D = 5120;
    uint32_t nD = n * D;              // 20480
    uint32_t phiRows = n * n + 2 * n; // 24

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPre",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
            {{{phiRows, nD}, {phiRows, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // alpha
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},         // bias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                    // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // hin
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_post
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_res
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // inv_rms (optional)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_mix (optional)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // h_pre (optional)
        },
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, D},       // hin: [B, S, D]
        {B, S, n},       // h_post: [B, S, n]
        {B, S, n, n},    // h_res: [B, S, n, n]
        {B, S},          // inv_rms: [B, S]
        {B, S, phiRows}, // h_mix: [B, S, phiRows]
        {B, S, n}        // h_pre: [B, S, n]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

/*
 * 测试用例描述：BSND场景正常Shape推导 - 不同batch size
 * 输入：BSND格式 [2, 512, 4, 2560]
 * 预期结果：成功
 */
TEST_F(MhcPreProto, Ut_Check_mHCPreProto_BSND_DifferentBatch)
{
    uint32_t B = 2;
    uint32_t S = 512;
    uint32_t n = 4;
    uint32_t D = 2560;
    uint32_t nD = n * D;              // 10240
    uint32_t phiRows = n * n + 2 * n; // 24

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPre",
        {
            {{{B, S, n, D}, {B, S, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x
            {{{phiRows, nD}, {phiRows, nD}}, ge::DT_FLOAT, ge::FORMAT_ND}, // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // alpha
            {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},         // bias
            {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}                    // gamma (optional)
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // hin
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_post
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_res
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // inv_rms (optional)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // h_mix (optional)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}    // h_pre (optional)
        },
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {B, S, D},       // hin: [B, S, D]
        {B, S, n},       // h_post: [B, S, n]
        {B, S, n, n},    // h_res: [B, S, n, n]
        {B, S},          // inv_rms: [B, S]
        {B, S, phiRows}, // h_mix: [B, S, phiRows]
        {B, S, n}        // h_pre: [B, S, n]
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MhcPreProto, Ut_Check_mHCPreProto_InvalidAlphaShape)
{
    constexpr uint32_t T = 1;
    constexpr uint32_t n = 4;
    constexpr uint32_t D = 2560;
    constexpr uint32_t nD = n * D;
    constexpr uint32_t phiRows = n * n + 2 * n;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPre",
        {{{{T, n, D}, {T, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phiRows, nD}, {phiRows, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
TEST_F(MhcPreProto, Ut_Check_mHCPreProto_InvalidAlphaRank)
{
    constexpr uint32_t T = 1;
    constexpr uint32_t n = 4;
    constexpr uint32_t D = 2560;
    constexpr uint32_t nD = n * D;
    constexpr uint32_t phiRows = n * n + 2 * n;

    gert::InfershapeContextPara infershapeContextPara(
        "MhcPre",
        {{{{T, n, D}, {T, n, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phiRows, nD}, {phiRows, nD}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3, 1}, {3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phiRows}, {phiRows}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{nD}, {nD}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
