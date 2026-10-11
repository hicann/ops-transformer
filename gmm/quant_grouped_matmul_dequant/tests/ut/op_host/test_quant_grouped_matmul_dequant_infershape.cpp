/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "infer_shape_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;

namespace {

constexpr int64_t kM = 256;
constexpr int64_t kK = 256;
constexpr int64_t kN = 256;
constexpr int64_t kG = 16;

class QuantGroupedMatmulDequantInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "QuantGroupedMatmulDequantInfershapeTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "QuantGroupedMatmulDequantInfershapeTest TearDown" << endl;
    }
};

TEST_F(QuantGroupedMatmulDequantInfershapeTest, test_float16_pertoken_all_inputs)
{
    gert::InfershapeContextPara para(
        "QuantGroupedMatmulDequant",
        {{{{kM, kK}, {kM, kK}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{kG, kK / 32, kN / 16, 16, 32}, {kG, kK / 32, kN / 16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
         {{{kG, kN}, {kG, kN}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{kG}, {kG}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{kN}, {kN}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{kM}, {kM}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{kM}, {kM}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{kK}, {kK}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<string>("pertoken")},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{kM, kN}});
}

TEST_F(QuantGroupedMatmulDequantInfershapeTest, test_float16_pertensor_minimal)
{
    constexpr int64_t M = 128;
    constexpr int64_t K = 1344;
    constexpr int64_t N = 512;
    constexpr int64_t G = 8;
    gert::InfershapeContextPara para(
        "QuantGroupedMatmulDequant",
        {{{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{G, K / 32, N / 16, 16, 32}, {G, K / 32, N / 16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
         {{{G, N}, {G, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{G}, {G}}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{K}, {K}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<string>("pertensor")},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{M, N}});
}

} // namespace
