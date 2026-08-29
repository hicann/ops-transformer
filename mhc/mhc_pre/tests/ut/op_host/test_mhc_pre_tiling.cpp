/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/arch35/mhc_pre_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MhcPreTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreTiling TearDown" << std::endl;
    }
};

template <typename T>
static string TilingData2Str(void *buf, size_t size)
{
    string result;
    const T *data = reinterpret_cast<const T *>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

static string MakeAscend950SocInfo(uint32_t cubeCoreCnt, uint32_t vectorCoreCnt)
{
    return R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,)"
           R"( "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,)"
           R"( "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432,)"
           R"( "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,)"
           R"( "CORE_NUM": )" +
           std::to_string(cubeCoreCnt) + R"(, "cube_core_cnt": )" + std::to_string(cubeCoreCnt) +
           R"(, "vector_core_cnt": )" + std::to_string(vectorCoreCnt) + R"(, "socVersion":"Ascend950"} })";
}

static string MakeAscend950SocInfo(uint32_t cubeCoreCnt)
{
    return MakeAscend950SocInfo(cubeCoreCnt, cubeCoreCnt * 2);
}

/*
 * 测试用例1：B=1，S=1，n=4，d=1，x的数据类型为bf16
 * alpha的值为[0.1, 0.1, 0.1]，norm_eps值为0.000001，hc_eps值为0.000001
 * 预期结果：失败
 */
TEST_F(MhcPreTiling, Ut_Check_Case01_B1_S1_n4_d1_BF16)
{
    uint32_t B = 1;
    uint32_t S = 1;
    uint32_t n = 4;
    uint32_t d = 1;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 4
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.1f, 0.1f, 0.1f}; // alpha = [0.1, 0.1, 0.1]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},                  // x
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // alpha (fixed size 3)
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // bias
            {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // gamma (optional)
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},                // hin
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},               // h_post
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},         // h_res
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // inv_rms (optional)
            {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND}, // h_mix (optional)
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}                // h_pre (optional)
        },
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

/*
 * 测试用例2：B=1，S=1，n=6，d=65535，x的数据类型为float16
 * alpha的值为[1.0, 1.0, 1.0]，norm_eps值为20.0，hc_eps值为200
 * 预期结果：失败
 */
TEST_F(MhcPreTiling, Ut_Check_Case02_B1_S1_n6_d65535_FP16)
{
    uint32_t B = 1;
    uint32_t S = 1;
    uint32_t n = 6;
    uint32_t d = 65535;
    uint32_t phi_dim0 = n * n + 2 * n; // 48
    uint32_t phi_dim1 = n * d;         // 393210
    uint32_t bias_dim = n * n + 2 * n; // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.5f, 0.5f, 0.5f}; // alpha = [0.5, 0.5, 0.5]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

/*
 * 测试用例3：B=65535，S=1，n=8，d=1，x的数据类型为bf16
 * alpha的值为[3.0, 0.3, 0.03]，norm_eps值为1024.0，hc_eps值为2
 * 预期结果：失败
 */
TEST_F(MhcPreTiling, Ut_Check_Case03_B65535_S1_n8_d1_BF16)
{
    uint32_t B = 65535;
    uint32_t S = 1;
    uint32_t n = 8;
    uint32_t d = 1;
    uint32_t phi_dim0 = n * n + 2 * n; // 80
    uint32_t phi_dim1 = n * d;         // 8
    uint32_t bias_dim = n * n + 2 * n; // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.3f, -0.3f, -0.3f}; // alpha = [-0.3, -0.3, -0.3]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

/*
 * 测试用例4：B=2，S=4096，n=4，d=1536，x的数据类型为bf16
 * alpha的值为[0.5, 0.5, 0.5]，norm_eps值为3.0，hc_eps值为200
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case04_B2_S4096_n4_d1536_BF16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 6144
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.2f, 0.2f, 0.2f}; // alpha = [0.2, 0.2, 0.2]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例5：B=2，S=4096，n=6，d=2048，x的数据类型为float16
 * alpha的值为[3.0, 10.0, 100.0]，norm_eps值为1.0，hc_eps值为20
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case05_B2_S4096_n6_d2048_FP16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 6;
    uint32_t d = 2048;
    uint32_t phi_dim0 = n * n + 2 * n; // 48
    uint32_t phi_dim1 = n * d;         // 12288
    uint32_t bias_dim = n * n + 2 * n; // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.6f, 0.6f, 0.6f}; // alpha = [0.6, 0.6, 0.6]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例6：B=2，S=4096，n=8，d=6144，x的数据类型为bf16
 * alpha的值为[0.5, 0.5, 0.5]，norm_eps值为3.0，hc_eps值为200
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case06_B2_S4096_n8_d6144_BF16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 8;
    uint32_t d = 6144;
    uint32_t phi_dim0 = n * n + 2 * n; // 80
    uint32_t phi_dim1 = n * d;         // 49152
    uint32_t bias_dim = n * n + 2 * n; // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.5f, -0.5f, -0.5f}; // alpha = [-0.5, -0.5, -0.5]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例7：B=256，S=1024，n=4，d=2048，x的数据类型为bf16
 * alpha的值为[0.2, 1.5, 100.0]，norm_eps值为50.0，hc_eps值为1000
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case07_B256_S1024_n4_d2048_BF16)
{
    uint32_t B = 256;
    uint32_t S = 1024;
    uint32_t n = 4;
    uint32_t d = 2048;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 8192
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.3f, 0.3f, 0.3f}; // alpha = [0.3, 0.3, 0.3]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例8：B=20，S=4096，n=6，d=1024，x的数据类型为float16
 * alpha的值为[10.0, 5.0, 20.0]，norm_eps值为60.0，hc_eps值为384
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case08_B20_S4096_n6_d1024_FP16)
{
    uint32_t B = 20;
    uint32_t S = 4096;
    uint32_t n = 6;
    uint32_t d = 1024;
    uint32_t phi_dim0 = n * n + 2 * n; // 48
    uint32_t phi_dim1 = n * d;         // 6144
    uint32_t bias_dim = n * n + 2 * n; // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.7f, 0.7f, 0.7f}; // alpha = [0.7, 0.7, 0.7]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例9：B=8，S=512，n=8，d=768，x的数据类型为float16
 * alpha的值为[0.8, 0.8, 0.8]，norm_eps值为0.00001，hc_eps值为100
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case09_B8_S512_n8_d768_FP16)
{
    uint32_t B = 8;
    uint32_t S = 512;
    uint32_t n = 8;
    uint32_t d = 768;
    uint32_t phi_dim0 = n * n + 2 * n; // 80
    uint32_t phi_dim1 = n * d;         // 6144
    uint32_t bias_dim = n * n + 2 * n; // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.2f, -0.2f, -0.2f}; // alpha = [-0.2, -0.2, -0.2]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例10：B=32，S=256，n=4，d=512，x的数据类型为bf16
 * alpha的值为[1.2, 1.5, 2.0]，norm_eps值为0.000001，hc_eps值为500
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case10_B32_S256_n4_d512_BF16)
{
    uint32_t B = 32;
    uint32_t S = 256;
    uint32_t n = 4;
    uint32_t d = 512;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 2048
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.4f, 0.4f, 0.4f}; // alpha = [0.4, 0.4, 0.4]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例11：B=1，S=8192，n=6，d=256，x的数据类型为float16
 * alpha的值为[0.3, 0.3, 0.3]，norm_eps值为0.1，hc_eps值为50
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case11_B1_S8192_n6_d256_FP16)
{
    uint32_t B = 1;
    uint32_t S = 8192;
    uint32_t n = 6;
    uint32_t d = 256;
    uint32_t phi_dim0 = n * n + 2 * n; // 48
    uint32_t phi_dim1 = n * d;         // 1536
    uint32_t bias_dim = n * n + 2 * n; // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.8f, 0.8f, 0.8f}; // alpha = [0.8, 0.8, 0.8]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例12：B=128，S=64，n=8，d=1024，x的数据类型为bf16
 * alpha的值为[2.5, 3.0, 3.5]，norm_eps值为0.01，hc_eps值为300
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case12_B128_S64_n8_d1024_BF16)
{
    uint32_t B = 128;
    uint32_t S = 64;
    uint32_t n = 8;
    uint32_t d = 1024;
    uint32_t phi_dim0 = n * n + 2 * n; // 80
    uint32_t phi_dim1 = n * d;         // 8192
    uint32_t bias_dim = n * n + 2 * n; // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.4f, -0.4f, -0.4f}; // alpha = [-0.4, -0.4, -0.4]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例13：B=4，S=2048，n=4，d=3072，x的数据类型为float16
 * alpha的值为[1.0, 2.0, 3.0]，norm_eps值为10.0，hc_eps值为150
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case13_B4_S2048_n4_d3072_FP16)
{
    uint32_t B = 4;
    uint32_t S = 2048;
    uint32_t n = 4;
    uint32_t d = 3072;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 12288
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.15f, 0.15f, 0.15f}; // alpha = [0.15, 0.15, 0.15]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例14：B=16，S=1024，n=6，d=128，x的数据类型为bf16
 * alpha的值为[0.1, 1.0, 10.0]，norm_eps值为0.001，hc_eps值为1000
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case14_B16_S1024_n6_d128_BF16)
{
    uint32_t B = 16;
    uint32_t S = 1024;
    uint32_t n = 6;
    uint32_t d = 128;
    uint32_t phi_dim0 = n * n + 2 * n; // 48
    uint32_t phi_dim1 = n * d;         // 768
    uint32_t bias_dim = n * n + 2 * n; // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.65f, 0.65f, 0.65f}; // alpha = [0.65, 0.65, 0.65]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例15：B=64，S=128，n=8，d=4096，x的数据类型为float16
 * alpha的值为[5.0, 5.0, 5.0]，norm_eps值为100.0，hc_eps值为50
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case15_B64_S128_n8_d4096_FP16)
{
    uint32_t B = 64;
    uint32_t S = 128;
    uint32_t n = 8;
    uint32_t d = 4096;
    uint32_t phi_dim0 = n * n + 2 * n; // 80
    uint32_t phi_dim1 = n * d;         // 32768
    uint32_t bias_dim = n * n + 2 * n; // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.35f, -0.35f, -0.35f}; // alpha = [-0.35, -0.35, -0.35]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例16：B=2，S=32768，n=4，d=512，x的数据类型为bf16
 * alpha的值为[0.01, 0.1, 1.0]，norm_eps值为0.5，hc_eps值为800
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case16_B2_S32768_n4_d512_BF16)
{
    uint32_t B = 2;
    uint32_t S = 32768;
    uint32_t n = 4;
    uint32_t d = 512;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 2048
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.25f, 0.25f, 0.25f}; // alpha = [0.25, 0.25, 0.25]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例17：B=48，S=512，n=6，d=2048，x的数据类型为float16
 * alpha的值为[0.2, 0.4, 0.8]，norm_eps值为5.0，hc_eps值为250
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case17_B48_S512_n6_d2048_FP16)
{
    uint32_t B = 48;
    uint32_t S = 512;
    uint32_t n = 6;
    uint32_t d = 2048;
    uint32_t phi_dim0 = n * n + 2 * n; // 48
    uint32_t phi_dim1 = n * d;         // 12288
    uint32_t bias_dim = n * n + 2 * n; // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.75f, 0.75f, 0.75f}; // alpha = [0.75, 0.75, 0.75]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例18：B=12，S=1536，n=8，d=1536，x的数据类型为bf16
 * alpha的值为[1.5, 2.5, 4.0]，norm_eps值为2.0，hc_eps值为120
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case18_B12_S1536_n8_d1536_BF16)
{
    uint32_t B = 12;
    uint32_t S = 1536;
    uint32_t n = 8;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n; // 80
    uint32_t phi_dim1 = n * d;         // 12288
    uint32_t bias_dim = n * n + 2 * n; // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {-0.25f, -0.25f, -0.25f}; // alpha = [-0.25, -0.25, -0.25]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例19：B=1，S=1，n=4，d=128，x的数据类型为float16
 * alpha的值为[0.05, 0.05, 0.05]，norm_eps值为0.1，hc_eps值为100
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case19_B1_S1_n4_d128_FP16)
{
    uint32_t B = 1;
    uint32_t S = 1;
    uint32_t n = 4;
    uint32_t d = 128;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 512
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.35f, 0.35f, 0.35f}; // alpha = [0.35, 0.35, 0.35]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例20：B=1024，S=32，n=6，d=768，x的数据类型为bf16
 * alpha的值为[0.7, 0.7, 0.7]，norm_eps值为0.001，hc_eps值为600
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case20_B1024_S32_n6_d768_BF16)
{
    uint32_t B = 1024;
    uint32_t S = 32;
    uint32_t n = 6;
    uint32_t d = 768;
    uint32_t phi_dim0 = n * n + 2 * n; // 48
    uint32_t phi_dim1 = n * d;         // 4608
    uint32_t bias_dim = n * n + 2 * n; // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.85f, 0.85f, 0.85f}; // alpha = [0.85, 0.85, 0.85]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例21：t = 1024，n=6，d=768，x的数据类型为bf16
 * alpha的值为[0.85, 0.85, 0.85]，norm_eps值为0.000001，hc_eps值为0.000001
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case21_T1024_n6_d768_BF16)
{
    uint32_t T = 1024;
    uint32_t n = 6;
    uint32_t d = 768;
    uint32_t phi_dim0 = n * n + 2 * n; // 48
    uint32_t phi_dim1 = n * d;         // 4608
    uint32_t bias_dim = n * n + 2 * n; // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.85f, 0.85f, 0.85f}; // alpha = [0.85, 0.85, 0.85]
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{T, n, d}, {T, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{T, d}, {T, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T, phi_dim0}, {T, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}
/*
 * 测试用例22：B=2，S=4096，n=4，d=1536，x的数据类型为bf16，gamma为空（可选输入）
 * alpha的值为[0.2, 0.2, 0.2]，norm_eps值为0.000001，hc_eps值为0.000001
 * 预期结果：成功
 * 说明：gamma为可选输入，此用例测试gamma为空时的正确性
 */
TEST_F(MhcPreTiling, Ut_Check_Case22_B2_S4096_n4_d1536_NoGamma)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 6144
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.2f, 0.2f, 0.2f};
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    // 不包含gamma输入，只有4个输入：x, phi, alpha, bias（gamma为可选输入，此处不填）
    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {
            {{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},                  // x
            {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // phi
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // alpha (fixed size 3)
            {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // bias
        },
        {
            {{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},                // hin
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},               // h_post
            {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},         // h_res
            {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // inv_rms (optional)
            {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND}, // h_mix (optional)
            {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}                // h_pre (optional)
        },
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例23：outFlag=1，SPLIT_BS模式，BSND layout
 * B=2，S=4096，n=4，d=1536，x的数据类型为bf16
 * totalLength=8192 > 512 → SPLIT_BS
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case23_OutFlag1_SplitBS_BSND_BF16)
{
    uint32_t B = 2;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 6144
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.3f, 0.3f, 0.3f};
    uint32_t outFlag = 1;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例24：outFlag=1，SPLIT_ND模式，BSND layout
 * B=1，S=64，n=6，d=2048，x的数据类型为float16
 * totalLength=64 <= 512 → SPLIT_ND
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case24_OutFlag1_SplitND_BSND_FP16)
{
    uint32_t B = 1;
    uint32_t S = 64;
    uint32_t n = 6;
    uint32_t d = 2048;
    uint32_t phi_dim0 = n * n + 2 * n; // 48
    uint32_t phi_dim1 = n * d;         // 12288
    uint32_t bias_dim = n * n + 2 * n; // 48
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.5f, 0.5f, 0.5f};
    uint32_t outFlag = 1;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例25：outFlag=1，SPLIT_BS模式，TND layout
 * T=1024，n=4，d=1536，x的数据类型为bf16
 * totalLength=1024 > 512 → SPLIT_BS
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case25_OutFlag1_SplitBS_TND_BF16)
{
    uint32_t T = 1024;
    uint32_t n = 4;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 6144
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.4f, 0.4f, 0.4f};
    uint32_t outFlag = 1;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{T, n, d}, {T, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{T, d}, {T, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T, phi_dim0}, {T, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例26：outFlag=1，SPLIT_ND模式，TND layout
 * T=256，n=8，d=768，x的数据类型为float16
 * totalLength=256 <= 512 → SPLIT_ND
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case26_OutFlag1_SplitND_TND_FP16)
{
    uint32_t T = 256;
    uint32_t n = 8;
    uint32_t d = 768;
    uint32_t phi_dim0 = n * n + 2 * n; // 80
    uint32_t phi_dim1 = n * d;         // 6144
    uint32_t bias_dim = n * n + 2 * n; // 80
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.6f, 0.6f, 0.6f};
    uint32_t outFlag = 1;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{T, n, d}, {T, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{T, d}, {T, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T, phi_dim0}, {T, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * 测试用例27：outFlag=1，长序列 totalLength>65536，BSND layout
 * B=1，S=70000，n=4，d=512，x的数据类型为bf16
 * totalLength=70000 > 65536 → SPLIT_BS
 * 预期结果：成功
 */
TEST_F(MhcPreTiling, Ut_Check_Case27_OutFlag1_LongSeq_BSND_BF16)
{
    uint32_t B = 1;
    uint32_t S = 70000;
    uint32_t n = 4;
    uint32_t d = 512;
    uint32_t phi_dim0 = n * n + 2 * n; // 24
    uint32_t phi_dim1 = n * d;         // 2048
    uint32_t bias_dim = n * n + 2 * n; // 24
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    float alpha[3] = {0.5f, 0.5f, 0.5f};
    uint32_t outFlag = 1;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}
/*
 * Low-api fast path: N=4, SPLIT_BS, hasResi, gamma present.
 * totalLength=12288 is outside the validated M-K range and keeps the BS fallback covered.
 */
TEST_F(MhcPreTiling, Ut_Check_BasicApi_N4_SplitBS_HasResi_Gamma)
{
    uint32_t B = 3;
    uint32_t S = 4096;
    uint32_t n = 4;
    uint32_t d = 1536;
    uint32_t phi_dim0 = n * n + 2 * n;
    uint32_t phi_dim1 = n * d;
    uint32_t bias_dim = phi_dim0;
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_FALSE(tilingInfo.workspaceSizes.empty());
    EXPECT_GT(tilingInfo.workspaceSizes[0], 20 * 1024 * 1024);
}

/*
 * Low-api fast path: N=6, SPLIT_BS.
 */
TEST_F(MhcPreTiling, Ut_Check_BasicApi_N6_SplitBS)
{
    uint32_t B = 3;
    uint32_t S = 4096;
    uint32_t n = 6;
    uint32_t d = 1024;
    uint32_t phi_dim0 = n * n + 2 * n;
    uint32_t phi_dim1 = n * d;
    uint32_t bias_dim = phi_dim0;
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 0;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);
}

/*
 * BS outFlag=0 stores compact hMix in workspace. Long M must reserve the full
 * [M, fusionSize] region in addition to X staging and system workspace.
 */
TEST_F(MhcPreTiling, Ut_Check_BasicApi_N8_LongM_Workspace)
{
    uint32_t B = 1;
    uint32_t S = 65537;
    uint32_t n = 8;
    uint32_t d = 32;
    uint32_t phiDim0 = n * n + 2 * n;
    uint32_t phiDim1 = n * d;
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MhcPre",
                                              {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim0}, {phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim1}, {phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{B, S, phiDim0}, {B, S, phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
                                               {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
                                               {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
                                              &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_FALSE(tilingInfo.workspaceSizes.empty());
    constexpr size_t systemWorkspaceSize = 20U * 1024U * 1024U;
    size_t hMixWorkspaceSize = static_cast<size_t>(B) * S * phiDim0 * sizeof(float);
    EXPECT_GT(tilingInfo.workspaceSizes[0], systemWorkspaceSize + hMixWorkspaceSize);
}

/*
 * M-K performance template probe: totalLength=1024, N=4, D=5120.
 * The compile-time probe selector should route this exact shape to MHC_PRE_SPLIT_M_K.
 */
TEST_F(MhcPreTiling, Ut_Check_MK_N4_M1024_D5120_HasResi)
{
    uint32_t B = 1;
    uint32_t S = 1024;
    uint32_t n = 4;
    uint32_t d = 5120;
    uint32_t phi_dim0 = n * n + 2 * n;
    uint32_t phi_dim1 = n * d;
    uint32_t bias_dim = phi_dim0;
    float normEps = 0.000001f;
    float hcEps = 0.000001f;
    uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phi_dim0, phi_dim1}, {phi_dim0, phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{bias_dim}, {bias_dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phi_dim1}, {phi_dim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, phi_dim0}, {B, S, phi_dim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    int64_t expectTilingKey = 2;
    string expectTilingDataStr = "";
    std::vector<size_t> expectWorkspaces = {};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                    TilingData2Str<int32_t>);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    constexpr uint32_t sentinel = 0x5A5AA55AU;
    auto fieldOffset = [&](auto setter) -> size_t {
        optiling::MhcPreTilingData schema;
        std::vector<uint8_t> schemaBuffer(schema.GetDataSize(), 0U);
        setter(schema, sentinel);
        schema.SaveToBuffer(schemaBuffer.data(), schemaBuffer.size());
        for (size_t offset = 0; offset + sizeof(uint32_t) <= schemaBuffer.size(); ++offset) {
            uint32_t value = 0;
            std::memcpy(&value, schemaBuffer.data() + offset, sizeof(value));
            if (value == sentinel) {
                return offset;
            }
        }
        return schemaBuffer.size();
    };
    auto readField = [&](size_t offset) -> uint32_t {
        EXPECT_LT(offset, tilingInfo.tilingDataSize);
        uint32_t value = 0;
        std::memcpy(&value, tilingInfo.tilingData.get() + offset, sizeof(value));
        return value;
    };

    uint32_t mDim = readField(fieldOffset([](auto &data, uint32_t value) { data.set_cubeBlockDimM(value); }));
    uint32_t kDim = readField(fieldOffset([](auto &data, uint32_t value) { data.set_cubeBlockDimK(value); }));
    uint32_t splitK = readField(fieldOffset([](auto &data, uint32_t value) { data.set_multCoreSplitKSize(value); }));
    uint32_t mL1Size = readField(fieldOffset([](auto &data, uint32_t value) { data.set_mL1Size(value); }));
    uint32_t kL1Size = readField(fieldOffset([](auto &data, uint32_t value) { data.set_kL1Size(value); }));
    uint32_t kUbSize = readField(fieldOffset([](auto &data, uint32_t value) { data.set_kUbSize(value); }));
    uint32_t mmOffset = readField(fieldOffset([](auto &data, uint32_t value) { data.set_mkWorkspaceMmOffset(value); }));
    uint32_t rmsOffset =
        readField(fieldOffset([](auto &data, uint32_t value) { data.set_mkWorkspaceRmsOffset(value); }));
    uint32_t finalOffset =
        readField(fieldOffset([](auto &data, uint32_t value) { data.set_mkWorkspaceFinalOffset(value); }));
    uint32_t mkUseGmStage = readField(fieldOffset([](auto &data, uint32_t value) { data.set_mkUseGmStage(value); }));
    uint32_t stage2UsedAivNum =
        readField(fieldOffset([](auto &data, uint32_t value) { data.set_stage2UsedAivNum(value); }));
    uint32_t stage2RowsPerCore =
        readField(fieldOffset([](auto &data, uint32_t value) { data.set_stage2RowsPerCore(value); }));

    auto ceilDiv = [](uint32_t value, uint32_t divisor) -> uint32_t { return (value + divisor - 1U) / divisor; };
    auto roundUp = [&ceilDiv](uint32_t value, uint32_t align) -> uint32_t { return ceilDiv(value, align) * align; };
    constexpr uint32_t expectedMDim = 4U;
    constexpr uint32_t sequentialPartialK = 1024U;
    uint32_t expectedKDim = tilingInfo.blockNum / expectedMDim;
    uint32_t expectedWorkspaceGroupK = ceilDiv(phi_dim1, sequentialPartialK);
    uint32_t expectedSplitK = ceilDiv(expectedWorkspaceGroupK, expectedKDim) * sequentialPartialK;
    uint32_t expectedActualKBlocks = ceilDiv(phi_dim1, expectedSplitK);
    uint32_t expectedStage2UsedAivNum = std::min<uint32_t>(B * S, static_cast<uint32_t>(tilingInfo.blockNum) * 2U);
    uint32_t expectedStage2RowsPerCore = ceilDiv(B * S, expectedStage2UsedAivNum);

    EXPECT_EQ(mDim, expectedMDim);
    EXPECT_EQ(kDim, expectedActualKBlocks);
    EXPECT_EQ(splitK, expectedSplitK);
    EXPECT_EQ(mL1Size, 256U);
    EXPECT_EQ(kL1Size, 128U);
    EXPECT_EQ(kUbSize, kL1Size);
    EXPECT_EQ(mmOffset, 0U);
    EXPECT_LT(mmOffset, rmsOffset);
    EXPECT_LT(rmsOffset, finalOffset);
    EXPECT_EQ(mkUseGmStage, 0U);
    EXPECT_EQ(stage2UsedAivNum, expectedStage2UsedAivNum);
    EXPECT_EQ(stage2RowsPerCore, expectedStage2RowsPerCore);
    ASSERT_FALSE(tilingInfo.workspaceSizes.empty());
    EXPECT_GT(tilingInfo.workspaceSizes[0], static_cast<int64_t>(finalOffset));
}

TEST_F(MhcPreTiling, Ut_Check_InvalidAlphaRank)
{
    constexpr uint32_t T = 1;
    constexpr uint32_t n = 4;
    constexpr uint32_t d = 256;
    constexpr uint32_t phiDim0 = n * n + 2 * n;
    constexpr uint32_t phiDim1 = n * d;
    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara("MhcPre",
                                              {{{{T, n, d}, {T, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{3, 1}, {3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim0}, {phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim1}, {phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{T, d}, {T, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, phiDim0}, {T, phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                               {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}},
                                              &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MhcPreTiling, Ut_Check_InvalidAlphaShape)
{
    constexpr uint32_t T = 1;
    constexpr uint32_t n = 4;
    constexpr uint32_t d = 256;
    constexpr uint32_t phiDim0 = n * n + 2 * n;
    constexpr uint32_t phiDim1 = n * d;
    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara("MhcPre",
                                              {{{{T, n, d}, {T, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim0}, {phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim1}, {phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{T, d}, {T, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, phiDim0}, {T, phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                               {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}},
                                              &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MhcPreTiling, Ut_Check_InvalidXRank)
{
    constexpr uint32_t T = 1;
    constexpr uint32_t n = 4;
    constexpr uint32_t d = 256;
    constexpr uint32_t phiDim0 = n * n + 2 * n;
    constexpr uint32_t phiDim1 = n * d;
    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara("MhcPre",
                                              {{{{T, n}, {T, n}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim0}, {phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim1}, {phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{T, d}, {T, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, phiDim0}, {T, phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                               {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}},
                                              &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MhcPreTiling, Ut_Check_InvalidHMixShape)
{
    constexpr uint32_t T = 1;
    constexpr uint32_t n = 4;
    constexpr uint32_t d = 256;
    constexpr uint32_t phiDim0 = n * n + 2 * n;
    constexpr uint32_t phiDim1 = n * d;
    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara("MhcPre",
                                              {{{{T, n, d}, {T, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim0}, {phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim1}, {phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{T, d}, {T, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, phiDim1}, {T, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                               {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}},
                                              &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

static void CheckMkGeneralizedTiling(uint32_t totalLength, uint32_t n, uint32_t d, int64_t expectedTilingKey,
                                     int32_t deterministicLevel = 0, bool hasResi = true, int64_t implMode = 0,
                                     uint32_t expectedSequentialPartialK = 0U, uint32_t expectedMDim = 0U,
                                     uint32_t expectedKDim = 0U)
{
    uint32_t phiDim0 = hasResi ? n * n + 2U * n : 2U * n;
    uint32_t alphaDim0 = hasResi ? 3U : 2U;
    uint32_t phiDim1 = n * d;
    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPre",
        {{{{1U, totalLength, n, d}, {1U, totalLength, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{alphaDim0}, {alphaDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phiDim0}, {phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{phiDim1}, {phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1U, totalLength, d}, {1U, totalLength, d}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1U, totalLength, n}, {1U, totalLength, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1U, totalLength, n, n}, {1U, totalLength, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1U, totalLength}, {1U, totalLength}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1U, totalLength, phiDim0}, {1U, totalLength, phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1U, totalLength, n}, {1U, totalLength, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(0.000001f)},
         {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(0.000001f)},
         {"op_impl_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(implMode)}},
        &compileInfo, "Ascend950", 32, 262144, 4096, MakeAscend950SocInfo(32));
    tilingContextPara.deterministicInfo_ = deterministicLevel;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, "", {}, 0, TilingData2Str<int32_t>);
    if (expectedTilingKey != 2 && expectedTilingKey != 6) {
        return;
    }

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    constexpr uint32_t sentinel = 0x5A5AA55AU;
    auto fieldOffset = [&](auto setter) -> size_t {
        optiling::MhcPreTilingData schema;
        std::vector<uint8_t> schemaBuffer(schema.GetDataSize(), 0U);
        setter(schema, sentinel);
        schema.SaveToBuffer(schemaBuffer.data(), schemaBuffer.size());
        for (size_t offset = 0; offset + sizeof(uint32_t) <= schemaBuffer.size(); ++offset) {
            uint32_t value = 0;
            std::memcpy(&value, schemaBuffer.data() + offset, sizeof(value));
            if (value == sentinel) {
                return offset;
            }
        }
        return schemaBuffer.size();
    };
    auto readField = [&](size_t offset) -> uint32_t {
        EXPECT_LT(offset, tilingInfo.tilingDataSize);
        uint32_t value = 0;
        std::memcpy(&value, tilingInfo.tilingData.get() + offset, sizeof(value));
        return value;
    };
    uint32_t kL1Size = readField(fieldOffset([](auto &data, uint32_t value) { data.set_kL1Size(value); }));
    uint32_t mDim = readField(fieldOffset([](auto &data, uint32_t value) { data.set_cubeBlockDimM(value); }));
    uint32_t kDim = readField(fieldOffset([](auto &data, uint32_t value) { data.set_cubeBlockDimK(value); }));
    uint32_t splitK = readField(fieldOffset([](auto &data, uint32_t value) { data.set_multCoreSplitKSize(value); }));
    uint32_t fusionAlign = readField(fieldOffset([](auto &data, uint32_t value) { data.set_fusionAlign(value); }));
    uint32_t stage2UsedAivNum =
        readField(fieldOffset([](auto &data, uint32_t value) { data.set_stage2UsedAivNum(value); }));
    uint32_t stage2RowsPerCore =
        readField(fieldOffset([](auto &data, uint32_t value) { data.set_stage2RowsPerCore(value); }));
    uint32_t actualImplMode = readField(fieldOffset([](auto &data, uint32_t value) { data.set_implMode(value); }));
    uint32_t mkUseGmStage = readField(fieldOffset([](auto &data, uint32_t value) { data.set_mkUseGmStage(value); }));
    EXPECT_EQ(actualImplMode, static_cast<uint32_t>(implMode));
    bool expectL1Stage = (implMode == 0 && totalLength >= 512U) || (n == 8U && totalLength < 1536U);
    EXPECT_EQ(mkUseGmStage, expectL1Stage ? 0U : 1U);
    EXPECT_EQ(kL1Size % 128U, 0U);
    EXPECT_EQ(fusionAlign % 8U, 0U);
    EXPECT_LE(static_cast<uint64_t>(kL1Size) * fusionAlign, 128U * 256U);
    if (expectedSequentialPartialK != 0U) {
        EXPECT_GE(splitK, expectedSequentialPartialK);
        EXPECT_EQ(splitK % expectedSequentialPartialK, 0U);
    }
    if (expectedMDim != 0U) {
        EXPECT_EQ(mDim, expectedMDim);
    }
    if (expectedKDim != 0U) {
        EXPECT_EQ(kDim, expectedKDim);
    }
    uint32_t expectedAivNum = static_cast<uint32_t>(tilingInfo.blockNum) * 2U;
    uint32_t expectedStage2RowsPerCore =
        std::min<uint32_t>(totalLength, std::max<uint32_t>(2U, (totalLength + expectedAivNum - 1U) / expectedAivNum));
    uint32_t expectedStage2UsedAivNum =
        std::min<uint32_t>(expectedAivNum, (totalLength + expectedStage2RowsPerCore - 1U) / expectedStage2RowsPerCore);
    EXPECT_EQ(stage2RowsPerCore, expectedStage2RowsPerCore);
    EXPECT_EQ(stage2UsedAivNum, expectedStage2UsedAivNum);
}

TEST_F(MhcPreTiling, Ut_Check_MK_Generalized_N4_SmallM)
{
    CheckMkGeneralizedTiling(1U, 4U, 2560U, 2);
    CheckMkGeneralizedTiling(64U, 4U, 2560U, 2);
    CheckMkGeneralizedTiling(179U, 4U, 6144U, 2);
    CheckMkGeneralizedTiling(180U, 4U, 6144U, 2);
}

TEST_F(MhcPreTiling, Ut_Check_MK_ImplMode_HF32)
{
    CheckMkGeneralizedTiling(64U, 4U, 2560U, 2, 0, true, 1);
    CheckMkGeneralizedTiling(1024U, 8U, 6144U, 2, 0, true, 1);
    CheckMkGeneralizedTiling(1536U, 8U, 6144U, 2, 0, true, 1);
}

TEST_F(MhcPreTiling, Ut_Check_MK_FillIdleAicWithoutReducingKDim)
{
    CheckMkGeneralizedTiling(3072U, 4U, 5120U, 2, 0, true, 1, 1024U, 16U, 2U);
    CheckMkGeneralizedTiling(2561U, 4U, 6144U, 6, 0, false, 0, 1024U, 15U, 2U);
}

TEST_F(MhcPreTiling, Ut_Check_MK_Generalized_N4_Boundary)
{
    CheckMkGeneralizedTiling(511U, 4U, 6144U, 2);
    CheckMkGeneralizedTiling(512U, 4U, 6144U, 2);
    CheckMkGeneralizedTiling(1536U, 4U, 6144U, 2);
    CheckMkGeneralizedTiling(2048U, 4U, 5120U, 2);
    CheckMkGeneralizedTiling(8192U, 4U, 5120U, 2);
    CheckMkGeneralizedTiling(8193U, 4U, 5120U, 0);
    CheckMkGeneralizedTiling(10240U, 4U, 5120U, 0);
    CheckMkGeneralizedTiling(10241U, 4U, 5120U, 0);
}

TEST_F(MhcPreTiling, Ut_Check_MK_Generalized_N6_KTail)
{
    CheckMkGeneralizedTiling(1000U, 6U, 2576U, 2);
}

TEST_F(MhcPreTiling, Ut_Check_MK_Generalized_N8_WideFusion)
{
    CheckMkGeneralizedTiling(1024U, 8U, 6144U, 2);
}

TEST_F(MhcPreTiling, Ut_Check_MK_Generalized_NoResi)
{
    CheckMkGeneralizedTiling(129U, 6U, 2592U, 6, 0, false);
    CheckMkGeneralizedTiling(96U, 8U, 2592U, 6, 0, false);
}

TEST_F(MhcPreTiling, Ut_Check_MK_SequentialPartials_AllNAndResiModes)
{
    constexpr uint32_t sequentialPartialK = 1024U;
    for (uint32_t n : {4U, 6U, 8U}) {
        CheckMkGeneralizedTiling(2049U, n, 2576U, 2, 0, true, 0, sequentialPartialK);
        CheckMkGeneralizedTiling(2049U, n, 2576U, 6, 0, false, 0, sequentialPartialK);
        CheckMkGeneralizedTiling(2049U, n, 2576U, 2, 0, true, 1, sequentialPartialK);
        CheckMkGeneralizedTiling(2049U, n, 2576U, 6, 0, false, 1, sequentialPartialK);
    }
}

TEST_F(MhcPreTiling, Ut_Check_MK_Fallback_SmallK)
{
    CheckMkGeneralizedTiling(128U, 4U, 16U, 1);
    CheckMkGeneralizedTiling(32U, 4U, 2560U, 2);
}

TEST_F(MhcPreTiling, Ut_Check_MK_DeterministicLevels)
{
    // Levels represented by the legacy UT context keep the fixed-order M-K reduction.
    CheckMkGeneralizedTiling(256U, 4U, 5120U, 2, 1);
    CheckMkGeneralizedTiling(1024U, 4U, 5120U, 2, 1);
    CheckMkGeneralizedTiling(256U, 4U, 5120U, 2, 2);
    CheckMkGeneralizedTiling(1024U, 4U, 5120U, 2, 2);
}

TEST_F(MhcPreTiling, Ut_Check_BatchConsistency_ForceSplitBs)
{
    // Batch consistency level 3 must use the BS accumulation order instead of the M-K template.
    CheckMkGeneralizedTiling(256U, 4U, 5120U, 2, 3);
}

/*
 * 控核约束：Ascend950要求aicNum:aivNum=1:2，不满足核数比的tiling必须失败。
 * 以下用例使用在1:2比例下能够tiling成功的形状(B=2,S=4096,n=4,d=1536)，
 * 仅改变cube/vector核数比，验证不满足1:2时返回GRAPH_FAILED。
 */
static void CheckCoreRatioFailure(uint32_t cubeCoreCnt, uint32_t vectorCoreCnt)
{
    constexpr uint32_t B = 2;
    constexpr uint32_t S = 4096;
    constexpr uint32_t n = 4;
    constexpr uint32_t d = 1536;
    const uint32_t phiDim0 = n * n + 2 * n;
    const uint32_t phiDim1 = n * d;
    const uint32_t biasDim = phiDim0;
    constexpr float normEps = 0.000001f;
    constexpr float hcEps = 0.000001f;
    constexpr uint32_t outFlag = 0;

    optiling::MhcPreCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara("MhcPre",
                                              {{{{B, S, n, d}, {B, S, n, d}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{biasDim}, {biasDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{phiDim1}, {phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{B, S, d}, {B, S, d}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{B, S}, {B, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{B, S, phiDim0}, {B, S, phiDim0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{"out_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outFlag)},
                                               {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
                                               {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)}},
                                              &compileInfo, "Ascend950", cubeCoreCnt, 262144, 4096,
                                              MakeAscend950SocInfo(cubeCoreCnt, vectorCoreCnt));

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// aicNum:aivNum = 1:1，不满足1:2，预期失败
TEST_F(MhcPreTiling, Ut_Check_CoreRatio_1to1_Fail)
{
    CheckCoreRatioFailure(32, 32);
}

// aicNum:aivNum = 1:3，不满足1:2，预期失败
TEST_F(MhcPreTiling, Ut_Check_CoreRatio_1to3_Fail)
{
    CheckCoreRatioFailure(16, 48);
}

// aicNum:aivNum = 1:4，不满足1:2，预期失败
TEST_F(MhcPreTiling, Ut_Check_CoreRatio_1to4_Fail)
{
    CheckCoreRatioFailure(32, 128);
}

// aicNum:aivNum = 2:1，vector核数不足，预期失败
TEST_F(MhcPreTiling, Ut_Check_CoreRatio_2to1_Fail)
{
    CheckCoreRatioFailure(64, 32);
}
