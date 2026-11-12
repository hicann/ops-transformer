/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/op_tiling/mhc_pre_sinkhorn_backward_tiling.h"
#include "../../../../op_kernel/arch35/mhc_pre_sinkhorn_backward_data_arch35.h"
#include "../../../../op_kernel/arch92/mhc_pre_sinkhorn_backward_data_arch92.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// Ascend 950 uses the arch35 tiling template; SUCCESS cases skip tilingKey validation.
constexpr uint64_t SKIP_TILING_KEY_VALIDATION = UINT64_MAX;
constexpr char kSocVersion950[] = "Ascend950";
// Match Ascend950's 256 KiB L0C while preserving the other generic UT platform settings.
constexpr char kAscend950SocInfo[] = R"({"hardware_info":{
    "BT_SIZE":0,"load3d_constraints":"1","Intrinsic_fix_pipe_l0c2out":false,
    "Intrinsic_data_move_l12ub":true,"Intrinsic_data_move_l0c2ub":true,
    "Intrinsic_data_move_out2l1_nd2nz":false,"UB_SIZE":262144,"L2_SIZE":33554432,
    "L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":262144,
    "CORE_NUM":64,"socVersion":"Ascend950"}})";

class MhcPreSinkhornBackwardTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreSinkhornBackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreSinkhornBackwardTiling TearDown" << std::endl;
    }
};

TEST_F(MhcPreSinkhornBackwardTiling, test_tiling_B2_S128_N4_C256)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, kSocVersion950);

    tilingContextPara.socInfoString_ = kAscend950SocInfo;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY_VALIDATION);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_tiling_B1_S512_N4_C128)
{
    int64_t B = 1;
    int64_t S = 512;
    int64_t N = 4;
    int64_t C = 128;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, kSocVersion950);

    tilingContextPara.socInfoString_ = kAscend950SocInfo;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY_VALIDATION);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_tiling_B4_S64_N4_C512)
{
    int64_t B = 4;
    int64_t S = 64;
    int64_t N = 4;
    int64_t C = 512;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, kSocVersion950);

    tilingContextPara.socInfoString_ = kAscend950SocInfo;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY_VALIDATION);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_tiling_B1_S64_N9_C128_fail)
{
    int64_t B = 1;
    int64_t S = 64;
    int64_t N = 9;
    int64_t C = 128;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, kSocVersion950);

    tilingContextPara.socInfoString_ = kAscend950SocInfo;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_tiling_B1_S35_N1_C512)
{
    int64_t B = 1;
    int64_t S = 35;
    int64_t N = 1;
    int64_t C = 512;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, kSocVersion950);

    tilingContextPara.socInfoString_ = kAscend950SocInfo;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY_VALIDATION);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_tiling_B2_S64_N2_C256)
{
    int64_t B = 2;
    int64_t S = 64;
    int64_t N = 2;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, kSocVersion950);

    tilingContextPara.socInfoString_ = kAscend950SocInfo;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY_VALIDATION);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_tiling_B1_S64_N8_C128)
{
    int64_t B = 1;
    int64_t S = 64;
    int64_t N = 8;
    int64_t C = 128;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, kSocVersion950);

    tilingContextPara.socInfoString_ = kAscend950SocInfo;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY_VALIDATION);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_tiling_T256_N4_C256_3d)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t T = B * S;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{T, C}, {T, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, hcMix}, {T, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, 1}, {T, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N}, {skIterCount * 2, T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N, N}, {skIterCount * 2, T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, kSocVersion950);

    tilingContextPara.socInfoString_ = kAscend950SocInfo;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY_VALIDATION);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_tiling_T64_N8_C128_3d)
{
    int64_t B = 1;
    int64_t S = 64;
    int64_t T = B * S;
    int64_t N = 8;
    int64_t C = 128;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{T, C}, {T, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, hcMix}, {T, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, 1}, {T, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N}, {skIterCount * 2, T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N, N}, {skIterCount * 2, T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, kSocVersion950);

    tilingContextPara.socInfoString_ = kAscend950SocInfo;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY_VALIDATION);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_tiling_T64_N9_C128_3d_fail)
{
    int64_t B = 1;
    int64_t S = 64;
    int64_t T = B * S;
    int64_t N = 9;
    int64_t C = 128;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{T, C}, {T, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, hcMix}, {T, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, 1}, {T, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N}, {skIterCount * 2, T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, T, N, N}, {skIterCount * 2, T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{T, N, C}, {T, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, kSocVersion950);

    tilingContextPara.socInfoString_ = kAscend950SocInfo;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

namespace {
void CheckPlatformRouting(const char* soc, int32_t deterministic, uint64_t expectedKey, int64_t batchSize = 2,
                          int64_t sequenceLength = 128)
{
    int64_t B = batchSize;
    int64_t S = sequenceLength;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    float eps = 1e-6f;

    int64_t hcMix = N * N + 2 * N;
    int64_t phiDim0 = hcMix;
    int64_t phiDim1 = N * C;

    optiling::MhcPreSinkhornBackwardCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "MhcPreSinkhornBackward",
        {
            {{{B, S, C}, {B, S, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, hcMix}, {B, S, hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, 1}, {B, S, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N}, {skIterCount * 2, B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{skIterCount * 2, B, S, N, N}, {skIterCount * 2, B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{B, S, N, C}, {B, S, N, C}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{phiDim0, phiDim1}, {phiDim0, phiDim1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hcMix}, {hcMix}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
        },
        &compileInfo, soc);

    if (tilingContextPara.socVersion_ == kSocVersion950) {
        tilingContextPara.socInfoString_ = kAscend950SocInfo;
    }
    tilingContextPara.deterministicInfo_ = deterministic;
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info));
    ASSERT_EQ(info.tilingKey, expectedKey);
    if (tilingContextPara.socVersion_ == kSocVersion950) {
        ASSERT_GE(info.tilingDataSize, sizeof(MhcPreSinkhornBackwardArch35TilingData));
        const auto* data = reinterpret_cast<const MhcPreSinkhornBackwardArch35TilingData*>(info.tilingData.get());
        const uint32_t expectedSmallM = B * S > 0 && B * S <= 512 ? 1U : 0U;
        EXPECT_EQ(data->smallM, expectedSmallM);
        using MhcPreSinkhornBackwardSinkhornNs::SinkhornPath;
        EXPECT_EQ(data->sinkhornPlan.path,
                  expectedSmallM != 0U ? SinkhornPath::N4_COMPACT : SinkhornPath::N4_ROW_PACKED);
    }
    if (expectedKey == 4001) {
        ASSERT_GE(info.tilingDataSize, sizeof(MhcPreSinkhornBackward960DeterminiticTilingData));
        const auto* data =
            reinterpret_cast<const MhcPreSinkhornBackward960DeterminiticTilingData*>(info.tilingData.get());
        EXPECT_EQ(data->n, N);
        EXPECT_EQ(data->c, C);
    }
}
} // namespace

TEST_F(MhcPreSinkhornBackwardTiling, test_platform_routing_960_arch92)
{
    CheckPlatformRouting("Ascend960DT", 0, 4001);
    CheckPlatformRouting("Ascend960DT", 3, 4001);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_platform_routing_950_arch35)
{
    CheckPlatformRouting(kSocVersion950, 0, 3000);
}

TEST_F(MhcPreSinkhornBackwardTiling, test_runtime_small_m_boundaries_950_arch35)
{
    for (int64_t rows : {0, 1, 511, 512, 513, 8192}) {
        CheckPlatformRouting(kSocVersion950, 0, 3000, 1, rows);
    }
}
