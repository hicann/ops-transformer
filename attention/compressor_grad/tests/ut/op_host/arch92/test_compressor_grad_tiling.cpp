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
#include <limits>

#include "../../../../op_host/compressor_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// A6 (Ascend960DT/Ascend960PR) SoC info: 32 AIC / 32 AIV.
// CompressorGrad 在 ascend960dt/pr 上复用 arch35(PyPTO) tiling 实现。
std::string CompressorGrad_tiling_A6SocInfo = "{\n"
                                              "  \"hardware_info\": {\n"
                                              "    \"BT_SIZE\": 0,\n"
                                              "    \"load3d_constraints\": \"1\",\n"
                                              "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                              "    \"Intrinsic_data_move_l12ub\": true,\n"
                                              "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                              "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                              "    \"UB_SIZE\": 262144,\n"
                                              "    \"L2_SIZE\": 134217728,\n"
                                              "    \"L1_SIZE\": 524288,\n"
                                              "    \"L0A_SIZE\": 65536,\n"
                                              "    \"L0B_SIZE\": 65536,\n"
                                              "    \"L0C_SIZE\": 262144,\n"
                                              "    \"CORE_NUM\": 32,\n"
                                              "    \"vector_core_cnt\": 32,\n"
                                              "    \"cube_core_cnt\": 32,\n"
                                              "    \"socVersion\": \"Ascend960DT\"\n"
                                              "  }\n"
                                              "}";

// tiling key encoding (compressor_grad.py CompressorGradTilingKey):
//   key = (coff-1) + layout*4 + dtype*8 + enableApe*32

class CompressorGradTilingArch92 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CompressorGradTilingArch92 SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "CompressorGradTilingArch92 TearDown" << std::endl;
    }
};

// A6 DT, BSH C4A bf16: key = 1 + 0 + 0 + 32 = 33
TEST_F(CompressorGradTilingArch92, a6_dt_bsh_c4a_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // x
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // wgate
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // d_cmp_kv
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // start_pos
        },
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // d_x
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // d_wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // d_wgate
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},      // d_ape
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"enable_ape", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "9201", CompressorGrad_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 33);
}

// A6 DT, BSH C4A fp16: key = 1 + 0 + 8 + 32 = 41
TEST_F(CompressorGradTilingArch92, a6_dt_bsh_c4a_fp16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 2, 512}, {2, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                  {"enable_ape", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo, "9201", CompressorGrad_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 41);
}

// A6 DT, BSH C4Li bf16: D=128, coff=2 -> key = 1 + 0 + 0 + 32 = 33
TEST_F(CompressorGradTilingArch92, a6_dt_bsh_c4li_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 128}, {2, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 128}, {2, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 128}, {2, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                  {"enable_ape", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo, "9201", CompressorGrad_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 33);
}

// A6 DT, TH C4A bf16: key = 1 + 4(layout) + 0 + 32 = 37
TEST_F(CompressorGradTilingArch92, a6_dt_th_c4a_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                  {"enable_ape", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo, "9201", CompressorGrad_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 37);
}

// A6 PR shares the same PyPTO tiling: key = 33
TEST_F(CompressorGradTilingArch92, a6_pr_bsh_c4a_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                  {"enable_ape", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo, "9202", CompressorGrad_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 33);
}

// A6 DT, unsupported cmp_ratio=3 -> GRAPH_FAILED
TEST_F(CompressorGradTilingArch92, a6_dt_bad_cmp_ratio)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                  {"enable_ape", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo, "9201", CompressorGrad_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}
