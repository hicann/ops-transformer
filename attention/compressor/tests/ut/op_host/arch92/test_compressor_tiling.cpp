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

#include "../../../../op_host/arch92/compressor_tiling_arch92.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// A6 (Ascend960DT/Ascend960PR) SoC info: 32 AIC / 32 AIV.
std::string Compressor_tiling_A6SocInfo = "{\n"
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

// tilingKey encoding of arch92 (compressor_template_tiling_key_arch92.h):
//   bit0     layout     0:BSH 1:TH
//   bit1-4   dtype      0:BF16 1:FP16
//   bit5-6   coff       1:off 2:on
//   bit7-8   cache_mode 1:LINEAR 2:RING
//   bit9-10  templateId 0:EMPTY_X 1:NORMAL 2:FULL_LOAD
//   bit11    grad_enabled

class CompressorTilingArch92 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CompressorTilingArch92 SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "CompressorTilingArch92 TearDown" << std::endl;
    }
};

// A6 DT, C4A bf16: B=2, S=8, H=4096, D=512, coff=2, cmp_ratio=4, LINEAR.
// key = 0(layout) + 0(dtype) + 1(coff)<<5 + 0(cache) + 1(template)<<9 = 544
TEST_F(CompressorTilingArch92, a6_dt_bsh_c4a_bf16)
{
    optiling::CompressorCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Compressor",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // x
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // wgate
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},           // ape
            {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},                 // state_block_table
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // start_pos
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // cmp_kv
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache (in-place)
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "9201", Compressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 544);
}

// A6 DT, C4A fp16: dtype index 1 -> 1<<1 = 2, key = 544 + 2 = 546
TEST_F(CompressorTilingArch92, a6_dt_bsh_c4a_fp16)
{
    optiling::CompressorCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Compressor",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "9201", Compressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 546);
}

// A6 DT, C4Li bf16: B=2, S=8, H=2048, D=128, coff=2, cmp_ratio=4, LINEAR.
// key = 544 (template NORMAL)
TEST_F(CompressorTilingArch92, a6_dt_bsh_c4li_bf16)
{
    optiling::CompressorCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Compressor",
        {
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 512}, {4, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 128}, {2, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 512}, {4, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 128}, {2, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 128}, {2, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        },
        &compileInfo, "9201", Compressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 544);
}

// A6 DT, C128A bf16: B=2, S=16, H=4096, D=512, coff=1, cmp_ratio=128, LINEAR.
// key = 0 + 0 + 0(coff=1) + 0(cache=1) + 1<<9 = 512
TEST_F(CompressorTilingArch92, a6_dt_bsh_c128a_bf16)
{
    optiling::CompressorCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Compressor",
        {
            {{{2, 16, 4096}, {2, 16, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128, 512}, {128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 1, 512}, {2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 128, 512}, {2, 1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 128, 512}, {2, 1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(131072)},
        },
        &compileInfo, "9201", Compressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 512);
}

// A6 DT, TH layout C4A bf16: T=8, H=4096, B=2, cu_seqlens=[0,4,8].
// key = 1(layout) + 544 = 545
TEST_F(CompressorTilingArch92, a6_dt_th_c4a_bf16)
{
    optiling::CompressorCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Compressor",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 512}, {2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 512}, {2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 512}, {2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "9201", Compressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 545);
}

// A6 DT, BSH FULL_LOAD template: B=2, S=4, H=4096, D=128, coff=2, RING.
// key = 1(coff)<<5 + 1(cache)<<7 + 2(template)<<9 = 32 + 128 + 1024 = 1184
TEST_F(CompressorTilingArch92, a6_dt_bsh_full_load)
{
    optiling::CompressorCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Compressor",
        {
            {{{2, 4, 4096}, {2, 4, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 4096}, {256, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 4096}, {256, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 4, 128}, {2, 1, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 4, 128}, {2, 1, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "9201", Compressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1184);
}

// A6 DT, grad_enabled=true: bit11 -> 544 + 2048 = 2592
TEST_F(CompressorTilingArch92, a6_dt_grad_enabled)
{
    optiling::CompressorCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Compressor",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
            {"grad_enabled", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "9201", Compressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2592);
}

// A6 PR also dispatches to the arch92 tiling: same config as a6_dt_bsh_c4a_bf16.
TEST_F(CompressorTilingArch92, a6_pr_bsh_c4a_bf16)
{
    optiling::CompressorCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Compressor",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "9202", Compressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 544);
}

// A6 DT, unsupported cmp_ratio=3 -> GRAPH_FAILED
TEST_F(CompressorTilingArch92, a6_dt_bad_cmp_ratio)
{
    optiling::CompressorCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Compressor",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "9201", Compressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// A6 DT, RING buffer with blockNum(1) < batchSize(2) -> GRAPH_FAILED
TEST_F(CompressorTilingArch92, a6_dt_ring_blocknum_lt_batch)
{
    optiling::CompressorCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "Compressor",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 2048}, {1, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 2048}, {1, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "9201", Compressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
