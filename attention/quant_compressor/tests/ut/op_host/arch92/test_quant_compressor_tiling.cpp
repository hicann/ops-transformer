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
#include "tiling_case_executor.h"

struct QuantCompressorUtCompileInfo {
    int64_t core_num;
};

// A6 (Ascend960DT/Ascend960PR) SoC info.
std::string QuantCompressor_tiling_A6SocInfo = "{\n"
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

// Full-load splitting needs a small core count to fit splitCoreParam[36].
std::string QuantCompressor_tiling_A6SocInfo8Core = "{\n"
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
                                                    "    \"CORE_NUM\": 8,\n"
                                                    "    \"vector_core_cnt\": 8,\n"
                                                    "    \"cube_core_cnt\": 8,\n"
                                                    "    \"socVersion\": \"Ascend960DT\"\n"
                                                    "  }\n"
                                                    "}";

// tilingKey encoding of arch92 (quant_compressor_template_tiling_key_arch92.h):
//   bit0     layout     0:BSH 1:TH
//   bit1-4   dtype      0:HIFP8
//   bit5-6   coff       1:off 2:on
//   bit7-8   cache_mode 1:LINEAR 2:RING
//   bit9     quant_mode 1:HIFP8
//   bit10-11 templateId 0:EMPTY_X 1:NORMAL 2:FULL_LOAD

class QuantCompressorTilingArch92 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantCompressorTilingArch92 SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "QuantCompressorTilingArch92 TearDown" << std::endl;
    }
};

// A6 DT, BSH C4Li (D=128, coff=2, cmp_ratio=4, LINEAR).
// key = 1(coff)<<5 + 1(template)<<10 = 32 + 1024 = 1056
TEST_F(QuantCompressorTilingArch92, a6_dt_bsh_c4li)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "QuantCompressor",
        {
            {{{1, 128, 4096}, {1, 128, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // x
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},       // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},       // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},      // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},                // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                    // state_block_table
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // cu_seqlens (absent for BSH)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // start_pos (absent)
        },
        {
            {{{1, 32, 128}, {1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // cmp_kv
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "9201", QuantCompressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1056);
}

// A6 DT, BSH C128A (D=512, coff=1, cmp_ratio=128, RING).
// key = 0(coff) + 1(cache)<<7 + 1(template)<<10 = 128 + 1024 = 1152
TEST_F(QuantCompressorTilingArch92, a6_dt_bsh_c128a)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "QuantCompressor",
        {
            {{{1, 128, 4096}, {1, 128, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // x
            {{{512, 4096}, {512, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},       // wkv
            {{{512, 4096}, {512, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},       // wgate
            {{{1, 256, 1024}, {1, 256, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},    // state_cache
            {{{128, 512}, {128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},            // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // x_descale
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // wkv_descale
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // wgate_descale
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                          // state_block_table [B] for RING
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // cu_seqlens (absent for BSH)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // start_pos (absent)
        },
        {
            {{{1, 1, 512}, {1, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // cmp_kv
            {{{1, 256, 1024}, {1, 256, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "9201", QuantCompressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1152);
}

// A6 DT, BSH FULL_LOAD: seqSize<=4 && tokenSize<=256 selects FULL_LOAD only on DAV_9201.
// key = 1(coff)<<5 + 2(template)<<10 = 32 + 2048 = 2080
TEST_F(QuantCompressorTilingArch92, a6_dt_bsh_full_load)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "QuantCompressor",
        {
            {{{1, 4, 4096}, {1, 4, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // x (S=4)
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},            // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                // state_block_table
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                          // cu_seqlens (absent for BSH)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                          // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                          // start_pos (absent)
        },
        {
            {{{1, 1, 128}, {1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},      // cmp_kv (Sr=ceil(4/4)=1)
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "9201", QuantCompressor_tiling_A6SocInfo8Core, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2080);
}

// A6 PR keeps NORMAL template for the same small shape (FULL_LOAD is DAV_9201-only).
// key = 1(coff)<<5 + 1(template)<<10 = 32 + 1024 = 1056
TEST_F(QuantCompressorTilingArch92, a6_pr_bsh_small_normal_template)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "QuantCompressor",
        {
            {{{1, 4, 4096}, {1, 4, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // x (S=4)
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},            // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                // state_block_table
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                          // cu_seqlens (absent for BSH)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                          // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                          // start_pos (absent)
        },
        {
            {{{1, 1, 128}, {1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},      // cmp_kv (Sr=ceil(4/4)=1)
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "9202", QuantCompressor_tiling_A6SocInfo8Core, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1056);
}

// A6 DT, TH layout C4Li: layout bit0 = 1 -> key = 1056 + 1 = 1057
TEST_F(QuantCompressorTilingArch92, a6_dt_th_c4li)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "QuantCompressor",
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},  // x [T,H]
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},  // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},  // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},           // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},               // state_block_table
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                     // cu_seqlens [B+1]
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                         // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                         // start_pos (absent)
        },
        {
            {{{33, 128}, {33, 128}}, ge::DT_BF16, ge::FORMAT_ND},          // cmp_kv [Sr,D]
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "9201", QuantCompressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1057);
}

// A6 DT, unsupported cmp_ratio=3 -> GRAPH_FAILED
TEST_F(QuantCompressorTilingArch92, a6_dt_bad_cmp_ratio)
{
    QuantCompressorUtCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "QuantCompressor",
        {
            {{{1, 128, 4096}, {1, 128, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // x
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},       // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},       // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},      // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},                // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                    // state_block_table
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // cu_seqlens (absent for BSH)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // seqused (absent)
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // start_pos (absent)
        },
        {
            {{{1, 43, 128}, {1, 43, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // cmp_kv
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo, "9201", QuantCompressor_tiling_A6SocInfo, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
