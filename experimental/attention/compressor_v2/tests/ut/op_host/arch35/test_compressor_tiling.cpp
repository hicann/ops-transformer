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

#include "../../../../op_host/arch35/compressor_v2_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// 构造版本
std::string Compressor_tiling_A5SocInfo = "{\n"
                                          "  \"hardware_info\": {\n"
                                          "    \"BT_SIZE\": 0,\n"
                                          "    \"load3d_constraints\": \"1\",\n"
                                          "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                          "    \"Intrinsic_data_move_l12ub\": true,\n"
                                          "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                          "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                          "    \"4096\": 196608,\n"
                                          "    \"L2_SIZE\": 201326592,\n"
                                          "    \"L1_SIZE\": 524288,\n"
                                          "    \"L0A_SIZE\": 65536,\n"
                                          "    \"L0B_SIZE\": 65536,\n"
                                          "    \"L0C_SIZE\": 131072,\n"
                                          "    \"vector_core_cnt\": 40,\n"
                                          "    \"cube_core_cnt\": 20,\n"
                                          "    \"socVersion\": \"Ascend950\"\n"
                                          "  }\n"
                                          "}";

// ====================================================================
// BSH Layout Tiling Tests
// ====================================================================

class CompressorTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CompressorTilingArch35 SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "CompressorTilingArch35 TearDown" << std::endl;
    }
};

// C4A bf16: B=2, S=8, H=4096, D=512, coff=2, cmp_ratio=4
TEST_F(CompressorTilingArch35, test1)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // x
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},        // wkv
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},        // wgate
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                       // state_block_table
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // start_pos
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // cmp_kv
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache (in-place)
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    int64_t expectTilingKey = 32;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ====================================================================
// Additional Error Cases — err msg branch coverage (EZ00xx)
// ====================================================================

// Non-x empty tensor (wkv empty) -> EZ0009 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
TEST_F(CompressorTilingArch35, test_empty_wkv)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{0, 4096}, {0, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wkv empty
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// Invalid blockSize=0 -> EZ0026 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
TEST_F(CompressorTilingArch35, test_bad_block_size_zero)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 0, 1024}, {4, 0, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, // blockSize=0
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 0, 1024}, {4, 0, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// RING_BUFFER: blockNum < batchSize -> EZ0026 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
TEST_F(CompressorTilingArch35, test_ring_blocknum_lt_batch)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // B=2
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, // blockNum=1 < B=2
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                       // 1D for RING
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1024}, {1, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// RING_BUFFER: stateBlockTable dimNum=2 (should be 1) -> EZ0012 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
TEST_F(CompressorTilingArch35, test_ring_bad_block_table_dim)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND}, // 2D, wrong for RING
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// BSH with cu_seqlens present (should be absent) -> EZ0037 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
TEST_F(CompressorTilingArch35, test_bsh_with_cu_seqlens)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND}, // cu_seqlens present in BSH
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// TH without cu_seqlens (should be present) -> EZ0037 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
TEST_F(CompressorTilingArch35, test_th_without_cu_seqlens)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // TH layout
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // cu_seqlens absent in TH
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 512}, {2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// wkv dtype != x dtype -> EZ0020 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
TEST_F(CompressorTilingArch35, test_dtype_consistency_wkv)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wkv fp16, x bf16
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// cmpKv dimNum != x dimNum -> EZ0012 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
TEST_F(CompressorTilingArch35, test_dimnum_consistency_cmpkv)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // x 3D
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 512}, {2, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // cmpKv 2D, x 3D
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// wkv dim1 != hiddenSize -> EZ0009 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
TEST_F(CompressorTilingArch35, test_shape_consistency_wkv_hidden)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // H=4096
            {{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND},   // wkv dim1=2048 != 4096
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// Invalid x dtype (DT_FLOAT not supported) -> EZ0019 OP_LOGE_FOR_INVALID_DTYPE
TEST_F(CompressorTilingArch35, test_bad_dtype_x)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x DT_FLOAT unsupported
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// Invalid x dimNum=4 (not 2 or 3) -> EZ0012 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
TEST_F(CompressorTilingArch35, test_bad_x_dimnum)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096, 1}, {2, 8, 4096, 1}}, ge::DT_BF16, ge::FORMAT_ND}, // x 4D
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// BSH fp16: B=2, S=8, H=4096, D=512, cmp_ratio=4
TEST_F(CompressorTilingArch35, test2)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    int64_t expectTilingKey = 34;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// BSH bf16: B=2, S=8, H=2048, D=128, cmp_ratio=4
TEST_F(CompressorTilingArch35, test3)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 256}, {4, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 128}, {2, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 256}, {4, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    int64_t expectTilingKey = 32;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// BSH bf16: B=2, S=16, H=4096, D=512, cmp_ratio=128
TEST_F(CompressorTilingArch35, test4)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 16, 4096}, {2, 16, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 1, 512}, {2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(131072)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    int64_t expectTilingKey = 32;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// Non-divisible S: B=1, S=6, H=2048, D=128, cmp_ratio=4, Sr=ceil(6/4)=2
TEST_F(CompressorTilingArch35, test5)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{1, 6, 2048}, {1, 6, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 2, 128}, {1, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 128, 256}, {2, 128, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    int64_t expectTilingKey = 32;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ====================================================================
// TH Layout Tiling Tests
// ====================================================================

// TH bf16: T=8, H=4096, B=2, cu_seqlens=[0,4,8]
TEST_F(CompressorTilingArch35, test6)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 512}, {2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    int64_t expectTilingKey = 33;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ====================================================================
// Error Cases — expected FAIL
// ====================================================================

// Unsupported cmp_ratio=129 (out of [2,128])
TEST_F(CompressorTilingArch35, test7)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(129)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    int64_t expectTilingKey = 32;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// Unsupported headDim=256 (only 128 and 512 supported)
TEST_F(CompressorTilingArch35, test9)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 512}, {4, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 256}, {2, 2, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 512}, {4, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    int64_t expectTilingKey = 32;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}

// Unsupported hiddenSize=3000 (not 512-aligned)
TEST_F(CompressorTilingArch35, test10)
{
    optiling::CompressorV2CompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorV2",
        {
            {{{2, 8, 3000}, {2, 8, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 3000}, {512, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 3000}, {512, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(262144)},
        },
        &compileInfo, "Ascend950", Compressor_tiling_A5SocInfo, 4096);
    int64_t expectTilingKey = 32;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey);
}
