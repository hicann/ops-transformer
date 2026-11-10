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
#include <string>

// 引入自动生成的 Tiling 头文件
#include "../../../op_host/block_sparse_attention_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class BlockSparseAttentionGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- BlockSparseAttentionGradTiling UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- BlockSparseAttentionGradTiling UT TearDown ---" << std::endl;
    }
};

namespace {
void RunArch35BnsdTilingCase(int64_t s, int64_t s_kv, int64_t blockX, int64_t blockY, ge::graphStatus expectedResult,
                             int64_t nKv = 2, bool isHif8 = false, int32_t deterministic = 0, int64_t quantMode = -1)
{
    optiling::BlockSparseAttentionGradCompileInfo compileInfo;
    compileInfo.socVersion = platform_ascendc::SocVersion::ASCEND950;

    // Arch35 NonZero tiling requires separate cube/vector core counts and vector register width.
    const std::string socInfo = R"({
        "hardware_info": {
            "CORE_NUM": 36, "cube_core_cnt": 36, "vector_core_cnt": 72,
            "UB_SIZE": 253952, "VECTOR_REG_WIDTH": 256,
            "L2_SIZE": 33554432, "L1_SIZE": 524288,
            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072
        }
    })";

    int64_t b = 1, n = 4, n_kv = nKv, d = 128;
    int64_t ceilQ = (s + blockX - 1) / blockX;
    int64_t ceilKv = (s_kv + blockY - 1) / blockY;

    int64_t blockShapeData[2] = {blockX, blockY};
    int64_t actualSeqData[1] = {s};
    int64_t actualSeqKvData[1] = {s_kv};
    float scaleData[1] = {1.0F};
    ge::DataType inputType = isHif8 ? ge::DT_HIFLOAT8 : ge::DT_FLOAT16;
    ge::DataType attentionType = isHif8 ? ge::DT_BF16 : ge::DT_FLOAT16;
    ge::DataType outputType = isHif8 ? ge::DT_BF16 : ge::DT_FLOAT16;

    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttentionGrad",
        {// --- Input Info ---
         {{{b, n, s, d}, {b, n, s, d}}, inputType, ge::FORMAT_ND},
         {{{b, n, s, d}, {b, n, s, d}}, inputType, ge::FORMAT_ND},
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, inputType, ge::FORMAT_ND},
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, inputType, ge::FORMAT_ND},
         {{{b, n, s, d}, {b, n, s, d}}, attentionType, ge::FORMAT_ND},
         {{{b, n, s}, {b, n, s}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, n, ceilQ, ceilKv}, {b, n, ceilQ, ceilKv}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND},

         // 将真实的内存地址传给框架，供 Tiling 内部解析
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, (void*)blockShapeData},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void*)actualSeqData},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void*)actualSeqKvData},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, (void*)scaleData},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, (void*)scaleData},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, (void*)scaleData},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, (void*)scaleData},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, (void*)scaleData},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, (void*)scaleData}},
        {// --- Output Info ---
         {{{b, n, s, d}, {b, n, s, d}}, outputType, ge::FORMAT_ND},
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, outputType, ge::FORMAT_ND},
         {{{b, n_kv, s_kv, d}, {b, n_kv, s_kv, d}}, outputType, ge::FORMAT_ND}},
        {// --- Attr Info ---

         {"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n_kv)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"quantMode",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode == -1 ? (isHif8 ? 5 : 0) : quantMode)}},
        &compileInfo, "3510", socInfo, 4096, deterministic);

    const uint64_t expectTilingKey = (isHif8 ? 2002UL : 1003UL) + (deterministic == 1 ? 8UL : 0UL);
    ExecuteTestCase(tilingContextPara, expectedResult, expectTilingKey);
}
} // namespace

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_block64_regression)
{
    RunArch35BnsdTilingCase(128, 128, 64, 64, ge::GRAPH_SUCCESS);
}

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_block64_deterministic)
{
    RunArch35BnsdTilingCase(128, 128, 64, 64, ge::GRAPH_SUCCESS, 2, false, 1);
}

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_hif8_block64_deterministic)
{
    RunArch35BnsdTilingCase(128, 128, 64, 64, ge::GRAPH_SUCCESS, 2, true, 1);
}

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_block16_with_sequence_tail)
{
    RunArch35BnsdTilingCase(70, 70, 16, 16, ge::GRAPH_SUCCESS);
}

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_asymmetric_block16x32)
{
    RunArch35BnsdTilingCase(70, 70, 16, 32, ge::GRAPH_SUCCESS);
}

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_rejects_non_multiple_of_16)
{
    RunArch35BnsdTilingCase(96, 96, 24, 16, ge::GRAPH_FAILED);
}

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_hif8_block64)
{
    RunArch35BnsdTilingCase(128, 128, 64, 64, ge::GRAPH_SUCCESS, 2, true);
}

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_hif8_rejects_block16)
{
    RunArch35BnsdTilingCase(96, 96, 16, 16, ge::GRAPH_FAILED, 2, true);
}

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_rejects_hif8_with_quant_mode_zero)
{
    RunArch35BnsdTilingCase(128, 128, 64, 64, ge::GRAPH_FAILED, 2, true, 0, 0);
}

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_rejects_fp16_with_quant_mode_five)
{
    RunArch35BnsdTilingCase(128, 128, 64, 64, ge::GRAPH_FAILED, 2, false, 0, 5);
}

TEST_F(BlockSparseAttentionGradTilingTest, tiling_arch35_rejects_unsupported_quant_mode)
{
    RunArch35BnsdTilingCase(128, 128, 64, 64, ge::GRAPH_FAILED, 2, true, 0, 6);
}
