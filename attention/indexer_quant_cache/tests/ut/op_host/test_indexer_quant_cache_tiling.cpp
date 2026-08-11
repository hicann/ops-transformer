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
#include "tiling/platform/platform_ascendc.h"
#include "tiling_case_executor.h"

// ---------------------------------------------------------------------------
// 4D-only 契约 (mirror kv_compress_epilog):
//   cache 与 cache_scale 逻辑 shape 必须恰为 4D [blockNum, blockSize, 1, headDim]。
//   - cache headDim(末维, 每 token 行宽) >= 每行写出的量化-x 长度:
//       mode0/1/2(fp8/uint8 元素==字节) -> >= d; mode3 MX-FP4(cache 末维为 fp4 元素) -> >= d。
//   - cache_scale headDim >= scaleCol: mode0/3(MX) = CeilDiv(d,32); mode1/2 = 1。
//   - 所有模式(0/1/2/3)均校验 cache_scale; mode2(HiFloat8) 无例外(scaleCol=1, 末维须 >= 1)。
// 下列用例 x=[1024,128] (d=128): scaleCol mode0/3=4, mode1/2=1; cacheCol mode3=64, 其余=128。
// ---------------------------------------------------------------------------

class IndexerQuantCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "IndexerQuantCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IndexerQuantCacheTiling TearDown" << std::endl;
    }
};

// cache/cache_scale 均为 4D [blockNum, blockSize, 1, headDim]; num_slots = 2048。
// mode1 Normal: scaleCol=1, cache headDim>=d(128), scale headDim>=1。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_normal)
{
    gert::TilingContextPara optilingContextPara("IndexerQuantCache",
                                                {
                                                    {{{2048, 1, 1, 128}, {2048, 1, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                    {{{2048, 1, 1, 1}, {2048, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                },
                                                nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// mode0 MX-FP8: scaleCol=CeilDiv(128,32)=4, scale headDim 必须 >= 4。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_mxfp8)
{
    gert::TilingContextPara optilingContextPara("IndexerQuantCache",
                                                {
                                                    {{{2048, 1, 1, 128}, {2048, 1, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                    {{{2048, 1, 1, 4}, {2048, 1, 1, 4}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                                    {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                    {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                    {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                },
                                                nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// mode3 MX-FP4: cache 末维为 fp4 元素(>=d=128), scaleCol=4。cacheCol(byte)=64, rowStride=128/2=64>=64。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_mxfp4)
{
    gert::TilingContextPara optilingContextPara("IndexerQuantCache",
                                                {
                                                    {{{2048, 1, 1, 128}, {2048, 1, 1, 128}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
                                                    {{{2048, 1, 1, 4}, {2048, 1, 1, 4}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                                    {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{}, {}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
                                                    {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                },
                                                nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// mode3 MX-FP4: cache headDim(fp4元素)为奇数(129) -> GRAPH_FAILED。
// FP4每字节打包2个元素, headDim为奇数时逐行起点会交替落在字节高/低半字节, 现有整字节寻址
// 实现无法表达该偏移, 故直接拒绝而非产出错误数据。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_mxfp4_odd_headdim_rejected)
{
    gert::TilingContextPara optilingContextPara("IndexerQuantCache",
                                                {
                                                    {{{2048, 1, 1, 129}, {2048, 1, 1, 129}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
                                                    {{{2048, 1, 1, 4}, {2048, 1, 1, 4}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                                    {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{}, {}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
                                                    {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                    {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                },
                                                nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// cache headDim > d (256 > 128) -> 成功 (Reading B: 行可更宽, 只写 d, 其余保留)。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_cache_wide_headdim_ok)
{
    gert::TilingContextPara optilingContextPara("IndexerQuantCache",
                                                {
                                                    {{{2048, 1, 1, 256}, {2048, 1, 1, 256}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                    {{{2048, 1, 1, 1}, {2048, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                    {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                    {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                },
                                                nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// HIFLOAT8(mode2) 无例外: cache 4D 合法, cache_scale 为 2D(非 4D) -> GRAPH_FAILED。
// kernel 在 mode2 下仍为每 token 散写 1 个 float scale, 故必须校验 cache_scale 为合法 4D。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_hifloat_scale_non4d_rejected)
{
    gert::TilingContextPara optilingContextPara("IndexerQuantCache",
                                                {
                                                    {{{2048, 1, 1, 128}, {2048, 1, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                    {{{2048, 1}, {2048, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 2D scale, mode2 现已校验 -> 拒绝
                                                    {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                },
                                                nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// HIFLOAT8(mode2): cache 4D 合法 + cache_scale 合法 4D (scaleCol=1, 末维>=1) -> 成功。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_hifloat_scale_4d_ok)
{
    gert::TilingContextPara optilingContextPara("IndexerQuantCache",
                                                {
                                                    {{{2048, 1, 1, 128}, {2048, 1, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                    {{{2048, 1, 1, 1}, {2048, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 4D scale, scaleCol=1
                                                    {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                    {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                },
                                                {
                                                    {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                    {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                },
                                                {
                                                    {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                    {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                    {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                },
                                                nullptr, "Ascend950");
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// ---------------------------------------------------------------------------
// 拒绝用例
// ---------------------------------------------------------------------------
// cache 为 2D(旧 [num_slots, d] 布局) -> GRAPH_FAILED。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_cache_2d_rejected)
{
    gert::TilingContextPara para("IndexerQuantCache",
                                 {
                                     {{{2048, 128}, {2048, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{2048, 1, 1, 1}, {2048, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                     {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                     {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                 },
                                 nullptr, "Ascend950");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// cache 4D 但倒数第二维 != 1 ([128,16,2,128]) -> GRAPH_FAILED。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_cache_dim2_not_one_rejected)
{
    gert::TilingContextPara para("IndexerQuantCache",
                                 {
                                     {{{128, 16, 2, 128}, {128, 16, 2, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{128, 16, 1, 1}, {128, 16, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                     {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                     {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                 },
                                 nullptr, "Ascend950");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// cache headDim(64) < d(128) -> GRAPH_FAILED (行宽不足, 写出越界)。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_cache_headdim_lt_d_rejected)
{
    gert::TilingContextPara para("IndexerQuantCache",
                                 {
                                     {{{2048, 1, 1, 64}, {2048, 1, 1, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{2048, 1, 1, 1}, {2048, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                     {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                     {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                 },
                                 nullptr, "Ascend950");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// mode0 MX-FP8: cache_scale headDim(2) < scaleCol(4) -> GRAPH_FAILED。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_scale_headdim_lt_scalecol_rejected)
{
    gert::TilingContextPara para("IndexerQuantCache",
                                 {
                                     {{{2048, 1, 1, 128}, {2048, 1, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{2048, 1, 1, 2}, {2048, 1, 1, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}, // 2 < scaleCol(4)
                                     {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                 },
                                 {
                                     {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                     {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                 },
                                 nullptr, "Ascend950");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 非 HIFLOAT8(mode1) 下 cache_scale 为 2D(非 4D) -> GRAPH_FAILED。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_scale_non4d_rejected)
{
    gert::TilingContextPara para("IndexerQuantCache",
                                 {
                                     {{{2048, 1, 1, 128}, {2048, 1, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{2048, 1}, {2048, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 2D scale, mode1 必须 4D
                                     {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                     {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                     {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                 },
                                 nullptr, "Ascend950");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// d tail 维上界 (mirror kv_compress_epilog D_LENGTH_FULL_LOAD=8192):
//   d=8192 (32 对齐, 边界) -> GRAPH_SUCCESS。mode0 MX-FP8: scaleCol=CeilDiv(8192,32)=256。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_d_eq_8192_ok)
{
    gert::TilingContextPara para("IndexerQuantCache",
                                 {
                                     {{{2048, 1, 1, 8192}, {2048, 1, 1, 8192}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{2048, 1, 1, 256}, {2048, 1, 1, 256}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                     {{{1024, 8192}, {1024, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                 },
                                 {
                                     {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                     {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                 },
                                 nullptr, "Ascend950");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// d=8224 (32 对齐: 8224=32*257, 但 > 8192) -> GRAPH_FAILED (超过 D_LENGTH_FULL_LOAD 上界)。
TEST_F(IndexerQuantCacheTiling, indexer_quant_cache_tiling_d_gt_8192_rejected)
{
    gert::TilingContextPara para("IndexerQuantCache",
                                 {
                                     {{{2048, 1, 1, 8224}, {2048, 1, 1, 8224}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{2048, 1, 1, 257}, {2048, 1, 1, 257}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                     {{{1024, 8224}, {1024, 8224}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
                                 },
                                 {
                                     {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                     {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                 },
                                 nullptr, "Ascend950");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
