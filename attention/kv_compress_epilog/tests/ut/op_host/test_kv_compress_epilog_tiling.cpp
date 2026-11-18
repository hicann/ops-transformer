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
#include "../../../op_host/kv_compress_epilog_tiling.h"
#include "../../../op_host/arch22/kv_compress_epilog_tiling_arch22.h"
#include "../../../op_host/arch35/kv_compress_epilog_tiling_arch35.h"

static optiling::KvCompressEpilogCompileInfo compileInfo{48, 196608};

class KvCompressEpilogTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvCompressEpilogTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvCompressEpilogTiling TearDown" << std::endl;
    }
};

TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_e8m0)
{
    gert::TilingContextPara optilingContextPara(
        "KvCompressEpilog",
        {
            // cache: 4D [blockNum, blockSize, 1, headDim] = [128,16,1,384], num_slots=2048
            {{{128, 16, 1, 384}, {128, 16, 1, 384}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{128, 16, 1, 384}, {128, 16, 1, 384}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
    ExecuteTestCase(optilingContextPara, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// mode2: rope hifloat8 静态量化 + nope FLOAT4_E2M1 per-group 动态量化。
// d=256 -> nope=192, 可被 16/32/64 整除; tiling key 应为 2。
static gert::TilingContextPara MakeMode2Para(int64_t groupSize)
{
    return gert::TilingContextPara("KvCompressEpilog",
                                   {
                                       // cache: 4D [blockNum, blockSize, 1, headDim] = [128,16,1,384], num_slots=2048
                                       {{{128, 16, 1, 384}, {128, 16, 1, 384}}, ge::DT_UINT8, ge::FORMAT_ND},
                                       {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},
                                       {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                   },
                                   {
                                       {{{128, 16, 1, 384}, {128, 16, 1, 384}}, ge::DT_UINT8, ge::FORMAT_ND},
                                   },
                                   {
                                       {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupSize)},
                                       {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                       {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                       {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(2.0f)},
                                   },
                                   &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
}

TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_hif8_fp4_group64)
{
    auto para = MakeMode2Para(64);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL); // 单一 tiling key 0
}

TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_hif8_fp4_group32)
{
    auto para = MakeMode2Para(32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL); // 单一 tiling key 0
}

TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_hif8_fp4_group16)
{
    auto para = MakeMode2Para(16);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL); // 单一 tiling key 0
}

// mode2 仅支持 group_size in {16,32,64}, 其它取值应 tiling 失败。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_hif8_fp4_bad_group)
{
    auto para = MakeMode2Para(128);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// ---------------------------------------------------------------------------
// Reading B + 4D-only 契约: cache 必须为 4D [blockNum, blockSize, 1, headDim]。
//   headDim(末维, 每 token 行宽) 由 cache 自身决定; 算子每行写出 kvCacheCol 字节:
//   - E8M0(mode1) 不补齐: kvCacheCol = concatCol;
//   - 其它模式 32B 对齐:  kvCacheCol = RoundUp(concatCol, 32);
//   单条校验(Reading B): headDim >= kvCacheCol (否则散写越界到下一行)。
// 下列用例统一使用 x=[bs,256] bf16(d=256), 通过 cache 末维直接控制 headDim。
//   d=256 时: scaleCol=CeilDiv(256-64,64)=3
//     mode1(E8M0) concatCol = (256-64)+128+3*1 = 323, kvCacheCol = 323(不补齐);
//     mode0(bf16)  concatCol = (256-64)+128+3*2 = 326, kvCacheCol = RoundUp(326,32)=352。
// cache 4D 用 [2048,1,1,headDim] (blockNum=2048, blockSize=1) 直接以末维控制 headDim。
// ---------------------------------------------------------------------------
static gert::TilingContextPara MakeHeadDimPara(int64_t quantMode, int64_t headDim, int64_t groupSize = 128)
{
    return gert::TilingContextPara("KvCompressEpilog",
                                   {
                                       // cache 4D, 末维=headDim
                                       {{{2048, 1, 1, headDim}, {2048, 1, 1, headDim}}, ge::DT_UINT8, ge::FORMAT_ND},
                                       {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND}, // x: d=256
                                       {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                   },
                                   {
                                       {{{2048, 1, 1, headDim}, {2048, 1, 1, headDim}}, ge::DT_UINT8, ge::FORMAT_ND},
                                   },
                                   {
                                       {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupSize)},
                                       {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                                       {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                       {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                   },
                                   &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
}

// (a) E8M0 + 非 32 对齐 headDim(=concatCol+1=324) -> 成功。
//     若仍按旧逻辑 RoundUp(concatCol,128)=384, 则 kvCacheCol(384) > headDim(324) 会触发越界校验失败;
//     成功本身即证明 E8M0 不补齐(kvCacheCol == concatCol == 323 <= 324)。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_e8m0_headdim_unaligned_ok)
{
    auto para = MakeHeadDimPara(/*quantMode=*/1, /*headDim=*/324);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// (b) mode0(group_bf16) + 32 对齐 kvCacheCol(=352) 且 headDim(384) > kvCacheCol -> 成功。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_mode0_headdim_aligned_ok)
{
    auto para = MakeHeadDimPara(/*quantMode=*/0, /*headDim=*/384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// (c) 错误: headDim(320) < kvCacheCol(=concatCol=323) -> GRAPH_FAILED (cache 行宽不足, 写出越界)。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_headdim_le_concat_fail)
{
    auto para = MakeHeadDimPara(/*quantMode=*/1, /*headDim=*/320);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// (d) 错误: 非 E8M0, concatCol(326) < headDim(340) < RoundUp(concatCol,32)(352) -> GRAPH_FAILED
//     (能容纳 concat 数据, 但 32B 对齐后 kvCacheCol(352) > headDim(340) 写出会越界)。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_mode0_pad_overflow_fail)
{
    auto para = MakeHeadDimPara(/*quantMode=*/0, /*headDim=*/340);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// (e) 边界成功: headDim == kvCacheCol。
//     E8M0 d=256 -> kvCacheCol=323, headDim=323 (恰好相等) -> 成功 (headDim>=kvCacheCol)。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_e8m0_headdim_eq_kvcachecol_ok)
{
    auto para = MakeHeadDimPara(/*quantMode=*/1, /*headDim=*/323);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// (e2) 边界成功: mode0 d=256 -> kvCacheCol=352, headDim=352 (恰好相等) -> 成功。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_mode0_headdim_eq_kvcachecol_ok)
{
    auto para = MakeHeadDimPara(/*quantMode=*/0, /*headDim=*/352);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::numeric_limits<uint64_t>::max());
}

// ---------------------------------------------------------------------------
// 4D-only 契约门禁: cache 逻辑 shape 必须恰为 4D [blockNum, blockSize, 1, headDim]。
// ---------------------------------------------------------------------------
// (f) 错误: 2D cache (旧 [num_slots, kvCacheCol] 布局) -> GRAPH_FAILED (维数 != 4)。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_cache_2d_rejected)
{
    gert::TilingContextPara para("KvCompressEpilog",
                                 {
                                     {{{2048, 384}, {2048, 384}}, ge::DT_UINT8, ge::FORMAT_ND}, // 2D cache, 不再支持
                                     {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2048, 384}, {2048, 384}}, ge::DT_UINT8, ge::FORMAT_ND},
                                 },
                                 {
                                     {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                     {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                     {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                     {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                 },
                                 &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// (g) 错误: 4D cache 但倒数第二维 != 1 ([128,16,2,384]) -> GRAPH_FAILED。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_cache_dim2_not_one_rejected)
{
    gert::TilingContextPara para(
        "KvCompressEpilog",
        {
            {{{128, 16, 2, 384}, {128, 16, 2, 384}}, ge::DT_UINT8, ge::FORMAT_ND}, // dim2=2, 非法
            {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{128, 16, 2, 384}, {128, 16, 2, 384}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, "Ascend950", compileInfo.coreNum, compileInfo.ubSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// ---------------------------------------------------------------------------
// arch22 (DAV_2201, SoC Ascend910B) quant_mode=1: rope bf16 位型直通 + nope per-group int8 动态量化。
//   x 必须 2D [bs, 512] (前 448 列 nope, 后 64 列 rope); quant_group_size 恒 64;
//   行格式 128B(rope) + 448B(nope int8) + 14B(scale bf16 x7) + 18B(pad) = kvCacheCol 608, headDim >= 608。
// ---------------------------------------------------------------------------
static gert::TilingContextPara MakeInt8Para(int64_t bs, int64_t d, int64_t headDim, int64_t groupSize,
                                            int64_t quantMode, ge::DataType slotDtype, const char* soc,
                                            int64_t blockNum = 128, int64_t blockSize = 16, int64_t dim2 = 1)
{
    return gert::TilingContextPara(
        "KvCompressEpilog",
        {
            {{{blockNum, blockSize, dim2, headDim}, {blockNum, blockSize, dim2, headDim}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{bs, d}, {bs, d}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{bs}, {bs}}, slotDtype, ge::FORMAT_ND},
        },
        {
            {{{blockNum, blockSize, dim2, headDim}, {blockNum, blockSize, dim2, headDim}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupSize)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
            {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        },
        &compileInfo, soc, compileInfo.coreNum, compileInfo.ubSize);
}

// (h) 正常路径: INT8 + slot int32, headDim=608 (=kvCacheCol), bs=1024 -> 成功, tilingKey=0。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_int8_int32)
{
    auto para = MakeInt8Para(1024, 512, 608, 64, 1, ge::DT_INT32, "Ascend910B");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// (i) 正常路径: INT8 + slot int64 -> 成功, tilingKey=0 (dtype 差异由 kernel 模板承载)。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_int8_int64)
{
    auto para = MakeInt8Para(1024, 512, 608, 64, 1, ge::DT_INT64, "Ascend910B");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// (j) 正常路径: bs=1 单核边界 -> 成功。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_int8_bs1)
{
    auto para = MakeInt8Para(1, 512, 608, 64, 1, ge::DT_INT32, "Ascend910B");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// (k) 正常路径: headDim=640 > kvCacheCol(608), 尾部保留 -> 成功。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_int8_headdim640)
{
    auto para = MakeInt8Para(1024, 512, 640, 64, 1, ge::DT_INT32, "Ascend910B");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// (l) 正常路径: blockNum 维非连续分页 cache (blockStride=10240 > blockSize*headDim=16*640) -> 成功。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_int8_noncontiguous_block)
{
    auto para = MakeInt8Para(1024, 512, 608, 64, 1, ge::DT_INT32, "Ascend910B", 5, 16, 1);
    // view shape [5,16,1,608], storage 行宽 640: strides = {16*640, 640, 640, 1}
    para.inputTensorDesc_[0].hasStride_ = true;
    para.inputTensorDesc_[0].stride_.SetDimNum(4);
    para.inputTensorDesc_[0].stride_.SetStride(0, 16 * 640);
    para.inputTensorDesc_[0].stride_.SetStride(1, 640);
    para.inputTensorDesc_[0].stride_.SetStride(2, 640);
    para.inputTensorDesc_[0].stride_.SetStride(3, 1);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// (m) 错误: d != 512 (d=448) -> GRAPH_FAILED。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_int8_d_not_512)
{
    auto para = MakeInt8Para(1024, 448, 608, 64, 1, ge::DT_INT32, "Ascend910B");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// (n) 错误: quant_group_size != 64 (group=32) -> GRAPH_FAILED。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_int8_group_not_64)
{
    auto para = MakeInt8Para(1024, 512, 608, 32, 1, ge::DT_INT32, "Ascend910B");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// (o) 错误: headDim(576) < kvCacheCol(608) -> GRAPH_FAILED。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_int8_headdim_lt_608)
{
    auto para = MakeInt8Para(1024, 512, 576, 64, 1, ge::DT_INT32, "Ascend910B");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// (p) 错误: cache dim2 != 1 -> GRAPH_FAILED。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_int8_dim2_not_one)
{
    auto para = MakeInt8Para(1024, 512, 608, 64, 1, ge::DT_INT32, "Ascend910B", 128, 16, 2);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// (q) 错误: arch22 上 quant_mode 0/2 显式报错 -> GRAPH_FAILED。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_mode0_rejected)
{
    auto para = MakeInt8Para(1024, 512, 608, 64, 0, ge::DT_INT32, "Ascend910B");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch22_mode2_rejected)
{
    auto para = MakeInt8Para(1024, 512, 608, 64, 2, ge::DT_INT32, "Ascend910B");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// (r) 错误: arch35 (Ascend950) 上未定义的 quant_mode 取值显式报错 -> GRAPH_FAILED。
TEST_F(KvCompressEpilogTiling, kv_compress_epilog_tiling_arch35_mode7_rejected)
{
    auto para = MakeInt8Para(1024, 512, 608, 64, 7, ge::DT_INT32, "Ascend950");
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
