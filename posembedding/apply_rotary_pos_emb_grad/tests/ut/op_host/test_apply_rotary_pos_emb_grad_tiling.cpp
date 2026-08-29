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
#include "../../../op_host/apply_rotary_pos_emb_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class ApplyRotaryPosEmbGradBabTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRotaryPosEmbGradBabTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ApplyRotaryPosEmbGradBabTiling TearDown" << std::endl;
    }
};

// dCosFlag=0 (4 inputs, 2 outputs) — tilingKey=41281
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 1; // dcos0: TilingReduce 跳过 → EMPTY pattern
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 128 4 4 2 1 22 3 3 1 4 4 44 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp32_bsnd_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 1; // dcos0: TilingReduce 跳过 → EMPTY pattern
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 128 4 4 2 1 22 3 3 1 4 4 44 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

// dCosFlag=1 (6 inputs, 4 outputs) — tilingKey=17179910465
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_dcos1)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17179910465;
    string expectTilingData = "2 64 1 1 1 1 1 8192 55808 6912 137438953520 4467570830351532032 1 1 1 1 1 "
                              "2 64 4 128 65536 65536 65536 65536 65536 32768 512 128 1 8192 8192 8192 8192 8192 8192 "
                              "128 128 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 "
                              "2 64 4 128 65536 65536 65536 65536 65536 32768 512 128 1 "
                              "2 64 128 4 4 2 1 22 3 3 1 4 4 44 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

// AB template, dCosFlag=1 (SBND layout, 6 inputs, 4 outputs) — covers AB -> TilingReduce path
TEST_F(ApplyRotaryPosEmbGradBabTiling, ab_fp16_sbnd_dcos1)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{64, 2, 4, 128}, {64, 2, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 2, 4, 128}, {64, 2, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 2, 4, 128}, {64, 2, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 2, 4, 128}, {64, 2, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 2, 4, 128}, {64, 2, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 2, 4, 128}, {64, 2, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1, 1, 128}, {64, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17246988369;
    string expectTilingData =
        "2 64 1 1 1 1 1 8192 55808 6912 137438953520 4467570830351532032 64 8 128 0 0 "
        "0 0 0 0 1024 128 1 0 0 0 0 0 0 128 128 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 64 8 128 0 0 0 0 0 0 1024 128 1 0 0 0 "
        "0 0 0 "
        "2 64 128 4 4 0 0 0 0 0 0 0 0 0 0 0 0 2 64 128 8 8 128 2 32 2 2 1 8 8 2 8 32 0 4294967297 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_bf16_bsnd_dcos1)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17179910465;
    string expectTilingData = "2 64 1 1 1 1 1 8192 55808 6912 137438953520 4467570830351532032 1 1 1 1 1 "
                              "2 64 4 128 65536 65536 65536 65536 65536 32768 512 128 1 8192 8192 8192 8192 8192 8192 "
                              "128 128 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 "
                              "2 64 4 128 65536 65536 65536 65536 65536 32768 512 128 1 "
                              "2 64 128 4 4 2 1 22 3 3 1 4 4 44 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

// nQ != nK — maxN=8
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_nq_neq_nk)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17179910465;
    string expectTilingData = "2 64 1 1 1 1 1 8192 57600 3584 137438953520 4431542033332568064 1 1 1 1 1 "
                              "2 64 8 128 131072 131072 131072 131072 131072 65536 1024 128 1 8192 8192 8192 8192 8192 "
                              "8192 128 128 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 "
                              "2 64 8 128 131072 131072 131072 131072 131072 65536 1024 128 1 "
                              "2 64 128 8 4 2 1 22 3 3 1 8 8 44 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

// TND (3D) — B=1
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_tnd_dcos1)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17179879505;
    string expectTilingData = "2 64 1 1 1 1 1 8192 52736 13056 137438953520 4503599627370496000 64 4 128 0 0 0 0 0 0 "
                              "512 128 1 0 0 0 0 0 0 128 128 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 "
                              "64 4 128 0 0 0 0 0 0 512 128 1 0 0 0 0 0 0 1 "
                              "64 128 4 4 1 1 32 2 2 1 4 4 32 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

// D 非对齐 — D=48
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_d48_unaligned)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17179910465;
    string expectTilingData = "1 32 2 1 1 1 1 3072 55808 6912 137438953520 4467570830351532032 1 1 1 1 1 "
                              "2 64 4 48 24576 24576 24576 24576 24576 12288 192 48 1 3072 3072 3072 3072 3072 3072 48 "
                              "48 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 "
                              "2 64 4 48 24576 24576 24576 24576 24576 12288 192 48 1 "
                              "2 64 48 4 4 2 1 22 3 3 1 4 4 44 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_d48_unaligned_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 1; // dcos0: TilingReduce 跳过 → EMPTY pattern
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 48 4 4 2 1 22 3 3 1 4 4 44 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, invalid_layout)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 2, 128}, {2, 64, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 2, 128}, {2, 64, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, invalid_layout_bsnd_with_3d_input)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, invalid_layout_tnd_with_4d_input)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 2, 128}, {2, 64, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 2, 128}, {2, 64, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, invalid_half_mode_odd_d)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 127}, {2, 64, 4, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 2, 127}, {2, 64, 2, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 127}, {1, 64, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 64, 1, 127}, {1, 64, 1, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 127}, {2, 64, 4, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 2, 127}, {2, 64, 2, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, ab_fp16_bsnd_d48_unaligned)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 1, 48}, {2, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 1, 48}, {2, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 1, 48}, {2, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 64, 1, 48}, {2, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 17246988369;
    string expectTilingData =
        "2 64 2 1 1 1 1 6144 52736 13056 137438953520 4503599627370496000 128 4 48 0 0 0 0 0 0 192 48 1 0 0 0 0 0 0 48 "
        "48 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 128 4 48 0 0 0 0 0 0 192 48 1 0 0 0 0 0 0 2 64 48 4 4 0 0 0 0 0 0 0 0 0 0 "
        "0 0 2 64 48 4 4 64 2 43 3 2 1 4 4 3 4 43 0 4294967297 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, ab_fp32_bsnd_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 67108865; // dcos0: TilingReduce 跳过 → EMPTY pattern
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 64 128 4 4 0 0 0 0 0 0 0 0 0 0 0 0 2 64 "
                              "128 4 4 128 2 43 3 2 1 4 4 3 4 43 0 4294967296 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}

TEST_F(ApplyRotaryPosEmbGradBabTiling, a_fp32_bsnd_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
                                 {
                                     {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &compileInfo, "Ascend950", 48, 253952);
    uint64_t expectTilingKey = 134217729; // dcos0: TilingReduce 跳过 → EMPTY pattern
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 128 1 128 1 1 43 3 1 1 3 1 1 1 43 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12884901888 ";
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, {});
}
