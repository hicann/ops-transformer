/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstring>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/flash_attn_grad_tiling.h"

using namespace std;

static const char* A5SocInfo = R"JSON({
  "hardware_info": {
    "BT_SIZE": 0,
    "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false,
    "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true,
    "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 262144,
    "L2_SIZE": 134217728,
    "L1_SIZE": 524288,
    "L0A_SIZE": 65536,
    "L0B_SIZE": 65536,
    "L0C_SIZE": 262144,
    "vector_core_cnt": 40,
    "cube_core_cnt": 20,
    "socVersion": "Ascend950"
  }
})JSON";

// attn_mask 压缩掩码固定边长（fag_attr_checker 要求 [ATTN_MASK_DIM, ATTN_MASK_DIM]）
static constexpr int64_t ATTN_MASK_HW = 2048;
// Base metadata header (16 words) plus the FAG body (412 words).
static constexpr int64_t FAG_METADATA_WORDS = 428;

class FlashAttnGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// ---------------------------------------------------------------------------
// 辅助：构造 BSND 布局的 TilingContextPara（不带 attn_mask）
// ---------------------------------------------------------------------------
static optiling::FlashAttnGradCompileInfo MakeCompileInfo()
{
    optiling::FlashAttnGradCompileInfo compileInfo = {};
    compileInfo.aivNum = 64;
    compileInfo.aicNum = 32;
    compileInfo.l2CacheSize = 134217728;
    compileInfo.libapiWorkspaceSize = 0;
    return compileInfo;
}

static gert::TilingContextPara MakeBsndContext(optiling::FlashAttnGradCompileInfo* compileInfo, int64_t b, int64_t s1,
                                               int64_t s2, int64_t n1, int64_t n2, int64_t d, int64_t dv,
                                               int64_t maskMode = 0, const std::string& layout = "BSND",
                                               float scale = 0.0f, ge::DataType dtype = ge::DT_FLOAT16)
{
    return gert::TilingContextPara("FlashAttnGrad",
                                   {
                                       // q: [B, S1, N1, D]
                                       {{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND},
                                       // k: [B, S2, N2, D]
                                       {{{b, s2, n2, d}, {b, s2, n2, d}}, dtype, ge::FORMAT_ND},
                                       // v: [B, S2, N2, Dv]
                                       {{{b, s2, n2, dv}, {b, s2, n2, dv}}, dtype, ge::FORMAT_ND},
                                       // dout: [B, S1, N1, Dv]
                                       {{{b, s1, n1, dv}, {b, s1, n1, dv}}, dtype, ge::FORMAT_ND},
                                       // attn_out: [B, S1, N1, Dv]
                                       {{{b, s1, n1, dv}, {b, s1, n1, dv}}, dtype, ge::FORMAT_ND},
                                       // softmax_lse: [B, N1, S1]
                                       {{{b, n1, s1}, {b, n1, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // cu_seqlens_q: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cu_seqlens_kv: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // seqused_q: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // seqused_kv: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // sinks: optional
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // attn_mask: optional（需要 mask 的用例走内联构造）
                                       {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                       // metadata: required one-dimensional int32 tensor
                                       {{{FAG_METADATA_WORDS}, {FAG_METADATA_WORDS}}, ge::DT_INT32, ge::FORMAT_ND},
                                   },
                                   {
                                       // dq
                                       {{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND},
                                       // dk
                                       {{{b, s2, n2, d}, {b, s2, n2, d}}, dtype, ge::FORMAT_ND},
                                       // dv
                                       {{{b, s2, n2, dv}, {b, s2, n2, dv}}, dtype, ge::FORMAT_ND},
                                       // dsink
                                       {{{n1}, {n1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                   },
                                   {
                                       {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(scale)},
                                       {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode)},
                                       {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                       {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                       {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                       {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                       {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
                                       {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
                                       {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
                                   },
                                   compileInfo, "Ascend950", A5SocInfo, 81920);
}

// ---------------------------------------------------------------------------
// 基础：BSND D=128 无掩码
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_D128_NoMask)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 256, 256, 4, 4, 128, 128, 0, "BSND");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// D=100（padWs 触发：100 % 8 != 0）
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_D100_PadWs)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 256, 256, 4, 4, 100, 100, 0, "BSND");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// D=72（padWs 不触发：72 % 8 == 0）
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_D72_NoPad)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 256, 256, 4, 4, 72, 72, 0, "BSND");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// D=192（192 档，前移激活但 padWs 不触发）
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_D192)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 256, 256, 4, 4, 192, 192, 0, "BSND");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// D=192 Dv=128（唯一合法 D≠Dv 组合）
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_D192_Dv128)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 256, 256, 4, 4, 192, 128, 0, "BSND");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// causal mask（mask_mode=3）：attn_mask 必须提供且为 [2048,2048] INT8，
// win_left/win_right 必须为 -1；S1<=S2 走 FAG_SPARSE_CASUAL 路由。
// 采用内联字面量构造（同 BNSD/WithSink 用例模式），S1=256, S2=512
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_D128_Causal)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "FlashAttnGrad",
        {
            // q: [B, S1, N1, D]
            {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: [B, S2, N2, D]
            {{{1, 512, 4, 128}, {1, 512, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: [B, S2, N2, Dv]
            {{{1, 512, 4, 128}, {1, 512, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dout: [B, S1, N1, Dv]
            {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attn_out: [B, S1, N1, Dv]
            {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // softmax_lse: [B, N1, S1]
            {{{1, 4, 256}, {1, 4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // cu_seqlens_q: optional
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // cu_seqlens_kv: optional
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // seqused_q: optional
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // seqused_kv: optional
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // sinks: optional
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attn_mask: [2048,2048] INT8（mask_mode=3 必需）
            {{{ATTN_MASK_HW, ATTN_MASK_HW}, {ATTN_MASK_HW, ATTN_MASK_HW}}, ge::DT_INT8, ge::FORMAT_ND},
            // metadata: required one-dimensional int32 tensor
            {{{FAG_METADATA_WORDS}, {FAG_METADATA_WORDS}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // dq
            {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk
            {{{1, 512, 4, 128}, {1, 512, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dv
            {{{1, 512, 4, 128}, {1, 512, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dsink
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
        },
        &compileInfo, "Ascend950", A5SocInfo, 81920);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 非法：D > 768（checker 应拦截）
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Reject_D_Exceeds_Max)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 256, 256, 4, 4, 769, 769, 0, "BSND");
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 非法：D != Dv（非 192/128 组合）
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Reject_D_Neq_Dv)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 256, 256, 4, 4, 128, 64, 0, "BSND");
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// GQA：N1=8, N2=2, G=4
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_GQA_G4_D128)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 256, 256, 8, 2, 128, 128, 0, "BSND");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// S2=128 恰好走 BN2 模板
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_S128_BN2)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 128, 128, 4, 4, 128, 128, 0, "BSND");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// S2=129 强制走 GS1S2 模板
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_S129_GS1S2)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 128, 129, 4, 4, 128, 128, 0, "BSND");
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// dtype 矩阵：bf16（接口支持 float16/bfloat16；dq/dk/dv dtype 须与 q 一致）
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_D128_Bf16)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndContext(&compileInfo, 1, 256, 256, 4, 4, 128, 128, 0, "BSND", 0.0f, ge::DT_BF16);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// BNSD 布局：q [B,N1,S1,D]、k/v [B,N2,S2,D]、dout/attn_out [B,N1,S1,Dv]、
// lse [B,N1,S1]（BSND/BNSD 场景 lse 同构）
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bnsd_D128_NoMask)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("FlashAttnGrad",
                                 {
                                     // q: [B, N1, S1, D]
                                     {{{1, 4, 256, 128}, {1, 4, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // k: [B, N2, S2, D]
                                     {{{1, 4, 256, 128}, {1, 4, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // v: [B, N2, S2, Dv]
                                     {{{1, 4, 256, 128}, {1, 4, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // dout: [B, N1, S1, Dv]
                                     {{{1, 4, 256, 128}, {1, 4, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // attn_out: [B, N1, S1, Dv]
                                     {{{1, 4, 256, 128}, {1, 4, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // softmax_lse: [B, N1, S1]
                                     {{{1, 4, 256}, {1, 4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     // cu_seqlens_q: optional
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // cu_seqlens_kv: optional
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // seqused_q: optional
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // seqused_kv: optional
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // sinks: optional
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     // attn_mask: optional
                                     {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                     // metadata: required one-dimensional int32 tensor
                                     {{{FAG_METADATA_WORDS}, {FAG_METADATA_WORDS}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     // dq: [B, N1, S1, D]
                                     {{{1, 4, 256, 128}, {1, 4, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // dk: [B, N2, S2, D]
                                     {{{1, 4, 256, 128}, {1, 4, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // dv: [B, N2, S2, Dv]
                                     {{{1, 4, 256, 128}, {1, 4, 256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // dsink
                                     {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                     {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                     {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                     {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                     {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                     {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
                                     {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
                                     {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
                                 },
                                 &compileInfo, "Ascend950", A5SocInfo, 81920);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 可选输入交互：sinks [N1] FP32 提供则成功（sink 由 tilingkey bit15 携带）
// ---------------------------------------------------------------------------
TEST_F(FlashAttnGradTiling, Arch35_Bsnd_D128_WithSink)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("FlashAttnGrad",
                                 {
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // softmax_lse: [B, N1, S1]
                                     {{{1, 4, 256}, {1, 4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // sinks: [N1] FP32
                                     {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                     // metadata: required one-dimensional int32 tensor
                                     {{{FAG_METADATA_WORDS}, {FAG_METADATA_WORDS}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // dsink: [N1]
                                     {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f)},
                                     {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                     {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                     {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                     {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                     {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                 },
                                 &compileInfo, "Ascend950", A5SocInfo, 81920);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}
