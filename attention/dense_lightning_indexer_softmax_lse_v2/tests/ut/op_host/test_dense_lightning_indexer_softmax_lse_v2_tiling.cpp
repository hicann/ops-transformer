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
#include "../../../op_host/dense_lightning_indexer_softmax_lse_v2_tiling.h"

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

class DenseLISoftmaxLseV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 全局 compileInfo：builder 要求非空指针（传 nullptr 会损坏 ComputeNodeInfo 段）。
// dense 的 tiling 不读取 compileInfo 内容，空结构体即可。
static optiling::DenseLISoftmaxLseV2CompileInfo g_compileInfo = {};

// ---------------------------------------------------------------------------
// 辅助：构造 BSND 布局上下文。
// tiling ParamCheck 硬约束：d==128、n2==1、n1∈[1,128]、cmpRatio∈[1,128]、
// maskMode∈{0,3}、layoutQ==layoutK、weight[B,S1,N1]、output[B,N2,S1]。
// ---------------------------------------------------------------------------
static gert::TilingContextPara MakeBsndContext(int64_t b, int64_t s1, int64_t s2, int64_t n1, int64_t n2, int64_t d,
                                               int64_t maskMode = 0, int64_t cmpRatio = 1,
                                               const std::string& layoutQ = "BSND", const std::string& layoutK = "BSND",
                                               ge::DataType dtype = ge::DT_FLOAT16)
{
    return gert::TilingContextPara("DenseLightningIndexerSoftmaxLseV2",
                                   {
                                       // query_index: [B, S1, N1, D]
                                       {{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND},
                                       // key_index: [B, S2, N2, D]
                                       {{{b, s2, n2, d}, {b, s2, n2, d}}, dtype, ge::FORMAT_ND},
                                       // weight: [B, S1, N1] fp32（tiling 按 dim1=S1、dim2=N1 读取）
                                       {{{b, s1, n1}, {b, s1, n1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // cu_seq_lens_q: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cu_seq_lens_k: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // seq_used_q: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // seq_used_k: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cmp_residual_k: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // metadata: 必选，1D INT32
                                       {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                   },
                                   {
                                       // softmax_lse: [B, N2, S1] fp32
                                       {{{b, n2, s1}, {b, n2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                   },
                                   {
                                       {"layoutQ", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
                                       {"layoutK", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutK)},
                                       {"maskMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode)},
                                       {"cmpRatio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cmpRatio)},
                                   },
                                   &g_compileInfo, "Ascend950", A5SocInfo, 4096);
}

// ---------------------------------------------------------------------------
// 基础：BSND D=128 N2=1 无掩码
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Bsnd_D128_NoMask)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 128);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 小尺寸：B=2, S=128
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Bsnd_Small_S128)
{
    auto para = MakeBsndContext(2, 128, 128, 2, 1, 128);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 非对称长度：S1=128, S2=256
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Bsnd_Asym_S1_128_S2_256)
{
    auto para = MakeBsndContext(1, 128, 256, 4, 1, 128);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// causal mask（mask_mode=3）
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Bsnd_D128_Causal)
{
    auto para = MakeBsndContext(1, 256, 512, 4, 1, 128, 3);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// cmpRatio=4（压缩比 4 倍）
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Bsnd_D128_CmpRatio4)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 128, 0, 4);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// GQA：N1=8, N2=1（tiling 要求 N2 恒为 1，N1>N2 即 GQA）
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Bsnd_GQA_N1_8)
{
    auto para = MakeBsndContext(1, 256, 256, 8, 1, 128);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 非法：维度数不匹配（q 是 3 维，BSND 应为 4 维）
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Reject_WrongDimNum)
{
    gert::TilingContextPara para("DenseLightningIndexerSoftmaxLseV2",
                                 {
                                     // query_index: 3 维（错误）
                                     {{{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // key_index: 4 维
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // weight: [B, S1, N1]
                                     {{{1, 256, 4}, {1, 256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // metadata: 必选
                                     {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1, 256}, {1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutQ", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"layoutK", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"maskMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"cmpRatio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &g_compileInfo, "Ascend950", A5SocInfo, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 非法：head_dim != 128（ParamCheck L43 约束回归）
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Reject_D_Not128)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 64);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 非法：N2 != 1（ParamCheck L57 约束回归）
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Reject_N2_Not1)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 4, 128);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 辅助：TND 布局上下文（query [T1,N1,D]、key [T2,N2,D]、weight [T1,N1]、
// output [N2,T1]、cu_seq_lens_q/k [B+1] 必选、metadata 必选）
// ---------------------------------------------------------------------------
static gert::TilingContextPara MakeTndContext(int64_t b, int64_t t1, int64_t t2, int64_t n1, int64_t n2, int64_t d,
                                              int64_t maskMode = 0, int64_t cmpRatio = 1,
                                              ge::DataType dtype = ge::DT_FLOAT16)
{
    return gert::TilingContextPara("DenseLightningIndexerSoftmaxLseV2",
                                   {
                                       // query_index: [T1, N1, D]
                                       {{{t1, n1, d}, {t1, n1, d}}, dtype, ge::FORMAT_ND},
                                       // key_index: [T2, N2, D]
                                       {{{t2, n2, d}, {t2, n2, d}}, dtype, ge::FORMAT_ND},
                                       // weight: [T1, N1] fp32
                                       {{{t1, n1}, {t1, n1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // cu_seq_lens_q: [B+1] TND 必选
                                       {{{b + 1}, {b + 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cu_seq_lens_k: [B+1] TND 必选
                                       {{{b + 1}, {b + 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // seq_used_q: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // seq_used_k: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cmp_residual_k: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // metadata: 必选，1D INT32
                                       {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                   },
                                   {
                                       // softmax_lse: [N2, T1] fp32
                                       {{{n2, t1}, {n2, t1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                   },
                                   {
                                       {"layoutQ", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                       {"layoutK", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                       {"maskMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode)},
                                       {"cmpRatio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cmpRatio)},
                                   },
                                   &g_compileInfo, "Ascend950", A5SocInfo, 4096);
}

// ---------------------------------------------------------------------------
// dtype 矩阵：bf16
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Bsnd_D128_Bf16)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 128, 0, 1, "BSND", "BSND", ge::DT_BF16);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// TND 布局：B=2、T1=512、T2=384（非对称 token 数）
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Tnd_D128_NoMask)
{
    auto para = MakeTndContext(2, 512, 384, 4, 1, 128);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 可选输入交互：maskMode=3 + cmpRatio=4 时 cmp_residual_k [B] 必选——提供则成功
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Bsnd_Causal_Cmp4_WithResidual)
{
    gert::TilingContextPara para("DenseLightningIndexerSoftmaxLseV2",
                                 {
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 512, 1, 128}, {1, 512, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 4}, {1, 256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // cmp_residual_k: [B] INT32（maskMode=3 且 cmpRatio>1 必选）
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // metadata: 必选，1D INT32
                                     {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1, 256}, {1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutQ", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"layoutK", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"maskMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                     {"cmpRatio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                 },
                                 &g_compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 可选输入缺失：maskMode=3 + cmpRatio=4 但未提供 cmp_residual_k → 拒绝
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Reject_Causal_Cmp4_NoResidual)
{
    auto para = MakeBsndContext(1, 256, 512, 4, 1, 128, 3, 4);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 可选输入交互：seq_used_q [B] 提供则成功
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Bsnd_WithSequsedQ)
{
    gert::TilingContextPara para("DenseLightningIndexerSoftmaxLseV2",
                                 {
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 4}, {1, 256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // seq_used_q: [B] INT32
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // metadata: 必选，1D INT32
                                     {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1, 256}, {1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutQ", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"layoutK", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"maskMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"cmpRatio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &g_compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 必选输入缺失：metadata 未提供 → 拒绝（L321 约束回归）
// ---------------------------------------------------------------------------
TEST_F(DenseLISoftmaxLseV2Tiling, Arch35_Reject_NoMetadata)
{
    gert::TilingContextPara para("DenseLightningIndexerSoftmaxLseV2",
                                 {
                                     {{{1, 256, 4, 128}, {1, 256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 4}, {1, 256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // metadata: 未提供（空）→ 应被拒绝
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 1, 256}, {1, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutQ", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"layoutK", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"maskMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"cmpRatio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 &g_compileInfo, "Ascend950", A5SocInfo, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}
