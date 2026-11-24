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
#include "../../../op_host/sparse_flash_mla_softmax_l1_norm_tiling.h"

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

static const char* A2SocInfo = R"JSON({
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
    "socVersion": "Ascend910B3"
  }
})JSON";

// Atlas A3 uses the same Arch22 kernel as A2, but has a distinct SOC string.
static const char* A3SocInfo = R"JSON({
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
    "socVersion": "Ascend910_93"
  }
})JSON";

class SparseMlaSoftmaxL1NormTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 全局 compileInfo：builder 要求非空指针（传 nullptr 会损坏 ComputeNodeInfo 段）。
// UT 路径 platformInfo 恒非空，tiling 不读取 compileInfo 内容，空结构体即可。
static optiling::smlasoftmaxl1norm::SparseFlashMlaSoftmaxL1NormCompileInfo g_compileInfo = {};

// ---------------------------------------------------------------------------
// 辅助：构造 BSND 布局上下文。
// tiling 硬约束：D∈{256,512}、N2==1、G∈[1,128]、maskMode∈{0,3}、cmpRatio∈[1,128]、
// lse 为 4 维 [B, N2, Sq, G]、BSND 下 cu_seqlens 必须为空。
// ---------------------------------------------------------------------------
static gert::TilingContextPara MakeBsndContext(int64_t b, int64_t sq, int64_t sk, int64_t g, int64_t n2, int64_t d,
                                               int64_t maskMode = 0, int64_t cmpRatio = 1,
                                               const std::string& layoutQ = "BSND", const std::string& layoutK = "BSND",
                                               float softmaxScale = 1.0f, int64_t maxSeqlenK = 0,
                                               ge::DataType dtype = ge::DT_FLOAT16,
                                               const char* socVersion = "Ascend950", const char* socInfo = A5SocInfo)
{
    return gert::TilingContextPara("SparseFlashMlaSoftmaxL1Norm",
                                   {
                                       // q: [B, Sq, G, D]
                                       {{{b, sq, g, d}, {b, sq, g, d}}, dtype, ge::FORMAT_ND},
                                       // k: [B, Sk, N2, D]
                                       {{{b, sk, n2, d}, {b, sk, n2, d}}, dtype, ge::FORMAT_ND},
                                       // softmax_lse: [B, N2, Sq, G] fp32（4 维）
                                       {{{b, n2, sq, g}, {b, n2, sq, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // sparse_indices: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cu_seqlens_q: BSND 下必须为空
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cu_seqlens_k: BSND 下必须为空
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // seqused_q: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // seqused_k: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cmp_residual_k: optional（maskMode=3 且 cmpRatio>1 时必选）
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // topk_length: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // metadata: 必选，1D INT32
                                       {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                   },
                                   {
                                       // softmax_l1_norm: [B, Sq, G, D] fp32
                                       {{{b, sq, g, d}, {b, sq, g, d}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                   },
                                   {
                                       {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(softmaxScale)},
                                       {"max_seqlen_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxSeqlenK)},
                                       {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cmpRatio)},
                                       {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode)},
                                       {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
                                       {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutK)},
                                   },
                                   &g_compileInfo, socVersion, socInfo, 4096);
}

// ---------------------------------------------------------------------------
// 基础：BSND D=512 N2=1 无掩码
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Bsnd_D512_NoMask)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 512);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// D=256 is supported by the native A2/A3 kernel.
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch22_Bsnd_D256_NoMask)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 256, 0, 1, "BSND", "BSND", 1.0f, 0, ge::DT_FLOAT16, "Ascend910B3",
                                A2SocInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

TEST_F(SparseMlaSoftmaxL1NormTiling, Arch22_A3_Bsnd_D256_NoMask)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 256, 0, 1, "BSND", "BSND", 1.0f, 0, ge::DT_FLOAT16, "Ascend910_93",
                                A3SocInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Reject_D256)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 256);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 小尺寸：B=2, Sq=Sk=128
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Bsnd_Small_S128)
{
    auto para = MakeBsndContext(2, 128, 128, 2, 1, 512);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 非对称长度：Sq=128, Sk=256
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Bsnd_Asym_Sq128_Sk256)
{
    auto para = MakeBsndContext(1, 128, 256, 4, 1, 512);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// causal mask（mask_mode=3，cmpRatio=1 无需 cmp_residual_k）
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Bsnd_D512_Causal)
{
    auto para = MakeBsndContext(1, 256, 512, 4, 1, 512, 3);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// cmpRatio=4（压缩比 4 倍，maskMode=0 无需 cmp_residual_k）
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Bsnd_D512_CmpRatio4)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 512, 0, 4);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// GQA：G=8, N2=1（tiling 要求 N2 恒为 1）
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Bsnd_GQA_G8)
{
    auto para = MakeBsndContext(1, 256, 256, 8, 1, 512);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// G=128 上边界（N1∈[1,128]）
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Bsnd_G128_Boundary)
{
    auto para = MakeBsndContext(1, 128, 128, 128, 1, 512);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 非法：layout_q 不支持（BNSD 应被 ValidateLayout 拦截）
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Reject_UnsupportedLayout)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 512, 0, 1, "BNSD", "BSND");
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 非法：D 不在 {256,512}（ValidateShapeRanges 约束回归）
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Reject_D_Not512)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 128);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 非法：N2 != 1（ValidateShapeRanges 约束回归）
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Reject_N2_Not1)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 4, 512);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 辅助：TND 布局上下文（query [T1,G,D]、key [T2,N2,D]、lse [N2,T1,G]、
// cu_seqlens_q/k [B+1] 必选、metadata 必选）
// ---------------------------------------------------------------------------
static gert::TilingContextPara MakeTndContext(int64_t b, int64_t t1, int64_t t2, int64_t g, int64_t n2, int64_t d,
                                              int64_t maskMode = 0, int64_t cmpRatio = 1,
                                              ge::DataType dtype = ge::DT_FLOAT16, const char* socVersion = "Ascend950",
                                              const char* socInfo = A5SocInfo)
{
    return gert::TilingContextPara("SparseFlashMlaSoftmaxL1Norm",
                                   {
                                       // q: [T1, G, D]
                                       {{{t1, g, d}, {t1, g, d}}, dtype, ge::FORMAT_ND},
                                       // k: [T2, N2, D]
                                       {{{t2, n2, d}, {t2, n2, d}}, dtype, ge::FORMAT_ND},
                                       // softmax_lse: [N2, T1, G] fp32（3 维）
                                       {{{n2, t1, g}, {n2, t1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // sparse_indices: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cu_seqlens_q: [B+1] TND 必选
                                       {{{b + 1}, {b + 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cu_seqlens_k: [B+1] TND 必选
                                       {{{b + 1}, {b + 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // seqused_q: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // seqused_k: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // cmp_residual_k: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // topk_length: optional
                                       {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                       // metadata: 必选，1D INT32
                                       {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                   },
                                   {
                                       // softmax_l1_norm: [T1, G, D] fp32
                                       {{{t1, g, d}, {t1, g, d}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                   },
                                   {
                                       {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                       {"max_seqlen_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(t2)},
                                       {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cmpRatio)},
                                       {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode)},
                                       {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                       {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                   },
                                   &g_compileInfo, socVersion, socInfo, 4096);
}

// ---------------------------------------------------------------------------
// dtype 矩阵：bf16
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Bsnd_D512_Bf16)
{
    auto para = MakeBsndContext(1, 256, 256, 4, 1, 512, 0, 1, "BSND", "BSND", 1.0f, 0, ge::DT_BF16);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// TND layout with empty preceding optional inputs.  The tiling implementation
// must use GetOptionalInputShape for cu_seqlens, otherwise older CANN releases
// read the compact input list and derive a wrong batch size.
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Tnd_D512)
{
    auto para = MakeTndContext(2, 512, 384, 4, 1, 512);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

TEST_F(SparseMlaSoftmaxL1NormTiling, Arch22_Tnd_D256_Bf16)
{
    auto para = MakeTndContext(2, 512, 384, 8, 1, 256, 0, 4, ge::DT_BF16, "Ascend910B3", A2SocInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 可选输入交互：maskMode=3 + cmpRatio=4 时 cmp_residual_k [B] 必选——提供则成功
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Bsnd_Causal_Cmp4_WithResidual)
{
    gert::TilingContextPara para("SparseFlashMlaSoftmaxL1Norm",
                                 {
                                     {{{1, 256, 4, 512}, {1, 256, 4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 512, 1, 512}, {1, 512, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 1, 256, 4}, {1, 1, 256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // cmp_residual_k: [B] INT32（maskMode=3 且 cmpRatio>1 必选）
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // metadata: 必选，1D INT32
                                     {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 256, 4, 512}, {1, 256, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                     {"max_seqlen_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                     {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                     {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                 },
                                 &g_compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 可选输入缺失：maskMode=3 + cmpRatio=4 但未提供 cmp_residual_k → 拒绝
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Reject_Causal_Cmp4_NoResidual)
{
    auto para = MakeBsndContext(1, 256, 512, 4, 1, 512, 3, 4);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 可选输入交互：seqused_q [B] 提供则成功
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Bsnd_WithSequsedQ)
{
    gert::TilingContextPara para("SparseFlashMlaSoftmaxL1Norm",
                                 {
                                     {{{1, 256, 4, 512}, {1, 256, 4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 1, 512}, {1, 256, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 1, 256, 4}, {1, 1, 256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // seqused_q: [B] INT32
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // metadata: 必选，1D INT32
                                     {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 256, 4, 512}, {1, 256, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                     {"max_seqlen_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                     {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                 },
                                 &g_compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// ---------------------------------------------------------------------------
// 必选输入缺失：metadata 未提供 → 拒绝（L128-131 约束回归）
// ---------------------------------------------------------------------------
TEST_F(SparseMlaSoftmaxL1NormTiling, Arch35_Reject_NoMetadata)
{
    gert::TilingContextPara para("SparseFlashMlaSoftmaxL1Norm",
                                 {
                                     {{{1, 256, 4, 512}, {1, 256, 4, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 1, 512}, {1, 256, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 1, 256, 4}, {1, 1, 256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     // metadata: 未提供（空）→ 应被拒绝
                                     {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 256, 4, 512}, {1, 256, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                     {"max_seqlen_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                     {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                     {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                 },
                                 &g_compileInfo, "Ascend950", A5SocInfo, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}
