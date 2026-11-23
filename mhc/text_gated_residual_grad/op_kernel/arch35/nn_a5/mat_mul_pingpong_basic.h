/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mat_mul_pingpong_basic.h
 * \brief
 */
#pragma once

#include "kernels.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_fixpipe_opti.h"
#include "block_scheduler_matmul_basic.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace HcGradNnA5 {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t FULL_LOAD_MODE = 0, uint64_t FUSED_OP_TYPE = 0, uint64_t NON_CONTIGIOUS_TYPE = 0>
__aicore__ inline void MatMulBasicKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                         const HcGradA5NnTiling& tilingData, int64_t batch = 0)
{
    // Matrix element types and layouts.
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using LayoutBias = LayoutC;

    // The problem shape stores M, N, K, and batch.
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool isNDFormat = !(Blaze::Gemm::IsWeightNz<LayoutB>::value);
    static constexpr bool isFp32 = (AscendC::Std::is_same_v<BType, float>);
    // Scheduler type.
    using BlockScheduler =
        Blaze::Gemm::Block::HcGradBlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE, isFp32, isNDFormat>;

    // Block MMAD type.
    using DispatchPolicy =
        Blaze::Gemm::MatmulMultiBlockBasic<FULL_LOAD_MODE, FUSED_OP_TYPE, Blaze::Gemm::KernelMmadMultiBlockBasic,
                                           NON_CONTIGIOUS_TYPE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC,
                                                    BiasType, LayoutBias>;

    // Block epilogue type.
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;

    // MatMul kernel type.
    using MatmulKernel = Blaze::Gemm::Kernel::HcGradNnGemm<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {
        {tilingData.m, tilingData.n, tilingData.k, batch}, // shape
        {aGM, bGM, cGM, biasGM, nullptr, nullptr, tilingData.mL1, tilingData.nL1, tilingData.kL1, tilingData.baseM,
         tilingData.baseN, tilingData.baseK, tilingData.l1BufferNum, tilingData.l0cDB, nullptr, tilingData.rowStride},
        {}, // epilogue args
        {tilingData.mL1, tilingData.nL1, tilingData.kL1, tilingData.baseM, tilingData.baseN, tilingData.baseK,
         tilingData.mTailCnt, tilingData.nTailCnt, tilingData.mBaseTailSplitCnt, tilingData.nBaseTailSplitCnt,
         tilingData.mTailMain, tilingData.nTailMain, tilingData.mmadParam,
         static_cast<uint32_t>(tilingData.l2CacheDisable), tilingData.sliceM, tilingData.srcNdStride,
         tilingData.innerBatch, tilingData.aswWindowLen}}; // scheduler params
    params.fusionLda = tilingData.rowStride;
    params.schParams.usedCoreNum = tilingData.usedCoreNum;
    MatmulKernel mm;
    mm(params);
}

template <class D_T, class C_T, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT, uint32_t BLOCKED_B_ROWS,
          bool BYPASS_OUTPUT_L2>
__aicore__ inline void MatMulBasicTrafficKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, const HcGradA5NnTiling& t)
{
    using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Scheduler = Blaze::Gemm::Block::HcGradBlockSchedulerMatmulBasic<Shape, 0, false, true>;
    using Policy = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelMmadMultiBlockBasic, 0>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, D_T, A_LAYOUT, D_T, B_LAYOUT, C_T, C_LAYOUT, float, C_LAYOUT>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using MatmulKernel = Blaze::Gemm::Kernel::HcGradNnGemm<Shape, Mmad, Epilogue, Scheduler>;
    typename MatmulKernel::Params params = {
        {t.m, t.n, t.k, 0},
        {aGM, bGM, cGM, nullptr, nullptr, nullptr, t.mL1, t.nL1, t.kL1, t.baseM, t.baseN, t.baseK, t.l1BufferNum,
         t.l0cDB, nullptr, t.rowStride},
        {},
        {t.mL1, t.nL1, t.kL1, t.baseM, t.baseN, t.baseK, t.mTailCnt, t.nTailCnt, t.mBaseTailSplitCnt,
         t.nBaseTailSplitCnt, t.mTailMain, t.nTailMain, t.mmadParam, static_cast<uint32_t>(t.l2CacheDisable), t.sliceM,
         t.srcNdStride, t.innerBatch, t.aswWindowLen}};
    params.fusionLda = t.rowStride;
    params.schParams.usedCoreNum = t.usedCoreNum;
    MatmulKernel mm;
    mm.template RunTraffic<BLOCKED_B_ROWS, BYPASS_OUTPUT_L2>(params);
}
template <class D_T>
__aicore__ inline void MatMulC1aToV3Ub(GM_ADDR dup, GM_ADDR wUp, const HcGradA5NnTiling& t, uint32_t tileBegin,
                                       uint32_t tileRows, uint32_t ubOffset)
{
    namespace Te = asc::te;
    namespace Sync = Blaze::Gemm::Sync;
    using ND = Te::nd_ext_layout_ptn;
    using Policy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, D_T, ND, D_T, ND, float, ND, float, ND>;
    static_assert(Sync::FIXPIPE_AIV_ACK_FLAG_BASE == TextGatedResidualGrad::HC_GRAD_C1A_UB_FREE &&
                  Sync::FIXPIPE_AIC_READY_FLAG_BASE == TextGatedResidualGrad::HC_GRAD_C1A_UB_READY &&
                  Sync::SYNC_MODE_INTRA == 4U && Sync::MIX_FLAG_ID_MAX == 16U);
    using LayoutA = Te::frame_layout_format<ND, AscendC::Std::Int<Te::c0_element<D_T>>>;
    using LayoutOut = Te::frame_layout_format<ND, AscendC::Std::Int<Te::c0_element<float>>>;
    auto a = Te::make_tensor(
        Te::make_mem_ptr<Te::location::gm>(reinterpret_cast<__gm__ D_T*>(dup) + static_cast<uint64_t>(tileBegin) * t.k),
        LayoutA{}(tileRows, t.k));
    auto b =
        Te::make_tensor(Te::make_mem_ptr<Te::location::gm>(reinterpret_cast<__gm__ D_T*>(wUp)), LayoutA{}(t.k, t.n));
    auto bias =
        Te::make_tensor(Te::make_mem_ptr<Te::location::gm>(static_cast<__gm__ float*>(nullptr)), LayoutOut{}(1L, t.n));
    auto out = Te::make_tensor(Te::make_mem_ptr<Te::location::ub, float>(static_cast<uint64_t>(ubOffset)),
                               LayoutOut{}(tileRows, t.n));
    a.set_l2_cache_hint(asc::te::cache_mode::disable);
    typename Mmad::TupleShape shape{tileRows, t.n, t.k, 1};
    typename Mmad::Params params;
    params.oriK = t.k;
    params.mL1 = t.mL1;
    params.nL1 = t.nL1;
    params.kL1 = t.kL1;
    params.mL0 = t.baseM;
    params.nL0 = t.baseN;
    params.kL0 = t.baseK;
    params.l1Stages = t.l1BufferNum;
    params.l0cStages = t.l0cDB;
    params.splitM = 1U;
    params.ubDB = 1U;
    params.rowStride = t.k;
    Mmad mm;
    mm.Init(params);
    mm(a, b, bias, out, shape);
}

} // namespace HcGradNnA5
