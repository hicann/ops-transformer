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
 * \file mat_mul_al1_full_load.h
 * \brief
 */
#ifndef MAT_MUL_FULL_LOAD_TENSORAPI_H
#define MAT_MUL_FULL_LOAD_TENSORAPI_H

#include "kernels.h"
#include "blaze/gemm/block/block_mmad.h"
#include "block_scheduler_matmul_basic.h"
#include "blaze/gemm/policy/dispatch_policy.h"

namespace HcGradNnA5 {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t FULL_LOAD_MODE = 0, uint64_t FUSED_OP_TYPE = 0>
__aicore__ inline void MatMulAL1FullLoadKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                               GM_ADDR workspaceGM, const HcGradA5NnTiling& tilingData,
                                               int64_t batch = 0)
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

    static constexpr bool IS_FP32 = (AscendC::Std::is_same_v<BType, float>);
    static constexpr bool IS_ND_FORMAT = !(Blaze::Gemm::IsWeightNz<LayoutB>::value);
    // Scheduler type.
    using BlockScheduler =
        Blaze::Gemm::Block::HcGradBlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE, IS_FP32, IS_ND_FORMAT>;

    // Block MMAD type.
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockAFullLoad<FULL_LOAD_MODE, FUSED_OP_TYPE,
                                                                  Blaze::Gemm::KernelMmadMultiBlockAFullLoad>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC,
                                                    BiasType, LayoutBias>;

    // Block epilogue type.
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;

    // MatMul kernel type.
    using MatmulKernel = Blaze::Gemm::Kernel::HcGradNnGemm<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {
        {tilingData.m, tilingData.n, tilingData.k, batch}, // shape
        {aGM, bGM, cGM, biasGM, tilingData.k, tilingData.mL1, tilingData.kL1, tilingData.baseM, tilingData.baseN,
         tilingData.baseK, tilingData.l1BufferNum, tilingData.l0cDB, tilingData.rowStride},
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
// Reuse the complete K dimension of a row tile across adjacent DX column tiles.
template <class D_T>
__aicore__ inline void MatMulDxAPanelL1Kernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, const HcGradA5NnTiling& t)
{
    static_assert(AscendC::Std::is_same_v<D_T, bfloat16_t> || AscendC::Std::is_same_v<D_T, half>);
    using ND = asc::te::nd_ext_layout_ptn;
    using Policy = Blaze::Gemm::MatmulMultiBlockAFullLoad<1, 0, Blaze::Gemm::KernelMmadMultiBlockAFullLoad>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, D_T, ND, D_T, ND, float, ND, float, ND>;
    using MakeLayoutB = asc::te::frame_layout_format<ND, AscendC::Std::Int<asc::te::c0_element<D_T>>>;
    using MakeLayoutC = asc::te::frame_layout_format<ND, AscendC::Std::Int<asc::te::c0_element<float>>>;

    constexpr int64_t M = 5120;
    constexpr int64_t N = 10240;
    constexpr int64_t K = 324;
    constexpr int64_t TILE = 256;
    constexpr int64_t CORE_COUNT = 28;
    constexpr int64_t N_TILES = N / TILE;
    constexpr int64_t BLOCKS = (M / TILE) * N_TILES;
    constexpr int64_t TILES_PER_CORE = BLOCKS / CORE_COUNT;
    constexpr int64_t REMAINDER = BLOCKS % CORE_COUNT;
    static_assert(M % TILE == 0 && N % TILE == 0);
    static_assert(TILE * 336U * sizeof(D_T) + 2U * TILE * 256U * sizeof(D_T) <= 512U * 1024U);

    const int64_t core = AscendC::GetBlockIdx();
    if (core >= CORE_COUNT) {
        return;
    }
    const int64_t begin = core * TILES_PER_CORE + AscendC::Std::min(core, REMAINDER);
    const int64_t end = begin + TILES_PER_CORE + (core < REMAINDER);
    auto aLayout = asc::te::make_pattern_layout<ND, asc::te::layout_trait_default<>>(
        asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, M), asc::te::make_shape(asc::te::_1{}, K)),
        asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, static_cast<uint64_t>(t.rowStride)),
                             asc::te::make_stride(asc::te::_0{}, asc::te::_1{})));
    auto gmA =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ D_T*>(aGM)), aLayout);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ D_T*>(bGM)),
                                    MakeLayoutB{}(K, N));
    auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ float*>(cGM)),
                                    MakeLayoutC{}(M, N));
    auto gmBias = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(static_cast<__gm__ float*>(nullptr)), MakeLayoutC{}(1L, TILE));
    gmC.set_l2_cache_hint(asc::te::cache_mode::disable);

    typename Mmad::Params params = {aGM,     bGM,     cGM,     nullptr,       K,       TILE,       t.kL1,
                                    t.baseM, t.baseN, t.baseK, t.l1BufferNum, t.l0cDB, t.rowStride};
    typename Mmad::TupleShape tileShape{TILE, TILE, K, 1L};
    Blaze::Gemm::SetHF32(t.mmadParam);
    Mmad mm;
    for (int64_t block = begin; block < end;) {
        const int64_t row = block / N_TILES * TILE;
        const int64_t panelEnd = AscendC::Std::min(end, (block / N_TILES + 1) * N_TILES);
        auto a = gmA.slice(asc::te::make_coord(row, 0L), asc::te::make_shape(TILE, K));
        mm.Init(params);
        for (; block < panelEnd; ++block) {
            const int64_t col = block % N_TILES * TILE;
            auto b = gmB.slice(asc::te::make_coord(0L, col), asc::te::make_shape(K, TILE));
            auto c = gmC.slice(asc::te::make_coord(row, col), asc::te::make_shape(TILE, TILE));
            mm(a, b, gmBias, c, tileShape);
        }
        if (block < end) {
            // Drain this AIC before reusing its AL1/BL1 and rotating the buffer indices.
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    Blaze::Gemm::UnsetHF32(t.mmadParam);
}

} // namespace HcGradNnA5
#endif
