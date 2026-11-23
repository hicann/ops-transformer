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
 * \file mat_mul_bl1_full_load.h
 * \brief
 */
#ifndef MAT_MUL_BL1_FULL_LOAD_H
#define MAT_MUL_BL1_FULL_LOAD_H

#include "kernels.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/epilogue/block/block_epilogue_fixpipe.h"
#include "block_scheduler_matmul_basic.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace HcGradNnA5 {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t FULL_LOAD_MODE, uint64_t L0C2OUT_MODEL = 0, uint64_t FUSED_OP_TYPE = 0>
__aicore__ inline void MatMulBL1FullLoadKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
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

    static constexpr bool isFp32 = (AscendC::Std::is_same_v<BType, float>);
    static constexpr bool isNDFormat = !(Blaze::Gemm::IsWeightNz<LayoutB>::value);
    // Scheduler type.
    using BlockScheduler =
        Blaze::Gemm::Block::HcGradBlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE, isFp32, isNDFormat>;

    // Block MMAD type.
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBFullLoad<L0C2OUT_MODEL, FUSED_OP_TYPE,
                                                                  Blaze::Gemm::KernelMmadMultiBlockBFullLoad>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC,
                                                    BiasType, LayoutBias>;

    // Block epilogue type.
    using BlockEpilogue =
        AscendC::Std::conditional_t<(L0C2OUT_MODEL != Blaze::Gemm::ON_THE_FLY),
                                    Blaze::Epilogue::Block::BlockEpilogueFixpipe<OutType, OutType, DispatchPolicy>,
                                    Blaze::Epilogue::Block::BlockEpilogueEmpty>;

    // MatMul kernel type.
    using MatmulKernel = Blaze::Gemm::Kernel::HcGradNnGemm<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{tilingData.m, tilingData.n, tilingData.k, batch}, // shape
                     {aGM, bGM, cGM, biasGM, nullptr, nullptr, tilingData.k, tilingData.mL1, tilingData.nL1,
                      tilingData.kL1, tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.l1BufferNum,
                      tilingData.l0cDB, tilingData.splitM, tilingData.ubDB, tilingData.rowStride},
                     {}, // epilogue args (set below for fixpipe mode)
                     {tilingData.mL1, tilingData.nL1, tilingData.kL1, tilingData.baseM, tilingData.baseN,
                      tilingData.baseK, tilingData.mTailCnt, tilingData.nTailCnt, tilingData.mBaseTailSplitCnt,
                      tilingData.nBaseTailSplitCnt, tilingData.mTailMain, tilingData.nTailMain, tilingData.mmadParam,
                      static_cast<uint32_t>(tilingData.l2CacheDisable), tilingData.sliceM, tilingData.srcNdStride,
                      tilingData.innerBatch, tilingData.aswWindowLen}}; // scheduler params
    if constexpr (L0C2OUT_MODEL != Blaze::Gemm::ON_THE_FLY) {
        params.epilogueParams = {cGM};
    }
    params.fusionLda = tilingData.rowStride;
    params.schParams.usedCoreNum = tilingData.usedCoreNum;
    MatmulKernel mm;
    mm(params);
}
} // namespace HcGradNnA5
#endif
