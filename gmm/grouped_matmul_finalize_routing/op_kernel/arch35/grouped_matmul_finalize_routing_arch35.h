/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_finalize_routing_arch35.h
 * \brief TensorAPI entry point for GroupedMatmulFinalizeRouting MX.
 */

#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_ARCH35_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_ARCH35_H

#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_finalize_routing.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"

namespace {
constexpr uint64_t GMM_FR_MX_SCALE_FACTOR_BIT = 8UL;
constexpr uint64_t GMM_FR_MX_SCALE_FACTOR_MASK = 0xFFUL;

__aicore__ inline uint64_t GetGmmFrScaleKL1(const TCubeTiling &matmulTiling)
{
    const uint64_t scaleFactorA = matmulTiling.mxTypePara & GMM_FR_MX_SCALE_FACTOR_MASK;
    const uint64_t scaleFactorB = (matmulTiling.mxTypePara >> GMM_FR_MX_SCALE_FACTOR_BIT) & GMM_FR_MX_SCALE_FACTOR_MASK;
    const uint64_t kAL1 = static_cast<uint64_t>(matmulTiling.stepKa) * static_cast<uint64_t>(matmulTiling.baseK);
    const uint64_t kBL1 = static_cast<uint64_t>(matmulTiling.stepKb) * static_cast<uint64_t>(matmulTiling.baseK);
    return Blaze::Gemm::Min(Blaze::Gemm::Max(scaleFactorA * kAL1, scaleFactorB * kBL1),
                            static_cast<uint64_t>(matmulTiling.Ka));
}
} // namespace

template <typename layoutA, typename layoutB, typename logitType>
__aicore__ inline void grouped_matmul_finalize_routing_mx(GM_ADDR x, GM_ADDR w, GM_ADDR wScale, GM_ADDR bias,
                                                          GM_ADDR xScale, GM_ADDR groupList, GM_ADDR shareInput,
                                                          GM_ADDR logit, GM_ADDR rowIndex, GM_ADDR offset, GM_ADDR y,
                                                          GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)offset;
    (void)workspaceGM;
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    const auto &gmmParamsIn = tilingData.gmmFinalizeRoutingDataParams;
    const auto &matmulTiling = tilingData.matmulTiling;

    using AType = DTYPE_X;
    using BType = DTYPE_W;
    using CType = DTYPE_Y;
    // Keep Cube accumulation and the epilogue input in FP32. The requested
    // output dtype is applied by the vector epilogue after logit processing.
    using C1Type = float;
    using LogitType = logitType;
    using RowIndexType = AscendC::Std::conditional_t<AscendC::Std::is_same_v<CType, bfloat16_t>, int32_t, int64_t>;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = bfloat16_t;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockMmadPolicy =
        Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelQgmmMxMixFinalizeRouting>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, C1Type, LayoutC,
                                                    BiasType, LayoutBias>;
    using BlockPrologue = Blaze::Gemm::Kernel::BlockPrologueFinalizeRouting<CType, BiasType>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<CType, C1Type, LogitType, RowIndexType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using BlockPrologueEpilogue = AscendC::Std::tuple<BlockPrologue, BlockEpilogue>;
    using GmmKernel =
        Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockPrologueEpilogue, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;
    using BlockMmadParams = typename BlockMmad::Params;

    const uint32_t kAL1 = static_cast<uint32_t>(matmulTiling.stepKa) * static_cast<uint32_t>(matmulTiling.baseK);
    const uint32_t kBL1 = static_cast<uint32_t>(matmulTiling.stepKb) * static_cast<uint32_t>(matmulTiling.baseK);
    const uint32_t scaleKL1 = static_cast<uint32_t>(GetGmmFrScaleKL1(matmulTiling));

    GMMTiling gmmParams{static_cast<uint32_t>(gmmParamsIn.groupNum),
                        static_cast<uint32_t>(gmmParamsIn.batch),
                        static_cast<uint32_t>(gmmParamsIn.sharedInputOffset),
                        static_cast<uint32_t>(gmmParamsIn.sharedInputLen),
                        gmmParamsIn.residualScale,
                        static_cast<uint32_t>(matmulTiling.baseM),
                        static_cast<uint32_t>(matmulTiling.baseN),
                        static_cast<uint32_t>(matmulTiling.baseK),
                        kAL1,
                        kBL1,
                        scaleKL1,
                        scaleKL1,
                        static_cast<uint8_t>(gmmParamsIn.hasBias),
                        static_cast<uint8_t>(matmulTiling.dbL0C),
                        static_cast<uint8_t>(gmmParamsIn.groupListType)};
    BlockMmadParams blockMmadParams{x, w, y, bias, xScale, wScale};
    Params params = {{static_cast<int64_t>(matmulTiling.M), static_cast<int64_t>(matmulTiling.N),
                      static_cast<int64_t>(matmulTiling.Ka), static_cast<int64_t>(1)},
                     blockMmadParams,
                     {shareInput, y, gmmParamsIn.sharedInputOffset, gmmParamsIn.sharedInputLen, matmulTiling.N,
                      gmmParamsIn.batch, gmmParamsIn.residualScale},
                     {y, wScale, xScale, bias, logit, rowIndex, matmulTiling.baseM, matmulTiling.baseN},
                     groupList,
                     gmmParams};
    GmmKernel gmm;
    gmm(params);
}
#endif
