/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_finalize_routing_pertoken_blaze.h
 * \brief TensorAPI entry point for GroupedMatmulFinalizeRouting per-token dequant
 *        (INT8 x INT8 / FP8 E4M3FN x FP8 E4M3FN / HIFLOAT8 x HIFLOAT8 full quantization).
 */

#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_PERTOKEN_BLAZE_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_PERTOKEN_BLAZE_H

#include "blaze/epilogue/block/block_epilogue_finalize_routing.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_mix_pertoken_finalize_routing.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"

namespace GroupedMatmulFinalizeRouting {

template <typename LayoutA_, typename LayoutB_, int ScaleType, int RowIndexTypeValue, bool IsDeterministic_>
__aicore__ inline void InvokeBlazePertokenKernelImpl(
    GM_ADDR x, GM_ADDR w, GM_ADDR wScale, GM_ADDR bias, GM_ADDR xScale, GM_ADDR groupList, GM_ADDR shareInput,
    GM_ADDR logit, GM_ADDR rowIndex, GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM,
    const GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData& tilingData)
{
    (void)offset;
    const auto& gmmParamsIn = tilingData.gmmFinalizeRoutingDataParams;
    const auto& matmulTiling = tilingData.matmulTiling;

    using AType = DTYPE_X;
    using BType = DTYPE_W;
    using CType = DTYPE_Y;
    // Keep the MMAD accumulator in int32 for INT8 inputs and float for FP8/HIFLOAT8.
    using C1Type = AscendC::Std::conditional_t<AscendC::Std::is_same_v<AType, int8_t>, int32_t, float>;
    constexpr int SCALE_TYPE_BF16 = 2;
    using WeightScaleType = AscendC::Std::conditional_t<ScaleType == SCALE_TYPE_BF16, bfloat16_t, float>;
    using BiasType = bfloat16_t;
    using RowIndexType = AscendC::Std::conditional_t<RowIndexTypeValue == 1, int32_t, int64_t>;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockMmadPolicy =
        Blaze::Gemm::MatmulWithScaleMix<0, false, Blaze::Gemm::KernelQgmmMixPertokenFinalizeRouting>;
    // The Mix block takes B as a tuple; the per-token path has no fixpipe scale, so the
    // tuple only carries the weight dtype.
    using BTypeTuple = AscendC::Std::tuple<BType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BTypeTuple, LayoutB, C1Type,
                                                    LayoutC, BiasType, LayoutBias>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;

    const uint32_t kAL1 = static_cast<uint32_t>(matmulTiling.stepKa) * static_cast<uint32_t>(matmulTiling.baseK);
    const uint32_t kBL1 = static_cast<uint32_t>(matmulTiling.stepKb) * static_cast<uint32_t>(matmulTiling.baseK);
    constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

    typename BlockMmad::Params mmadParams{};
    mmadParams.aGmAddr = x;
    mmadParams.bGmAddr = w;

    using BlockEpilogue =
        Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<CType, C1Type, float, RowIndexType, WeightScaleType,
                                                             IsDeterministic_,
                                                             Blaze::Epilogue::Block::FinalizeRoutingMode::PERTOKEN>;
    using GmmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    typename GmmKernel::SharedInputParams sharedInputParams{shareInput,
                                                            y,
                                                            gmmParamsIn.sharedInputOffset,
                                                            gmmParamsIn.sharedInputLen,
                                                            matmulTiling.N,
                                                            gmmParamsIn.batch,
                                                            gmmParamsIn.residualScale};

    typename BlockEpilogue::Params epilogueParams{};
    epilogueParams.yGmAddr = y;
    epilogueParams.x2ScaleGmAddr = wScale;
    epilogueParams.x1ScaleGmAddr = xScale;
    epilogueParams.biasGmAddr = bias;
    epilogueParams.logitGmAddr = logit;
    epilogueParams.rowIndexGmAddr = rowIndex;
    if constexpr (IsDeterministic_) {
        epilogueParams.workspaceGmAddr = AscendC::GetUserWorkspace(workspaceGM);
    }

    typename GmmKernel::GmmTiling gmmTiling{gmmParamsIn.groupNum,
                                            gmmParamsIn.batch,
                                            gmmParamsIn.sharedInputOffset,
                                            gmmParamsIn.sharedInputLen,
                                            gmmParamsIn.residualScale,
                                            static_cast<uint32_t>(matmulTiling.baseM),
                                            static_cast<uint32_t>(matmulTiling.baseN),
                                            static_cast<uint32_t>(matmulTiling.baseK),
                                            kAL1,
                                            kBL1,
                                            DOUBLE_BUFFER_NUM,
                                            gmmParamsIn.hasBias,
                                            static_cast<uint8_t>(matmulTiling.dbL0C),
                                            gmmParamsIn.groupListType,
                                            IsDeterministic_ ? gmmParamsIn.deterWorkspaceSize : 0,
                                            IsDeterministic_ ? static_cast<uint32_t>(matmulTiling.usedCoreNum) : 0};
    typename GmmKernel::Params params{
        ProblemShape{static_cast<int64_t>(matmulTiling.M), static_cast<int64_t>(matmulTiling.N),
                     static_cast<int64_t>(matmulTiling.Ka), static_cast<int64_t>(1)},
        mmadParams,
        sharedInputParams,
        epilogueParams,
        groupList,
        gmmTiling};
    GmmKernel gmm;
    gmm(params);
    (void)workspaceGM;
}

template <typename LayoutA_, typename LayoutB_, int ScaleType, int RowIndexTypeValue>
__aicore__ inline void InvokeBlazePertokenKernel(GM_ADDR x, GM_ADDR w, GM_ADDR wScale, GM_ADDR bias, GM_ADDR xScale,
                                                 GM_ADDR groupList, GM_ADDR shareInput, GM_ADDR logit, GM_ADDR rowIndex,
                                                 GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    if (tilingData.gmmFinalizeRoutingDataParams.deterministicFlag == 1) {
        InvokeBlazePertokenKernelImpl<LayoutA_, LayoutB_, ScaleType, RowIndexTypeValue, true>(
            x, w, wScale, bias, xScale, groupList, shareInput, logit, rowIndex, offset, y, workspaceGM, tilingData);
    } else {
        InvokeBlazePertokenKernelImpl<LayoutA_, LayoutB_, ScaleType, RowIndexTypeValue, false>(
            x, w, wScale, bias, xScale, groupList, shareInput, logit, rowIndex, offset, y, workspaceGM, tilingData);
    }
}
} // namespace GroupedMatmulFinalizeRouting

#endif
