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
 * \file grouped_matmul_finalize_routing_a8w4_blaze.h
 * \brief TensorAPI entry point for GroupedMatmulFinalizeRouting MX A8W4 weight quant
 *        (FP8 E4M3FN x FP4 E2M1, WeightNZ). Replaces the legacy
 *        GMMFRWeightQuantResplitController assembly with Blaze components.
 */

#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_A8W4_BLAZE_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_A8W4_BLAZE_H

#include "blaze/epilogue/block/block_epilogue_finalize_routing.h"
#include "blaze/gemm/block/block_mmad_weight_prologue_mx.h"
#include "blaze/gemm/kernel/kernel_wqgmm_mx_mix_finalize_routing.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "weight_quant_basic_block/gmm_fr_weight_quant_tiling_data.h"

namespace GroupedMatmulFinalizeRouting {

template <typename XType_, typename WType_, typename ScaleType_, typename PerTokenScaleType_, typename BiasType_,
          typename YType_, typename SharedInputType_, typename LogitType_, typename RowIndexType_, bool HasBias>
__aicore__ inline void InvokeBlazeA8W4Kernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR logitsAddr, GM_ADDR rowIndexAddr, GM_ADDR y, GM_ADDR shareInput,
    const GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData* __restrict tiling)
{
    using AType = XType_;                // fp8_e4m3fn_t activation
    using BType = WType_;                // fp4x2_e2m1_t packed weight
    using ScaleType = ScaleType_;        // fp8_e8m0_t
    using BiasType = BiasType_;          // bfloat16_t
    using SharedType = SharedInputType_; // bfloat16_t
    using LogitType = LogitType_;        // float
    using RowIndexType = RowIndexType_;  // int64_t
    static_assert(AscendC::IsSameType<LogitType, float>::value, "A8W4 FR kernel requires float logits");
    static_assert(AscendC::IsSameType<RowIndexType, int64_t>::value, "A8W4 FR kernel requires int64 row index");
    static_assert(AscendC::IsSameType<SharedType, bfloat16_t>::value, "A8W4 FR kernel requires bf16 shared input");

    static_assert(AscendC::Std::is_same_v<PerTokenScaleType_, ScaleType_>, "MX scale types must match");
    static_assert(AscendC::Std::is_same_v<YType_, float>, "A8W4 output must be float");
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulFinalizeRoutingWithWeightQuantMx;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::zn_layout_ptn;
    using LayoutScaleA = asc::te::scalea_nd_layout_ptn;
    using LayoutScaleB = asc::te::scaleb_dn_layout_ptn;
    using ProblemShape = decltype(asc::te::make_shape(0UL, 0UL, 0UL));
    using BlockScheduler =
        Blaze::Gemm::Kernel::BlockSchedulerWqgmmNResplit<decltype(asc::te::make_shape(0UL, 0UL, 0UL))>;
    using BlockMmad =
        Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AscendC::Std::tuple<AType, ScaleType>,
                                      AscendC::Std::tuple<LayoutA, LayoutScaleA>, AscendC::Std::tuple<BType, ScaleType>,
                                      AscendC::Std::tuple<LayoutB, LayoutScaleB>, float, asc::te::nd_ext_layout_ptn,
                                      BiasType, asc::te::nd_ext_layout_ptn>;
    using BlockPrologue = Blaze::Gemm::Kernel::GmmWqgmmFrWeightPrologue<AType, BType, BiasType>;
    using BlockEpilogue =
        Blaze::Epilogue::Block::BlockEpilogueFinalizeRouting<float, float, float, int64_t, float, false,
                                                             Blaze::Epilogue::Block::FinalizeRoutingMode::A8W4>;
    using BlockOutputInit = Blaze::Gemm::Kernel::GmmWqgmmFrOutputInit;
    using KernelImpl =
        Blaze::Gemm::Kernel::GmmWqgmmMxFinalizeRoutingKernel<ProblemShape, BlockMmad, BlockScheduler, BlockPrologue,
                                                             BlockEpilogue, BlockOutputInit>;

    typename BlockMmad::Params mmadParams{};
    mmadParams.aGmAddr = x;
    mmadParams.scaleAGmAddr = perTokenScale;
    mmadParams.scaleBGmAddr = antiquantScale;
    mmadParams.biasGmAddr = bias;
    mmadParams.cGmAddr = y; // consumed by the AIV epilogue; AIC writes the UB fixpipe buffer

    constexpr uint64_t MAIN_BLOCK_N = 256;
    constexpr int32_t BASE_M = 256;
    typename BlockScheduler::Params schedulerParams{};
    schedulerParams.mainBlockCount = Blaze::Gemm::CeilDiv(tiling->nSize, MAIN_BLOCK_N);
    schedulerParams.mainBlockSize = MAIN_BLOCK_N;
    schedulerParams.coreNum = static_cast<uint64_t>(tiling->coreNum);
    schedulerParams.baseM = BASE_M;
    schedulerParams.nSize = static_cast<uint64_t>(tiling->nSize);
    typename BlockPrologue::Params prologueParams{reinterpret_cast<__gm__ BType*>(weight)};
    typename BlockEpilogue::Params epilogueParams{};
    epilogueParams.yGmAddr = y;
    epilogueParams.logitGmAddr = logitsAddr;
    epilogueParams.rowIndexGmAddr = rowIndexAddr;
    epilogueParams.nSize = tiling->nSize;
    typename BlockOutputInit::Params outputInitParams{};
    outputInitParams.ptrY = reinterpret_cast<__gm__ float*>(y);
    outputInitParams.ptrSharedInput = reinterpret_cast<__gm__ SharedType*>(shareInput);
    outputInitParams.nSize = tiling->nSize;
    outputInitParams.sharedInputOffset = tiling->sharedInputOffset;
    outputInitParams.sharedInputLen = tiling->sharedInputLen;
    outputInitParams.sharedInputWeight = tiling->sharedInputWeight;

    typename KernelImpl::Params params{
        ProblemShape{static_cast<uint64_t>(tiling->kSize), static_cast<uint64_t>(tiling->nSize),
                     static_cast<uint64_t>(tiling->groupNum)},
        mmadParams,
        schedulerParams,
        prologueParams,
        epilogueParams,
        outputInitParams,
        groupList,
        static_cast<uint32_t>(tiling->groupListType),
        static_cast<uint32_t>(tiling->hasBias)};
    KernelImpl kernelImpl;
    kernelImpl(params);
}

} // namespace GroupedMatmulFinalizeRouting

#endif // GROUPED_MATMUL_FINALIZE_ROUTING_A8W4_BLAZE_H
