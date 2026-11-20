/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM2_COMBINE_ARCH92_H
#define MEGA_MOE_GMM2_COMBINE_ARCH92_H

#include "../../arch35_92/stage/mega_moe_gmm2_combine.h"

namespace MegaMoeImpl {
namespace GmmKernel {

template <bool UsePersistentMmad, typename BlockMmad, typename ElementC, bool IsLayered = false, bool IsShared = false,
          bool NotifyCombineTileReady = false, typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecGeneric(Scheduler& scheduler, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                       uint32_t startLoopIdx, uint32_t tileNum,
                                       BlockMmadContext<BlockMmad>* blockMmadContext, bool allowWeightL2Bypass,
                                       uint32_t rowOffsetInExpert = 0U, const Params* params = nullptr,
                                       uint32_t expertTokenCount = 0U)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmA = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)),
        layouts.a);
    auto gmB = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)),
        layouts.b);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
                                         layouts.scaleA);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
                                         layouts.scaleB);
    if constexpr (Config::IS_WAVE_FLAG_GRAINED && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    auto gmBias = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BiasType*>(0UL)), layouts.bias);
    auto gmC = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm2OutGlobal)),
        layouts.c);

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        auto runMmad = [&](auto* context) __attribute__((cce_aicore))
        {
            context->template Init<GmmStage::GMM2>(config);
            Gmm2AicMmadGeneric<BlockMmad, IsShared, IsLayered, NotifyCombineTileReady>(
                context->blockMmad, workSet, gmmAddrInfo, config, startLoopIdx, tileNum, rowOffsetInExpert,
                expertTokenCount);
        };
        if constexpr (UsePersistentMmad) {
            runMmad(blockMmadContext);
        } else {
            BlockMmadContext<BlockMmad> context;
            runMmad(&context);
        }
    } else if constexpr (NotifyCombineTileReady) {
        // NotifyCombineTileReady 为真时唯一 AIV 逐 tile 锁步消费。combine 去重的独立消费路径
        // （RunWaveCombineStageDedup）由调用方把本参数实例化为 false，整个分支编译期裁掉。
        using MakeLayoutC = typename KernelConfig::MakeLayoutC;
        CombineTokenRange<ElementC, MakeLayoutC>(workSet.scheduler, workSet.gmC, *params, gmmAddrInfo, config,
                                                 startLoopIdx, tileNum);
    }
}

} // namespace GmmKernel

template <bool UsePersistentMmad, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool IsShared = false, bool IsWaveFlagGrained = false, bool NotifyCombineTileReady = false>
__aicore__ inline void RunGmm2Generic(uint8_t combineQuantMode, bool topkWeightsPrefetch,
                                      const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                      const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                      const BlockJobContext& blockJob, void* blockMmadContext,
                                      bool allowWeightL2Bypass = false, uint32_t rowOffsetInExpert = 0U,
                                      const Params* params = nullptr,
                                      uint32_t activationOutputDivisor = ACTIVATION_N_HALF,
                                      uint32_t expertTokenCount = 0U)
{
    static_assert(!IsShared || !NotifyCombineTileReady, "Shared expert GMM2 has its own completion counter");
    static_assert(!IsLayered || !NotifyCombineTileReady, "Layered GMM2 selects its synchronization through IsLayered");
    using GmmConfig = GmmKernel::Config<false, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                                        IsWeightNZ, IsShared, IsLayered, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM, activationOutputDivisor);
    config.combineQuantMode = combineQuantMode;
    config.topkWeightsPrefetch = topkWeightsPrefetch;
    config.activationTileM = topkWeightsPrefetch ? L1_TILE_M_128 : Gmm1TileM;

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{asc::te::make_coord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if constexpr (g_coreType == AscendC::AIV && !NotifyCombineTileReady) {
        if (combineQuantMode == COMBINE_NO_QUANT) {
            startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
            return;
        }
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = GmmKernel::BlockMmadContext<BlockMmad>;
    auto* typedBlockMmadContext = reinterpret_cast<MmadContext*>(blockMmadContext);
    GmmKernel::Gmm2ExecGeneric<UsePersistentMmad, BlockMmad, ElementC, IsLayered, IsShared, NotifyCombineTileReady>(
        scheduler, gmmAddrInfo, config, startLoopIdx, tileNum, typedBlockMmadContext, allowWeightL2Bypass,
        rowOffsetInExpert, params, expertTokenCount);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_ARCH92_H
