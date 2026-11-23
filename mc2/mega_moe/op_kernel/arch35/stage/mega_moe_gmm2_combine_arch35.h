/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM2_COMBINE_ARCH35_H
#define MEGA_MOE_GMM2_COMBINE_ARCH35_H

#include "../../arch35_92/stage/mega_moe_gmm2_combine.h"

namespace MegaMoeImpl {
namespace GmmKernel {

template <typename BlockMmad, typename ElementC, bool IsLayered = false, bool IsShared = false,
          bool NotifyCombineTileReady = false, typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecGeneric(Scheduler& scheduler, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                       uint32_t startLoopIdx, uint32_t tileNum, bool allowWeightL2Bypass,
                                       uint32_t rowOffsetInExpert, const Params* params, uint32_t expertTokenCount,
                                       BlockMmadContext<BlockMmad>* waveContext)
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
        auto runMmad = [&](BlockMmadContext<BlockMmad> & mmadContext) __attribute__((cce_aicore))
        {
            mmadContext.template Init<GmmStage::GMM2>(config);
            Gmm2AicMmadGeneric<BlockMmad, IsShared, IsLayered, NotifyCombineTileReady>(
                mmadContext.blockMmad, workSet, gmmAddrInfo, config, startLoopIdx, tileNum, rowOffsetInExpert,
                expertTokenCount);
        };
        if (waveContext == nullptr) {
            // 这些调用在本阶段持有 MMAD，完成后归还 L1 和事件。
            BlockMmadContext<BlockMmad> localContext;
            runMmad(localContext);
        } else {
            runMmad(*waveContext);
        }
    } else if constexpr (NotifyCombineTileReady) {
        // NotifyCombineTileReady 为真时 AIV1 逐 tile 锁步消费。combine 去重的独立消费路径
        // （RunWaveCombineStageDedup）由调用方把本参数实例化为 false，整个分支编译期裁掉。
        if (MegaMoeHardwarePolicy::GetSubBlockIdx() == 1U) {
            using MakeLayoutC = typename KernelConfig::MakeLayoutC;
            CombineTokenRange<ElementC, MakeLayoutC>(workSet.scheduler, workSet.gmC, *params, gmmAddrInfo, config,
                                                     startLoopIdx, tileNum);
        }
    }
}

template <typename BlockMmad, typename BlockPrologue, typename ElementC, bool IsShared, bool IsLayered,
          bool NotifyCombineTileReady, typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecA8W4(Scheduler& scheduler, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                    uint32_t startLoopIdx, uint32_t tileNum, uint32_t expertTokenCount,
                                    uint32_t rowOffsetInExpert, const Params* params,
                                    const A8W4BlockContext<BlockMmad, BlockPrologue>& pipeline,
                                    bool allowPackedWeightL2Bypass)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmC = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm2OutGlobal)),
        layouts.c);
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
    auto gmBias = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BiasType*>(0UL)), layouts.bias);

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        auto& blockMmad = *pipeline.block;
        typename BlockMmad::BlockShape l0TileShape{config.blockMmadTiling.tileM, config.blockMmadTiling.tileN,
                                                   L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        // GMM2 has no gate/up concatenation; preserve the persistent buffer indices.
        blockMmad.template Init<false>(matmulShape, l0TileShape, false);
        Gmm2AicMmadA8W4<IsShared, IsLayered, NotifyCombineTileReady, BlockMmad, decltype(workSet.scheduler),
                        decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                        std::remove_reference_t<decltype(workSet.gmC)>, Config>(
            blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC, gmmAddrInfo,
            config, startLoopIdx, tileNum, expertTokenCount, rowOffsetInExpert);
    } else {
        if (MegaMoeHardwarePolicy::GetSubBlockIdx() == 0U) {
            // 缓存提示只作用于 GM 搬到 UB 的 packed W2。
            const bool bypassPackedWeightL2 = allowPackedWeightL2Bypass && rowOffsetInExpert == 0U &&
                                              config.m == expertTokenCount && config.m <= config.tileM;
            workSet.gmB.set_l2_cache_hint(bypassPackedWeightL2 ? asc::te::cache_mode::disable :
                                                                 asc::te::cache_mode::normal);
            Gmm2Aiv0PrologueA8W4(*pipeline.block, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
        } else if constexpr (NotifyCombineTileReady) {
            // 同上：W4 路径去重开关走运行时判断（嵌套模板参数触发编译器代码生成问题）。
            if (!IsCombineDedupOn(params->tilingData->compute.dedupMode)) {
                using MakeLayoutC = typename KernelConfig::MakeLayoutC;
                CombineTokenRange<ElementC, MakeLayoutC>(workSet.scheduler, workSet.gmC, *params, gmmAddrInfo, config,
                                                         startLoopIdx, tileNum);
            }
        }
    }
}

} // namespace GmmKernel

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsWeightNZ = false, bool IsLayered = false, uint32_t Gmm1TileM = L1_TILE_M_256, bool IsShared = false,
          bool IsWaveFlagGrained = false, bool NotifyCombineTileReady = false,
          typename GmmConfig = GmmKernel::Config<false, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                                                 IsWeightNZ, IsShared, IsLayered, IsWaveFlagGrained>>
__aicore__ inline void RunGmm2Generic(uint8_t combineQuantMode, bool topkWeightsPrefetch,
                                      const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                      const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                      const BlockJobContext& blockJob, bool allowWeightL2Bypass = false,
                                      uint32_t rowOffsetInExpert = 0U, const Params* params = nullptr,
                                      uint32_t activationOutputDivisor = ACTIVATION_N_HALF,
                                      uint32_t expertTokenCount = 0U,
                                      typename GmmConfig::MmadContext* waveContext = nullptr)
{
    static_assert(!IsShared || !NotifyCombineTileReady, "Shared expert GMM2 has its own completion counter");
    static_assert(!IsLayered || !NotifyCombineTileReady, "Layered GMM2 selects its synchronization through IsLayered");
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
    GmmKernel::Gmm2ExecGeneric<BlockMmad, ElementC, IsLayered, IsShared, NotifyCombineTileReady>(
        scheduler, gmmAddrInfo, config, startLoopIdx, tileNum, allowWeightL2Bypass, rowOffsetInExpert, params,
        expertTokenCount, waveContext);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool IsShared = false, bool IsLayered = false,
          bool IsWaveFlagGrained = false, bool NotifyCombineTileReady = false, typename BlockContext>
__aicore__ inline void RunGmm2A8W4(bool topkWeightsPrefetch, const BlockContext& pipeline,
                                   const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                   const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                   const BlockJobContext& blockJob, uint32_t expertTokenCount,
                                   uint32_t rowOffsetInExpert, const Params* params = nullptr,
                                   uint32_t activationOutputDivisor = ACTIVATION_N_HALF,
                                   bool allowPackedWeightL2Bypass = false)
{
    static_assert(!IsShared || !NotifyCombineTileReady, "Shared expert GMM2 has its own completion counter");
    static_assert(!IsLayered || !NotifyCombineTileReady, "Layered GMM2 selects its synchronization through IsLayered");
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = GmmKernel::Config<true, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                                        IsShared, IsLayered, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM, activationOutputDivisor);
    config.topkWeightsPrefetch = topkWeightsPrefetch;
    config.activationTileM = topkWeightsPrefetch ? L1_TILE_M_128 : Gmm1TileM;

    using BlockMmad = typename GmmConfig::BlockMmad;
    using BlockPrologue = typename GmmConfig::BlockPrologue;
    GmmKernel::BlockScheduler scheduler({config.m, config.outputN, config.k},
                                        GmmKernel::BlockScheduler::Params{asc::te::make_coord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    // tileNum 与 mega_moe_utils.h 的 GetGmmProblemTileCount 同源，预判无活的口径不得各算一份。
    uint32_t tileNum = GetGmmProblemTileCount(config.m, config.outputN);
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    GmmKernel::Gmm2ExecA8W4<BlockMmad, BlockPrologue, ElementC, IsShared, IsLayered, NotifyCombineTileReady,
                            GmmKernel::BlockScheduler, decltype(config)>(scheduler, gmmAddrInfo, config, startLoopIdx,
                                                                         tileNum, expertTokenCount, rowOffsetInExpert,
                                                                         params, pipeline, allowPackedWeightL2Bypass);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

// 默认由 AIV1 复用既有配置，仅为 AIV0 绑定最终 Combine 的 UB 视图。steady Wave 使用较小行环，
// 调用方排空旧行环后通过 ReinitializeAllAiv 让两个 AIV 都恢复最终 Wave 的最大容量。
template <bool ReinitializeAllAiv = false>
__aicore__ inline WaveCombineBufferConfig PrepareFinalWaveCombineBuffers(
    uint8_t combineMode, const MoeStageCommonConfig& common, const WaveCombineBufferConfig& currentBufferConfig,
    WaveCombineScratch& scratch)
{
    if constexpr (g_coreType == AIC) {
        return currentBufferConfig;
    }
    if (ReinitializeAllAiv || MegaMoeHardwarePolicy::GetSubBlockIdx() == 0U) {
        return InitWaveCombineBuffers(combineMode, common, scratch);
    }
    return currentBufferConfig;
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_ARCH35_H
