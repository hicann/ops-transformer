/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM1_ACTIVATION_ARCH35_H
#define MEGA_MOE_GMM1_ACTIVATION_ARCH35_H

#include "../../arch35_92/stage/mega_moe_gmm1_activation.h"

namespace MegaMoeImpl {
namespace GmmKernel {

template <typename BlockMmad, typename ElementC, typename MakeLayoutC, bool IsShared = false,
          bool IsWaveFlagGrained = false, typename Scheduler, typename Config, typename ActivationQuantOp,
          typename LayoutBundle>
__aicore__ inline void Gmm1ExecGeneric(Scheduler& scheduler, const Params& params, const GMMAddrInfo& gmmAddrInfo,
                                       const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                       ActivationQuantOp& activationQuantOp, int32_t& vecSetSyncCom,
                                       uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t& pingpongIdx,
                                       bool allowWeightL2Bypass, const LayoutBundle& layouts)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
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
    if constexpr (IsWaveFlagGrained && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    GM_ADDR cGlobal = 0UL;
    if (config.topkWeightsPrefetch) {
        cGlobal = gmmAddrInfo.gmm1OutGlobal;
    }
    auto gmC = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementC*>(cGlobal)), layouts.c);
    __gm__ float* metaInfoGm = nullptr;
    if constexpr (g_coreType == AscendC::AIV) {
        if (config.topkWeightsPrefetch) {
            metaInfoGm = reinterpret_cast<__gm__ float*>(gmmAddrInfo.metaInfoGlobal);
        }
    }

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmC), decltype(metaInfoGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmC, metaInfoGm};
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmadContext<BlockMmad> localBlockMmadContext;
        if (config.topkWeightsPrefetch) {
            Gmm1AicMmadToGmGeneric<BlockMmadContext<BlockMmad>, IsShared>(
                workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, expertIdx, &localBlockMmadContext);
        } else {
            Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsShared>(workSet, gmmAddrInfo, config, startLoopIdx,
                                                                      tileNum, vecSetSyncCom, pingpongIdx,
                                                                      &localBlockMmadContext);
        }
    } else {
        if (config.topkWeightsPrefetch) {
            Gmm1Aiv0PrefetchEpilogueGeneric<MakeLayoutC, ElementC, IsWaveFlagGrained>(
                workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, activationQuantOp, expertBeforeCnt,
                expertIdx);
        } else {
            Gmm1Aiv0EpilogueGeneric<IsWaveFlagGrained>(workSet, gmmAddrInfo, config, startLoopIdx, tileNum,
                                                       activationQuantOp, pingpongIdx);
        }
    }
}

template <typename BlockMmad, typename BlockPrologue, typename ElementC, typename MakeLayoutC, bool IsShared,
          typename Scheduler, typename Config, bool IsWaveFlagGrained, typename ActivationQuantOp,
          typename LayoutBundle>
__aicore__ inline void Gmm1ExecA8W4(Scheduler& scheduler, const Params& params, const GMMAddrInfo& gmmAddrInfo,
                                    const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                    ActivationQuantOp& activationQuantOp, uint32_t expertBeforeCnt, uint32_t expertIdx,
                                    const LayoutBundle& layouts,
                                    const A8W4BlockContext<BlockMmad, BlockPrologue>& pipeline)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    auto gmC = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm1OutGlobal)),
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
    __gm__ float* metaInfoGm = nullptr;
    if constexpr (g_coreType == AscendC::AIV) {
        if (config.topkWeightsPrefetch) {
            metaInfoGm = reinterpret_cast<__gm__ float*>(gmmAddrInfo.metaInfoGlobal);
        }
    }

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmC), decltype(metaInfoGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmC, metaInfoGm};
    if constexpr (g_coreType == AscendC::AIC) {
        Gmm1AicMmadA8W4<BlockMmad, IsShared>(*pipeline.block, workSet, params, gmmAddrInfo, config, startLoopIdx,
                                             tileNum, expertIdx);
    } else if (MegaMoeHardwarePolicy::GetSubBlockIdx() == 0U) {
        Gmm1Aiv0PrologueA8W4<BlockPrologue>(*pipeline.block, workSet.scheduler, workSet.gmB, config, startLoopIdx,
                                            tileNum);
    } else {
        if (config.topkWeightsPrefetch) {
            Gmm1Aiv1PrefetchEpilogueA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(
                workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, activationQuantOp, expertBeforeCnt,
                expertIdx);
        } else {
            Gmm1Aiv1EpilogueA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(workSet, gmmAddrInfo, config, startLoopIdx,
                                                                           tileNum, activationQuantOp);
        }
    }
}

} // namespace GmmKernel

template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256, bool IsShared = false,
          bool IsWaveFlagGrained = false, typename BlockEpilogue, typename... LayoutArgs>
__aicore__ inline void RunGmm1Generic(bool topkWeightsPrefetch, BlockEpilogue& epilogueOp, const Params& params,
                                      const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                      const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom,
                                      const BlockJobContext& blockJob, uint32_t expertBeforeCnt, uint32_t expertIdx,
                                      uint16_t& pingpongIdx, uint32_t activationOutputDivisor = ACTIVATION_N_HALF,
                                      bool allowWeightL2Bypass = false, const LayoutArgs&... layoutArgs)
{
    using GmmConfig = GmmKernel::Config<false, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                                        IsWeightNZ, IsShared, false, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM, activationOutputDivisor);
    config.topkWeightsPrefetch = topkWeightsPrefetch;
    config.activationTileM = topkWeightsPrefetch ? L1_TILE_M_128 : Gmm1TileM;

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{asc::te::make_coord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if (MegaMoeHardwarePolicy::GetSubBlockIdx() != 0U) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }
    if constexpr (g_coreType == AIV) {
        epilogueOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;
    auto layouts = GmmConfig::BuildLayouts(config, layoutArgs...);
    GmmKernel::Gmm1ExecGeneric<BlockMmad, ElementC, MakeLayoutC, IsShared, IsWaveFlagGrained>(
        scheduler, params, gmmAddrInfo, config, startLoopIdx, tileNum, epilogueOp, vecSetSyncCom, expertBeforeCnt,
        expertIdx, pingpongIdx, allowWeightL2Bypass, layouts);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256, bool IsShared = false,
          bool IsWaveFlagGrained = false, typename BlockEpilogue>
__aicore__ inline void RunGmm1Generic(bool topkWeightsPrefetch, BlockEpilogue& epilogueOp, const Params& params,
                                      const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                      const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom,
                                      uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t& pingpongIdx,
                                      uint32_t activationOutputDivisor = ACTIVATION_N_HALF,
                                      bool allowWeightL2Bypass = false)
{
    BlockJobContext blockJob{MegaMoeHardwarePolicy::GetAicBlockIdx(),
                             static_cast<uint32_t>(MegaMoeHardwarePolicy::GetBlockNum())};
    RunGmm1Generic<ElementA, EpilogueElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ,
                   Gmm1TileM, IsShared, IsWaveFlagGrained>(
        topkWeightsPrefetch, epilogueOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom, blockJob,
        expertBeforeCnt, expertIdx, pingpongIdx, activationOutputDivisor, allowWeightL2Bypass);
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool IsShared = false, bool IsWaveFlagGrained = false,
          typename BlockContext, typename BlockEpilogue, typename... LayoutArgs>
__aicore__ inline void RunGmm1A8W4(bool topkWeightsPrefetch, const BlockContext& pipeline,
                                   BlockEpilogue& activationQuantOp, const Params& params,
                                   const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                   const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                   const BlockJobContext& blockJob, uint32_t expertBeforeCnt, uint32_t expertIdx,
                                   uint32_t activationOutputDivisor = ACTIVATION_N_HALF,
                                   const LayoutArgs&... layoutArgs)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = GmmKernel::Config<true, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                                        IsShared, false, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM, activationOutputDivisor);
    config.topkWeightsPrefetch = topkWeightsPrefetch;
    config.activationTileM = topkWeightsPrefetch ? L1_TILE_M_128 : Gmm1TileM;

    if constexpr (g_coreType == AIV) {
        activationQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    using BlockMmad = typename GmmConfig::BlockMmad;
    using BlockPrologue = typename GmmConfig::BlockPrologue;
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{asc::te::make_coord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    auto layouts = GmmConfig::BuildLayouts(config, layoutArgs...);
    GmmKernel::Gmm1ExecA8W4<BlockMmad, BlockPrologue, ElementC, MakeLayoutC, IsShared, GmmKernel::BlockScheduler,
                            decltype(config), IsWaveFlagGrained>(scheduler, params, gmmAddrInfo, config, startLoopIdx,
                                                                 tileNum, activationQuantOp, expertBeforeCnt, expertIdx,
                                                                 layouts, pipeline);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool IsShared = false, bool IsWaveFlagGrained = false,
          typename BlockContext, typename BlockEpilogue>
__aicore__ inline void RunGmm1A8W4(bool topkWeightsPrefetch, const BlockContext& pipeline,
                                   BlockEpilogue& activationQuantOp, const Params& params,
                                   const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                   const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, uint32_t expertBeforeCnt,
                                   uint32_t expertIdx = 0, uint32_t activationOutputDivisor = ACTIVATION_N_HALF)
{
    BlockJobContext blockJob{MegaMoeHardwarePolicy::GetAicBlockIdx(),
                             static_cast<uint32_t>(MegaMoeHardwarePolicy::GetBlockNum())};
    RunGmm1A8W4<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM, IsShared, IsWaveFlagGrained>(
        topkWeightsPrefetch, pipeline, activationQuantOp, params, problemShape, gmmAddrInfo, startBlockIdx, blockJob,
        expertBeforeCnt, expertIdx, activationOutputDivisor);
}

// A5 的地址准备与 Activation 绑定按本架构 AIV 分工执行。
template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          uint32_t ActivationElementsPerByte, bool UseA8W4Gmm1, typename BlockEpilogue>
__aicore__ inline void UpdateMoeExpertGmm1GlobalBuffer(bool topkWeightsPrefetch, const GmmExecutionConfig& gmmConfig,
                                                       const MoeSyncWorkspaceLayout& syncLayout,
                                                       const WorkspaceLayout& workspace, GM_ADDR workspaceBase,
                                                       const ExpertWeightTensorListAddrs& weights,
                                                       BlockEpilogue& epilogueOp, GMMAddrInfo& gmmAddrInfo,
                                                       const ExpertLoopState& state, uint32_t activationOutputWidth,
                                                       uint32_t rowOffsetInExpert = 0U)
{
    if constexpr (g_coreType == AIV && !UseA8W4Gmm1) {
        if (MegaMoeHardwarePolicy::GetSubBlockIdx() != 0U) {
            return;
        }
    }

    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElementsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    int64_t n = Get<N_VALUE>(state.problemShape);
    int64_t k = Get<K_VALUE>(state.problemShape);
    int64_t expertOffset = state.expertIdx;
    int64_t globalTokenStartIndex = state.globalTokenStartIndex + rowOffsetInExpert;
    uint64_t physicalTokenStartIndex =
        gmmConfig.ring.gmm1 ?
            CalcMteRingPhysicalRow(state.paddedTokenStartIndex + rowOffsetInExpert, gmmConfig.ring.rows) :
            static_cast<uint64_t>(globalTokenStartIndex);
    // Dispatch 输入按自身循环开关选择行号；GMM1 输出和激活仍按各自的循环物理行寻址。
    const uint64_t dispatchTokenStartIndex =
        gmmConfig.ring.dispatch ? physicalTokenStartIndex : static_cast<uint64_t>(globalTokenStartIndex);
    uint32_t expertMGroupOffset = rowOffsetInExpert / L1_TILE_M_256;
    int64_t scaleK = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    if (UseA8W4Gmm1 || topkWeightsPrefetch) {
        gmmAddrInfo.gmm1OutGlobal =
            workspace.routed.gmm1Output.get_ptr(workspaceBase) + physicalTokenStartIndex * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.metaInfoGlobal = workspace.routing.receivedTokenMetadata.get_ptr(workspaceBase);
    gmmAddrInfo.activationToGmm2Flag =
        reinterpret_cast<__gm__ int32_t*>(workspace.sync.activationReady.get_ptr(workspaceBase)) +
        expertOffset * syncLayout.activationFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.aGlobal = workspace.routed.gmm1Input.data.get_ptr(workspaceBase) +
                          dispatchTokenStartIndex * k / ActivationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = workspace.routed.gmm1Input.scale.get_ptr(workspaceBase) +
                               dispatchTokenStartIndex * scaleK * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<WeightType>(weights.weight1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                        static_cast<uint64_t>(state.expertIdx) * n * k / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal =
        GetExpertWeightAddr<QuantScaleType>(weights.weightScales1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                            static_cast<uint64_t>(state.expertIdx) * n * scaleK);
    if constexpr (g_coreType == AIV) {
        bool runsActivation = true;
        if constexpr (UseA8W4Gmm1) {
            runsActivation = MegaMoeHardwarePolicy::GetSubBlockIdx() == 1U;
        }
        if (runsActivation) {
            int64_t scaleN = Ops::Base::CeilDiv(static_cast<int64_t>(activationOutputWidth),
                                                static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                             MXFP_MULTI_BASE_SIZE;
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                static_cast<int64_t>(physicalTokenStartIndex) * activationOutputWidth / outputElementsPerByte,
                static_cast<int64_t>(physicalTokenStartIndex) * scaleN,
                expertOffset * syncLayout.activationFlagSlotCountPerExpert / INT_CACHELINE + expertMGroupOffset,
                0L,
                0L,
                0L};
            epilogueOp.UpdateGlobalAddr(vecBaseOffset);
        }
    }
    if constexpr (g_coreType == AIC) {
        gmmAddrInfo.dispatchToGmm1Flag =
            reinterpret_cast<__gm__ int32_t*>(workspace.sync.dispatchReady.get_ptr(workspaceBase)) +
            expertOffset * syncLayout.dispatchFlagSlotCountPerExpert +
            static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM1_ACTIVATION_ARCH35_H
