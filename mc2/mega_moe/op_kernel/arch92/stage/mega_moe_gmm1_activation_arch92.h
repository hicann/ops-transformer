/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM1_ACTIVATION_ARCH92_H
#define MEGA_MOE_GMM1_ACTIVATION_ARCH92_H

#include "../../arch35_92/stage/mega_moe_gmm1_activation.h"

namespace MegaMoeImpl {
namespace GmmKernel {

// Arch92 routed GMM1 固定写 GM，Activation 使用既有预取或顺序消费协议。
template <typename BlockMmad, typename ElementC, typename MakeLayoutC, bool TopkWeightsPrefetch, bool IsShared = false,
          bool IsWaveFlagGrained = false, typename Scheduler, typename Config, typename ActivationQuantOp,
          typename LayoutBundle>
__aicore__ inline void Gmm1ExecToGm(Scheduler& scheduler, const Params& params, const GMMAddrInfo& gmmAddrInfo,
                                    const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                    ActivationQuantOp& activationQuantOp, uint32_t expertBeforeCnt, uint32_t expertIdx,
                                    uint16_t& pingpongIdx, BlockMmadContext<BlockMmad>* blockMmadContext,
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
    GM_ADDR cGlobal = gmmAddrInfo.gmm1OutGlobal;
    auto gmC = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementC*>(cGlobal)), layouts.c);
    __gm__ float* metaInfoGm = nullptr;
    if constexpr (TopkWeightsPrefetch && g_coreType == AscendC::AIV) {
        metaInfoGm = reinterpret_cast<__gm__ float*>(gmmAddrInfo.metaInfoGlobal);
    }

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmC), decltype(metaInfoGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmC, metaInfoGm};
    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            Gmm1AicMmadToGmGeneric<BlockMmadContext<BlockMmad>, IsShared, TopkWeightsPrefetch>(
                workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, expertIdx, blockMmadContext);
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            Gmm1AicMmadToGmGeneric<BlockMmadContext<BlockMmad>, IsShared, TopkWeightsPrefetch>(
                workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, expertIdx, &localBlockMmadContext);
        }
    } else {
        if constexpr (TopkWeightsPrefetch) {
            Gmm1Aiv0PrefetchEpilogueGeneric<MakeLayoutC, ElementC, IsWaveFlagGrained>(
                workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, activationQuantOp, expertBeforeCnt,
                expertIdx);
        } else {
            auto& sync = *gmmAddrInfo.gmm1ActivationSync;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
                auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
                auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
                uint32_t mLoc = Get<M_VALUE>(blockCoord);
                uint32_t nLoc = Get<N_VALUE>(blockCoord);
                sync.WaitForGmm1();
                Gmm1AivReadGmEpilogueTileGeneric<ElementC, MakeLayoutC, IsWaveFlagGrained>(
                    activationQuantOp, workSet.gmC, gmmAddrInfo, config, actualShape, mLoc, nLoc);
                sync.NotifyGmm1();
            }
        }
    }
}

} // namespace GmmKernel

// 构造 routed GMM1 的矩阵配置和分核状态；共享专家仍使用公共入口。
template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsWaveFlagGrained = false, typename BlockEpilogue, typename... LayoutArgs>
__aicore__ inline void RunGmm1ToGm(BlockEpilogue& epilogueOp, const Params& params,
                                   const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                   const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                   const BlockJobContext& blockJob, uint32_t expertBeforeCnt, uint32_t expertIdx,
                                   uint16_t& pingpongIdx, uint32_t activationOutputDivisor = ACTIVATION_N_HALF,
                                   void* blockMmadContext = nullptr, bool allowWeightL2Bypass = false,
                                   const LayoutArgs&... layoutArgs)
{
    using GmmConfig =
        GmmKernel::Config<false, COMBINE_NO_QUANT, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                          IsWeightNZ, TopkWeightsPrefetch, IsShared, false, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM, activationOutputDivisor);

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{asc::te::make_coord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if constexpr (g_coreType == AIV) {
        epilogueOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = GmmKernel::BlockMmadContext<BlockMmad>;
    auto* typedBlockMmadContext = reinterpret_cast<MmadContext*>(blockMmadContext);
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;
    auto layouts = GmmConfig::BuildLayouts(config, layoutArgs...);
    GmmKernel::Gmm1ExecToGm<BlockMmad, ElementC, MakeLayoutC, TopkWeightsPrefetch, IsShared, IsWaveFlagGrained>(
        scheduler, params, gmmAddrInfo, config, startLoopIdx, tileNum, epilogueOp, expertBeforeCnt, expertIdx,
        pingpongIdx, typedBlockMmadContext, allowWeightL2Bypass, layouts);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

// 绑定 routed GMM1 的输入、权重、GM 结果及 Activation 输出地址。
template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          uint32_t ActivationElementsPerByte, typename BlockEpilogue>
__aicore__ inline void UpdateMoeExpertGmm1GlobalBufferToGm(const GmmExecutionConfig& gmmConfig,
                                                           const MoeSyncWorkspaceLayout& syncLayout,
                                                           const WorkspaceLayout& workspace, GM_ADDR workspaceBase,
                                                           const ExpertWeightTensorListAddrs& weights,
                                                           BlockEpilogue& epilogueOp, GMMAddrInfo& gmmAddrInfo,
                                                           const ExpertLoopState& state, uint32_t activationOutputWidth,
                                                           uint32_t rowOffsetInExpert = 0U)
{
    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElementsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    int64_t n = Get<N_VALUE>(state.problemShape);
    int64_t k = Get<K_VALUE>(state.problemShape);
    int64_t expertOffset = state.expertIdx;
    int64_t globalTokenStartIndex = state.globalTokenStartIndex + rowOffsetInExpert;
    uint32_t expertMGroupOffset = rowOffsetInExpert / L1_TILE_M_256;
    int64_t scaleK = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    gmmAddrInfo.gmm1OutGlobal =
        workspace.routed.gmm1Output.get_ptr(workspaceBase) + globalTokenStartIndex * n * sizeof(bfloat16_t);
    gmmAddrInfo.metaInfoGlobal = workspace.routing.receivedTokenMetadata.get_ptr(workspaceBase);
    gmmAddrInfo.activationToGmm2Flag =
        reinterpret_cast<__gm__ int32_t*>(workspace.sync.activationReady.get_ptr(workspaceBase)) +
        expertOffset * syncLayout.activationFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.aGlobal = workspace.routed.gmm1Input.data.get_ptr(workspaceBase) +
                          globalTokenStartIndex * k / ActivationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = workspace.routed.gmm1Input.scale.get_ptr(workspaceBase) +
                               globalTokenStartIndex * scaleK * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<WeightType>(weights.weight1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                        static_cast<uint64_t>(state.expertIdx) * n * k / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal =
        GetExpertWeightAddr<QuantScaleType>(weights.weightScales1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                            static_cast<uint64_t>(state.expertIdx) * n * scaleK);
    if constexpr (g_coreType == AIV) {
        int64_t scaleN =
            Ops::Base::CeilDiv(static_cast<int64_t>(activationOutputWidth), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            globalTokenStartIndex * activationOutputWidth / outputElementsPerByte,
            globalTokenStartIndex * scaleN,
            expertOffset * syncLayout.activationFlagSlotCountPerExpert / INT_CACHELINE + expertMGroupOffset,
            0L,
            0L,
            0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);
    }
    if constexpr (g_coreType == AIC) {
        gmmAddrInfo.dispatchToGmm1Flag =
            reinterpret_cast<__gm__ int32_t*>(workspace.sync.dispatchReady.get_ptr(workspaceBase)) +
            expertOffset * syncLayout.dispatchFlagSlotCountPerExpert +
            static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM1_ACTIVATION_ARCH92_H
