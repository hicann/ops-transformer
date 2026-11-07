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
 * \file mega_moe_wave.h
 * \brief MegaMoe arch92 1C1V wave 流水
 */

#ifndef MEGA_MOE_ARCH92_WAVE_H
#define MEGA_MOE_ARCH92_WAVE_H

#include <type_traits>

#include "../arch35_92/common/mega_moe_utils.h"
#include "mega_moe_arch92.h"
#include "stage/mega_moe_gmm1_activation_arch92.h"

namespace MegaMoeImpl {

constexpr uint32_t GMM2_LAG_MIN_TOKEN_NUM = 4096U;

#define TemplateMegaMoeArch92WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename MoeWeightType, int32_t MoeQuantMode, \
        typename SharedWeightType, int32_t SharedQuantMode, int32_t MoeWeight1Format, int32_t MoeWeight2Format, \
        int32_t SharedWeight1Format, int32_t SharedWeight2Format, int32_t CombineQuantMode, bool TopkWeightsPrefetch, \
        typename TopkIndexType
#define TemplateMegaMoeArch92WaveTypeFunc \
    XType, OutputType, TopkWeightsType, MoeWeightType, MoeQuantMode, SharedWeightType, SharedQuantMode, \
        MoeWeight1Format, MoeWeight2Format, SharedWeight1Format, SharedWeight2Format, CombineQuantMode, \
        TopkWeightsPrefetch, TopkIndexType

/*
 * Init、输入准备、共享专家和 Unpermute 由 MegaMoe 基类统一实现；本类保留 arch92 的
 * 专家切片级 Dispatch/Activation 交替、GMM1/GMM2 交错调度及量化 Combine 编排。
 */
template <TemplateMegaMoeArch92WaveTypeClass>
class MegaMoeArch92Wave : public MegaMoe<TemplateMegaMoeArch92WaveTypeFunc> {
private:
    using MegaMoeBase = MegaMoe<TemplateMegaMoeArch92WaveTypeFunc>;
    friend MegaMoeBase;

public:
    using MegaMoeBase::Init;
    __aicore__ inline void Process();

private:
    using CombineBufferConfig = WaveCombineBufferConfig;
    using QuantOutType = typename MegaMoeBase::QuantOutType;
    using ActivationType = typename MegaMoeBase::ActivationType;
    using QuantScaleOutType = typename MegaMoeBase::QuantScaleOutType;
    using ActivationQuantOutType = typename MegaMoeBase::ActivationQuantOutType;
    using SharedQuantConfig = typename MegaMoeBase::SharedQuantConfig;
    using MoeQuantConfig = typename MegaMoeBase::MoeQuantConfig;
    using SharedActivationOutType = typename MegaMoeBase::SharedActivationOutType;
    using SharedQuantScaleType = typename MegaMoeBase::SharedQuantScaleType;

    using Gmm1PersistentConfig =
        GmmKernel::Config<false, COMBINE_NO_QUANT, QuantOutType, MoeWeightType, bfloat16_t, QuantScaleOutType,
                          QuantScaleOutType, MoeWeight1Format != FORMAT_ND, TopkWeightsPrefetch, false, false, true>;
    using Gmm2PersistentConfig =
        GmmKernel::Config<false, CombineQuantMode, ActivationQuantOutType, MoeWeightType, bfloat16_t, QuantScaleOutType,
                          QuantScaleOutType, MoeWeight2Format != FORMAT_ND, TopkWeightsPrefetch, false, false, true>;
    using SharedGmm1PersistentConfig =
        GmmKernel::Config<false, COMBINE_NO_QUANT, typename SharedQuantConfig::QuantOutType, SharedWeightType,
                          bfloat16_t, SharedQuantScaleType, SharedQuantScaleType, SharedWeight1Format != FORMAT_ND,
                          false, true, false, true>;
    using SharedGmm2PersistentConfig =
        GmmKernel::Config<false, COMBINE_NO_QUANT, SharedActivationOutType, SharedWeightType, bfloat16_t,
                          SharedQuantScaleType, SharedQuantScaleType, SharedWeight2Format != FORMAT_ND, false, true,
                          false, true>;
    using Gmm1BlockMmad = typename Gmm1PersistentConfig::BlockMmad;
    using Gmm2BlockMmad = typename Gmm2PersistentConfig::BlockMmad;
    using SharedGmm1BlockMmad = typename SharedGmm1PersistentConfig::BlockMmad;
    using SharedGmm2BlockMmad = typename SharedGmm2PersistentConfig::BlockMmad;
    static_assert(MoeQuantConfig::AXW_MODE != AxWMode::A8W4 && SharedQuantConfig::AXW_MODE != AxWMode::A8W4,
                  "arch92 A8W4 requires the direct mixed FP8/FP4 Tensor BlockMmad interface");
    struct EmptyMmadContext {};
    template <typename BlockMmad>
    using MmadContextStorage =
        std::conditional_t<g_coreType == AIC, GmmKernel::BlockMmadContext<BlockMmad>, EmptyMmadContext>;
    static constexpr bool ROUTED_AND_SHARED_USE_SAME_MMAD = std::is_same_v<Gmm1BlockMmad, SharedGmm1BlockMmad>;
    static constexpr bool SHARED_GMM_STAGES_USE_SAME_MMAD = std::is_same_v<SharedGmm1BlockMmad, SharedGmm2BlockMmad>;
    static_assert(std::is_same_v<Gmm1BlockMmad, Gmm2BlockMmad>,
                  "Arch92 routed GMM1/GMM2 must share the persistent MMAD context type");
    static_assert(SHARED_GMM_STAGES_USE_SAME_MMAD,
                  "Arch92 shared GMM1/GMM2 must share the persistent MMAD context type");
    using SharedMmadContextStorage =
        std::conditional_t<g_coreType == AIC && !ROUTED_AND_SHARED_USE_SAME_MMAD,
                           GmmKernel::BlockMmadContext<SharedGmm1BlockMmad>, EmptyMmadContext>;

    __aicore__ inline void* GetRoutedMmadContext()
    {
        if constexpr (g_coreType == AIC) {
            return static_cast<void*>(&routedMmadContext_);
        }
        return nullptr;
    }

    __aicore__ inline void* GetRoutedGmm2MmadContext()
    {
        return GetRoutedMmadContext();
    }

    __aicore__ inline void* GetSharedMmadContext()
    {
        if constexpr (g_coreType == AIC) {
            if constexpr (ROUTED_AND_SHARED_USE_SAME_MMAD) {
                return static_cast<void*>(&routedMmadContext_);
            }
            return static_cast<void*>(&sharedMmadContext_);
        }
        return nullptr;
    }

    static constexpr uint32_t GMM1_TILE_M = MegaMoeBase::GMM1_TILE_M;
    static constexpr uint32_t EPILOGUE_TILE_M = MegaMoeBase::EPILOGUE_TILE_M;

    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::PrepareMoeExpertTokenCountTable;
    using MegaMoeBase::EnterSteadyDispatch;
    using MegaMoeBase::commonConfig_;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::exceptionDump_;
    using MegaMoeBase::gmm1PingPongIdx_;
    using MegaMoeBase::gmmExecutionConfig_;
    using MegaMoeBase::gmmLoopCount_;
    using MegaMoeBase::mGroupsPerWave_;
    using MegaMoeBase::moeExpertPerRank_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::startBlockIdx_;
    using MegaMoeBase::syncWorkspaceLayout_;
    using MegaMoeBase::tokenDispatchConfig_;
    using MegaMoeBase::tokenDispatchScratch_;
    using MegaMoeBase::waveCombineJob_;
    using MegaMoeBase::waveCombineScratch_;
    using MegaMoeBase::waveCombineDedupScratch_;

    struct Gmm2WaveState {
        ExpertTokenPosition position{};
        ExpertLoopState expertState{};
        GMMAddrInfo addrInfo{};
        ExpertLoopState combineExpertState{};
        GMMAddrInfo combineAddrInfo{};
        uint32_t combineBeginExpertIndex = 0U;
        uint32_t finalCombineExpertIndex = 0U;
        uint64_t loopCount = 0U;
    };

    __aicore__ inline uint32_t GetMoeExpertTokenCount(uint32_t expertIdx) const;
    __aicore__ inline uint32_t FindFinalCombineExpert();
    __aicore__ inline void ProcessGmm2AndCombineWave(const ExpertTokenPosition& waveEndPosition, Gmm2WaveState& state,
                                                     const CombineBufferConfig& combineBufferConfig);
    /*
     * 唯一 AIV 按阶段复用临时 UB；Dispatch cumsum 与专家 token 数常驻物理 UB 尾部。
     * 量化 Combine 沿用公共布局：行环与量化 scratch 位于 [64, 184 KiB)，ready 扫描从
     * 184 KiB 开始，metadata 从 200 KiB 开始。稳态使用一个行槽，最后一个非空专家可扩容
     * 至自适应的最多六槽，但必须先排空旧行环。这些地址不代表 A6 的 UB 总容量。
     */
    __aicore__ inline CombineBufferConfig InitCombineBuffers();
    __aicore__ inline void ProcessMoeExpertStages(Gmm1ActivationSync& gmm1ActivationSync,
                                                  Gmm2CombineSync& gmm2CombineSync);
    __aicore__ inline bool IsSameExpertTokenPosition(const ExpertTokenPosition& currentPosition,
                                                     const ExpertTokenPosition& targetPosition) const;
    __aicore__ inline ExpertTokenPosition DispatchFirstWave();
    __aicore__ inline void DispatchNextWaveExpertSlice(ExpertTokenPosition& dispatchPosition,
                                                       uint32_t& dispatchWaveMGroupCount);
    __aicore__ inline ExpertTokenPosition ProcessNextDispatchAndCurrentGmm1(ExpertTokenPosition& dispatchPosition,
                                                                            ExpertTokenPosition& gmm1Position,
                                                                            ExpertLoopState& gmm1ExpertState,
                                                                            GMMAddrInfo& gmm1AddrInfo);
    __aicore__ inline void ProcessGmm2Wave(ExpertTokenPosition& gmm2Position,
                                           const ExpertTokenPosition& waveEndPosition, ExpertLoopState& gmm2ExpertState,
                                           GMMAddrInfo& gmm2AddrInfo, uint32_t& startBlockIdx);
    __aicore__ inline void ProcessGmm2WaveProblem(const ExpertTokenPosition& gmm2Position,
                                                  uint32_t waveEndTokenIndexInExpert,
                                                  const ExpertLoopState& gmm2ExpertState, GMMAddrInfo& gmm2AddrInfo,
                                                  uint32_t& startBlockIdx);
    __aicore__ inline void AdvanceGmm2WavePosition(ExpertTokenPosition& gmm2Position,
                                                   uint32_t waveEndTokenIndexInExpert,
                                                   const ExpertLoopState& gmm2ExpertState);
    __aicore__ inline void ProcessCombineExperts(uint32_t expertBegin, uint32_t expertEnd,
                                                 ExpertLoopState& combineState, GMMAddrInfo& combineAddrInfo,
                                                 const CombineBufferConfig& bufferConfig,
                                                 uint32_t finalCombineExpertIndex);

    uint32_t gmm1TilesPerMGroup_ = 1U;
    uint32_t gmm2TilesPerMGroup_ = 1U;
    MmadContextStorage<Gmm1BlockMmad> routedMmadContext_;
    SharedMmadContextStorage sharedMmadContext_;
};

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline typename MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::CombineBufferConfig
MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::InitCombineBuffers()
{
    if constexpr (g_coreType == AIV) {
        return InitWaveCombineBuffers<CombineQuantMode, WAVE_COMBINE_STEADY_ROW_BUFFER_COUNT>(commonConfig_,
                                                                                              waveCombineScratch_);
    }
    return {};
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline bool MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::IsSameExpertTokenPosition(
    const ExpertTokenPosition& currentPosition, const ExpertTokenPosition& targetPosition) const
{
    return currentPosition.expertIdx == targetPosition.expertIdx &&
           currentPosition.tokenIndexInExpert == targetPosition.tokenIndexInExpert;
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline uint32_t MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::GetMoeExpertTokenCount(
    uint32_t expertIdx) const
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    if constexpr (g_coreType == AIV) {
        return static_cast<uint32_t>(tokenDispatchScratch_.expertTokenNumsOutTensor[expertIdx]);
    } else {
        return GetExpertTokenCountFromWorkspace(workspace.routing.recvCounts.get_ptr(buffer), countWorkspace_,
                                                moeExpertPerRank_, expertIdx);
    }
}

// 先 Dispatch W0，再交替执行下一 Wave 的 Dispatch 和当前 Wave 的 Activation。
template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline ExpertTokenPosition MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::DispatchFirstWave()
{
    ExpertTokenPosition dispatchPosition{};
    ExpertTokenRange dispatchRange{dispatchPosition, dispatchPosition};
    ExpertTokenPosition plannedDispatchPosition = dispatchPosition;
    uint32_t dispatchWaveMGroupCount = 0U;
    while (plannedDispatchPosition.expertIdx < moeExpertPerRank_ && dispatchWaveMGroupCount < mGroupsPerWave_) {
        uint32_t expertTokenCount = GetMoeExpertTokenCount(plannedDispatchPosition.expertIdx);
        if (expertTokenCount == 0U || plannedDispatchPosition.tokenIndexInExpert >= expertTokenCount) {
            ++plannedDispatchPosition.expertIdx;
            plannedDispatchPosition.tokenIndexInExpert = 0U;
        } else {
            AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(expertTokenCount, mGroupsPerWave_, dispatchWaveMGroupCount,
                                                          plannedDispatchPosition);
        }
        dispatchRange.end = plannedDispatchPosition;
    }
    DispatchTokenRange<TopkIndexType, ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
        tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_, params_,
        g_winRankAddr_, tokenDispatchScratch_, dispatchRange);
    // 整 WAVE 的数据和 ready flag 发布完成后，再提交 Dispatch 进度。
    dispatchPosition = plannedDispatchPosition;
    return dispatchPosition;
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::DispatchNextWaveExpertSlice(
    ExpertTokenPosition& dispatchPosition, uint32_t& dispatchWaveMGroupCount)
{
    ExpertTokenPosition plannedDispatchPosition = dispatchPosition;
    while (plannedDispatchPosition.expertIdx < moeExpertPerRank_ && dispatchWaveMGroupCount < mGroupsPerWave_) {
        uint32_t expertTokenCount = GetMoeExpertTokenCount(plannedDispatchPosition.expertIdx);
        if (expertTokenCount == 0U || plannedDispatchPosition.tokenIndexInExpert >= expertTokenCount) {
            ++plannedDispatchPosition.expertIdx;
            plannedDispatchPosition.tokenIndexInExpert = 0U;
            continue;
        }

        ExpertTokenRange dispatchRange{plannedDispatchPosition, plannedDispatchPosition};
        AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(expertTokenCount, mGroupsPerWave_, dispatchWaveMGroupCount,
                                                      plannedDispatchPosition);
        dispatchRange.end = plannedDispatchPosition;
        // cumsum 和专家 token 数表在 UB 尾部预留区中持续有效。
        DispatchTokenRange<TopkIndexType, ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
            tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_, params_,
            g_winRankAddr_, tokenDispatchScratch_, dispatchRange);
        break;
    }
    dispatchPosition = plannedDispatchPosition;
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline ExpertTokenPosition
MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessNextDispatchAndCurrentGmm1(
    ExpertTokenPosition& dispatchPosition, ExpertTokenPosition& gmm1Position, ExpertLoopState& gmm1ExpertState,
    GMMAddrInfo& gmm1AddrInfo)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    uint32_t processedMGroupCount = 0U;
    uint32_t dispatchWaveMGroupCount = 0U;
    bool nextWaveNeedsDispatch = false;
    if constexpr (g_coreType == AIV) {
        nextWaveNeedsDispatch = dispatchPosition.expertIdx < moeExpertPerRank_;
    }
    while ((gmm1Position.expertIdx < moeExpertPerRank_ && processedMGroupCount < mGroupsPerWave_) ||
           nextWaveNeedsDispatch) {
        if constexpr (g_coreType == AIV) {
            if (nextWaveNeedsDispatch) {
                DispatchNextWaveExpertSlice(dispatchPosition, dispatchWaveMGroupCount);
                nextWaveNeedsDispatch =
                    dispatchPosition.expertIdx < moeExpertPerRank_ && dispatchWaveMGroupCount < mGroupsPerWave_;
            }
        }
        if (gmm1Position.expertIdx >= moeExpertPerRank_ || processedMGroupCount >= mGroupsPerWave_) {
            continue;
        }
        if (gmm1Position.tokenIndexInExpert == 0U) {
            if (!gmm1ExpertState.expertCountTableReady) {
                WaitForMoeExpertTokenCountReady(workspace.sync.countsReady.get_ptr(buffer), countWorkspace_, 0U);
                gmm1ExpertState.expertCountTableReady = true;
            }
            uint32_t expertTokenCount = GetMoeExpertTokenCount(gmm1Position.expertIdx);
            UpdateExpertLoopState(gmm1ExpertState, gmm1Position.expertIdx, expertTokenCount);
        }

        uint64_t expertRowCount = Get<M_VALUE>(gmm1ExpertState.problemShape);
        if (expertRowCount == 0U || gmm1Position.tokenIndexInExpert >= expertRowCount) {
            ++gmm1Position.expertIdx;
            gmm1Position.tokenIndexInExpert = 0U;
            continue;
        }

        uint32_t remainingMGroupCount = mGroupsPerWave_ - processedMGroupCount;
        uint32_t waveEndTokenIndexInExpert = GetWaveEndRowOffsetInExpert(
            expertRowCount, gmm1Position.tokenIndexInExpert, remainingMGroupCount, GMM1_TILE_M);
        uint32_t waveRowCount = waveEndTokenIndexInExpert - gmm1Position.tokenIndexInExpert;
        uint32_t problemMGroupCount = GetMGroupCountForRows(waveRowCount, GMM1_TILE_M);
        bool skipGmm1Problem = false;
        if constexpr (g_coreType == AIC) {
            uint32_t problemTileCount = problemMGroupCount * gmm1TilesPerMGroup_;
            skipGmm1Problem =
                HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx_);
        }
        if (skipGmm1Problem) {
            processedMGroupCount += problemMGroupCount;
            gmm1Position.tokenIndexInExpert = waveEndTokenIndexInExpert;
            gmm1Position.globalTokenIndex += waveRowCount;
            if (gmm1Position.tokenIndexInExpert >= expertRowCount) {
                ++gmm1Position.expertIdx;
                gmm1Position.tokenIndexInExpert = 0U;
            }
            continue;
        }
        ProblemShape gmm1WaveProblemShape = gmm1ExpertState.problemShape;
        Get<M_VALUE>(gmm1WaveProblemShape) = waveEndTokenIndexInExpert - gmm1Position.tokenIndexInExpert;
        UpdateMoeExpertGmm1GlobalBufferToGm<ActivationType, MoeWeightType, ActivationQuantOutType, QuantScaleOutType,
                                            PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE>(
            gmmExecutionConfig_, syncWorkspaceLayout_, workspace, buffer, moeWeightTensorListAddrs_, epilogueOp_,
            gmm1AddrInfo, gmm1ExpertState, commonConfig_.gmm1OutputDim / commonConfig_.activationOutputDivisor,
            gmm1Position.tokenIndexInExpert);
        /*
         * 只有该 problem 从第 0 行覆盖到专家尾，后续 Wave 才不会再次读取同一份 GMM1 权重。
         * 热点专家的任一 Wave problem 都为 false，即使本次不足 256 行也保留正常 L2 cache。
         */
        bool isWholeExpert =
            gmm1Position.tokenIndexInExpert == 0U && static_cast<uint64_t>(waveEndTokenIndexInExpert) == expertRowCount;
        uint32_t waveTokenStartIndex =
            static_cast<uint32_t>(gmm1ExpertState.globalTokenStartIndex) + gmm1Position.tokenIndexInExpert;
        int32_t vecSetSyncCom = 0;
        void* blockMmadContext = GetRoutedMmadContext();
        RunGmm1Generic<QuantOutType, ActivationType, MoeWeightType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                       MoeWeight1Format != FORMAT_ND, GMM1_TILE_M, EPILOGUE_TILE_M, TopkWeightsPrefetch, false, true>(
            epilogueOp_, params_, gmm1WaveProblemShape, gmm1AddrInfo, startBlockIdx_, vecSetSyncCom,
            gmmExecutionConfig_.blockJob, waveTokenStartIndex, gmm1Position.expertIdx, gmm1PingPongIdx_,
            commonConfig_.activationOutputDivisor, blockMmadContext, isWholeExpert);

        processedMGroupCount += problemMGroupCount;
        gmm1Position.tokenIndexInExpert = waveEndTokenIndexInExpert;
        gmm1Position.globalTokenIndex += waveRowCount;
        if (gmm1Position.tokenIndexInExpert >= expertRowCount) {
            ++gmm1Position.expertIdx;
            gmm1Position.tokenIndexInExpert = 0U;
        }
    }
    return gmm1Position;
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessGmm2WaveProblem(
    const ExpertTokenPosition& gmm2Position, uint32_t waveEndTokenIndexInExpert, const ExpertLoopState& gmm2ExpertState,
    GMMAddrInfo& gmm2AddrInfo, uint32_t& startBlockIdx)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    uint64_t expertRowCount = Get<M_VALUE>(gmm2ExpertState.problemShape);
    uint32_t waveRowCount = waveEndTokenIndexInExpert - gmm2Position.tokenIndexInExpert;
    uint32_t problemMGroupCount = GetMGroupCountForRows(waveRowCount, GMM1_TILE_M);
    bool skipGmm2Problem = false;
    if constexpr (g_coreType == AIC) {
        uint32_t problemTileCount = problemMGroupCount * gmm2TilesPerMGroup_;
        skipGmm2Problem = HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx);
    }
    if (skipGmm2Problem) {
        return;
    }

    ProblemShape gmm2WaveProblemShape = gmm2ExpertState.problemShape;
    Get<M_VALUE>(gmm2WaveProblemShape) = waveEndTokenIndexInExpert - gmm2Position.tokenIndexInExpert;
    UpdateMoeExpertGmm2GlobalBuffer<MoeWeightType, ActivationQuantOutType, QuantScaleOutType>(
        gmmExecutionConfig_, syncWorkspaceLayout_, workspace, buffer, moeWeightTensorListAddrs_, gmm2AddrInfo,
        gmm2ExpertState, commonConfig_.gmm1OutputDim / commonConfig_.activationOutputDivisor,
        gmm2Position.tokenIndexInExpert);
    // GMM2 与 GMM1 使用相同保护：只有完整专家 problem 才允许进一步判断是否绕过 L2。
    bool isWholeExpert =
        gmm2Position.tokenIndexInExpert == 0U && static_cast<uint64_t>(waveEndTokenIndexInExpert) == expertRowCount;
    void* blockMmadContext = GetRoutedGmm2MmadContext();
    if (IsCombineDedupOn(static_cast<int32_t>(params_.tilingData->compute.dedupMode))) {
        RunGmm2Generic<COMBINE_NO_QUANT, ActivationQuantOutType, MoeWeightType, bfloat16_t, QuantScaleOutType,
                       QuantScaleOutType, MoeWeight2Format != FORMAT_ND, false, GMM1_TILE_M, TopkWeightsPrefetch, false,
                       true, false>(gmm2WaveProblemShape, gmm2AddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob,
                                    blockMmadContext, isWholeExpert, gmm2Position.tokenIndexInExpert, &params_,
                                    commonConfig_.activationOutputDivisor);
    } else {
        RunGmm2Generic<COMBINE_NO_QUANT, ActivationQuantOutType, MoeWeightType, bfloat16_t, QuantScaleOutType,
                       QuantScaleOutType, MoeWeight2Format != FORMAT_ND, false, GMM1_TILE_M, TopkWeightsPrefetch, false,
                       true, CombineQuantMode == COMBINE_NO_QUANT>(
            gmm2WaveProblemShape, gmm2AddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob, blockMmadContext,
            isWholeExpert, gmm2Position.tokenIndexInExpert, &params_, commonConfig_.activationOutputDivisor);
    }
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::AdvanceGmm2WavePosition(
    ExpertTokenPosition& gmm2Position, uint32_t waveEndTokenIndexInExpert, const ExpertLoopState& gmm2ExpertState)
{
    uint64_t expertRowCount = Get<M_VALUE>(gmm2ExpertState.problemShape);
    uint32_t waveRowCount = waveEndTokenIndexInExpert - gmm2Position.tokenIndexInExpert;
    gmm2Position.tokenIndexInExpert = waveEndTokenIndexInExpert;
    gmm2Position.globalTokenIndex += waveRowCount;
    if (gmm2Position.tokenIndexInExpert >= expertRowCount) {
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            NotifyWaveGmm2Ready(waveCombineJob_, params_, gmm2Position.expertIdx);
        } else if (IsCombineDedupOn(static_cast<int32_t>(params_.tilingData->compute.dedupMode))) {
            NotifyWaveGmm2Ready(waveCombineJob_, params_, gmm2Position.expertIdx);
        }
        ++gmm2Position.expertIdx;
        gmm2Position.tokenIndexInExpert = 0U;
    }
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessGmm2Wave(
    ExpertTokenPosition& gmm2Position, const ExpertTokenPosition& waveEndPosition, ExpertLoopState& gmm2ExpertState,
    GMMAddrInfo& gmm2AddrInfo, uint32_t& startBlockIdx)
{
    while (!IsSameExpertTokenPosition(gmm2Position, waveEndPosition) && gmm2Position.expertIdx < moeExpertPerRank_) {
        if (gmm2Position.tokenIndexInExpert == 0U) {
            uint32_t expertTokenCount = GetMoeExpertTokenCount(gmm2Position.expertIdx);
            UpdateExpertLoopState(gmm2ExpertState, gmm2Position.expertIdx, expertTokenCount);
        }

        uint64_t expertRowCount = Get<M_VALUE>(gmm2ExpertState.problemShape);
        if (expertRowCount == 0U || gmm2Position.tokenIndexInExpert >= expertRowCount) {
            ++gmm2Position.expertIdx;
            gmm2Position.tokenIndexInExpert = 0U;
            continue;
        }

        uint32_t waveEndTokenIndexInExpert = static_cast<uint32_t>(expertRowCount);
        if (gmm2Position.expertIdx == waveEndPosition.expertIdx) {
            waveEndTokenIndexInExpert = waveEndPosition.tokenIndexInExpert;
        }
        ProcessGmm2WaveProblem(gmm2Position, waveEndTokenIndexInExpert, gmm2ExpertState, gmm2AddrInfo, startBlockIdx);
        AdvanceGmm2WavePosition(gmm2Position, waveEndTokenIndexInExpert, gmm2ExpertState);
    }
}

// arch92 只有一个 AIV 角色：同一 AIV 在 GMM2 epilogue 后继续执行当前 WAVE 的 Combine。
template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessCombineExperts(
    uint32_t expertBegin, uint32_t expertEnd, ExpertLoopState& combineState, GMMAddrInfo& combineAddrInfo,
    const CombineBufferConfig& bufferConfig, uint32_t finalCombineExpertIndex)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    if constexpr (g_coreType == AIC) {
        return;
    }
    CombineBufferConfig activeBufferConfig = bufferConfig;
    uint32_t rowSequence = 0U;
    const bool dedupCombine = IsCombineDedupOn(static_cast<int32_t>(params_.tilingData->compute.dedupMode));
    if (dedupCombine) {
        ArmWaveCombineDedupEvents();
    }
    for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
        uint32_t expertTokenCount = GetMoeExpertTokenCount(expertIdx);
        UpdateExpertLoopState(combineState, expertIdx, expertTokenCount);
        if (expertTokenCount == 0U) {
            continue;
        }
        UpdateMoeExpertCombineGlobalBuffer(workspace, buffer, combineAddrInfo, combineState);
        if (dedupCombine) {
            RunWaveCombineStageDedup(commonConfig_, waveCombineJob_, activeBufferConfig, waveCombineScratch_,
                                     waveCombineDedupScratch_, params_, combineState, expertIdx, rowSequence);
            continue;
        }
        if (expertIdx == finalCombineExpertIndex) {
            // 扩容前按旧槽数消费全部在途事件，并将 rowSequence 归零后重新开始填充。
            DrainCombineRowBuffers(rowSequence, activeBufferConfig.rowBufferCount);
            activeBufferConfig = InitWaveCombineBuffers<CombineQuantMode>(commonConfig_, waveCombineScratch_);
        }
        RunWaveCombineStage<CombineQuantMode>(commonConfig_, waveCombineJob_, activeBufferConfig, waveCombineScratch_,
                                              params_, combineAddrInfo, combineState, combineState.expertIdx,
                                              rowSequence);
    }
    if (dedupCombine) {
        DrainCombineRowBuffersDedup(rowSequence, activeBufferConfig.rowBufferCount);
    } else {
        DrainCombineRowBuffers(rowSequence, activeBufferConfig.rowBufferCount);
    }
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline uint32_t MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::FindFinalCombineExpert()
{
    if constexpr (g_coreType == AIV) {
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            for (uint32_t expertEnd = moeExpertPerRank_; expertEnd > 0U; --expertEnd) {
                uint32_t expertIdx = expertEnd - 1U;
                if (GetMoeExpertTokenCount(expertIdx) != 0U) {
                    return expertIdx;
                }
            }
        }
    }

    return moeExpertPerRank_;
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessGmm2AndCombineWave(
    const ExpertTokenPosition& waveEndPosition, Gmm2WaveState& state, const CombineBufferConfig& combineBufferConfig)
{
    ProcessGmm2Wave(state.position, waveEndPosition, state.expertState, state.addrInfo, startBlockIdx_);
    UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++state.loopCount);
    if (CombineQuantMode != COMBINE_NO_QUANT ||
        IsCombineDedupOn(static_cast<int32_t>(params_.tilingData->compute.dedupMode))) {
        ProcessCombineExperts(state.combineBeginExpertIndex, waveEndPosition.expertIdx, state.combineExpertState,
                              state.combineAddrInfo, combineBufferConfig, state.finalCombineExpertIndex);
        state.combineBeginExpertIndex = waveEndPosition.expertIdx;
    }
}

/*
 * 按动态 WAVE 边界滚动执行 Arch92 MoE 流水。唯一 AIV 启动时只 Dispatch W0，稳态按专家切片交替
 * Dispatch 下一 WAVE 和处理当前 WAVE 的 Activation；AIC 只执行当前 WAVE 的 GMM1。
 * 两侧各自推进 WAVE 容量，均完成后才进入 GMM2/Combine；后者仍使用当前 GMM1 的 WAVE 终点。
 */
template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::ProcessMoeExpertStages(
    Gmm1ActivationSync& gmm1ActivationSync, Gmm2CombineSync& gmm2CombineSync)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const uint32_t gmm1SchedulerWidth = commonConfig_.gmm1OutputDim;
    gmm1TilesPerMGroup_ = Ops::Base::CeilDiv(gmm1SchedulerWidth, static_cast<uint32_t>(L1_TILE_N));
    gmm2TilesPerMGroup_ = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));

    // GMM1/GMM2 交错流水只记录一次阶段入口，各 Wave 完成轮次由独立计数记录。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::MOE_GMM1_ACTIVATION);
    uint64_t gmm1Count = 0U;
    DispatchBuffInit();
    CombineBufferConfig combineBufferConfig{};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        combineBufferConfig = InitCombineBuffers();
    } else if (IsCombineDedupOn(static_cast<int32_t>(params_.tilingData->compute.dedupMode))) {
        combineBufferConfig = InitWaveCombineBuffersDedup(commonConfig_, waveCombineScratch_, waveCombineDedupScratch_);
    }
    PrepareMoeExpertTokenCountTable();
    if (tokenDispatchConfig_.dedup.dedupMode != 0) {
        // Arch92 没有独立的 AIV0 工作核，也不使用完成计数的高 16 位。
        if constexpr (g_coreType == AIV) {
            BuildDedupRowDescTable<TopkIndexType>(tokenDispatchConfig_.dedup, commonConfig_,
                                                  gmmExecutionConfig_.blockJob, static_cast<int32_t>(1),
                                                  syncWorkspaceLayout_, params_, tokenDispatchScratch_);
        }
    }

    GMMAddrInfo gmm1AddrInfo{};
    ExpertLoopState gmm1ExpertState = CreateExpertLoopState(commonConfig_);
    Gmm2WaveState gmm2State{};
    gmm1AddrInfo.gmm1ActivationSync = &gmm1ActivationSync;
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if (!IsCombineDedupOn(static_cast<int32_t>(params_.tilingData->compute.dedupMode))) {
            gmm2State.addrInfo.gmm2CombineSync = &gmm2CombineSync;
        }
    }

    gmm2State.expertState = CreateExpertLoopState(commonConfig_);
    gmm2State.combineExpertState = CreateExpertLoopState(commonConfig_);
    ExpertTokenPosition dispatchPosition{};

    ExpertTokenPosition waveBeginPosition{};
    ExpertTokenPosition gmm1Position{};
    // GMM2 的滞后消费目标（见循环内注释）；首轮为零位置，GMM2 空转一轮。
    // 沿用按 token 数开启 lag 的门控；A6 上的性能收益仍需单独测量。
    const bool gmm2LagActive = commonConfig_.tokenNum >= GMM2_LAG_MIN_TOKEN_NUM;
    // AIC 与唯一 AIV 整体滞后一拍的缓存：上一 wave 的边界（见循环内注释）。
    ExpertTokenPosition gmm2PendingWaveEnd{};
    bool hasPendingGmm2Wave = false;

    if constexpr (g_coreType == AIV) {
        WaitForMoeExpertTokenCountReady(workspace.sync.countsReady.get_ptr(buffer), countWorkspace_, 0U);
    }
    gmm2State.finalCombineExpertIndex = FindFinalCombineExpert();
    if constexpr (g_coreType == AIV) {
        dispatchPosition = DispatchFirstWave();
        EnterSteadyDispatch();
    }

    while (waveBeginPosition.expertIdx < moeExpertPerRank_) {
        const uint32_t waveStartBlockIndex = startBlockIdx_;
        ExpertTokenPosition waveEndPosition =
            ProcessNextDispatchAndCurrentGmm1(dispatchPosition, gmm1Position, gmm1ExpertState, gmm1AddrInfo);
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM1, ++gmm1Count);
        const uint32_t gmm1EndBlockIndex = startBlockIdx_;

        /*
         * GMM2/Combine 的 AIC/AIV 整体滞后一拍（大 bs 门控）：wave w 延至 GMM1(w+1)
         * 之后执行，用下一 wave 的 GMM1 覆盖 AIC 等激活发布的自旋（唯一 AIV 在末批 MMAD
         * 后仍需一段向量时间才发布 activationToGmm2Flag，紧跟消费同一 wave 会让 AIC 反复自旋）。
         * AIC 与唯一 AIV 以相同调用序同拍延后：ProcessGmm2Wave 内部的 pairwise tile
         * 序号、分核游标与 combine 状态在 C/V 两侧保持一致推进，不增加跳过 GMM1 的游标补偿。
         */
        if (gmm2LagActive) {
            if (hasPendingGmm2Wave) {
                ProcessGmm2AndCombineWave(gmm2PendingWaveEnd, gmm2State, combineBufferConfig);
            }
            gmm2PendingWaveEnd = waveEndPosition;
            hasPendingGmm2Wave = true;
        } else {
            ProcessGmm2AndCombineWave(waveEndPosition, gmm2State, combineBufferConfig);
        }

        const bool hasNextWave = waveEndPosition.expertIdx < moeExpertPerRank_;
        const bool fixedRoleResonance =
            startBlockIdx_ == waveStartBlockIndex && gmm1EndBlockIndex != waveStartBlockIndex;
        if (hasNextWave && fixedRoleResonance) {
            startBlockIdx_ = gmm1EndBlockIndex;
        }

        waveBeginPosition = waveEndPosition;
    }

    // 滞后流水收尾：AIC 与唯一 AIV 共同补跑最后一个 wave 的 GMM2/Combine。
    if (hasPendingGmm2Wave) {
        ProcessGmm2AndCombineWave(gmm2PendingWaveEnd, gmm2State, combineBufferConfig);
    }

    gmm1PingPongIdx_ = 0;
}

template <TemplateMegaMoeArch92WaveTypeClass>
__aicore__ inline void MegaMoeArch92Wave<TemplateMegaMoeArch92WaveTypeFunc>::Process()
{
    this->ProcessWave(*this, GetSharedMmadContext());
}

#undef TemplateMegaMoeArch92WaveTypeClass
#undef TemplateMegaMoeArch92WaveTypeFunc

} // namespace MegaMoeImpl
#endif
