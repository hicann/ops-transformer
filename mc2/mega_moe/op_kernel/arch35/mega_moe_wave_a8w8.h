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
 * \file mega_moe_wave_a8w8.h
 * \brief MegaMoe A8W8 wave 流水实现
 */

#ifndef MEGA_MOE_WAVE_A8W8_H
#define MEGA_MOE_WAVE_A8W8_H

#include "common/mega_moe_utils.h"
#include "mega_moe_arch35.h"

namespace MegaMoeImpl {

constexpr uint32_t GMM2_LAG_MIN_TOKEN_NUM = 4096U;

#define TemplateMegaMoeA8W8WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename MoeWeightType, int32_t MoeQuantMode, \
        typename SharedWeightType, int32_t SharedQuantMode, int32_t MoeWeight1Format, int32_t MoeWeight2Format, \
        int32_t SharedWeight1Format, int32_t SharedWeight2Format, int32_t CombineQuantMode, bool TopkWeightsPrefetch, \
        bool IsGmm1Interleaved
#define TemplateMegaMoeA8W8WaveTypeFunc \
    XType, OutputType, TopkWeightsType, MoeWeightType, MoeQuantMode, SharedWeightType, SharedQuantMode, \
        MoeWeight1Format, MoeWeight2Format, SharedWeight1Format, SharedWeight2Format, CombineQuantMode, \
        TopkWeightsPrefetch, IsGmm1Interleaved

/*
 * Init、输入准备、共享专家和 Unpermute 由 MegaMoe 基类统一实现；本类保留 A8W8 特有的
 * 双 Wave lookahead、GMM1/GMM2 交错调度及量化 Combine 编排。
 */
template <TemplateMegaMoeA8W8WaveTypeClass>
class MegaMoeA8W8Wave : public MegaMoe<TemplateMegaMoeA8W8WaveTypeFunc> {
private:
    using MegaMoeBase = MegaMoe<TemplateMegaMoeA8W8WaveTypeFunc>;
    friend MegaMoeBase;

public:
    using MegaMoeBase::Init;
    __aicore__ inline void Process();

private:
    using CombineBufferConfig = WaveCombineBufferConfig;
    using QuantOutType = typename MegaMoeBase::QuantOutType;
    using ActivationType = typename MegaMoeBase::ActivationType;
    using QuantScaleOutType = typename MegaMoeBase::QuantScaleOutType;

    static constexpr uint32_t GMM1_TILE_M = MegaMoeBase::GMM1_TILE_M;
    static constexpr uint32_t EPILOGUE_TILE_M = MegaMoeBase::EPILOGUE_TILE_M;

    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::EnterSteadyDispatch;
    using MegaMoeBase::commonConfig_;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::exceptionDump_;
    using MegaMoeBase::gmm1PingPongIdx_;
    using MegaMoeBase::gmmExecutionConfig_;
    using MegaMoeBase::gmmLoopCount_;
    using MegaMoeBase::gmmTileSequence_;
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

    /*
     * AIV1 上 Dispatch 与 Combine 分阶段复用 UB，进入 Combine 前 Dispatch 的动态 ring 已经排空：
     *   [0, 64 KiB)       Dispatch 的 cumsum 等全流程常驻状态；MoE 流水结束后复用其中最多 36 KiB，
     *                     从 GM 恢复并压紧最多 1024 个专家的 token count；
     *   [64, 160 KiB)     非量化 Combine 的 6 个 BF16 row buffer（H 最大 8 KiB）；
     *                     量化 Combine 使用 2 个 [BF16 row | FP8 data + scale] 槽及共享量化 scratch；
     *   [160, 184 KiB)    空闲；
     *   [184, 187.5 KiB)  GMM2-ready 序号的 GM 搬入与逐 AIC lane 检查区；
     *   [187.5, 200 KiB)  空闲；
     *   [200, 248 KiB)    Combine 共用的 meta-info，共 1536 token * 8 int32；
     *   [248, 256 KiB)    硬件保留，不使用。
     */
    __aicore__ inline CombineBufferConfig InitCombineBuffers();
    __aicore__ inline void ProcessMoeExpertStages();
    __aicore__ inline bool IsSameExpertTokenPosition(const ExpertTokenPosition &currentPosition,
                                                     const ExpertTokenPosition &targetPosition) const;
    __aicore__ inline ExpertTokenPosition DispatchNextWave(ExpertTokenPosition &dispatchPosition);
    __aicore__ inline ExpertTokenPosition ProcessGmm1Wave(ExpertTokenPosition &gmm1Position,
                                                          ExpertLoopState &gmm1ExpertState, GMMAddrInfo &gmm1AddrInfo,
                                                          GmmRuntimeState &runtimeState);
    __aicore__ inline void AdvanceStartBlockIdxForSkippedGmm1(const ExpertTokenPosition &waveBeginPosition,
                                                              const ExpertTokenPosition &waveEndPosition);
    __aicore__ inline void ProcessGmm2Wave(ExpertTokenPosition &gmm2Position,
                                           const ExpertTokenPosition &waveEndPosition, ExpertLoopState &gmm2ExpertState,
                                           GMMAddrInfo &gmm2AddrInfo, uint32_t &startBlockIdx, int32_t &gmmTileSequence,
                                           uint32_t allCoreCombineExpertIndex,
                                           ExpertLoopState &allCoreCombineExpertState);
    __aicore__ inline void ProcessCombineExperts(uint32_t expertBegin, uint32_t expertEnd,
                                                 ExpertLoopState &combineState, GMMAddrInfo &combineAddrInfo,
                                                 const CombineBufferConfig &bufferConfig,
                                                 uint32_t allCoreCombineExpertIndex,
                                                 const ExpertLoopState &allCoreCombineExpertState);

    uint32_t gmm1TilesPerMGroup_ = 1U;
    uint32_t gmm2TilesPerMGroup_ = 1U;
};

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline typename MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::CombineBufferConfig
MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::InitCombineBuffers()
{
    return InitWaveCombineBuffers<CombineQuantMode, false, WAVE_COMBINE_STEADY_ROW_BUFFER_COUNT>(commonConfig_,
                                                                                                 waveCombineScratch_);
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline bool MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::IsSameExpertTokenPosition(
    const ExpertTokenPosition &currentPosition, const ExpertTokenPosition &targetPosition) const
{
    return currentPosition.expertIdx == targetPosition.expertIdx &&
           currentPosition.tokenIndexInExpert == targetPosition.tokenIndexInExpert;
}

// 规划并 Dispatch 紧接着的一个完整 WAVE，返回更新后的全局专家位置。
template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline ExpertTokenPosition MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::DispatchNextWave(
    ExpertTokenPosition &dispatchPosition)
{
    ExpertTokenRange dispatchRange{dispatchPosition, dispatchPosition};
    ExpertTokenPosition plannedDispatchPosition = dispatchPosition;
    uint32_t dispatchWaveMGroupCount = 0U;
    while (plannedDispatchPosition.expertIdx < moeExpertPerRank_ && dispatchWaveMGroupCount < mGroupsPerWave_) {
        ExpertTokenRange nextDispatchRange = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
            params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, moeExpertPerRank_, mGroupsPerWave_,
            dispatchWaveMGroupCount, plannedDispatchPosition);
        plannedDispatchPosition = nextDispatchRange.end;
        dispatchRange.end = nextDispatchRange.end;
    }
    DispatchTokenRange<ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
        tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_, params_,
        g_winRankAddr_, tokenDispatchScratch_, dispatchRange);
    // 整 WAVE 的数据和 ready flag 发布完成后，再提交 Dispatch 进度。
    dispatchPosition = plannedDispatchPosition;
    return dispatchPosition;
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::AdvanceStartBlockIdxForSkippedGmm1(
    const ExpertTokenPosition &waveBeginPosition, const ExpertTokenPosition &waveEndPosition)
{
    if (IsSameExpertTokenPosition(waveBeginPosition, waveEndPosition) || gmmExecutionConfig_.blockJob.totalJobs == 0U) {
        return;
    }
    // AIV1 在 GMM1 时段执行 Dispatch，未进入 GMM1 scheduler。补上各 problem 的 tile 数，
    // 使后续 GMM2 与配对 AIC 从相同的 startBlockIdx 开始，不执行任何 GMM1 搬运。
    uint32_t lastExpertIdx =
        waveEndPosition.tokenIndexInExpert != 0U ? waveEndPosition.expertIdx : waveEndPosition.expertIdx - 1U;
    for (uint32_t expertIdx = waveBeginPosition.expertIdx; expertIdx <= lastExpertIdx && expertIdx < moeExpertPerRank_;
         ++expertIdx) {
        uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr,
                                                                     countWorkspace_, moeExpertPerRank_, expertIdx);
        uint32_t partBegin = expertIdx == waveBeginPosition.expertIdx ? waveBeginPosition.tokenIndexInExpert : 0U;
        uint32_t partEnd = (expertIdx == waveEndPosition.expertIdx && waveEndPosition.tokenIndexInExpert != 0U) ?
                               waveEndPosition.tokenIndexInExpert :
                               expertTokenCount;
        if (partEnd <= partBegin) {
            continue;
        }
        uint32_t problemMGroupCount = GetMGroupCountForRows(partEnd - partBegin, GMM1_TILE_M);
        uint32_t problemTileCount = problemMGroupCount * gmm1TilesPerMGroup_;
        startBlockIdx_ = (startBlockIdx_ + problemTileCount) % gmmExecutionConfig_.blockJob.totalJobs;
    }
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline ExpertTokenPosition MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessGmm1Wave(
    ExpertTokenPosition &gmm1Position, ExpertLoopState &gmm1ExpertState, GMMAddrInfo &gmm1AddrInfo,
    GmmRuntimeState &runtimeState)
{
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            return gmm1Position;
        }
    }

    uint32_t processedMGroupCount = 0U;
    while (gmm1Position.expertIdx < moeExpertPerRank_ && processedMGroupCount < mGroupsPerWave_) {
        if (gmm1Position.tokenIndexInExpert == 0U) {
            if (!gmm1ExpertState.expertCountTableReady) {
                WaitForMoeExpertTokenCountReady(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_,
                                                0U);
                gmm1ExpertState.expertCountTableReady = true;
            }
            uint32_t expertTokenCount =
                GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_,
                                                 moeExpertPerRank_, gmm1Position.expertIdx);
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
            skipGmm1Problem = HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob,
                                                           runtimeState.startBlockIdx);
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
        UpdateMoeExpertGmm1GlobalBuffer<ActivationType, MoeWeightType, ActivationType, QuantScaleOutType,
                                        PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE, false,
                                        TopkWeightsPrefetch>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
            gmm1AddrInfo, gmm1ExpertState, gmm1Position.tokenIndexInExpert, gmm1TilesPerMGroup_);
        /*
         * 只有该 problem 从第 0 行覆盖到专家尾，后续 Wave 才不会再次读取同一份 GMM1 权重。
         * 热点专家的任一 Wave problem 都为 false，即使本次不足 256 行也保留正常 L2 cache。
         */
        bool isWholeExpert =
            gmm1Position.tokenIndexInExpert == 0U && static_cast<uint64_t>(waveEndTokenIndexInExpert) == expertRowCount;
        uint32_t waveTokenStartIndex =
            static_cast<uint32_t>(gmm1ExpertState.globalTokenStartIndex) + gmm1Position.tokenIndexInExpert;
        RunGmm1Generic<QuantOutType, ActivationType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                       MoeWeight1Format != FORMAT_ND, GMM1_TILE_M, EPILOGUE_TILE_M, TopkWeightsPrefetch, false,
                       IsGmm1Interleaved, true>(
            epilogueOp_, params_, gmm1WaveProblemShape, gmm1AddrInfo, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, gmmExecutionConfig_.blockJob, waveTokenStartIndex, gmm1Position.expertIdx,
            runtimeState.pingpongIdx, nullptr, isWholeExpert);

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

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessGmm2Wave(
    ExpertTokenPosition &gmm2Position, const ExpertTokenPosition &waveEndPosition, ExpertLoopState &gmm2ExpertState,
    GMMAddrInfo &gmm2AddrInfo, uint32_t &startBlockIdx, int32_t &gmmTileSequence, uint32_t allCoreCombineExpertIndex,
    ExpertLoopState &allCoreCombineExpertState)
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            return;
        }
    }

    while (!IsSameExpertTokenPosition(gmm2Position, waveEndPosition) && gmm2Position.expertIdx < moeExpertPerRank_) {
        if (gmm2Position.tokenIndexInExpert == 0U) {
            uint32_t expertTokenCount =
                GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_,
                                                 moeExpertPerRank_, gmm2Position.expertIdx);
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
        uint32_t waveRowCount = waveEndTokenIndexInExpert - gmm2Position.tokenIndexInExpert;
        uint32_t problemMGroupCount = GetMGroupCountForRows(waveRowCount, GMM1_TILE_M);
        bool skipGmm2Problem = false;
        if constexpr (g_coreType == AIC) {
            uint32_t problemTileCount = problemMGroupCount * gmm2TilesPerMGroup_;
            skipGmm2Problem =
                HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx);
        }
        if (skipGmm2Problem) {
            gmm2Position.tokenIndexInExpert = waveEndTokenIndexInExpert;
            gmm2Position.globalTokenIndex += waveRowCount;
            if (gmm2Position.tokenIndexInExpert >= expertRowCount) {
                if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
                    NotifyWaveGmm2Ready(waveCombineJob_, params_, gmm2Position.expertIdx);
                }
                ++gmm2Position.expertIdx;
                gmm2Position.tokenIndexInExpert = 0U;
            }
            continue;
        }

        ProblemShape gmm2WaveProblemShape = gmm2ExpertState.problemShape;
        Get<M_VALUE>(gmm2WaveProblemShape) = waveEndTokenIndexInExpert - gmm2Position.tokenIndexInExpert;
        UpdateMoeExpertGmm2GlobalBuffer<MoeWeightType, ActivationType, QuantScaleOutType>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmm2AddrInfo,
            gmm2ExpertState, gmm2Position.tokenIndexInExpert);
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            gmm2AddrInfo.gmmToEpilogueFlag =
                reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                static_cast<uint64_t>(gmmExecutionConfig_.blockJob.jobIndex) * INT_CACHELINE;
        }
        // GMM2 与 GMM1 使用相同保护：只有完整专家 problem 才允许进一步判断是否绕过 L2。
        bool isWholeExpert =
            gmm2Position.tokenIndexInExpert == 0U && static_cast<uint64_t>(waveEndTokenIndexInExpert) == expertRowCount;
        int32_t *tileSequence = nullptr;
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            tileSequence = &gmmTileSequence;
        }
        RunGmm2Generic<COMBINE_NO_QUANT, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                       MoeWeight2Format != FORMAT_ND, false, GMM1_TILE_M, TopkWeightsPrefetch, false, IsGmm1Interleaved,
                       true, CombineQuantMode == COMBINE_NO_QUANT>(
            gmm2WaveProblemShape, gmm2AddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob, nullptr, isWholeExpert,
            gmm2Position.tokenIndexInExpert, &params_, tileSequence);

        gmm2Position.tokenIndexInExpert = waveEndTokenIndexInExpert;
        gmm2Position.globalTokenIndex += waveRowCount;
        if (gmm2Position.tokenIndexInExpert >= expertRowCount) {
            if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
                if constexpr (g_coreType == AIV) {
                    if (GetSubBlockIdx() == 0U && gmm2Position.expertIdx == allCoreCombineExpertIndex) {
                        allCoreCombineExpertState = gmm2ExpertState;
                    }
                }
                NotifyWaveGmm2Ready(waveCombineJob_, params_, gmm2Position.expertIdx);
            }
            ++gmm2Position.expertIdx;
            gmm2Position.tokenIndexInExpert = 0U;
        }
    }
}

// 量化 Combine 保持原有专家粒度：只消费当前 WAVE 已完整完成的专家，最后一个非空专家由全部 AIV 处理。
template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessCombineExperts(
    uint32_t expertBegin, uint32_t expertEnd, ExpertLoopState &combineState, GMMAddrInfo &combineAddrInfo,
    const CombineBufferConfig &bufferConfig, uint32_t allCoreCombineExpertIndex,
    const ExpertLoopState &allCoreCombineExpertState)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() == 0U) {
        if (allCoreCombineExpertIndex < expertBegin || allCoreCombineExpertIndex >= expertEnd) {
            return;
        }
        CombineBufferConfig activeBufferConfig =
            PrepareFinalWaveCombineBuffers<CombineQuantMode>(commonConfig_, bufferConfig, waveCombineScratch_);
        UpdateMoeExpertCombineGlobalBuffer(params_.workspaceInfo, combineAddrInfo, allCoreCombineExpertState);
        uint32_t rowSequence = 0U;
        RunWaveCombineStage<CombineQuantMode, true>(commonConfig_, waveCombineJob_, activeBufferConfig,
                                                    waveCombineScratch_, params_, combineAddrInfo,
                                                    allCoreCombineExpertState, allCoreCombineExpertIndex, rowSequence);
        DrainCombineRowBuffers(rowSequence, activeBufferConfig.rowBufferCount);
        return;
    }

    CombineBufferConfig activeBufferConfig = bufferConfig;
    uint32_t rowSequence = 0U;
    for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
        uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr,
                                                                     countWorkspace_, moeExpertPerRank_, expertIdx);
        UpdateExpertLoopState(combineState, expertIdx, expertTokenCount);
        if (expertTokenCount == 0U) {
            continue;
        }
        bool useAllAivCores = expertIdx == allCoreCombineExpertIndex;
        UpdateMoeExpertCombineGlobalBuffer(params_.workspaceInfo, combineAddrInfo, combineState);
        if (useAllAivCores) {
            activeBufferConfig =
                PrepareFinalWaveCombineBuffers<CombineQuantMode>(commonConfig_, bufferConfig, waveCombineScratch_);
            RunWaveCombineStage<CombineQuantMode, true>(commonConfig_, waveCombineJob_, activeBufferConfig,
                                                        waveCombineScratch_, params_, combineAddrInfo, combineState,
                                                        combineState.expertIdx, rowSequence);
        } else {
            RunWaveCombineStage<CombineQuantMode>(commonConfig_, waveCombineJob_, activeBufferConfig,
                                                  waveCombineScratch_, params_, combineAddrInfo, combineState,
                                                  combineState.expertIdx, rowSequence);
        }
    }
    DrainCombineRowBuffers(rowSequence, activeBufferConfig.rowBufferCount);
}

/*
 * 按动态 WAVE 边界滚动执行 A8W8 MoE 流水。AIV1 启动时连续准备 W0/W1，稳态消费已准备 WAVE 的同时
 * Dispatch 下一 WAVE，始终保持一轮 lookahead。每次先规划完整 WAVE 的 [begin, end) 范围，再通过统一的
 * DispatchTokenRange 执行搬运和 ready 发布；GMM1、Activation、GMM2 与 Combine 复用同一 WAVE 终点。
 */
template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessMoeExpertStages()
{
    const uint32_t gmm1SchedulerWidth =
        IsGmm1Interleaved ? commonConfig_.gmm1OutputDim : commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF;
    gmm1TilesPerMGroup_ = Ops::Base::CeilDiv(gmm1SchedulerWidth, static_cast<uint32_t>(L1_TILE_N));
    gmm2TilesPerMGroup_ = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));

    // GMM1/GMM2 交错流水只记录一次阶段入口，各 Wave 完成轮次由独立计数记录。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::MOE_GMM1_ACTIVATION);
    uint64_t gmm1Count = 0U;
    uint64_t gmm2Count = 0U;
    DispatchBuffInit();
    CombineBufferConfig combineBufferConfig{};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        combineBufferConfig = InitCombineBuffers();
    }
    PrepareMoeExpertTokenCountTable(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);

    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};
    GMMAddrInfo combineAddrInfo{};
    ExpertLoopState gmm1ExpertState = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2ExpertState = CreateExpertLoopState(commonConfig_);
    ExpertLoopState combineExpertState = CreateExpertLoopState(commonConfig_);
    ExpertLoopState allCoreCombineExpertState = CreateExpertLoopState(commonConfig_);
    int32_t vecSetSyncCom = 0;
    GmmRuntimeState gmm1RuntimeState{startBlockIdx_, vecSetSyncCom, gmm1PingPongIdx_};

    ExpertTokenPosition waveBeginPosition{};
    ExpertTokenPosition dispatchPosition{};
    ExpertTokenPosition gmm1Position{};
    ExpertTokenPosition gmm2Position{};
    ExpertTokenPosition preparedWaveEndPosition{};
    bool hasPreparedWave = false;
    // GMM2 的滞后消费目标（见循环内注释）；首轮为零位置，GMM2 空转一轮。
    // 形状门控：滞后收益来自消除 AIC 等激活（随 wave 数放大），代价是流水尾部固定多一个
    // GMM2 wave 深度——小 bs 下 wave 少、尾深占比大，实测 bs2048 净劣化（中位数 +45us）而
    // bs8192 净收益（-820us），故仅在每卡 token 数达到阈值时启用；2048/8192 之间未实测，
    // 阈值取中点偏保守，中间形状合入前需补测。
    const bool gmm2LagActive = commonConfig_.tokenNum >= GMM2_LAG_MIN_TOKEN_NUM;
    // 三角色整体滞后一拍的缓存：上一 wave 的边界（见循环内注释）。
    ExpertTokenPosition gmm2PendingWaveEnd{};
    bool hasPendingGmm2Wave = false;
    uint32_t combineBeginExpertIndex = 0U;
    uint32_t allCoreCombineExpertIndex = moeExpertPerRank_;

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 0U) {
            WaitForMoeExpertTokenCountReady(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_, 0U);
        }
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            for (uint32_t expertEnd = moeExpertPerRank_; expertEnd > 0U; --expertEnd) {
                uint32_t expertIdx = expertEnd - 1U;
                uint64_t countOffset =
                    GetExpertCountWorkspaceOffset(countWorkspace_, moeExpertPerRank_, expertIdx, true);
                __gm__ int32_t *expertTokenCountAddr =
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr) + countOffset;
                if (AscendC::ReadGmByPassDCache(expertTokenCountAddr) != 0) {
                    allCoreCombineExpertIndex = expertIdx;
                    break;
                }
            }
        }
    }

    while (waveBeginPosition.expertIdx < moeExpertPerRank_) {
        const uint32_t waveStartBlockIndex = startBlockIdx_;
        ExpertTokenPosition waveEndPosition =
            ProcessGmm1Wave(gmm1Position, gmm1ExpertState, gmm1AddrInfo, gmm1RuntimeState);
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                waveEndPosition = hasPreparedWave ? preparedWaveEndPosition : DispatchNextWave(dispatchPosition);
                if (waveEndPosition.expertIdx < moeExpertPerRank_) {
                    preparedWaveEndPosition = DispatchNextWave(dispatchPosition);
                    hasPreparedWave = true;
                } else {
                    hasPreparedWave = false;
                }
                if (gmm1Count == 0U) {
                    // Startup W0/W1 are fully dispatched using the host-selected ring depth.
                    EnterSteadyDispatch();
                }
                if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                    AdvanceStartBlockIdxForSkippedGmm1(waveBeginPosition, waveEndPosition);
                }
            }
        }
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM1, ++gmm1Count);
        const uint32_t gmm1EndBlockIndex = startBlockIdx_;

        /*
         * GMM2/Combine 三角色整体滞后一拍（大 bs 门控）：wave w 的 GMM2/Combine 延至 GMM1(w+1)
         * 之后执行，用下一 wave 的 GMM1 覆盖 AIC 等激活发布的自旋（AIV0 epilogue 在末批 MMAD
         * 后仍需一段向量时间才发布 activationToGmm2Flag，紧跟消费同一 wave 会让 AIC 反复自旋）。
         * 三角色（AIC/AIV0/AIV1）以相同调用序同拍延后：ProcessGmm2Wave 内部的 pairwise tile
         * 序号、分核游标与 combine 状态在三侧保持一致推进，无需任何游标配平（与统一后的
         * W4 wave-ahead 路径同款模式）。此前的 AIC 单方滞后在逐 wave dispatch 交错的新调度下
         * 与 AIV 侧 tile 序号错拍（实测 aic_scalar 0.30->0.54 净劣化），已废弃。
         */
        if (gmm2LagActive) {
            if (hasPendingGmm2Wave) {
                ProcessGmm2Wave(gmm2Position, gmm2PendingWaveEnd, gmm2ExpertState, gmm2AddrInfo, startBlockIdx_,
                                gmmTileSequence_, allCoreCombineExpertIndex, allCoreCombineExpertState);
                UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
                if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
                    uint32_t combineEndExpertIndex = gmm2PendingWaveEnd.expertIdx;
                    ProcessCombineExperts(combineBeginExpertIndex, combineEndExpertIndex, combineExpertState,
                                          combineAddrInfo, combineBufferConfig, allCoreCombineExpertIndex,
                                          allCoreCombineExpertState);
                    combineBeginExpertIndex = combineEndExpertIndex;
                }
            }
            gmm2PendingWaveEnd = waveEndPosition;
            hasPendingGmm2Wave = true;
        } else {
            ProcessGmm2Wave(gmm2Position, waveEndPosition, gmm2ExpertState, gmm2AddrInfo, startBlockIdx_,
                            gmmTileSequence_, allCoreCombineExpertIndex, allCoreCombineExpertState);
            UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
            if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
                uint32_t combineEndExpertIndex = waveEndPosition.expertIdx;
                ProcessCombineExperts(combineBeginExpertIndex, combineEndExpertIndex, combineExpertState,
                                      combineAddrInfo, combineBufferConfig, allCoreCombineExpertIndex,
                                      allCoreCombineExpertState);
                combineBeginExpertIndex = combineEndExpertIndex;
            }
        }

        const bool hasNextWave = waveEndPosition.expertIdx < moeExpertPerRank_;
        const bool fixedRoleResonance =
            startBlockIdx_ == waveStartBlockIndex && gmm1EndBlockIndex != waveStartBlockIndex;
        if (hasNextWave && fixedRoleResonance) {
            startBlockIdx_ = gmm1EndBlockIndex;
        }

        waveBeginPosition = waveEndPosition;
    }

    // 滞后流水收尾：三角色共同补跑最后一个 wave 的 GMM2/Combine（与循环内滞后分支同构）。
    if (hasPendingGmm2Wave) {
        ProcessGmm2Wave(gmm2Position, gmm2PendingWaveEnd, gmm2ExpertState, gmm2AddrInfo, startBlockIdx_,
                        gmmTileSequence_, allCoreCombineExpertIndex, allCoreCombineExpertState);
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            ProcessCombineExperts(combineBeginExpertIndex, gmm2PendingWaveEnd.expertIdx, combineExpertState,
                                  combineAddrInfo, combineBufferConfig, allCoreCombineExpertIndex,
                                  allCoreCombineExpertState);
        }
    }

    if constexpr (!TopkWeightsPrefetch) {
        EndSync<IsGmm1Interleaved>(vecSetSyncCom, gmm1PingPongIdx_);
    }
    gmm1PingPongIdx_ = 0;
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::Process()
{
    this->ProcessWave(*this);
}

#undef TemplateMegaMoeA8W8WaveTypeClass
#undef TemplateMegaMoeA8W8WaveTypeFunc

} // namespace MegaMoeImpl
#endif
