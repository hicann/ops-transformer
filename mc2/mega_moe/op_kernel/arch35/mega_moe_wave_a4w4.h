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
 * \file mega_moe_wave_a4w4.h
 * \brief MegaMoe A4W4 动态 Wave 调度
 */

#ifndef MEGA_MOE_WAVE_A4W4_H
#define MEGA_MOE_WAVE_A4W4_H

#include "common/mega_moe_utils.h"
#include "mega_moe_arch35.h"

namespace MegaMoeImpl {

#define TemplateMegaMoeA4W4WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeA4W4WaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

template <TemplateMegaMoeA4W4WaveTypeClass>
class MegaMoeA4W4Wave : public MegaMoe<TemplateMegaMoeA4W4WaveTypeFunc> {
private:
    using MegaMoeBase = MegaMoe<TemplateMegaMoeA4W4WaveTypeFunc>;
    friend MegaMoeBase;

public:
    using MegaMoeBase::Init;
    __aicore__ inline void Process();

private:
    using QuantOutType = typename MegaMoeBase::QuantOutType;
    using ActivationType = typename MegaMoeBase::ActivationType;
    using QuantScaleOutType = typename MegaMoeBase::QuantScaleOutType;
    using ActivationQuantOutType = typename MegaMoeBase::ActivationQuantOutType;

    static constexpr uint32_t A_ELEMS_PER_BYTE = MegaMoeBase::A_ELEMS_PER_BYTE;
    static constexpr uint32_t GMM1_TILE_M = MegaMoeBase::GMM1_TILE_M;
    static constexpr uint32_t EPILOGUE_TILE_M = MegaMoeBase::EPILOGUE_TILE_M;

    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::RunGmm2CombineForExpert;
    using MegaMoeBase::exceptionDump_;
    using MegaMoeBase::gmmLoopCount_;
    using MegaMoeBase::commonConfig_;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::gmmExecutionConfig_;
    using MegaMoeBase::mGroupsPerWave_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::tokenDispatchConfig_;
    using MegaMoeBase::tokenDispatchScratch_;
    using MegaMoeBase::waveCombineScratch_;
    using MegaMoeBase::syncWorkspaceLayout_;

    __aicore__ inline bool IsPositionWithinWave(const ExpertTokenPosition &position, uint32_t waveMGroupCount) const
    {
        return position.expertIdx < commonConfig_.moeExpertPerRank && waveMGroupCount < mGroupsPerWave_;
    }

    __aicore__ inline void RunGmm1ActivationForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                      GmmRuntimeState &runtimeState, uint32_t tokenStartIndexInExpert,
                                                      uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup);
    __aicore__ inline void ProcessMoeExpertStages();
};

// A4W4 的 GMM1 使用 generic kernel，Activation 将中间结果提升为 FP8 后供 GMM2 使用。
template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, GmmRuntimeState &runtimeState, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup)
{
    uint32_t problemTileCount = GetMGroupCountForRows(sliceTokenCount, GMM1_TILE_M) * gmm1TilesPerMGroup;
    if (HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, runtimeState.startBlockIdx)) {
        return;
    }
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, false, TopkWeightsPrefetch>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
        gmmAddrInfo, state, tokenStartIndexInExpert, gmm1TilesPerMGroup);
    ProblemShape sliceProblemShape = state.problemShape;
    Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
    RunGmm1GenericByWeightFormat<QuantOutType, ActivationQuantOutType, QuantScaleOutType, GMM1_TILE_M, EPILOGUE_TILE_M,
                                 TopkWeightsPrefetch, false, true>(
        gmmExecutionConfig_, params_, epilogueOp_, gmmAddrInfo, sliceProblemShape,
        static_cast<uint32_t>(state.globalTokenStartIndex) + tokenStartIndexInExpert, runtimeState, state.expertIdx);
}

/*
 * 按专家顺序将 token 划分为连续 Wave。Wave 以 256 行 M group 为单位，根据 GMM1 的 N tile 数
 * 控制总负载；空间不足时可在专家内部切分，但同一 Wave 的 GMM1 和 GMM2 始终处理相同的 token 范围。
 *
 * Dispatch 预取用于提前准备下一 Wave 的 GMM1 输入。启动阶段先 Dispatch 第一个完整 Wave；稳态阶段
 * 由 AIV1 按专家 slice 交替执行下一 Wave 的 Dispatch 和当前 Wave 的 Activation，同时 AIC/AIV0 执行当前
 * Wave 的 GMM1。整 Wave 与专家 slice 都向 DispatchTokenRange 传入显式 [begin, end) 范围。当前 Wave
 * 完成后执行其 GMM2/Combine，再切换到已经准备好的下一 Wave。
 */
template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::ProcessMoeExpertStages()
{
    // GMM1/GMM2 交错流水只记录一次阶段入口，各 Wave 完成轮次由独立计数记录。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::MOE_GMM1_ACTIVATION);
    uint64_t gmm1Count = 0U;
    uint64_t gmm2Count = 0U;
    DispatchBuffInit();
    PrepareMoeExpertTokenCountTable<true>(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
    WaveCombineBufferConfig combineBufferConfig{};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        combineBufferConfig = InitWaveCombineBuffers<CombineQuantMode>(commonConfig_, waveCombineScratch_);
    }
    uint32_t combineRowSequence = 0U;
    int32_t pairwiseTileSequence = 0;

    ExpertLoopState gmm1State = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2State = CreateExpertLoopState(commonConfig_);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // 同一 Block 内绑定的 1C2V 各自持有分核游标，必须按相同调用顺序推进并始终保持一致。
    // 某个角色即使不参与当前 GMM 计算，也必须使用该 problem 的 tile 数更新游标。
    uint32_t startBlockIdx = 0U;
    int32_t vecSetSyncCom = 0;
    uint16_t gmm1PingPongIdx = 0U;
    GmmRuntimeState gmm1RuntimeState{startBlockIdx, vecSetSyncCom, gmm1PingPongIdx};

    const uint32_t gmm1TilesPerMGroup =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(L1_TILE_N));

    ExpertTokenPosition dispatchPosition{};

    // 启动阶段：GMM1 开始消费输入前，由 AIV1 先 Dispatch 第一个完整 Wave。
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            ExpertTokenRange firstDispatchRange{dispatchPosition, dispatchPosition};
            ExpertTokenPosition plannedDispatchPosition = dispatchPosition;
            uint32_t firstDispatchWaveMGroupCount = 0U;
            while (IsPositionWithinWave(plannedDispatchPosition, firstDispatchWaveMGroupCount)) {
                ExpertTokenRange nextDispatchRange = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
                    params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, commonConfig_.moeExpertPerRank,
                    mGroupsPerWave_, firstDispatchWaveMGroupCount, plannedDispatchPosition);
                plannedDispatchPosition = nextDispatchRange.end;
                firstDispatchRange.end = nextDispatchRange.end;
            }
            // count-table 准备刚完成，UB prefix 仍有效；首 WAVE 无需从 GM 备份重复恢复。
            DispatchTokenRange<ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
                tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_, params_,
                g_winRankAddr_, tokenDispatchScratch_, firstDispatchRange);
            // 首 WAVE 的全部数据和 ready flag 发布完成后，再提交 Dispatch 进度。
            dispatchPosition = plannedDispatchPosition;
        }
    }

    ExpertTokenPosition gmm1Position{};
    while (gmm1Position.expertIdx < commonConfig_.moeExpertPerRank) {
        ExpertTokenPosition waveBeginPosition = gmm1Position;
        ExpertTokenPosition waveEndPosition = gmm1Position;
        uint32_t currentWaveMGroupCount = 0U;
        uint32_t waveLastActiveExpertIdx = commonConfig_.moeExpertPerRank;
        bool currentWaveNeedsGmm1 = true;

        uint32_t nextDispatchWaveMGroupCount = 0U;
        bool nextWaveNeedsDispatch = false;
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
            }
        }

        // 两侧进度可能不同：只有当前计算和下一 Wave 的 Dispatch 都结束后，才能切换到当前 Wave 的 GMM2。
        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            // AIV1 每轮先发送下一 Wave 的一个专家 slice，再处理当前 Wave 的一个专家 slice。
            if constexpr (g_coreType == AIV) {
                if (GetSubBlockIdx() == 1U && nextWaveNeedsDispatch) {
                    ExpertTokenRange nextExpertDispatchRange = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
                        params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, commonConfig_.moeExpertPerRank,
                        mGroupsPerWave_, nextDispatchWaveMGroupCount, dispatchPosition);
                    // 规划完成后再判断是否得到有效专家 slice，调度函数本身不参与流程分支。
                    if (nextExpertDispatchRange.end.globalTokenIndex > nextExpertDispatchRange.begin.globalTokenIndex) {
                        // 当前 WAVE 的 Activation 可能覆盖 prefix UB；只恢复本专家 slice 所需的前缀。
                        ReloadDispatchCumsumRange(commonConfig_, tokenDispatchScratch_,
                                                  nextExpertDispatchRange.begin.expertIdx,
                                                  nextExpertDispatchRange.begin.expertIdx);
                        DispatchTokenRange<ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
                            tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_,
                            params_, g_winRankAddr_, tokenDispatchScratch_, nextExpertDispatchRange);
                    }
                    // 有效 slice 完成 Dispatch，或仅跳过空专家后，提交本次规划的末尾位置。
                    dispatchPosition = nextExpertDispatchRange.end;
                    nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
                }
            }

            if (currentWaveNeedsGmm1) {
                if (gmm1Position.tokenIndexInExpert == 0U) {
                    this->template PrepareGmmExpertState<true>(gmm1State, gmm1Position.expertIdx);
                }
                uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape));
                uint32_t sliceTokenStartIndexInExpert = gmm1Position.tokenIndexInExpert;
                uint32_t sliceTokenCount = AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(
                    expertTokenCount, mGroupsPerWave_, currentWaveMGroupCount, gmm1Position);
                if (sliceTokenCount != 0U) {
                    waveLastActiveExpertIdx = gmm1State.expertIdx;
                    RunGmm1ActivationForExpert(gmm1State, gmm1AddrInfo, gmm1RuntimeState, sliceTokenStartIndexInExpert,
                                               sliceTokenCount, gmm1TilesPerMGroup);
                }
                waveEndPosition = gmm1Position;
                currentWaveNeedsGmm1 = IsPositionWithinWave(gmm1Position, currentWaveMGroupCount);
            }
        }
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM1, ++gmm1Count);

        // GMM2 调度与 Combine 量化模式无关：统一按当前 WAVE 覆盖的专家 slice 顺序推进。
        uint32_t waveGmm2ExpertEndExclusive =
            waveEndPosition.expertIdx + (waveEndPosition.tokenIndexInExpert == 0U ? 0U : 1U);
        for (uint32_t expertIdx = waveBeginPosition.expertIdx; expertIdx < waveGmm2ExpertEndExclusive; ++expertIdx) {
            uint32_t sliceTokenStartIndexInExpert =
                expertIdx == waveBeginPosition.expertIdx ? waveBeginPosition.tokenIndexInExpert : 0U;
            if (sliceTokenStartIndexInExpert == 0U) {
                this->template PrepareGmmExpertState<false>(gmm2State, expertIdx);
            }
            uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm2State.problemShape));
            uint32_t sliceTokenEndIndexInExpert =
                expertIdx == waveEndPosition.expertIdx ? waveEndPosition.tokenIndexInExpert : expertTokenCount;
            uint32_t sliceTokenCount = sliceTokenEndIndexInExpert - sliceTokenStartIndexInExpert;
            if (sliceTokenCount != 0U) {
                bool isFinalCombine = waveEndPosition.expertIdx >= commonConfig_.moeExpertPerRank &&
                                      expertIdx == waveLastActiveExpertIdx &&
                                      sliceTokenEndIndexInExpert >= expertTokenCount;
                // W4 的 GMM2/Combine 调度集中在基类，派生模板只负责提供当前专家 slice。
                RunGmm2CombineForExpert(gmm2State, gmm2AddrInfo, startBlockIdx, sliceTokenStartIndexInExpert,
                                        sliceTokenCount, combineBufferConfig, combineRowSequence, pairwiseTileSequence,
                                        isFinalCombine);
            }
        }
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            DrainCombineRowBuffers(combineRowSequence, combineBufferConfig.rowBufferCount);
        }
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
    }

    if constexpr (!TopkWeightsPrefetch) {
        EndSync(gmm1RuntimeState.vecSetSyncCom);
    }
}

template <TemplateMegaMoeA4W4WaveTypeClass>
__aicore__ inline void MegaMoeA4W4Wave<TemplateMegaMoeA4W4WaveTypeFunc>::Process()
{
    this->ProcessWave(*this);
}

} // namespace MegaMoeImpl

#undef TemplateMegaMoeA4W4WaveTypeClass
#undef TemplateMegaMoeA4W4WaveTypeFunc

#endif
