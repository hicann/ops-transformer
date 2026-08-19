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
 * \file mega_moe_wave_a8w4.h
 * \brief MegaMoe A8W4 动态 Wave 调度
 */

#ifndef MEGA_MOE_WAVE_A8W4_H
#define MEGA_MOE_WAVE_A8W4_H

#include "common/mega_moe_utils.h"
#include "mega_moe_arch35.h"

namespace MegaMoeImpl {

#define TemplateMegaMoeA8W4WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeA8W4WaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

// 每个逻辑 AIC 任务以两个 GMM1 scheduler tile 作为 Wave 目标负载。
constexpr uint32_t A8W4_GMM1_TILES_PER_JOB_TARGET = 2U;

// 按当前 Wave 的剩余容量推进专家 token 位置，返回本次推进覆盖的 token 数。
template <uint32_t TileM>
__aicore__ inline uint32_t AdvanceExpertTokenPositionInWave(uint32_t expertTokenCount, uint32_t waveMGroupTarget,
                                                            uint32_t &waveMGroupCount, ExpertTokenPosition &position)
{
    uint32_t expertRemainingTokenCount = expertTokenCount - position.tokenIndexInExpert;
    uint32_t waveRemainingTokenCapacity = (waveMGroupTarget - waveMGroupCount) * TileM;
    uint32_t sliceTokenCount =
        expertRemainingTokenCount < waveRemainingTokenCapacity ? expertRemainingTokenCount : waveRemainingTokenCapacity;
    waveMGroupCount += Ops::Base::CeilDiv(sliceTokenCount, TileM);
    position.tokenIndexInExpert += sliceTokenCount;
    if (position.tokenIndexInExpert >= expertTokenCount) {
        ++position.expertIdx;
        position.tokenIndexInExpert = 0U;
    }
    return sliceTokenCount;
}

template <TemplateMegaMoeA8W4WaveTypeClass>
class MegaMoeA8W4Wave : public MegaMoe<TemplateMegaMoeA8W4WaveTypeFunc> {
private:
    using MegaMoeBase = MegaMoe<TemplateMegaMoeA8W4WaveTypeFunc>;

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

    static_assert(Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value,
                  "MegaMoeA8W4Wave requires fp8 activation and fp4 weight");

    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::CrossRankSyncInWorldSize;
    using MegaMoeBase::InitTokenUnpermuteBuffers;
    using MegaMoeBase::ProcessSharedExpertGmm1;
    using MegaMoeBase::ProcessSharedExpertGmm2;
    using MegaMoeBase::SendAndQuantBuffInit;
    using MegaMoeBase::dispatchPrepareConfig_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::commonConfig_;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::gmmExecutionConfig_;
    using MegaMoeBase::syncWorkspaceLayout_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::quantProcessConfig_;
    using MegaMoeBase::quantScratch_;
    using MegaMoeBase::quantCombineBufferConfig_;
    using MegaMoeBase::quantCombineConfig_;
    using MegaMoeBase::resetTensor_;
    using MegaMoeBase::resetWorkspaceConfig_;
    using MegaMoeBase::sendMaskConfig_;
    using MegaMoeBase::sendMaskScratch_;
    using MegaMoeBase::sharedExpertNum_;
    using MegaMoeBase::sharedExpertPrepareConfig_;
    using MegaMoeBase::sharedExpertPrepareScratch_;
    using MegaMoeBase::tokenDispatchConfig_;
    using MegaMoeBase::tokenDispatchScratch_;
    using MegaMoeBase::tokenUnpermuteConfig_;
    using MegaMoeBase::tokenUnpermuteScratch_;
    uint32_t waveMGroupTarget_ = 0U;

    __aicore__ inline bool IsPositionWithinWave(const ExpertTokenPosition &position, uint32_t waveMGroupCount) const
    {
        return position.expertIdx < commonConfig_.moeExpertPerRank && waveMGroupCount < waveMGroupTarget_;
    }

    __aicore__ inline uint32_t DispatchMoeExpert(uint32_t expertIdx);
    template <bool WaitForTokenCountReady>
    __aicore__ inline void PrepareGmmExpertState(ExpertLoopState &state, uint32_t expertIdx);
    __aicore__ inline void RunGmm1ActivationForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                      uint32_t &startBlockIdx, int32_t &gmTileSequence,
                                                      uint32_t tokenStartIndexInExpert, uint32_t sliceTokenCount,
                                                      uint32_t gmm1TilesPerMGroup);
    __aicore__ inline void RunGmm2CombineForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                   uint32_t &startBlockIdx, int32_t &gmTileSequence,
                                                   uint32_t tokenStartIndexInExpert, uint32_t sliceTokenCount);
    __aicore__ inline void ProcessMoeExpertStages(int32_t &gmTileSequence);
};

// 计算当前专家的 token 数，并在非空时执行完整 Dispatch。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline uint32_t MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::DispatchMoeExpert(uint32_t expertIdx)
{
    uint32_t expertTokenCount = 0U;
    ComputeExpertTokenCountAndNotify<ActivationType, true, TopkWeightsPrefetch>(
        tokenDispatchConfig_, params_, tokenDispatchScratch_, expertIdx, expertTokenCount);
    if (expertTokenCount != 0U) {
        DispatchExpertTokensByRankShard<ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
            tokenDispatchConfig_, syncWorkspaceLayout_, params_, winRankAddr_, tokenDispatchScratch_, expertIdx);
    }
    return expertTokenCount;
}

// 从 workspace 读取 token 数并准备 GMM 专家状态；GMM1 可按需先等待 token count ready。
template <TemplateMegaMoeA8W4WaveTypeClass>
template <bool WaitForTokenCountReady>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::PrepareGmmExpertState(ExpertLoopState &state,
                                                                                               uint32_t expertIdx)
{
    if constexpr (WaitForTokenCountReady) {
        if (GetSubBlockIdx() == 0U) {
            WaitForMoeExpertTokenCountReady(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_,
                                            expertIdx);
        }
    }
    uint32_t expertTokenCount =
        GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, expertIdx);
    UpdateExpertLoopState(state, expertIdx, expertTokenCount);
}

// 更新当前专家切片的输入、输出及同步地址，并执行 GMM1 和 Activation。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence,
    uint32_t tokenStartIndexInExpert, uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup)
{
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, true, TopkWeightsPrefetch>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
        gmmAddrInfo, state, tokenStartIndexInExpert, gmm1TilesPerMGroup);
    ProblemShape sliceProblemShape = state.problemShape;
    Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
    RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                EPILOGUE_TILE_M, TopkWeightsPrefetch, false, true>(
        epilogueOp_, params_, sliceProblemShape, gmmAddrInfo, startBlockIdx, gmTileSequence,
        gmmExecutionConfig_.blockJob, static_cast<uint32_t>(state.globalTokenStartIndex) + tokenStartIndexInExpert,
        state.expertIdx);
}

// 对已完成 Wave 中的一个专家执行 GMM2 及后续 Combine 调度。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence,
    uint32_t tokenStartIndexInExpert, uint32_t sliceTokenCount)
{
    uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    constexpr bool configureCombineCounter = CombineQuantMode != COMBINE_NO_QUANT;
    UpdateMoeExpertGmm2GlobalBuffer<Weight1Type, ActivationQuantOutType, QuantScaleOutType, true,
                                    configureCombineCounter, true>(gmmExecutionConfig_, syncWorkspaceLayout_,
                                                                   params_.workspaceInfo, moeWeightTensorListAddrs_,
                                                                   gmmAddrInfo, state, tokenStartIndexInExpert);
    ProblemShape sliceProblemShape = state.problemShape;
    Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
    RunGmm2A8W4<CombineQuantMode, ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                GMM1_TILE_M, TopkWeightsPrefetch, false, false, true>(
        params_, sliceProblemShape, gmmAddrInfo, startBlockIdx, gmTileSequence, gmmExecutionConfig_.blockJob,
        expertTokenCount, tokenStartIndexInExpert);
    ScheduleQuantizedMoeExpertCombine<CombineQuantMode>(
        quantCombineConfig_, quantCombineBufferConfig_, params_, gmmAddrInfo, expertTokenCount,
        static_cast<uint32_t>(state.globalTokenStartIndex), state.expertIdx, tokenStartIndexInExpert, sliceTokenCount);
}

/*
 * 按 GMM1 调度负载将专家划分为动态 Wave。启动阶段先 Dispatch 第一个完整 Wave；稳态阶段由 AIV1
 * 交替执行下一 Wave 的一个专家 Dispatch 和当前 Wave 的一个专家 Activation，同时 AIC/AIV0 执行当前
 * Wave 的 GMM1。当前 Wave 完成后立即启动 GMM2/Combine，无需等待所有专家的 GMM1。
 */
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessMoeExpertStages(int32_t &gmTileSequence)
{
    DispatchBuffInit();

    ExpertLoopState gmm1State = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2State = CreateExpertLoopState(commonConfig_);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // A8W4 GMM1 与 GMM2 按相同的跨 Wave tile 顺序执行，因此共用分核起点。
    uint32_t startBlockIdx = 0U;

    const uint32_t gmm1TilesPerMGroup =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(L1_TILE_N));
    // Wave 以完整 M group 为切分单位；一个 M group 会产生 gmm1TilesPerMGroup 个 scheduler tile。
    waveMGroupTarget_ =
        Ops::Base::CeilDiv(A8W4_GMM1_TILES_PER_JOB_TARGET * gmmExecutionConfig_.blockJob.totalJobs, gmm1TilesPerMGroup);

    ExpertTokenPosition dispatchPosition{};

    // 启动阶段：GMM1 开始消费输入前，由 AIV1 先 Dispatch 第一个完整 Wave。
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            uint32_t dispatchedWaveMGroupCount = 0U;
            while (IsPositionWithinWave(dispatchPosition, dispatchedWaveMGroupCount)) {
                uint32_t expertTokenCount =
                    dispatchPosition.tokenIndexInExpert == 0U ?
                        DispatchMoeExpert(dispatchPosition.expertIdx) :
                        GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_,
                                                         dispatchPosition.expertIdx);
                AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(expertTokenCount, waveMGroupTarget_,
                                                              dispatchedWaveMGroupCount, dispatchPosition);
            }
        }
    }

    ExpertTokenPosition gmm1Position{};
    while (gmm1Position.expertIdx < commonConfig_.moeExpertPerRank) {
        ExpertTokenPosition waveBeginPosition = gmm1Position;
        ExpertTokenPosition waveEndPosition = gmm1Position;
        uint32_t currentWaveMGroupCount = 0U;
        bool currentWaveNeedsGmm1 = true;

        uint32_t nextDispatchWaveMGroupCount = 0U;
        bool nextWaveNeedsDispatch = false;
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
            }
        }

        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            // AIV1 交替执行下一 Wave 的 Dispatch 与当前 Wave 的 Activation，避免集中突发 Dispatch。
            if constexpr (g_coreType == AIV) {
                if (GetSubBlockIdx() == 1U && nextWaveNeedsDispatch) {
                    uint32_t expertTokenCount =
                        dispatchPosition.tokenIndexInExpert == 0U ?
                            DispatchMoeExpert(dispatchPosition.expertIdx) :
                            GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr,
                                                             countWorkspace_, dispatchPosition.expertIdx);
                    AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(expertTokenCount, waveMGroupTarget_,
                                                                  nextDispatchWaveMGroupCount, dispatchPosition);
                    nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
                }
            }

            if (currentWaveNeedsGmm1) {
                if (gmm1Position.tokenIndexInExpert == 0U) {
                    PrepareGmmExpertState<true>(gmm1State, gmm1Position.expertIdx);
                }
                uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape));
                uint32_t sliceTokenStartIndexInExpert = gmm1Position.tokenIndexInExpert;
                uint32_t sliceTokenCount = AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(
                    expertTokenCount, waveMGroupTarget_, currentWaveMGroupCount, gmm1Position);
                if (sliceTokenCount != 0U) {
                    RunGmm1ActivationForExpert(gmm1State, gmm1AddrInfo, startBlockIdx, gmTileSequence,
                                               sliceTokenStartIndexInExpert, sliceTokenCount, gmm1TilesPerMGroup);
                }
                waveEndPosition = gmm1Position;
                currentWaveNeedsGmm1 = IsPositionWithinWave(gmm1Position, currentWaveMGroupCount);
            }
        }
        // 当前 Wave 完成后立即消费，使其 GMM2/Combine 与下一 Wave 的输入预取重叠。
        // Wave 在专家内结束时需包含该专家；在专家边界结束时，waveEndPosition 已指向下一专家。
        uint32_t waveGmm2ExpertEndExclusive =
            waveEndPosition.expertIdx + (waveEndPosition.tokenIndexInExpert == 0U ? 0U : 1U);
        for (uint32_t expertIdx = waveBeginPosition.expertIdx; expertIdx < waveGmm2ExpertEndExclusive; ++expertIdx) {
            uint32_t sliceTokenStartIndexInExpert =
                expertIdx == waveBeginPosition.expertIdx ? waveBeginPosition.tokenIndexInExpert : 0U;
            // Wave 从专家起点开始时，准备该专家的 GMM2 状态。
            if (sliceTokenStartIndexInExpert == 0U) {
                PrepareGmmExpertState<false>(gmm2State, expertIdx);
            }
            uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm2State.problemShape));
            uint32_t sliceTokenEndIndexInExpert =
                expertIdx == waveEndPosition.expertIdx ? waveEndPosition.tokenIndexInExpert : expertTokenCount;
            uint32_t sliceTokenCount = sliceTokenEndIndexInExpert - sliceTokenStartIndexInExpert;
            if (sliceTokenCount != 0U) {
                RunGmm2CombineForExpert(gmm2State, gmm2AddrInfo, startBlockIdx, gmTileSequence,
                                        sliceTokenStartIndexInExpert, sliceTokenCount);
            }
        }
    }
}

template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();

    // 输入准备：量化本卡 token、发布路由 mask，并重置共享同步状态。
    QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
        dispatchPrepareConfig_, params_, quantProcessConfig_, quantScratch_);
    GatherAndSendExpertMasks(dispatchPrepareConfig_, params_, winRankAddr_, sendMaskConfig_, sendMaskScratch_);
    ResetSyncStatus<TopkWeightsPrefetch>(dispatchPrepareConfig_, params_, resetWorkspaceConfig_, resetTensor_);
    if (sharedExpertNum_ > 0U) {
        PrepareSharedExpertInput<ActivationType, QuantScaleOutType, A_ELEMS_PER_BYTE>(
            dispatchPrepareConfig_, params_, sharedExpertPrepareConfig_, sharedExpertPrepareScratch_);
    }
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();

    int32_t gmTileSequence = 0;
    if (sharedExpertNum_ > 0U) {
        ProcessSharedExpertGmm1(gmTileSequence);
    }

    CrossRankSyncInWorldSize();

    // MoE 专家流水：动态 Wave Dispatch、GMM1/Activation、GMM2 和 Combine。
    ProcessMoeExpertStages(gmTileSequence);

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U && tokenDispatchConfig_.countWorkspace.blockIdx == 0U) {
            ExportExpertTokenCounts<ActivationType, true>(tokenDispatchConfig_, tokenDispatchScratch_, params_);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    if (sharedExpertNum_ > 0U) {
        ProcessSharedExpertGmm2(gmTileSequence);
    }

    // 所有 rank 完成 Combine 结果发送后，再开始输出聚合。
    if constexpr (g_coreType == AIV) {
        CrossRankSyncInWorldSize();
        MegaMoeUnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

} // namespace MegaMoeImpl

#undef TemplateMegaMoeA8W4WaveTypeClass
#undef TemplateMegaMoeA8W4WaveTypeFunc

#endif
