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
#include "mega_moe.h"

namespace MegaMoeImpl {

#define TemplateMegaMoeA8W4WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeA8W4WaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

// 常规 Wave 以两轮完整调度为目标；若末尾仅剩一个专家，则将其并入当前 Wave。
constexpr uint32_t A8W4_GMM1_TILES_PER_JOB_TARGET = 2U;
constexpr uint32_t MIN_EXPERTS_FOR_NEXT_A8W4_WAVE = 2U;

template <uint32_t TileM>
__aicore__ inline uint32_t GetA8W4Gmm1TileCount(uint32_t tokenCount, uint32_t nTileCount)
{
    return Ops::Base::CeilDiv(tokenCount, TileM) * nTileCount;
}

__aicore__ inline bool IsA8W4WaveComplete(uint32_t nextExpert, uint32_t expertCount,
                                          uint32_t accumulatedTileCount, uint32_t tileTarget)
{
    return nextExpert >= expertCount ||
           (accumulatedTileCount >= tileTarget &&
            expertCount - nextExpert >= MIN_EXPERTS_FOR_NEXT_A8W4_WAVE);
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

    static_assert(Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value &&
                      Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value,
                  "MegaMoeA8W4Wave requires fp8 activation and fp4 weight");

    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::CrossRankSyncInWorldSize;
    using MegaMoeBase::InitTokenUnpermuteBuffers;
    using MegaMoeBase::ProcessSharedExpertGmm1;
    using MegaMoeBase::ProcessSharedExpertGmm2;
    using MegaMoeBase::SendAndQuantBuffInit;
    using MegaMoeBase::dispatchPrepareConfig_;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::gmm1Config_;
    using MegaMoeBase::gmm2Config_;
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

    __aicore__ inline void RunGmm1ActivationForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                      uint32_t &startBlockIdx, int32_t &gmTileSequence,
                                                      uint32_t expertIdx);
    __aicore__ inline void RunGmm2CombineForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                   uint32_t &startBlockIdx, int32_t &gmTileSequence,
                                                   uint32_t expertIdx);
    __aicore__ inline void ProcessMoeExpertStages(int32_t &gmTileSequence);
};

/*
 * 动态 Wave 使 Dispatch 与 GMM1 独立推进。AIV1 将每个专家的 token 数发布到 GM，AIC 和 AIV0
 * 等待对应 ready flag 后再读取。发布的 token 数用于确定当前专家的行范围以及是否执行 GMM1/Activation。
 */
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence,
    uint32_t expertIdx)
{
    const BlockWorkspaceContext &countWorkspace = gmm1Config_.countWorkspace;
    uint64_t countSlotIndex =
        static_cast<uint64_t>(expertIdx) * countWorkspace.blockNum + countWorkspace.blockIdx;
    if (GetSubBlockIdx() == 0U) {
        __gm__ int32_t *sendCountReadyFlagAddr =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
            countSlotIndex * INT_CACHELINE;
        WaitUntilGmFlagIsNonZero(sendCountReadyFlagAddr);
    }

    if (!UpdateExpertLoopStateFromWorkspace(params_.workspaceInfo, countWorkspace, state, expertIdx)) {
        return;
    }

    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, true, TopkWeightsPrefetch>(
        gmm1Config_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_, gmmAddrInfo, state, expertIdx);
    RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                GMM1_TILE_M, EPILOGUE_TILE_M, TopkWeightsPrefetch, false, false>(
        epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx, gmTileSequence,
        gmm1Config_.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx);
}

// 对已完成 Wave 中的一个专家执行 GMM2 及后续 Combine 调度。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence,
    uint32_t expertIdx)
{
    if (!UpdateExpertLoopStateFromWorkspace(
            params_.workspaceInfo, gmm2Config_.countWorkspace, state, expertIdx)) {
        return;
    }

    constexpr bool configureCombineCounter = CombineQuantMode != COMBINE_NO_QUANT;
    UpdateA8W4WaveGmm2GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                   configureCombineCounter>(
        gmm2Config_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmmAddrInfo, state, expertIdx);
    RunGmm2A8W4<CombineQuantMode, ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch, false>(
        params_, state.problemShape, gmmAddrInfo, startBlockIdx, gmTileSequence, gmm2Config_.blockJob);
    ScheduleQuantizedMoeExpertCombine<CombineQuantMode>(
        quantCombineConfig_, quantCombineBufferConfig_, params_, gmmAddrInfo,
        static_cast<uint32_t>(Get<M_VALUE>(state.problemShape)), static_cast<uint32_t>(state.rowOffset), expertIdx);
}

/*
 * 按 GMM1 调度负载将专家划分为动态 Wave。启动阶段先 Dispatch 第一个完整 Wave；稳态阶段由 AIV1
 * 交替执行下一 Wave 的一个专家 Dispatch 和当前 Wave 的一个专家 Activation，同时 AIC/AIV0 执行当前
 * Wave 的 GMM1。当前 Wave 完成后立即启动 GMM2/Combine，无需等待所有专家的 GMM1。
 */
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessMoeExpertStages(
    int32_t &gmTileSequence)
{
    DispatchBuffInit();

    ExpertLoopState gmm1State = CreateExpertLoopState(gmm1Config_.common);
    ExpertLoopState gmm2State = CreateExpertLoopState(gmm2Config_.common);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // A8W4 GMM1 与 GMM2 按相同的跨 Wave 顺序执行，因此共用调度游标。
    uint32_t startBlockIdx = 0U;

    const uint32_t expertCount = gmm1Config_.common.moeExpertPerRank;
    const uint32_t gmm1NTileCount = Ops::Base::CeilDiv(
        gmm1Config_.common.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(L1_TILE_N));
    // 每个逻辑 AIC 任务以两个调度 tile 作为已验证 Wave 策略的目标负载。
    const uint32_t waveTileTarget =
        A8W4_GMM1_TILES_PER_JOB_TARGET * gmm1Config_.blockJob.totalJobs;

    uint32_t dispatchExpert = 0U;

    // 启动阶段：GMM1 开始消费输入前，由 AIV1 先 Dispatch 第一个完整 Wave。
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            uint32_t dispatchedWaveTileCount = 0U;
            while (!IsA8W4WaveComplete(dispatchExpert, expertCount, dispatchedWaveTileCount, waveTileTarget)) {
                uint64_t dispatchedTokenCount = 0U;
                RunMoeExpertDispatchStage<ActivationType, QuantScaleOutType, true, GMM1_TILE_M,
                                          TopkWeightsPrefetch>(
                    tokenDispatchConfig_, params_, winRankAddr_, tokenDispatchScratch_, dispatchExpert,
                    dispatchedTokenCount);
                dispatchedWaveTileCount += GetA8W4Gmm1TileCount<GMM1_TILE_M>(
                    static_cast<uint32_t>(dispatchedTokenCount), gmm1NTileCount);
                ++dispatchExpert;
            }
        }
    }

    uint32_t currentWaveBegin = 0U;
    while (currentWaveBegin < expertCount) {
        uint32_t currentGmm1Expert = currentWaveBegin;
        uint32_t currentWaveEnd = currentWaveBegin;
        uint32_t currentWaveTileCount = 0U;
        bool currentWaveNeedsGmm1 = true;

        uint32_t nextDispatchWaveTileCount = 0U;
        bool nextWaveNeedsDispatch = false;
        if constexpr (g_coreType == AIV) {
            nextWaveNeedsDispatch = GetSubBlockIdx() == 1U && dispatchExpert < expertCount;
        }

        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            // AIV1 交替执行下一 Wave 的 Dispatch 与当前 Wave 的 Activation，避免集中突发 Dispatch。
            if constexpr (g_coreType == AIV) {
                if (nextWaveNeedsDispatch) {
                    uint64_t dispatchedTokenCount = 0U;
                    RunMoeExpertDispatchStage<ActivationType, QuantScaleOutType, true, GMM1_TILE_M,
                                              TopkWeightsPrefetch>(
                        tokenDispatchConfig_, params_, winRankAddr_, tokenDispatchScratch_,
                        dispatchExpert, dispatchedTokenCount);
                    nextDispatchWaveTileCount += GetA8W4Gmm1TileCount<GMM1_TILE_M>(
                        static_cast<uint32_t>(dispatchedTokenCount), gmm1NTileCount);
                    ++dispatchExpert;
                    nextWaveNeedsDispatch = !IsA8W4WaveComplete(
                        dispatchExpert, expertCount, nextDispatchWaveTileCount, waveTileTarget);
                }
            }

            if (currentWaveNeedsGmm1) {
                RunGmm1ActivationForExpert(
                    gmm1State, gmm1AddrInfo, startBlockIdx, gmTileSequence, currentGmm1Expert);
                currentWaveTileCount += GetA8W4Gmm1TileCount<GMM1_TILE_M>(
                    static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape)), gmm1NTileCount);
                ++currentGmm1Expert;
                currentWaveEnd = currentGmm1Expert;
                currentWaveNeedsGmm1 = !IsA8W4WaveComplete(
                    currentGmm1Expert, expertCount, currentWaveTileCount, waveTileTarget);
            }
        }
        // 当前 Wave 完成后立即消费，使其 GMM2/Combine 与下一 Wave 的输入预取重叠。
        for (uint32_t expertIdx = currentWaveBegin; expertIdx < currentWaveEnd; ++expertIdx) {
            RunGmm2CombineForExpert(gmm2State, gmm2AddrInfo, startBlockIdx, gmTileSequence, expertIdx);
        }
        currentWaveBegin = currentWaveEnd;
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
    ResetSyncStatus<TopkWeightsPrefetch>(
        dispatchPrepareConfig_, params_, resetWorkspaceConfig_, resetTensor_);
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
