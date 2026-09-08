/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_AIV_MODE_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_AIV_MODE_H

#include <cstdint>

#include "allto_allv_grouped_mat_mul_aiv_comm.h"
#include "allto_allv_grouped_mat_mul_catlass.h"
#include "allto_allv_grouped_mat_mul_tiling.h"

namespace AlltoAllvGroupedMatMulAivMode {

using ExpertMeta = AlltoAllvGroupedMatMulAiv::ExpertMeta;

struct ElementRange {
    uint64_t offset = 0U;
    uint64_t count = 0U;
};

#if defined(__CCE_AICORE__)
#define A2AVGMM_AIV_MODE_HOST_DEVICE __aicore__ inline
#else
#define A2AVGMM_AIV_MODE_HOST_DEVICE inline
#endif

template <typename PrefixPtr>
A2AVGMM_AIV_MODE_HOST_DEVICE bool BuildExpertMetaForIndex(PrefixPtr recvPrefix, uint32_t rankSize,
                                                          uint32_t expertPerRank, uint32_t expertIdx,
                                                          uint64_t tokenCapacity, ExpertMeta &expert)
{
    if (recvPrefix == nullptr || rankSize == 0U || expertPerRank == 0U || expertIdx >= expertPerRank) {
        return false;
    }

    uint64_t countNum64 = 0U;
    if (!AlltoAllvGroupedMatMulAiv::SafeMulU64(rankSize, expertPerRank, countNum64) || countNum64 == 0U ||
        countNum64 > A2AVGMM_MAX_COUNT_NUM) {
        return false;
    }
    const uint32_t countNum = static_cast<uint32_t>(countNum64);
    const uint32_t expertStartIndex = expertIdx * rankSize;
    const uint32_t expertEndIndex = expertStartIndex + rankSize - 1U;
    uint32_t expertBase = 0U;
    uint32_t firstSourceEnd = 0U;
    uint32_t lastSourceBegin = 0U;
    uint32_t expertEnd = 0U;
    if (!AlltoAllvGroupedMatMulAiv::PrefixRange(recvPrefix, countNum, expertStartIndex, expertBase, firstSourceEnd) ||
        !AlltoAllvGroupedMatMulAiv::PrefixRange(recvPrefix, countNum, expertEndIndex, lastSourceBegin, expertEnd)) {
        return false;
    }

    if (expertEnd < expertBase || static_cast<uint64_t>(expertEnd) > tokenCapacity) {
        return false;
    }
    expert.recvTokenBase = expertBase;
    expert.tokenCount = expertEnd - expertBase;
    expert.reserved = 0U;
    return true;
}

template <typename PrefixPtr>
A2AVGMM_AIV_MODE_HOST_DEVICE bool GetDestinationSourceTokenOffset(PrefixPtr recvPrefix, uint32_t rankSize,
                                                                  uint32_t expertPerRank, uint32_t expertIdx,
                                                                  uint32_t sourceRank, uint64_t tokenCapacity,
                                                                  uint64_t &tokenOffset)
{
    if (recvPrefix == nullptr || rankSize == 0U || expertPerRank == 0U || expertIdx >= expertPerRank ||
        sourceRank >= rankSize) {
        return false;
    }

    uint64_t countNum64 = 0U;
    if (!AlltoAllvGroupedMatMulAiv::SafeMulU64(rankSize, expertPerRank, countNum64) || countNum64 == 0U ||
        countNum64 > A2AVGMM_MAX_COUNT_NUM) {
        return false;
    }
    const uint32_t sourceIndex = expertIdx * rankSize + sourceRank;
    uint32_t begin = 0U;
    uint32_t end = 0U;
    if (!AlltoAllvGroupedMatMulAiv::PrefixRange(recvPrefix, static_cast<uint32_t>(countNum64), sourceIndex, begin,
                                                end) ||
        static_cast<uint64_t>(end) > tokenCapacity) {
        return false;
    }
    tokenOffset = begin;
    return true;
}

A2AVGMM_AIV_MODE_HOST_DEVICE bool PartitionElements(uint64_t totalElements, uint32_t workerIdx, uint32_t workerNum,
                                                    ElementRange &range)
{
    if (workerNum == 0U || workerIdx >= workerNum) {
        range = {};
        return false;
    }

    const uint64_t quotient = totalElements / workerNum;
    const uint64_t remainder = totalElements % workerNum;
    range.count = quotient + (workerIdx < remainder ? 1U : 0U);
    range.offset = workerIdx < remainder ? workerIdx : remainder;
    for (uint32_t worker = 0U; worker < workerIdx; ++worker) {
        range.offset += quotient;
    }
    return true;
}

A2AVGMM_AIV_MODE_HOST_DEVICE bool IsSourceWorker(uint32_t taskIdx, uint32_t taskRatio, uint32_t subBlockIdx,
                                                 uint32_t rankSize, uint32_t sourceRank)
{
    if (taskRatio == 0U || subBlockIdx != 0U || sourceRank >= rankSize) {
        return false;
    }
    const uint32_t blockIdx = taskIdx / taskRatio;
    return blockIdx < rankSize && blockIdx == sourceRank;
}

A2AVGMM_AIV_MODE_HOST_DEVICE bool GetProducerWorkerCount(uint32_t aivCoreNum, uint32_t taskRatio, uint32_t &workerNum)
{
    workerNum = 0U;
    if (aivCoreNum == 0U || taskRatio == 0U || aivCoreNum % taskRatio != 0U) {
        return false;
    }
    workerNum = aivCoreNum / taskRatio;
    return workerNum != 0U;
}

#undef A2AVGMM_AIV_MODE_HOST_DEVICE

} // namespace AlltoAllvGroupedMatMulAivMode

#if defined(__CCE_AICORE__) && !defined(__CCE_KT_TEST__)

namespace AlltoAllvGroupedMatMulAivMode {
namespace detail {

constexpr uint32_t kFlagBufferBytes = kCopyFlagBufferBytes;

__aicore__ inline int32_t LoadInvocationEpoch(const AlltoAllvGroupedMatMulAiv::PeerWindowContext &context,
                                              const AlltoAllvGroupedMatMulAiv::A2avWindowLayout &layout,
                                              AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    __gm__ int32_t *epochAddress = AlltoAllvGroupedMatMulAiv::GetFlagAddress(
        context.Window(context.RankId()), layout.controlOffset, AlltoAllvGroupedMatMulAiv::kEpochOffset);
    return AlltoAllvGroupedMatMulAiv::LoadFlag(epochAddress, flagBuffer);
}

__aicore__ inline void SignalAicExpertReady(uint32_t expertIdx)
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(ExpertReadyFlagId(expertIdx));
}

__aicore__ inline void WaitAicExpertReady(uint32_t expertIdx)
{
    AscendC::CrossCoreWaitFlag<0x2>(ExpertReadyFlagId(expertIdx));
}

__aicore__ inline void SignalAicSharedExpertReady()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kSharedReadyFlag);
}

__aicore__ inline void WaitAicSharedExpertReady()
{
    AscendC::CrossCoreWaitFlag<0x2>(kSharedReadyFlag);
}

__aicore__ inline void SignalAicRoutedDataReady()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kRoutedReadyFlag);
}

__aicore__ inline void WaitAicRoutedDataReady()
{
    AscendC::CrossCoreWaitFlag<0x2>(kRoutedReadyFlag);
}

__aicore__ inline bool ShouldUseExpertOverlap(const __gm__ AlltoAllvGmmAivTilingData &tiling)
{
    if (tiling.expertOverlapMode != A2AVGMM_EXPERT_OVERLAP_ENABLED || tiling.gmmInfo.expertPerRank < 2U ||
        tiling.gmmInfo.expertPerRank > A2AVGMM_MAX_EXPERT_OVERLAP_LOCAL_EXPERT_NUM) {
        return false;
    }

    uint32_t nonEmptyExperts = 0U;
    for (uint32_t expertIdx = 0U; expertIdx < tiling.gmmInfo.expertPerRank; ++expertIdx) {
        ExpertMeta expert = {};
        if (!BuildExpertMetaForIndex(tiling.recvPrefix, tiling.gmmInfo.rankSize, tiling.gmmInfo.expertPerRank,
                                     expertIdx, tiling.gmmInfo.maxOutputSize, expert)) {
            return false;
        }
        if (expert.tokenCount != 0U) {
            ++nonEmptyExperts;
        }
    }
    return tiling.gmmInfo.expertPerRank > A2AVGMM_LEGACY_LOCAL_EXPERT_NUM ? nonEmptyExperts != 0U :
                                                                            nonEmptyExperts >= 2U;
}

__aicore__ inline bool GetPeerSourceTokenOffset(AscendC::GlobalTensor<int32_t> &peerPrefix, uint32_t rankSize,
                                                uint32_t expertPerRank, uint32_t destinationRank, uint32_t expertIdx,
                                                uint64_t &tokenOffset, uint32_t &tokenCount)
{
    if (rankSize == 0U || expertPerRank == 0U || destinationRank >= rankSize || expertIdx >= expertPerRank) {
        return false;
    }

    const uint64_t countNum = AlltoAllvGroupedMatMulAiv::MulU32ToU64(rankSize, expertPerRank);
    if (countNum == 0U || countNum > A2AVGMM_MAX_COUNT_NUM) {
        return false;
    }
    const uint64_t target = AlltoAllvGroupedMatMulAiv::MulU32ToU64(destinationRank, expertPerRank) + expertIdx;
    const int32_t current = peerPrefix.GetValue(target);
    const int32_t previous = target == 0U ? 0 : peerPrefix.GetValue(target - 1U);
    if (previous < 0 || current < previous) {
        return false;
    }
    tokenOffset = static_cast<uint32_t>(previous);
    tokenCount = static_cast<uint32_t>(current - previous);
    return true;
}

template <typename T>
__aicore__ inline bool BuildDeviceWindowLayout(uint64_t inputTokens, uint32_t hiddenSize, uint32_t countNum,
                                               uint32_t rankSize, uint32_t expertPerRank, uint64_t windowBytes,
                                               AlltoAllvGroupedMatMulAiv::RuntimeControlLayout &control,
                                               AlltoAllvGroupedMatMulAiv::A2avWindowLayout &layout)
{
    if (sizeof(T) != AlltoAllvGroupedMatMulAiv::kInputElementBytes || inputTokens > 0xffffffffULL ||
        !AlltoAllvGroupedMatMulAiv::BuildRuntimeControlLayout(rankSize, expertPerRank, control)) {
        return false;
    }
    return AlltoAllvGroupedMatMulAiv::BuildWindowLayout(inputTokens, hiddenSize, countNum, control, windowBytes,
                                                        layout);
}

template <typename T>
__aicore__ inline uint32_t GetCopyQueueBytes(const __gm__ AlltoAllvGmmAivTilingData &tiling)
{
    const uint64_t payloadBytes = static_cast<uint64_t>(tiling.gmmCocTiling.ubMoveNum) * sizeof(T);
    if (payloadBytes == 0U || payloadBytes % 32U != 0U ||
        payloadBytes * kCopyBufferCount + kFlagBufferBytes > tiling.gmmInfo.totalUbSize) {
        return 0U;
    }
    return static_cast<uint32_t>(payloadBytes);
}

template <typename T>
__aicore__ inline bool CopyInitialWindowPayload(GM_ADDR gmmx, const __gm__ AlltoAllvGmmAivTilingData &tiling,
                                                const AlltoAllvGroupedMatMulAiv::PeerWindowContext &context,
                                                const AlltoAllvGroupedMatMulAiv::A2avWindowLayout &layout,
                                                uint32_t taskIdx, uint32_t subBlockIdx,
                                                AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuffer,
                                                AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuffer1,
                                                AscendC::TBuf<AscendC::TPosition::VECCALC> &countBuffer)
{
    if (tiling.countNum == 0U) {
        return false;
    }
    const int32_t tokenPrefix = tiling.sendPrefix[tiling.countNum - 1U];
    if (tokenPrefix < 0) {
        return false;
    }
    uint64_t inputElements = static_cast<uint32_t>(tokenPrefix);
    inputElements = AlltoAllvGroupedMatMulAiv::MulU32ToU64(static_cast<uint32_t>(inputElements), tiling.gmmInfo.K);

    ElementRange range = {};
    if (!PartitionElements(inputElements, taskIdx, tiling.gmmInfo.aivCoreNum, range)) {
        return false;
    }

    AscendC::GlobalTensor<T> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gmmx));
    AscendC::GlobalTensor<T> localWindow;
    localWindow.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(context.Window(context.RankId())));
    AlltoAllvGroupedMatMulAiv::CopyLocalGmToWindow(localWindow[range.offset], input[range.offset], range.count,
                                                   tiling.gmmCocTiling.ubMoveNum, copyBuffer, copyBuffer1);

    if (taskIdx == 0U && subBlockIdx == 0U) {
        AscendC::GlobalTensor<int32_t> windowPrefix;
        windowPrefix.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(context.Window(context.RankId()) + layout.countsOffset));
        AscendC::LocalTensor<int32_t> localPrefix = countBuffer.Get<int32_t>();
        constexpr uint32_t countsPerBatch = kFlagBufferBytes / sizeof(int32_t);
        for (uint32_t offset = 0U; offset < tiling.countNum; offset += countsPerBatch) {
            const uint32_t current =
                (tiling.countNum - offset) > countsPerBatch ? countsPerBatch : (tiling.countNum - offset);
            for (uint32_t index = 0U; index < current; ++index) {
                localPrefix.SetValue(index, tiling.sendPrefix[offset + index]);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
            AscendC::DataCopyExtParams params(1U, current * sizeof(int32_t), 0U, 0U, 0U);
            AscendC::DataCopyPad(windowPrefix[offset], localPrefix, params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        }
    }
    return true;
}

template <typename T>
__aicore__ inline bool CopyExpertFromSource(GM_ADDR recvBuffer, const __gm__ AlltoAllvGmmAivTilingData &tiling,
                                            const AlltoAllvGroupedMatMulAiv::PeerWindowContext &context,
                                            const AlltoAllvGroupedMatMulAiv::A2avWindowLayout &layout,
                                            uint32_t expertIdx, uint32_t sourceRank,
                                            AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuffer,
                                            AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuffer1)
{
    uint64_t destinationTokenOffset = 0U;
    if (!GetDestinationSourceTokenOffset(tiling.recvPrefix, tiling.gmmInfo.rankSize, tiling.gmmInfo.expertPerRank,
                                         expertIdx, sourceRank, tiling.gmmInfo.maxOutputSize, destinationTokenOffset)) {
        return false;
    }

    AscendC::GlobalTensor<int32_t> peerPrefix;
    peerPrefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(context.Window(sourceRank) + layout.countsOffset));
    uint64_t sourceTokenOffset = 0U;
    uint32_t sourceTokenCount = 0U;
    if (!GetPeerSourceTokenOffset(peerPrefix, tiling.gmmInfo.rankSize, tiling.gmmInfo.expertPerRank, context.RankId(),
                                  expertIdx, sourceTokenOffset, sourceTokenCount)) {
        return false;
    }

    const uint32_t countIndex = expertIdx * tiling.gmmInfo.rankSize + sourceRank;
    uint32_t expectedBegin = 0U;
    uint32_t expectedEnd = 0U;
    if (!AlltoAllvGroupedMatMulAiv::PrefixRange(tiling.recvPrefix, tiling.countNum, countIndex, expectedBegin,
                                                expectedEnd) ||
        expectedEnd - expectedBegin != sourceTokenCount) {
        return false;
    }

    if (sourceTokenOffset > 0xffffffffULL || destinationTokenOffset > 0xffffffffULL) {
        return false;
    }
    const uint64_t sourceElementOffset =
        AlltoAllvGroupedMatMulAiv::MulU32ToU64(static_cast<uint32_t>(sourceTokenOffset), tiling.gmmInfo.K);
    const uint64_t destinationElementOffset =
        AlltoAllvGroupedMatMulAiv::MulU32ToU64(static_cast<uint32_t>(destinationTokenOffset), tiling.gmmInfo.K);
    const uint64_t elementCount = AlltoAllvGroupedMatMulAiv::MulU32ToU64(sourceTokenCount, tiling.gmmInfo.K);

    AscendC::GlobalTensor<T> peerInput;
    peerInput.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(context.Window(sourceRank)));
    AscendC::GlobalTensor<T> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(recvBuffer));
    AlltoAllvGroupedMatMulAiv::CopyPeerGmToLocalGm(destination[destinationElementOffset],
                                                   peerInput[sourceElementOffset], elementCount,
                                                   tiling.gmmCocTiling.ubMoveNum, copyBuffer, copyBuffer1);
    return true;
}

template <typename T, bool GMM_TRANSPOSE_B, bool MM_TRANSPOSE_B>
__aicore__ inline void ProcessAic(GM_ADDR gmmweight, GM_ADDR mmx, GM_ADDR mmweight, GM_ADDR gmmy, GM_ADDR mmy,
                                  GM_ADDR recvBuffer, const __gm__ AlltoAllvGmmAivTilingData &tiling)
{
    AscendC::SetLoadDataPaddingValue(0);
    AscendC::SetAtomicNone();
    AscendC::SetFixpipeNz2ndFlag(1, 0, 0);

    // The shared-expert matmul is independent of the routed receive buffer.
    // Wait until AIV has staged and published the local payload, matching the
    // MegaMoE input-preparation boundary.  The shared MM then overlaps peer
    // publication waiting and routed-expert communication without competing
    // with the initial local HBM-to-window copy.
    if (tiling.gmmInfo.hasSharedExpert != 0U) {
        WaitAicSharedExpertReady();
        AlltoAllvGmmInfo mmInfo = {};
        mmInfo.M = tiling.mmInfo.M;
        mmInfo.K = tiling.mmInfo.K;
        mmInfo.N = tiling.mmInfo.N;
        mmInfo.hasSharedExpert = tiling.mmInfo.hasSharedExpert;
        AlltoAllvGmmCoCTiling mmCocTiling = {};
        mmCocTiling.m0 = tiling.mmCocTiling.m0;
        mmCocTiling.k0 = tiling.mmCocTiling.k0;
        mmCocTiling.n0 = tiling.mmCocTiling.n0;
        if (mmx == nullptr || mmweight == nullptr || mmy == nullptr ||
            !AlltoAllvGroupedMatMulCatlass::RunSharedExpertGemm<T, MM_TRANSPOSE_B>(mmx, mmweight, mmy, mmInfo,
                                                                                   mmCocTiling)) {
            return;
        }
    }

    const bool expertOverlap = ShouldUseExpertOverlap(tiling);
    AlltoAllvGmmInfo gmmInfo = {};
    gmmInfo.K = tiling.gmmInfo.K;
    gmmInfo.N = tiling.gmmInfo.N;
    gmmInfo.rankSize = tiling.gmmInfo.rankSize;
    gmmInfo.expertPerRank = tiling.gmmInfo.expertPerRank;
    gmmInfo.maxOutputSize = tiling.gmmInfo.maxOutputSize;
    gmmInfo.hasSharedExpert = tiling.gmmInfo.hasSharedExpert;
    AlltoAllvGmmCoCTiling gmmCocTiling = {};
    gmmCocTiling.m0 = tiling.gmmCocTiling.m0;
    gmmCocTiling.k0 = tiling.gmmCocTiling.k0;
    gmmCocTiling.n0 = tiling.gmmCocTiling.n0;
    if (!expertOverlap) {
        WaitAicRoutedDataReady();
    }
    uint32_t readySequence = 0U;
    for (uint32_t expertIdx = 0U; expertIdx < gmmInfo.expertPerRank; ++expertIdx) {
        ExpertMeta expert = {};
        if (!BuildExpertMetaForIndex(tiling.recvPrefix, gmmInfo.rankSize, gmmInfo.expertPerRank, expertIdx,
                                     gmmInfo.maxOutputSize, expert)) {
            return;
        }
        if (expert.tokenCount == 0U) {
            continue;
        }
        if (expertOverlap) {
            WaitAicExpertReady(readySequence);
        }
        if (!AlltoAllvGroupedMatMulCatlass::RunExpertGemm<T, GMM_TRANSPOSE_B>(recvBuffer, gmmweight, gmmy, expertIdx,
                                                                              expert, gmmInfo, gmmCocTiling)) {
            return;
        }
        // All ready counts in this window have been consumed before reuse.
        ++readySequence;
        if (expertOverlap && NeedsExpertEventDrain(readySequence)) {
            AscendC::SyncAll<false>();
        }
    }
}

template <typename T>
__aicore__ inline void ProcessAiv(GM_ADDR gmmx, GM_ADDR recvBuffer, const __gm__ AlltoAllvGmmAivTilingData &tiling,
                                  const AlltoAllvGroupedMatMulAiv::PeerWindowContext &context,
                                  const AlltoAllvGroupedMatMulAiv::A2avWindowLayout &layout,
                                  const AlltoAllvGroupedMatMulAiv::RuntimeControlLayout &control,
                                  AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuffer,
                                  AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuffer1,
                                  AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    const uint32_t taskIdx = AscendC::GetBlockIdx();
    const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    const uint32_t taskRatio = AscendC::GetTaskRation();
    const uint32_t blockIdx = taskRatio == 0U ? 0xffffffffU : taskIdx / taskRatio;
    uint32_t workerNum = 0U;
    if (!GetProducerWorkerCount(tiling.gmmInfo.aivCoreNum, taskRatio, workerNum)) {
        return;
    }

    int32_t epoch = 0;
    if (taskIdx == 0U && subBlockIdx == 0U) {
        epoch = AlltoAllvGroupedMatMulAiv::NextInvocationEpoch(context, layout, flagBuffer);
    }
    AscendC::SyncAll<true>();
    epoch = LoadInvocationEpoch(context, layout, flagBuffer);

    (void)CopyInitialWindowPayload<T>(gmmx, tiling, context, layout, taskIdx, subBlockIdx, copyBuffer, copyBuffer1,
                                      flagBuffer);
    AscendC::SyncAll<true>();
    AlltoAllvGroupedMatMulAiv::PublishPeerPhase(context, layout, control.publishSlotsOffset, epoch, blockIdx, workerNum,
                                                subBlockIdx, flagBuffer);
    AscendC::SyncAll<true>();
    if (tiling.gmmInfo.hasSharedExpert != 0U) {
        SignalAicSharedExpertReady();
    }
    if (taskIdx == 0U && subBlockIdx == 0U) {
        AlltoAllvGroupedMatMulAiv::WaitPeerPhase(context, layout, control.publishSlotsOffset, epoch, flagBuffer);
    }
    AscendC::SyncAll<true>();

    const bool expertOverlap = ShouldUseExpertOverlap(tiling);
    if (expertOverlap) {
        uint32_t readySequence = 0U;
        for (uint32_t expertIdx = 0U; expertIdx < tiling.gmmInfo.expertPerRank; ++expertIdx) {
            ExpertMeta expert = {};
            if (!BuildExpertMetaForIndex(tiling.recvPrefix, tiling.gmmInfo.rankSize, tiling.gmmInfo.expertPerRank,
                                         expertIdx, tiling.gmmInfo.maxOutputSize, expert)) {
                return;
            }
            // All local cores inspect the same immutable expert-major prefix.
            if (expert.tokenCount == 0U) {
                continue;
            }
            if (subBlockIdx == 0U && blockIdx < workerNum) {
                for (uint32_t rank = blockIdx; rank < context.RankSize(); rank += workerNum) {
                    (void)CopyExpertFromSource<T>(recvBuffer, tiling, context, layout, expertIdx, rank, copyBuffer,
                                                  copyBuffer1);
                }
            }
            AscendC::SyncAll<true>();
            SignalAicExpertReady(readySequence);
            ++readySequence;
            if (NeedsExpertEventDrain(readySequence)) {
                AscendC::SyncAll<false>();
            }
        }
    } else {
        for (uint32_t expertIdx = 0U; expertIdx < tiling.gmmInfo.expertPerRank; ++expertIdx) {
            // One-shot mode has no per-expert barrier to eliminate. Only the
            // assigned producers inspect source counts; avoid an all-core prefix scan.
            if (subBlockIdx == 0U && blockIdx < workerNum) {
                for (uint32_t rank = blockIdx; rank < context.RankSize(); rank += workerNum) {
                    (void)CopyExpertFromSource<T>(recvBuffer, tiling, context, layout, expertIdx, rank, copyBuffer,
                                                  copyBuffer1);
                }
            }
        }
        AscendC::SyncAll<true>();
        SignalAicRoutedDataReady();
    }

    AlltoAllvGroupedMatMulAiv::PublishPeerPhase(context, layout, control.releaseSlotsOffset, epoch, blockIdx, workerNum,
                                                subBlockIdx, flagBuffer);
    AscendC::SyncAll<true>();
    if (taskIdx == 0U && subBlockIdx == 0U) {
        AlltoAllvGroupedMatMulAiv::WaitPeerPhase(context, layout, control.releaseSlotsOffset, epoch, flagBuffer);
    }
}

} // namespace detail

template <typename T, bool GMM_TRANSPOSE_B, bool MM_TRANSPOSE_B>
__aicore__ inline void Run(GM_ADDR gmmx, GM_ADDR gmmweight, GM_ADDR mmx, GM_ADDR mmweight, GM_ADDR gmmy, GM_ADDR mmy,
                           GM_ADDR permuteOut, GM_ADDR userWorkspace, const __gm__ AlltoAllvGmmAivTilingData &tiling)
{
    GM_ADDR recvBuffer = tiling.gmmInfo.hasPermuteOut != 0U && permuteOut != nullptr ?
                             permuteOut :
                             userWorkspace + tiling.recvTokenOffset;

    if ASCEND_IS_AIV {
        AlltoAllvGroupedMatMulAiv::PeerWindowContext context;
        if (!context.Init(tiling.is910C) || context.RankSize() != tiling.gmmInfo.rankSize) {
            return;
        }

        AlltoAllvGroupedMatMulAiv::A2avWindowLayout layout = {};
        AlltoAllvGroupedMatMulAiv::RuntimeControlLayout control = {};
        if (tiling.countNum == 0U) {
            return;
        }
        const int32_t inputTokenPrefix = tiling.sendPrefix[tiling.countNum - 1U];
        if (inputTokenPrefix < 0) {
            return;
        }
        const uint64_t inputTokens = static_cast<uint32_t>(inputTokenPrefix);
        if (!detail::BuildDeviceWindowLayout<T>(inputTokens, tiling.gmmInfo.K, tiling.countNum, tiling.gmmInfo.rankSize,
                                                tiling.gmmInfo.expertPerRank, context.WindowBytes(), control, layout)) {
            return;
        }
        AscendC::TPipe pipe;

        AscendC::TBuf<AscendC::TPosition::VECCALC> flagBuffer;
        pipe.InitBuffer(flagBuffer, detail::kFlagBufferBytes);
        AscendC::TBuf<AscendC::TPosition::VECCALC> copyBuffer;
        const uint32_t queueBytes = detail::GetCopyQueueBytes<T>(tiling);
        if (queueBytes == 0U) {
            return;
        }
        pipe.InitBuffer(copyBuffer, queueBytes);
        AscendC::TBuf<AscendC::TPosition::VECCALC> copyBuffer1;
        pipe.InitBuffer(copyBuffer1, queueBytes);
        detail::ProcessAiv<T>(gmmx, recvBuffer, tiling, context, layout, control, copyBuffer, copyBuffer1, flagBuffer);
    }
    if ASCEND_IS_AIC {
        detail::ProcessAic<T, GMM_TRANSPOSE_B, MM_TRANSPOSE_B>(gmmweight, mmx, mmweight, gmmy, mmy, recvBuffer, tiling);
    }
}

} // namespace AlltoAllvGroupedMatMulAivMode

#endif // defined(__CCE_AICORE__) && !defined(__CCE_KT_TEST__)

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_AIV_MODE_H
