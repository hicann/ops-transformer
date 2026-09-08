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

#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_AIV_COMM_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_AIV_COMM_H

#include <cstdint>

namespace AlltoAllvGroupedMatMulAiv {

constexpr uint32_t kMaxRankSize = 128U;
constexpr uint32_t kMaxA2RankSize = 8U;
constexpr uint64_t kWindowAlignment = 512U;
constexpr uint64_t kDefaultWindowBytes = 200U * 1024U * 1024U;
constexpr uint64_t kFlagSlotBytes = 32U;
constexpr uint64_t kControlFlagStride = kFlagSlotBytes;
constexpr uint64_t kEpochOffset = 0U;
constexpr uint64_t kPublishSlotsOffset = kControlFlagStride;
constexpr uint64_t kInputElementBytes = 2U;

struct ExpertMeta {
    uint64_t recvTokenBase = 0U;
    uint32_t tokenCount = 0U;
    uint32_t reserved = 0U;
};

struct ExpertSourceMeta {
    uint64_t dstTokenOffset = 0U;
    uint64_t srcTokenOffset = 0U;
    uint32_t tokenCount = 0U;
    uint32_t sourceRank = 0U;
};

struct WorkspaceLayout {
    uint64_t payloadBytes = 0U;
    uint64_t totalBytes = 0U;
};

struct A2avWindowLayout {
    uint64_t inputBytes = 0U;
    uint64_t payloadBytes = 0U;
    uint64_t countsOffset = 0U;
    uint64_t countBytes = 0U;
    uint64_t controlOffset = 0U;
    uint64_t controlBytes = 0U;
    uint64_t readyOffset = 0U;
    uint64_t totalBytes = 0U;
    uint64_t requiredBytes = 0U;
};

struct RuntimeControlLayout {
    uint64_t epochOffset = 0U;
    uint64_t publishSlotsOffset = 0U;
    uint64_t releaseSlotsOffset = 0U;
    // Only epoch/publish/release live in the peer window.  This boundary must
    // stay independent of expertPerRank so consecutive shapes reuse one
    // cross-rank synchronization address.
    uint64_t peerControlBytes = 0U;
    // Expert-ready helpers address user workspace, not the peer window.  Keep
    // their compatibility layout separate from peer-window allocation.
    uint64_t firstExpertFlagOffset = 0U;
    uint64_t totalBytes = 0U;
};

struct PeerContextMetadata {
    uint32_t rankId = 0U;
    uint32_t rankSize = 0U;
    uint64_t windowBytes = 0U;
};

#if defined(__CCE_AICORE__)
#define A2AVGMM_HOST_DEVICE __aicore__ inline
#else
#define A2AVGMM_HOST_DEVICE inline
#endif

A2AVGMM_HOST_DEVICE bool IsSupportedPeerRankSize(uint32_t rankSize, bool isA3)
{
    const bool supportedByA2 = rankSize == 2U || rankSize == 4U || rankSize == kMaxA2RankSize;
    const bool supportedByA3 = rankSize == 16U || rankSize == 32U || rankSize == 64U || rankSize == kMaxRankSize;
    return supportedByA2 || (isA3 && supportedByA3);
}

A2AVGMM_HOST_DEVICE bool NormalizePeerContextMetadata(uint32_t rankId, uint32_t rankSize, uint64_t windowBytes,
                                                      PeerContextMetadata &metadata)
{
    const uint64_t normalizedWindowBytes = windowBytes == 0U ? kDefaultWindowBytes : windowBytes;
    if (rankSize == 0U || rankSize > kMaxRankSize || rankId >= rankSize || normalizedWindowBytes == 0U) {
        return false;
    }
    metadata.rankId = rankId;
    metadata.rankSize = rankSize;
    metadata.windowBytes = normalizedWindowBytes;
    return true;
}

A2AVGMM_HOST_DEVICE bool SafeAddU64(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    constexpr uint64_t maxValue = ~static_cast<uint64_t>(0U);
    if (lhs > maxValue - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

A2AVGMM_HOST_DEVICE bool SafeMulU64(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    constexpr uint64_t maxValue = ~static_cast<uint64_t>(0U);
    if (lhs != 0U && rhs > maxValue / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

A2AVGMM_HOST_DEVICE bool BuildRuntimeControlLayout(uint32_t rankSize, uint32_t expertPerRank,
                                                   RuntimeControlLayout &layout)
{
    if (rankSize == 0U || rankSize > kMaxRankSize || expertPerRank == 0U) {
        return false;
    }
    uint64_t rankSlotsBytes = 0U;
    uint64_t expertSlotsBytes = 0U;
    layout = {};
    layout.epochOffset = kEpochOffset;
    layout.publishSlotsOffset = kPublishSlotsOffset;
    if (!SafeMulU64(rankSize, kFlagSlotBytes, rankSlotsBytes) ||
        !SafeMulU64(expertPerRank, kFlagSlotBytes, expertSlotsBytes) ||
        !SafeAddU64(layout.publishSlotsOffset, rankSlotsBytes, layout.releaseSlotsOffset) ||
        !SafeAddU64(layout.releaseSlotsOffset, rankSlotsBytes, layout.peerControlBytes)) {
        layout = {};
        return false;
    }
    layout.firstExpertFlagOffset = layout.peerControlBytes;
    if (!SafeAddU64(layout.firstExpertFlagOffset, expertSlotsBytes, layout.totalBytes)) {
        layout = {};
        return false;
    }
    return true;
}

A2AVGMM_HOST_DEVICE bool IsRankAssignedToWorker(uint32_t rank, uint32_t workerIdx, uint32_t workerNum)
{
    return workerNum != 0U && workerIdx < workerNum && rank >= workerIdx && (rank - workerIdx) % workerNum == 0U;
}

A2AVGMM_HOST_DEVICE uint64_t MulU32ToU64(uint32_t lhs, uint32_t rhs)
{
    const uint32_t lhsLow = lhs & 0xffffU;
    const uint32_t lhsHigh = lhs >> 16U;
    const uint32_t rhsLow = rhs & 0xffffU;
    const uint32_t rhsHigh = rhs >> 16U;
    const uint64_t low = static_cast<uint64_t>(lhsLow * rhsLow);
    const uint64_t middle = static_cast<uint64_t>(lhsLow * rhsHigh) + static_cast<uint64_t>(lhsHigh * rhsLow);
    const uint64_t high = static_cast<uint64_t>(lhsHigh * rhsHigh);
    return low + (middle << 16U) + (high << 32U);
}

A2AVGMM_HOST_DEVICE bool AlignUpU64(uint64_t value, uint64_t alignment, uint64_t &result)
{
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        return false;
    }
    uint64_t valueWithPadding = 0U;
    if (!SafeAddU64(value, alignment - 1U, valueWithPadding)) {
        return false;
    }
    result = valueWithPadding & ~(alignment - 1U);
    return true;
}

A2AVGMM_HOST_DEVICE uint64_t AlignDownU64(uint64_t value, uint64_t alignment)
{
    return value & ~(alignment - 1U);
}

A2AVGMM_HOST_DEVICE uint64_t ExpertReadyBytes(uint32_t expertPerRank)
{
    return static_cast<uint64_t>(expertPerRank) * kControlFlagStride;
}

A2AVGMM_HOST_DEVICE uint64_t ExpertReadyOffset(uint64_t readyBase, uint32_t expertIdx)
{
    return readyBase + static_cast<uint64_t>(expertIdx) * kControlFlagStride;
}

A2AVGMM_HOST_DEVICE bool BuildWorkspaceLayout(uint64_t tokenNum, uint64_t hiddenSize, uint64_t dtypeBytes,
                                              WorkspaceLayout &layout)
{
    uint64_t elementNum = 0U;
    uint64_t rawPayloadBytes = 0U;
    uint64_t payloadBytes = 0U;
    layout = {};
    if (!SafeMulU64(tokenNum, hiddenSize, elementNum) || !SafeMulU64(elementNum, dtypeBytes, rawPayloadBytes) ||
        !AlignUpU64(rawPayloadBytes, kWindowAlignment, payloadBytes)) {
        return false;
    }
    layout.payloadBytes = payloadBytes;
    layout.totalBytes = payloadBytes;
    return true;
}

template <typename PrefixPtr>
A2AVGMM_HOST_DEVICE bool PrefixRange(PrefixPtr inclusivePrefix, uint32_t countNum, uint32_t index, uint32_t &begin,
                                     uint32_t &end)
{
    begin = 0U;
    end = 0U;
    if (inclusivePrefix == nullptr || countNum == 0U || index >= countNum) {
        return false;
    }

    const int32_t current = inclusivePrefix[index];
    const int32_t previous = index == 0U ? 0 : inclusivePrefix[index - 1U];
    if (previous < 0 || current < previous) {
        return false;
    }
    begin = static_cast<uint32_t>(previous);
    end = static_cast<uint32_t>(current);
    return true;
}

A2AVGMM_HOST_DEVICE bool BuildExpertMetadata(const int32_t *recvPrefix, uint32_t rankSize, uint32_t expertPerRank,
                                             uint64_t tokenCapacity, ExpertMeta *expertMeta,
                                             ExpertSourceMeta *sourceMeta)
{
    if (recvPrefix == nullptr || expertMeta == nullptr || sourceMeta == nullptr || rankSize == 0U ||
        expertPerRank == 0U) {
        return false;
    }

    uint64_t countNum64 = 0U;
    if (!SafeMulU64(rankSize, expertPerRank, countNum64) || countNum64 == 0U || countNum64 > 0xffffffffULL) {
        return false;
    }
    const uint32_t countNum = static_cast<uint32_t>(countNum64);
    uint32_t totalBegin = 0U;
    uint32_t totalEnd = 0U;
    if (!PrefixRange(recvPrefix, countNum, countNum - 1U, totalBegin, totalEnd) ||
        static_cast<uint64_t>(totalEnd) != tokenCapacity) {
        return false;
    }

    for (uint32_t expertIdx = 0U; expertIdx < expertPerRank; ++expertIdx) {
        const uint32_t firstSourceIndex = expertIdx * rankSize;
        const uint32_t lastSourceIndex = firstSourceIndex + rankSize - 1U;
        uint32_t expertBase = 0U;
        uint32_t firstSourceEnd = 0U;
        uint32_t expertEndBegin = 0U;
        uint32_t expertEnd = 0U;
        if (!PrefixRange(recvPrefix, countNum, firstSourceIndex, expertBase, firstSourceEnd) ||
            !PrefixRange(recvPrefix, countNum, lastSourceIndex, expertEndBegin, expertEnd)) {
            return false;
        }
        if (expertEnd < expertBase || static_cast<uint64_t>(expertEnd) > tokenCapacity ||
            static_cast<uint64_t>(expertEnd - expertBase) > 0xffffffffULL) {
            return false;
        }
        expertMeta[expertIdx].recvTokenBase = expertBase;
        expertMeta[expertIdx].tokenCount = expertEnd - expertBase;
        expertMeta[expertIdx].reserved = 0U;

        for (uint32_t sourceRank = 0U; sourceRank < rankSize; ++sourceRank) {
            const uint32_t sourceIndex = firstSourceIndex + sourceRank;
            uint32_t sourceBegin = 0U;
            uint32_t sourceEnd = 0U;
            if (!PrefixRange(recvPrefix, countNum, sourceIndex, sourceBegin, sourceEnd)) {
                return false;
            }
            ExpertSourceMeta &source = sourceMeta[static_cast<uint64_t>(expertIdx) * rankSize + sourceRank];
            source.dstTokenOffset = sourceBegin;
            source.srcTokenOffset = 0U;
            source.tokenCount = sourceEnd - sourceBegin;
            source.sourceRank = sourceRank;
        }
    }
    return true;
}

A2AVGMM_HOST_DEVICE bool GetPeerSourceTokenOffset(const int32_t *peerSendPrefix, uint32_t rankSize,
                                                  uint32_t expertPerRank, uint32_t dstRank, uint32_t expertIdx,
                                                  uint64_t &tokenOffset)
{
    if (peerSendPrefix == nullptr || rankSize == 0U || expertPerRank == 0U || dstRank >= rankSize ||
        expertIdx >= expertPerRank) {
        return false;
    }

    uint64_t countNum64 = 0U;
    if (!SafeMulU64(rankSize, expertPerRank, countNum64) || countNum64 == 0U || countNum64 > 0xffffffffULL) {
        return false;
    }
    const uint32_t target = dstRank * expertPerRank + expertIdx;
    uint32_t begin = 0U;
    uint32_t end = 0U;
    if (!PrefixRange(peerSendPrefix, static_cast<uint32_t>(countNum64), target, begin, end)) {
        return false;
    }
    tokenOffset = begin;
    return true;
}

A2AVGMM_HOST_DEVICE bool BuildWindowLayout(uint64_t inputTokenNum, uint64_t hiddenSize, uint64_t countNum,
                                           const RuntimeControlLayout &control, uint64_t windowBytes,
                                           A2avWindowLayout &layout)
{
    uint64_t rawCountBytes = 0U;
    uint64_t countAreaBytes = 0U;
    uint64_t requiredBytes = 0U;
    uint64_t payloadAndCounts = 0U;
    uint64_t controlBytes = 0U;
    WorkspaceLayout workspace = {};
    layout = {};
    if (!BuildWorkspaceLayout(inputTokenNum, hiddenSize, kInputElementBytes, workspace) ||
        !SafeMulU64(countNum, sizeof(int32_t), rawCountBytes) ||
        !AlignUpU64(rawCountBytes, kWindowAlignment, countAreaBytes) ||
        !AlignUpU64(control.peerControlBytes, kWindowAlignment, controlBytes) ||
        !SafeAddU64(workspace.payloadBytes, countAreaBytes, payloadAndCounts) ||
        !SafeAddU64(payloadAndCounts, controlBytes, requiredBytes)) {
        return false;
    }
    if (control.peerControlBytes == 0U || control.peerControlBytes > controlBytes) {
        return false;
    }

    layout.inputBytes = workspace.payloadBytes;
    layout.payloadBytes = workspace.payloadBytes;
    layout.countBytes = countAreaBytes;
    layout.controlBytes = controlBytes;
    layout.totalBytes = requiredBytes;
    layout.requiredBytes = requiredBytes;
    if (windowBytes == 0U) {
        layout.readyOffset = control.firstExpertFlagOffset;
        return true;
    }
    if (requiredBytes > windowBytes || controlBytes > windowBytes) {
        layout = {};
        return false;
    }
    layout.controlOffset = AlignDownU64(windowBytes - controlBytes, kWindowAlignment);
    if (layout.controlOffset < countAreaBytes) {
        layout = {};
        return false;
    }
    layout.countsOffset = layout.controlOffset - countAreaBytes;
    if (layout.payloadBytes > layout.countsOffset) {
        layout = {};
        return false;
    }
    layout.readyOffset = control.firstExpertFlagOffset;
    return true;
}

#undef A2AVGMM_HOST_DEVICE

} // namespace AlltoAllvGroupedMatMulAiv

#if defined(__CCE_AICORE__)

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"

namespace AlltoAllvGroupedMatMulAiv {

class PeerWindowContext {
public:
    __aicore__ inline bool Init(uint32_t is910C)
    {
        GM_ADDR contextAddress = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
        if (contextAddress == nullptr) {
            return false;
        }

        PeerContextMetadata metadata = {};
        if (is910C != 0U) {
            a3Context_ = reinterpret_cast<__gm__ AscendC::HcclContextDef::HcclOpResParam *>(contextAddress);
            if (!NormalizePeerContextMetadata(a3Context_->rankId, a3Context_->rankNum, a3Context_->winSize, metadata)) {
                return false;
            }
            isA3_ = true;
        } else {
            a2Context_ = reinterpret_cast<__gm__ AscendC::HcclCombineOpParam *>(contextAddress);
            if (!NormalizePeerContextMetadata(a2Context_->rankId, a2Context_->rankNum, a2Context_->winSize, metadata)) {
                return false;
            }
            // CANN may select the dynamic peer-window table even for an A2
            // EP8 group (for example, count_num == 128).  EP capability is
            // restricted separately by IsSupportedPeerRankSize; the address
            // representation must still follow the runtime HCCL context.
            if (a2Context_->multiFlag != 0U && a2Context_->data == nullptr) {
                return false;
            }
        }
        if (!IsSupportedPeerRankSize(metadata.rankSize, isA3_)) {
            return false;
        }

        rankId_ = metadata.rankId;
        rankSize_ = metadata.rankSize;
        windowBytes_ = metadata.windowBytes;
        for (uint32_t rank = 0U; rank < rankSize_; ++rank) {
            if (Window(rank) == nullptr) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint32_t RankId() const
    {
        return rankId_;
    }

    __aicore__ inline uint32_t RankSize() const
    {
        return rankSize_;
    }

    __aicore__ inline uint64_t WindowBytes() const
    {
        return windowBytes_;
    }

    __aicore__ inline GM_ADDR Window(uint32_t rank) const
    {
        if (rank >= rankSize_) {
            return nullptr;
        }
        if (isA3_) {
            if (rank == rankId_) {
                return reinterpret_cast<GM_ADDR>(a3Context_->localWindowsIn);
            }
            auto *relation = AscendC::GetRemoteRankAddrs(a3Context_, rank);
            return relation == nullptr ? nullptr : reinterpret_cast<GM_ADDR>(relation->windowsIn);
        }
        if (a2Context_->multiFlag == 0U) {
            return reinterpret_cast<GM_ADDR>(a2Context_->windowsIn[rank]);
        }
        return rank == rankId_ ? reinterpret_cast<GM_ADDR>(a2Context_->data[rank].localInput.addr) :
                                 reinterpret_cast<GM_ADDR>(a2Context_->data[rank].remoteInput.addr);
    }

private:
    __gm__ AscendC::HcclCombineOpParam *a2Context_ = nullptr;
    __gm__ AscendC::HcclContextDef::HcclOpResParam *a3Context_ = nullptr;
    uint32_t rankId_ = 0U;
    uint32_t rankSize_ = 0U;
    uint64_t windowBytes_ = 0U;
    bool isA3_ = false;
};

template <typename T>
__aicore__ inline void CopyPeerGmToLocalGm(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, uint64_t count,
                                           uint32_t ubMoveNum, AscendC::TBuf<AscendC::TPosition::VECCALC> &buffer0,
                                           AscendC::TBuf<AscendC::TPosition::VECCALC> &buffer1)
{
    if (count == 0U || ubMoveNum == 0U) {
        return;
    }

    // One tile cannot overlap another transfer. Avoid priming/draining a second event.
    if (count <= ubMoveNum) {
        AscendC::LocalTensor<T> local = buffer0.template Get<T>();
        AscendC::DataCopyExtParams copyParams(1U, static_cast<uint32_t>(count) * sizeof(T), 0U, 0U, 0U);
        AscendC::DataCopyPadExtParams<T> padParams;
        AscendC::DataCopyPad(local, src, copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst, local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        return;
    }
    AscendC::LocalTensor<T> local0 = buffer0.template Get<T>();
    AscendC::LocalTensor<T> local1 = buffer1.template Get<T>();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t pingPongId = 0U;
    for (uint64_t offset = 0U; offset < count; offset += ubMoveNum) {
        const uint32_t current = static_cast<uint32_t>((count - offset) > ubMoveNum ? ubMoveNum : (count - offset));
        const uint32_t copyBytes = current * sizeof(T);
        const AscendC::TEventID eventId = pingPongId == 0U ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<T> local = pingPongId == 0U ? local0 : local1;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        AscendC::DataCopyExtParams copyParams(1U, copyBytes, 0U, 0U, 0U);
        AscendC::DataCopyPadExtParams<T> padParams;
        AscendC::DataCopyPad(local, src[offset], copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::DataCopyPad(dst[offset], local, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        pingPongId ^= 1U;
    }
    // Drain both buffers before publishing ready or reusing these event IDs.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void CopyLocalGmToWindow(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, uint64_t count,
                                           uint32_t ubMoveNum, AscendC::TBuf<AscendC::TPosition::VECCALC> &buffer0,
                                           AscendC::TBuf<AscendC::TPosition::VECCALC> &buffer1)
{
    CopyPeerGmToLocalGm(dst, src, count, ubMoveNum, buffer0, buffer1);
}

__aicore__ inline __gm__ int32_t *GetFlagAddress(GM_ADDR window, uint64_t controlOffset, uint64_t flagOffset)
{
    return reinterpret_cast<__gm__ int32_t *>(window + controlOffset + flagOffset);
}

__aicore__ inline void StoreFlag(__gm__ int32_t *address, AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer,
                                 int32_t value)
{
    AscendC::LocalTensor<int32_t> local = flagBuffer.Get<int32_t>();
    local.SetValue(0U, value);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(address);
    AscendC::DataCopyExtParams copyParams(1U, sizeof(int32_t), 0U, 0U, 0U);
    AscendC::DataCopyPad(global, local, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
}

__aicore__ inline int32_t LoadFlag(__gm__ int32_t *address, AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    AscendC::LocalTensor<int32_t> local = flagBuffer.Get<int32_t>();
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(address);
    AscendC::DataCopyExtParams copyParams(1U, sizeof(int32_t), 0U, 0U, 0U);
    AscendC::DataCopyPadExtParams<int32_t> padParams;
    AscendC::DataCopyPad(local, global, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    return local.GetValue(0U);
}

__aicore__ inline void WaitFlagValue(__gm__ int32_t *address, AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer,
                                     int32_t value)
{
    while (LoadFlag(address, flagBuffer) != value) {
    }
}

__aicore__ inline int32_t NextInvocationEpoch(const PeerWindowContext &context, const A2avWindowLayout &layout,
                                              AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    __gm__ int32_t *epochAddress = GetFlagAddress(context.Window(context.RankId()), layout.controlOffset, kEpochOffset);
    const int32_t current = LoadFlag(epochAddress, flagBuffer);
    const int32_t next = (current <= 0 || current == 0x7fffffff) ? 1 : current + 1;
    StoreFlag(epochAddress, flagBuffer, next);
    return next;
}

__aicore__ inline void PublishPeerPhase(const PeerWindowContext &context, const A2avWindowLayout &layout,
                                        uint64_t phaseOffset, int32_t epoch, uint32_t workerIdx, uint32_t workerNum,
                                        uint32_t subBlockIdx, AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    if (subBlockIdx != 0U || workerNum == 0U || workerIdx >= workerNum) {
        return;
    }
    for (uint32_t rank = workerIdx; rank < context.RankSize(); rank += workerNum) {
        __gm__ int32_t *slot =
            GetFlagAddress(context.Window(rank), layout.controlOffset,
                           phaseOffset + static_cast<uint64_t>(context.RankId()) * kControlFlagStride);
        StoreFlag(slot, flagBuffer, epoch);
    }
}

__aicore__ inline void WaitPeerPhase(const PeerWindowContext &context, const A2avWindowLayout &layout,
                                     uint64_t phaseOffset, int32_t epoch,
                                     AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    for (uint32_t sourceRank = 0U; sourceRank < context.RankSize(); ++sourceRank) {
        __gm__ int32_t *slot = GetFlagAddress(context.Window(context.RankId()), layout.controlOffset,
                                              phaseOffset + static_cast<uint64_t>(sourceRank) * kControlFlagStride);
        WaitFlagValue(slot, flagBuffer, epoch);
    }
}

__aicore__ inline void PublishExpertReady(GM_ADDR workspace, uint64_t readyBase, uint32_t expertIdx, int32_t epoch,
                                          AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    __gm__ int32_t *ready = reinterpret_cast<__gm__ int32_t *>(workspace + ExpertReadyOffset(readyBase, expertIdx));
    StoreFlag(ready, flagBuffer, epoch);
}

__aicore__ inline void WaitExpertReady(GM_ADDR workspace, uint64_t readyBase, uint32_t expertIdx, int32_t epoch,
                                       AscendC::TBuf<AscendC::TPosition::VECCALC> &flagBuffer)
{
    __gm__ int32_t *ready = reinterpret_cast<__gm__ int32_t *>(workspace + ExpertReadyOffset(readyBase, expertIdx));
    WaitFlagValue(ready, flagBuffer, epoch);
}

} // namespace AlltoAllvGroupedMatMulAiv

#endif // defined(__CCE_AICORE__)

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_AIV_COMM_H
