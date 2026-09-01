/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_FETCH_GRAD_ARCH35_H
#define ENGRAM_FETCH_GRAD_ARCH35_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_ENGRAM_FETCH_GRAD_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_grad_tiling_data.h"
#include "../engram_fetch_grad_utils.h"
#include "adv_api/hccl/hccl.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif

#include "engram_fetch_grad_sort.h"
#include "engram_fetch_grad_unique.h"

namespace Mc2Kernel {

#if defined(ENABLE_ENGRAM_FETCH_GRAD_KERNEL)

using namespace AscendC;

template <AscendC::HardEvent event>
__aicore__ inline void EngramFetchGradSyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

constexpr uint32_t ENGRAM_GRAD_TIMEOUT_US = 5U * 1000U * 1000U;
constexpr uint32_t ENGRAM_GRAD_CYCLES_PER_US = 1000U;
constexpr uint32_t COMM_RETRY_COUNT = 3U;

class EngramFetchGradArch35 {
public:
    __aicore__ inline EngramFetchGradArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR gradFetched, GM_ADDR permOut, GM_ADDR sendCountsOut,
                                GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut,
                                GM_ADDR gradUniqueOut, GM_ADDR uniqueLocalEntryOut, GM_ADDR numUniqueOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const EngramFetchGradTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteNbiChecked(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void DrainChecked(uint64_t handle);
    __aicore__ inline void TimeoutCheck(uint64_t startTime);
    __aicore__ inline void UnsortGrad();
    __aicore__ inline uint32_t LoadGradChunk(int64_t pos, int64_t end, LocalTensor<uint8_t> &buf, int32_t bufIdx,
                                             LocalTensor<int32_t> &idxUb, uint32_t tokensPerBuf);
    __aicore__ inline void StoreGradChunk(int64_t base, uint32_t count, LocalTensor<uint8_t> &buf, int32_t bufIdx);
    __aicore__ inline void ExchangeGrad();
    __aicore__ inline void SendGradToPeers();
    __aicore__ inline void SendGradRemote(uint32_t dstRank, uint32_t senderIdx, int32_t sendCount, int64_t sdispl,
                                          GM_ADDR localWinBase);
    __aicore__ inline void DrainAndSendFlags();
    __aicore__ inline void RecvGradFromPeers();
    __aicore__ inline void FinishExchangeGrad();
    __aicore__ inline void InitCompanion(uint32_t numRecv);
    __aicore__ inline void RunSort(uint32_t numRecv);
    __aicore__ inline void InitFlagsAndDispls();
    __aicore__ inline void ClearWinCounters();
    __aicore__ inline void CrossRankBarrierIssue();
    __aicore__ inline void CrossRankBarrierWait();

    __aicore__ inline void WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount);
    __aicore__ inline void ClearStatusFlags(GM_ADDR statusWinBase);
    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset);
    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank, uint32_t senderIdx);
    __aicore__ inline int32_t ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset, uint32_t srcRank,
                                               uint32_t senderIdx);
    __aicore__ inline void WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset, int32_t value,
                                              uint32_t senderIdx);
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);

    TPipe *tpipe_{nullptr};
    GM_ADDR gradFetchedGM_{nullptr};
    GM_ADDR permOutGM_{nullptr};
    GM_ADDR sendCountsOutGM_{nullptr};
    GM_ADDR recvCountsOutGM_{nullptr};
    GM_ADDR recvLocalEntryOutGM_{nullptr};
    GM_ADDR numRecvOutGM_{nullptr};
    GM_ADDR gradUniqueOutGM_{nullptr};
    GM_ADDR uniqueLocalEntryOutGM_{nullptr};
    GM_ADDR numUniqueOutGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    uint32_t channelsPerRank_{1};
    int32_t numEntriesPerRank_{0};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};
    int64_t hiddenDim_{0};
    int64_t totalRecv_{0};
    int32_t inputDtype_{0};
    int32_t outputDtype_{0};
    uint64_t winSize_{0};
    uint64_t ubSize_{0};
    uint32_t tileBytes_{0};
    uint32_t gradSubBatch_{Mc2Kernel::GRAD_SUB_BATCH};

    uint64_t barrierFlagOffset_{0};
    uint64_t tokenWriteOffset_{0};
    uint64_t tokenReadOffset_{0};
    uint64_t tokenDataOffset_{0};
    uint64_t tokenSlotSize_{0};
    uint32_t maxTokensPerSlot_{0};

    uint32_t numSendCores_{0};
    uint32_t numRecvCores_{0};
    uint32_t sendersPerRank_{1};
    uint32_t pendingHandleCount_{0};
    uint64_t pendingHandles_[Mc2Kernel::MAX_PENDING_HANDLES]{0};
    bool isSender_{false};
    bool isReceiver_{false};
    bool isFlagCore_{false};

    GM_ADDR gradSortedGM_{0};
    GM_ADDR recvGradGM_{0};
    GM_ADDR sendCountsGM_{0};
    GM_ADDR recvCountsGM_{0};
    GM_ADDR sdisplsGM_{0};
    GM_ADDR rdisplsGM_{0};
    GM_ADDR recvLocalEntryGM_{0};
    GM_ADDR counterScratchGM_{0};
    GM_ADDR flagScratchGM_{0};
    GM_ADDR segCountGM_{0};
    GM_ADDR coreStartGM_{0};
    GM_ADDR sortCompanionGM_{0};
    GM_ADDR sortWorkspaceGm_{0};

    TBuf<> entryBuf_;
    TBuf<> gradBuf_;
    int32_t ppEvtMte2ToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToMte2_[2] = {0, 0};
    TBuf<> hcommBuf_;
    TBuf<> statusBuf_;
    TBuf<> tempBuf_;
    TBuf<> indicesBuf_;
    uint32_t indicesBufElements_{0};
    TBuf<> castFp32Buf_;
    TBuf<> accumBuf_;
    TBufPool<TPosition::VECCALC, 16> sortPool_;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    EngramFetchGradSort::EngramFetchGradSort sorter_;
    EngramFetchGradUnique::EngramFetchGradUnique uniqueScatter_;
};

__aicore__ inline void EngramFetchGradArch35::WriteNbiChecked(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    int32_t ret = hcomm_.WriteNbi(handle, dst, src, len);
    if (ret != 0) {
        for (uint32_t i = 0; i < COMM_RETRY_COUNT; i++) {
            ret = hcomm_.WriteNbi(handle, dst, src, len);
            if (ret == 0) {
                return;
            }
        }
        RUNTIME_ABORT("WriteNbi failed after %u retries, ret=%d, rankId=%u, aivId=%u", COMM_RETRY_COUNT, ret, rankId_,
                      aivId_);
    }
}

__aicore__ inline void EngramFetchGradArch35::DrainChecked(uint64_t handle)
{
    int32_t ret = hcomm_.Drain(handle);
    if (ret != 0) {
        for (uint32_t i = 0; i < COMM_RETRY_COUNT; i++) {
            ret = hcomm_.Drain(handle);
            if (ret == 0) {
                return;
            }
        }
        RUNTIME_ABORT("DrainChecked failed after %u retries, ret=%d, handle=%llu", COMM_RETRY_COUNT, ret, handle);
    }
}

__aicore__ inline void EngramFetchGradArch35::TimeoutCheck(uint64_t startTime)
{
    uint64_t nowUs = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
    if ((nowUs - startTime) >= ENGRAM_GRAD_TIMEOUT_US) {
        RUNTIME_ABORT("timeout, rankId=%u, aivId=%u, elapsed=%llu us", rankId_, aivId_, nowUs - startTime);
    }
}

__aicore__ inline GM_ADDR EngramFetchGradArch35::GetRemoteWinAddr(uint32_t dstRank, uint64_t offset)
{
    return (GM_ADDR)ctxPtr_->commBuffer[dstRank] + offset;
}

__aicore__ inline uint64_t EngramFetchGradArch35::GetCommHandle(uint32_t dstRank, uint32_t senderIdx)
{
    uint32_t roleOffset = isSender_ ? SENDER_CHANNEL_IDX : RECEIVER_CHANNEL_IDX;
    uint32_t channelIdx = senderIdx * 2U + roleOffset;
    if (channelIdx >= channelsPerRank_) {
        channelIdx = roleOffset;
    }
    return ctxPtr_->hcommHandle[dstRank * channelsPerRank_ + channelIdx];
}

__aicore__ inline int32_t EngramFetchGradArch35::ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset,
                                                                  uint32_t srcRank, uint32_t senderIdx)
{
    GM_ADDR counterAddr =
        winBase + counterOffset + (static_cast<uint64_t>(srcRank) * sendersPerRank_ + senderIdx) * STATE_OFFSET;
    GlobalTensor<int32_t> counterGM;
    counterGM.SetGlobalBuffer((__gm__ int32_t *)counterAddr);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counterGM);
    return counterGM.GetValue(0);
}

__aicore__ inline void EngramFetchGradArch35::WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset,
                                                                 int32_t value, uint32_t senderIdx)
{
    if (dstRank == rankId_) {
        return;
    }

    LocalTensor<int32_t> valLocal = statusBuf_.Get<int32_t>();
    valLocal.SetValue(0, value);
    EngramFetchGradSyncFunc<HardEvent::S_MTE3>();

    GM_ADDR srcAddr = counterScratchGM_ + static_cast<uint64_t>(aivId_) * UB_ALIGN;
    GlobalTensor<int32_t> srcGM;
    srcGM.SetGlobalBuffer((__gm__ int32_t *)srcAddr);
    DataCopyParams valParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
    DataCopyPad(srcGM, valLocal, valParams);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();

    uint64_t handle = GetCommHandle(dstRank, senderIdx);
    GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, counterOffset) +
                                (static_cast<uint64_t>(rankId_) * sendersPerRank_ + senderIdx) * STATE_OFFSET;
    WriteNbiChecked(handle, remoteCounterAddr, srcAddr, sizeof(int32_t));
    DrainChecked(handle);
}

__aicore__ inline void EngramFetchGradArch35::WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount)
{
    int32_t sumOfFlag = 0;
    int32_t compareFlag = static_cast<int32_t>(expectCount);
    uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
    while (sumOfFlag != compareFlag) {
        sumOfFlag = 0;
        for (uint32_t i = 0; i < numRanks_; i++) {
            GlobalTensor<int32_t> slotGM;
            slotGM.SetGlobalBuffer((__gm__ int32_t *)(statusWinBase + i * STATE_OFFSET));
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(slotGM);
            int32_t flagVal = slotGM.GetValue(0);
            sumOfFlag += flagVal;
        }
        TimeoutCheck(startTime);
    }
}

__aicore__ inline void EngramFetchGradArch35::ClearStatusFlags(GM_ADDR statusWinBase)
{
    LocalTensor<int32_t> cleanTensor = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(cleanTensor, 0, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> statusGM;
    statusGM.SetGlobalBuffer((__gm__ int32_t *)statusWinBase);
    DataCopy(statusGM, cleanTensor, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    if (len == 0) {
        return;
    }

    GlobalTensor<uint8_t> srcGm;
    GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);

    LocalTensor<uint8_t> buf0 = entryBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> buf1 = gradBuf_.Get<uint8_t>();

    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};

    uint64_t off = 0;
    uint32_t tileIdx = 0;
    uint64_t pendingOff = 0;
    uint32_t pendingLen = 0;
    uint32_t pendingBufIdx = 0;
    bool hasPendingMte3 = false;

    while (off < len) {
        uint32_t thisLen = (len - off > MAX_BLOCK_BYTES) ? MAX_BLOCK_BYTES : static_cast<uint32_t>(len - off);
        uint32_t bufIdx = tileIdx % 2U;
        LocalTensor<uint8_t> &buf = (bufIdx == 0U) ? buf0 : buf1;

        if (tileIdx >= 2U) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        }

        DataCopyExtParams mte2Params{1U, thisLen, 0U, 0U, 0U};
        DataCopyPad(buf, srcGm[off], mte2Params, pad);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);

        if (hasPendingMte3) {
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[pendingBufIdx]);
            LocalTensor<uint8_t> &pendBuf = (pendingBufIdx == 0U) ? buf0 : buf1;
            DataCopyExtParams mte3Params{1U, pendingLen, 0U, 0U, 0U};
            DataCopyPad(dstGm[pendingOff], pendBuf, mte3Params);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[pendingBufIdx]);
        }

        pendingOff = off;
        pendingLen = thisLen;
        pendingBufIdx = bufIdx;
        hasPendingMte3 = true;

        off += thisLen;
        tileIdx++;
    }

    if (hasPendingMte3) {
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[pendingBufIdx]);
        LocalTensor<uint8_t> &pendBuf = (pendingBufIdx == 0U) ? buf0 : buf1;
        DataCopyExtParams mte3Params{1U, pendingLen, 0U, 0U, 0U};
        DataCopyPad(dstGm[pendingOff], pendBuf, mte3Params);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[pendingBufIdx]);
    }

    uint32_t last = (tileIdx - 1U) % 2U;
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[last]);
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[last ^ 1U]);
    }
}

// Rows of one grad ping-pong half buffer: bounded by the half size and gradSubBatch.
__aicore__ inline static uint32_t MaxGradRowsPerPing(uint32_t hiddenBytes, uint32_t gradSubBatch)
{
    uint32_t maxByPong = Mc2Kernel::GRAD_PING_BYTES / hiddenBytes;
    if (maxByPong < 1U) {
        maxByPong = 1U;
    }
    if (maxByPong > gradSubBatch) {
        maxByPong = gradSubBatch;
    }
    return maxByPong;
}

__aicore__ inline static uint32_t CastBufBytes(uint32_t hiddenDim, uint32_t maxByPong)
{
    return Ceil(hiddenDim * sizeof(float) * maxByPong * 2U, UB_ALIGN) * UB_ALIGN;
}

__aicore__ inline static uint32_t AccumBufBytes(uint32_t hiddenDim)
{
    return Ceil(hiddenDim * sizeof(float), UB_ALIGN) * UB_ALIGN * Mc2Kernel::ACCUM_BUF_COPIES;
}

__aicore__ inline void EngramFetchGradArch35::Init(GM_ADDR commContext, GM_ADDR gradFetched, GM_ADDR permOut,
                                                   GM_ADDR sendCountsOut, GM_ADDR recvCountsOut,
                                                   GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut, GM_ADDR gradUniqueOut,
                                                   GM_ADDR uniqueLocalEntryOut, GM_ADDR numUniqueOut,
                                                   GM_ADDR workspaceGM, TPipe *pipe,
                                                   const EngramFetchGradTilingData *tilingData)
{
    tpipe_ = pipe;
    gradFetchedGM_ = gradFetched;
    permOutGM_ = permOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCountsOutGM_ = recvCountsOut;
    recvLocalEntryOutGM_ = recvLocalEntryOut;
    numRecvOutGM_ = numRecvOut;
    gradUniqueOutGM_ = gradUniqueOut;
    uniqueLocalEntryOutGM_ = uniqueLocalEntryOut;
    numUniqueOutGM_ = numUniqueOut;
    workspaceGM_ = workspaceGM;
    aivId_ = GetBlockIdx();
    totalBlocks_ = GetBlockNum();
    tileBytes_ = TILE_BYTES;

    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;
    // commContext 为设备侧外部数据，与 Host 侧 tiling 的 rankSize（sendCounts.dim0/8）互为独立来源，
    // 必须一致性校验，否则 workspace 的 displs 区按 Host rankSize 规划而 Kernel 按 numRanks_ 写入会越界
    if (numRanks_ == 0U || numRanks_ > Mc2Kernel::MAX_QP_SIZE || numRanks_ != tilingData->rankSize) {
        RUNTIME_ABORT("invalid rankSize: commContext=%u, tiling=%u", numRanks_, tilingData->rankSize);
    }
    channelsPerRank_ = ctxPtr_->channelsPerRank;
    if (channelsPerRank_ == 0) {
        channelsPerRank_ = 1;
    }

    numEntriesPerRank_ = tilingData->numEntriesPerRank;
    numTokens_ = tilingData->numTokens;
    hiddenBytes_ = tilingData->hiddenBytes;
    hiddenDim_ = tilingData->hiddenDim;
    totalRecv_ = tilingData->totalRecv;
    inputDtype_ = tilingData->inputDtype;
    outputDtype_ = tilingData->outputDtype;
    ubSize_ = tilingData->ubSize;
    gradSubBatch_ = tilingData->gradSubBatch;
    if (gradSubBatch_ == 0U) {
        // 旧 tiling 兼容回落：回落值必须同时受半缓冲容量约束，否则 subLen*hiddenBytes 越过 32KB 半缓冲
        uint32_t rowsPerPing = 1U;
        if (hiddenBytes_ > 0 && hiddenBytes_ <= static_cast<int64_t>(Mc2Kernel::GRAD_PING_BYTES)) {
            rowsPerPing = Mc2Kernel::GRAD_PING_BYTES / static_cast<uint32_t>(hiddenBytes_);
        }
        gradSubBatch_ = (rowsPerPing < Mc2Kernel::GRAD_SUB_BATCH) ? rowsPerPing : Mc2Kernel::GRAD_SUB_BATCH;
    }

    winSize_ = tilingData->commBufferSize;

    uint32_t halfBlocks = totalBlocks_ / 2U;
    if (halfBlocks == 0U) {
        halfBlocks = 1U;
    }
    sendersPerRank_ = halfBlocks / numRanks_;
    if (sendersPerRank_ < 1U) {
        sendersPerRank_ = 1U;
    }
    if (sendersPerRank_ > NUM_SLOTS) {
        sendersPerRank_ = NUM_SLOTS;
    }
    numSendCores_ = sendersPerRank_ * numRanks_;
    if (numSendCores_ > halfBlocks) {
        numSendCores_ = halfBlocks;
    }
    numRecvCores_ = totalBlocks_ - numSendCores_;
    if (numRecvCores_ < 1U) {
        numRecvCores_ = 1U;
        numSendCores_ = totalBlocks_ - 1U;
    }
    isSender_ = (aivId_ < numSendCores_) || (totalBlocks_ <= 1U);
    isReceiver_ = (aivId_ >= numSendCores_ && aivId_ < totalBlocks_ - 1U) || (totalBlocks_ <= 1U);
    isFlagCore_ = (aivId_ == totalBlocks_ - 1U);

    uint64_t totalCounterEntries = static_cast<uint64_t>(numRanks_) * static_cast<uint64_t>(sendersPerRank_);
    uint64_t totalRanksOffset = totalCounterEntries * STATE_OFFSET;
    barrierFlagOffset_ = 0;
    tokenWriteOffset_ = totalRanksOffset;
    tokenReadOffset_ = tokenWriteOffset_ + totalRanksOffset;
    tokenDataOffset_ = 3U * totalRanksOffset;

    uint64_t tokenAreaRaw = (winSize_ > tokenDataOffset_) ? (winSize_ - tokenDataOffset_) : 0U;
    uint64_t tokenArea = tokenAreaRaw;
    if (hiddenBytes_ > 0) {
        uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
        tokenArea = Ceil(tokenAreaRaw, hiddenBytes) * hiddenBytes;
        if (tokenArea > tokenAreaRaw) {
            tokenArea = (tokenAreaRaw / hiddenBytes) * hiddenBytes;
        }
    }
    tokenSlotSize_ = tokenArea / numRanks_ / NUM_SLOTS;
    maxTokensPerSlot_ = static_cast<uint32_t>(tokenSlotSize_ / static_cast<uint64_t>(hiddenBytes_));
    if (maxTokensPerSlot_ == 0U) {
        maxTokensPerSlot_ = 1U;
    }

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    ppEvtMte2ToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte2ToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte3ToMte2_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    ppEvtMte3ToMte2_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    sendCountsGM_ = sendCountsOutGM_;
    recvCountsGM_ = recvCountsOutGM_;
    recvLocalEntryGM_ = recvLocalEntryOutGM_;

    uint64_t wsOffset = 0;
    gradSortedGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(numTokens_) * static_cast<uint64_t>(hiddenBytes_);
    int64_t totalRecvUB = totalRecv_;
    recvGradGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(totalRecvUB) * static_cast<uint64_t>(hiddenBytes_);
    sdisplsGM_ = workspaceGM_ + wsOffset;
    wsOffset += numRanks_ * UB_ALIGN;
    rdisplsGM_ = workspaceGM_ + wsOffset;
    wsOffset += numRanks_ * UB_ALIGN;
    counterScratchGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * UB_ALIGN;
    flagScratchGM_ = workspaceGM_ + wsOffset;
    wsOffset += UB_ALIGN;
    segCountGM_ = workspaceGM_ + wsOffset;
    wsOffset += Ceil(static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    coreStartGM_ = workspaceGM_ + wsOffset;
    wsOffset += Ceil(static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;

    uint32_t maxSortCount = static_cast<uint32_t>(totalRecv_);
    uint64_t sortCompanionSize = Ceil(static_cast<uint64_t>(maxSortCount) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    sortCompanionGM_ = workspaceGM_ + wsOffset;
    wsOffset += sortCompanionSize;
    uint64_t sortWorkspaceSize = EngramFetchGradSort::EngramFetchGradSort::GetWorkspaceSize(maxSortCount, totalBlocks_);

    uint32_t statusBufSize = Ceil(numRanks_ * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    uint32_t tempBufSize = statusBufSize;
    uint32_t entryBatchBytes = ENTRY_BATCH_CAP * sizeof(int32_t);
    if (tempBufSize < entryBatchBytes) {
        tempBufSize = entryBatchBytes;
    }
    uint32_t coreArrayBytes = Ceil(static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t) * 2U, UB_ALIGN) * UB_ALIGN;
    if (tempBufSize < coreArrayBytes) {
        tempBufSize = coreArrayBytes;
    }
    // displs 批量构造区：sdisplUb+rdisplUb 两段各 numRanks_*UB_ALIGN 字节（tempBuf_ 容量必须覆盖）
    uint32_t displsBatchBytes = 2U * numRanks_ * UB_ALIGN;
    if (tempBufSize < displsBatchBytes) {
        tempBufSize = displsBatchBytes;
    }
    tpipe_->InitBuffer(tempBuf_, tempBufSize);
    uint32_t indicesBufSize = Mc2Kernel::IDX_BUF_BYTES;
    if (indicesBufSize < statusBufSize) {
        indicesBufSize = statusBufSize;
    }
    // countsBuf 暂存区：recvCounts 占 numRanks_*4B，sendCounts 暂存位于偏移 numRanks_*UB_ALIGN、
    // 长度 numRanks_*UB_ALIGN，尾部合计 2*numRanks_*UB_ALIGN 字节
    uint32_t stagingBytes = 2U * numRanks_ * UB_ALIGN;
    if (indicesBufSize < stagingBytes) {
        indicesBufSize = stagingBytes;
    }
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    indicesBufElements_ = indicesBufSize / sizeof(int32_t);

    uint32_t sortUbSize = EngramFetchGradSort::EngramFetchGradSort::GetUbSize(maxSortCount, totalBlocks_);
    uint32_t maxByPong = MaxGradRowsPerPing(static_cast<uint32_t>(hiddenBytes_), gradSubBatch_);
    uint32_t castBufSize = (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) ? CastBufBytes(hiddenDim_, maxByPong) : 0U;
    uint32_t accumBufSize = AccumBufBytes(hiddenDim_);
    uint32_t uniqueBufSize = Mc2Kernel::COMM_BUF_BYTES + castBufSize + accumBufSize;
    uint32_t poolSize = sortUbSize;
    if (uniqueBufSize > poolSize) {
        poolSize = uniqueBufSize;
    }
    // UB 池预算自检：池 + 常驻四缓冲必须落在 SetLocalMemorySize 授权范围内，超限确定性失败
    uint64_t permanentUsed = Mc2Kernel::HCOMM_INIT_SIZE + statusBufSize + tempBufSize + indicesBufSize;
    uint64_t budgetLeft = (ubSize_ > permanentUsed) ? (ubSize_ - permanentUsed) : 0U;
    if (static_cast<uint64_t>(poolSize) > budgetLeft) {
        RUNTIME_ABORT("UB pool overflow: pool=%u, permanent=%llu, ubSize=%llu", poolSize, permanentUsed, ubSize_);
    }
    tpipe_->InitBufPool(sortPool_, poolSize);
    sortPool_.InitBuffer(entryBuf_, Mc2Kernel::ENTRY_BUF_BYTES);
    sortPool_.InitBuffer(gradBuf_, Mc2Kernel::GRAD_BUF_BYTES);

    sortWorkspaceGm_ = workspaceGM_ + wsOffset;
    wsOffset += sortWorkspaceSize;

    uniqueScatter_.Init(aivId_, totalBlocks_, rankId_, numRanks_, numEntriesPerRank_, hiddenDim_, hiddenBytes_,
                        inputDtype_, outputDtype_, tpipe_, entryBuf_, gradBuf_, indicesBuf_, tempBuf_, statusBuf_,
                        castFp32Buf_, accumBuf_);
}

__aicore__ inline uint32_t EngramFetchGradArch35::LoadGradChunk(int64_t pos, int64_t end, LocalTensor<uint8_t> &buf,
                                                                int32_t bufIdx, LocalTensor<int32_t> &idxUb,
                                                                uint32_t tokensPerBuf)
{
    uint32_t n = 0;
    if (end > pos) {
        n = static_cast<uint32_t>(end - pos);
        if (n > tokensPerBuf) {
            n = tokensPerBuf;
        }
    }
    if (n == 0) {
        return 0;
    }

    GlobalTensor<int32_t> permGM;
    permGM.SetGlobalBuffer((__gm__ int32_t *)permOutGM_);
    DataCopyExtParams cpParams{1U, static_cast<uint32_t>(n) * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    DataCopyPad(idxUb, permGM[static_cast<uint32_t>(pos)], cpParams, cpPad);
    EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

    GlobalTensor<uint8_t> srcGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)gradFetchedGM_);
    DataCopyPadExtParams<uint8_t> gPad{false, 0, 0, 0};

    uint32_t j = 0;
    while (j < n) {
        int32_t runStartIdx = idxUb.GetValue(j);
        uint32_t runLen = 1;
        while (j + runLen < n) {
            int32_t nextIdx = idxUb.GetValue(j + runLen);
            if (nextIdx == runStartIdx + static_cast<int32_t>(runLen)) {
                runLen++;
            } else {
                break;
            }
        }

        uint64_t srcOff = static_cast<uint64_t>(runStartIdx) * static_cast<uint64_t>(hiddenBytes_);
        uint64_t dstOff = static_cast<uint64_t>(j) * static_cast<uint64_t>(hiddenBytes_);
        uint64_t runBytesLeft = static_cast<uint64_t>(runLen) * static_cast<uint64_t>(hiddenBytes_);
        uint64_t curSrcOff = srcOff;
        uint64_t curDstOff = dstOff;
        while (runBytesLeft > 0) {
            uint32_t chunkBytes =
                (runBytesLeft > MAX_BLOCK_BYTES) ? MAX_BLOCK_BYTES : static_cast<uint32_t>(runBytesLeft);
            DataCopyExtParams gParams{1U, chunkBytes, 0U, 0U, 0U};
            DataCopyPad(buf[curDstOff], srcGm[curSrcOff], gParams, gPad);
            curSrcOff += chunkBytes;
            curDstOff += chunkBytes;
            runBytesLeft -= chunkBytes;
        }

        j += runLen;
    }
    AscendC::SetFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
    return n;
}

__aicore__ inline void EngramFetchGradArch35::StoreGradChunk(int64_t base, uint32_t count, LocalTensor<uint8_t> &buf,
                                                             int32_t bufIdx)
{
    AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);

    GlobalTensor<uint8_t> dstGm;
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)gradSortedGM_);
    uint64_t dstOff = static_cast<uint64_t>(base) * static_cast<uint64_t>(hiddenBytes_);
    uint64_t totalBytes = static_cast<uint64_t>(count) * static_cast<uint64_t>(hiddenBytes_);
    uint64_t srcOff = 0;
    while (totalBytes > 0) {
        uint32_t chunkBytes = (totalBytes > MAX_BLOCK_BYTES) ? MAX_BLOCK_BYTES : static_cast<uint32_t>(totalBytes);
        DataCopyExtParams sParams{1U, chunkBytes, 0U, 0U, 0U};
        DataCopyPad(dstGm[dstOff + srcOff], buf[srcOff], sParams);
        srcOff += chunkBytes;
        totalBytes -= chunkBytes;
    }
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
}

__aicore__ inline void EngramFetchGradArch35::UnsortGrad()
{
    int64_t totalPerCore = (numTokens_ + static_cast<int64_t>(totalBlocks_) - 1) / static_cast<int64_t>(totalBlocks_);
    int64_t start = static_cast<int64_t>(aivId_) * totalPerCore;
    int64_t end = start + totalPerCore;
    if (end > numTokens_) {
        end = numTokens_;
    }
    if (start >= end) {
        return;
    }

    GlobalTensor<int32_t> permGM;
    permGM.SetGlobalBuffer((__gm__ int32_t *)permOutGM_);

    if (static_cast<uint64_t>(hiddenBytes_) > tileBytes_) {
        LocalTensor<int32_t> idxUb = indicesBuf_.Get<int32_t>();
        int64_t pos = start;
        while (pos < end) {
            uint32_t batch = static_cast<uint32_t>(end - pos);
            if (batch > indicesBufElements_) {
                batch = indicesBufElements_;
            }
            DataCopyExtParams cpParams{1U, batch * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            DataCopyPad(idxUb, permGM[static_cast<uint32_t>(pos)], cpParams, cpPad);
            EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

            uint32_t j = 0;
            while (j < batch) {
                int32_t runStartIdx = idxUb.GetValue(j);
                uint32_t runLen = 1;
                while (j + runLen < batch) {
                    int32_t nextIdx = idxUb.GetValue(j + runLen);
                    if (nextIdx == runStartIdx + static_cast<int32_t>(runLen)) {
                        runLen++;
                    } else {
                        break;
                    }
                }

                GM_ADDR src = gradFetchedGM_ + static_cast<uint64_t>(runStartIdx) * static_cast<uint64_t>(hiddenBytes_);
                GM_ADDR dst = gradSortedGM_ + static_cast<uint64_t>(pos + j) * static_cast<uint64_t>(hiddenBytes_);
                LocalCopySlice(dst, src, static_cast<uint64_t>(runLen) * static_cast<uint64_t>(hiddenBytes_));

                j += runLen;
            }
            pos += batch;
        }
        return;
    }

    uint32_t tokensPerBuf = static_cast<uint32_t>(tileBytes_ / static_cast<uint64_t>(hiddenBytes_));
    if (tokensPerBuf == 0) {
        tokensPerBuf = 1;
    }
    if (tokensPerBuf > indicesBufElements_) {
        tokensPerBuf = indicesBufElements_;
    }

    LocalTensor<uint8_t> buf0 = entryBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> buf1 = gradBuf_.Get<uint8_t>();
    LocalTensor<int32_t> idxUb = indicesBuf_.Get<int32_t>();

    int32_t curBuf = 0;
    int64_t pos = start;
    int64_t base0 = start;
    uint32_t n0 = LoadGradChunk(pos, end, buf0, curBuf, idxUb, tokensPerBuf);
    pos += n0;
    bool stored0 = false;
    bool stored1 = false;

    while (pos < end) {
        int32_t nxtBuf = 1 - curBuf;
        if (stored0 && nxtBuf == 0) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[0]);
        }
        if (stored1 && nxtBuf == 1) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[1]);
        }
        uint32_t n1 = (nxtBuf == 0) ? LoadGradChunk(pos, end, buf0, nxtBuf, idxUb, tokensPerBuf) :
                                      LoadGradChunk(pos, end, buf1, nxtBuf, idxUb, tokensPerBuf);
        pos += n1;

        if (curBuf == 0) {
            StoreGradChunk(base0, n0, buf0, curBuf);
            stored0 = true;
        } else {
            StoreGradChunk(base0, n0, buf1, curBuf);
            stored1 = true;
        }
        base0 += n0;
        curBuf = nxtBuf;
        n0 = n1;
    }

    if (curBuf == 0) {
        StoreGradChunk(base0, n0, buf0, curBuf);
        stored0 = true;
    } else {
        StoreGradChunk(base0, n0, buf1, curBuf);
        stored1 = true;
    }

    if (stored0) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[0]);
    }
    if (stored1) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[1]);
    }
}

__aicore__ inline void EngramFetchGradArch35::FinishExchangeGrad()
{
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];

    DrainAndSendFlags();

    SyncAll<true>();
}
__aicore__ inline void EngramFetchGradArch35::SendGradToPeers()
{
    pendingHandleCount_ = 0;
    if (!isSender_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t totalSendWorkUnits = numRanks_ * sendersPerRank_;
    if (totalSendWorkUnits == 0U) {
        return;
    }
    for (uint32_t wIdx = aivId_ % totalSendWorkUnits; wIdx < totalSendWorkUnits; wIdx += numSendCores_) {
        uint32_t dstRank = wIdx % numRanks_;
        uint32_t senderIdx = wIdx / numRanks_;

        GM_ADDR sSlot = sendCountsGM_ + dstRank * UB_ALIGN;
        GlobalTensor<int32_t> sSlotGM;
        sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
        int32_t totalSendCount = sSlotGM.GetValue(0);

        GM_ADDR sDisplSlot = sdisplsGM_ + dstRank * UB_ALIGN;
        GlobalTensor<int64_t> sDisplSlotGM;
        sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
        int64_t sdispl = sDisplSlotGM.GetValue(0);

        int32_t baseShare = totalSendCount / static_cast<int32_t>(sendersPerRank_);
        int32_t mySendCount = baseShare;
        int64_t mySdispl = sdispl + static_cast<int64_t>(senderIdx) * baseShare;
        if (senderIdx == sendersPerRank_ - 1U) {
            mySendCount = totalSendCount - static_cast<int32_t>(senderIdx) * baseShare;
        }

        if (mySendCount <= 0) {
            continue;
        }

        if (dstRank == rankId_) {
            GM_ADDR rDisplSlot = rdisplsGM_ + rankId_ * UB_ALIGN;
            GlobalTensor<int64_t> rDisplSlotGM;
            rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
            int64_t rdispl = rDisplSlotGM.GetValue(0);
            int64_t myRdispl = rdispl + static_cast<int64_t>(senderIdx) * baseShare;
            LocalCopySlice(recvGradGM_ + myRdispl * hiddenBytes_, gradSortedGM_ + mySdispl * hiddenBytes_,
                           static_cast<uint64_t>(mySendCount) * hiddenBytes_);
            continue;
        }

        SendGradRemote(dstRank, senderIdx, mySendCount, mySdispl, localWinBase);
    }
}

__aicore__ inline void EngramFetchGradArch35::SendGradRemote(uint32_t dstRank, uint32_t senderIdx, int32_t sendCount,
                                                             int64_t sdispl, GM_ADDR localWinBase)
{
    uint64_t handle = GetCommHandle(dstRank, senderIdx);
    uint32_t totalSent = 0;
    uint32_t localWriteCnt = 0;
    uint32_t slotsPerSender = NUM_SLOTS / sendersPerRank_;
    if (slotsPerSender == 0U) {
        slotsPerSender = 1U;
    }

    while (totalSent < static_cast<uint32_t>(sendCount)) {
        if (totalBlocks_ > 1U) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
            int32_t remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank, senderIdx);
            while (localWriteCnt >= static_cast<uint32_t>(remoteReadCnt) &&
                   localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= slotsPerSender) {
                remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank, senderIdx);
                TimeoutCheck(startTime);
            }
        }

        uint32_t remaining = static_cast<uint32_t>(sendCount) - totalSent;
        uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;
        ascendc_assert(chunkLen != 0U, "ExchangeGrad chunkLen is 0");

        uint64_t slotBase = static_cast<uint64_t>(rankId_) * NUM_SLOTS * tokenSlotSize_;
        uint64_t slotIdx =
            static_cast<uint64_t>(senderIdx) * slotsPerSender + (static_cast<uint64_t>(localWriteCnt) % slotsPerSender);
        uint64_t slotOffset = tokenDataOffset_ + slotBase + slotIdx * tokenSlotSize_;
        GM_ADDR remoteSlotAddr = GetRemoteWinAddr(dstRank, slotOffset);
        GM_ADDR srcAddr = gradSortedGM_ + (sdispl + totalSent) * hiddenBytes_;
        uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

        GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, tokenWriteOffset_) +
                                    (static_cast<uint64_t>(rankId_) * sendersPerRank_ + senderIdx) * STATE_OFFSET;
        int32_t ret = hcomm_.WriteWithNotifyNbi(handle, remoteSlotAddr, srcAddr, dataBytes, remoteCounterAddr,
                                                static_cast<uint64_t>(localWriteCnt + 1));
        if (ret != 0) {
            for (uint32_t i = 0; i < COMM_RETRY_COUNT; i++) {
                ret = hcomm_.WriteWithNotifyNbi(handle, remoteSlotAddr, srcAddr, dataBytes, remoteCounterAddr,
                                                static_cast<uint64_t>(localWriteCnt + 1));
                if (ret == 0) {
                    break;
                }
            }
            if (ret != 0) {
                RUNTIME_ABORT(
                    "WriteWithNotifyNbi failed after %u retries, ret=%d, tag=ExTok_data, rankId=%u, dstRank=%u",
                    COMM_RETRY_COUNT, ret, rankId_, dstRank);
            }
        }

        localWriteCnt++;
        totalSent += chunkLen;
    }

    // 单核 remote handle 数随 numRanks_/numSendCores_ 配置增长，必须守卫固定数组边界
    if (pendingHandleCount_ >= Mc2Kernel::MAX_PENDING_HANDLES) {
        RUNTIME_ABORT("pendingHandles overflow: count=%u, max=%u, rankId=%u, dstRank=%u", pendingHandleCount_,
                      Mc2Kernel::MAX_PENDING_HANDLES, rankId_, dstRank);
    }
    pendingHandles_[pendingHandleCount_] = handle;
    pendingHandleCount_++;
}

__aicore__ inline void EngramFetchGradArch35::DrainAndSendFlags()
{
    for (uint32_t i = 0; i < pendingHandleCount_; i++) {
        DrainChecked(pendingHandles_[i]);
    }
    pendingHandleCount_ = 0;
}

__aicore__ inline void EngramFetchGradArch35::RecvGradFromPeers()
{
    if (!isReceiver_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t recvIdx = aivId_ - numSendCores_;
    uint32_t totalWorkUnits = (numRanks_ - 1U) * sendersPerRank_;
    if (totalWorkUnits == 0U) {
        return;
    }

    // Each work unit must be owned by EXACTLY one receiver core: with numRecvCores_ >
    // totalWorkUnits, `recvIdx % totalWorkUnits` would map several cores onto the same
    // (srcRank, senderIdx) unit and race on the tokenRead counters (flow-control slots
    // get reused while the slower duplicate is still copying -> recvGrad corruption).
    for (uint32_t wIdx = recvIdx; wIdx < totalWorkUnits; wIdx += numRecvCores_) {
        uint32_t adjustedSrcRank = wIdx / sendersPerRank_;
        uint32_t srcRank = (adjustedSrcRank >= rankId_) ? (adjustedSrcRank + 1U) : adjustedSrcRank;
        uint32_t si = wIdx % sendersPerRank_;

        GM_ADDR rSlot = recvCountsGM_ + srcRank * sizeof(int32_t);
        GlobalTensor<int32_t> rSlotGM;
        rSlotGM.SetGlobalBuffer((__gm__ int32_t *)rSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rSlotGM);
        int32_t recvCount = rSlotGM.GetValue(0);

        GM_ADDR rDisplSlot = rdisplsGM_ + srcRank * UB_ALIGN;
        GlobalTensor<int64_t> rDisplSlotGM;
        rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
        int64_t rdispl = rDisplSlotGM.GetValue(0);

        int32_t baseShare = recvCount / static_cast<int32_t>(sendersPerRank_);
        uint32_t senderRecvCount = (si == sendersPerRank_ - 1U) ?
                                       static_cast<uint32_t>(recvCount - static_cast<int32_t>(si) * baseShare) :
                                       static_cast<uint32_t>(baseShare);
        if (senderRecvCount == 0) {
            continue;
        }

        int64_t senderBase = rdispl + static_cast<int64_t>(si) * baseShare;
        uint32_t slotsPerSender = NUM_SLOTS / sendersPerRank_;
        if (slotsPerSender == 0U) {
            slotsPerSender = 1U;
        }
        uint64_t slotAreaBase = tokenDataOffset_ + static_cast<uint64_t>(srcRank) * NUM_SLOTS * tokenSlotSize_;

        uint32_t senderReceived = 0;
        uint32_t localReadCnt = 0;

        while (senderReceived < senderRecvCount) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
            int32_t remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank, si);
            while (remoteWriteCnt <= 0 || static_cast<uint32_t>(remoteWriteCnt) <= localReadCnt) {
                remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank, si);
                TimeoutCheck(startTime);
            }

            uint32_t availSlots = static_cast<uint32_t>(remoteWriteCnt) - localReadCnt;
            uint32_t remaining = senderRecvCount - senderReceived;
            uint32_t remainingChunks = (remaining + maxTokensPerSlot_ - 1U) / maxTokensPerSlot_;
            uint32_t localSlotIdx = localReadCnt % slotsPerSender;
            uint32_t maxBatchFromHere = slotsPerSender - localSlotIdx;

            uint32_t batchSlots = availSlots;
            if (batchSlots > remainingChunks) {
                batchSlots = remainingChunks;
            }
            if (batchSlots > maxBatchFromHere) {
                batchSlots = maxBatchFromHere;
            }

            uint32_t slotsRead = 0;
            uint32_t tokensRead = 0;
            // The contiguous multi-slot read assumes consecutive slots are exactly
            // maxTokensPerSlot_*hiddenBytes_ apart. tokenSlotSize_ = tokenArea/numRanks_/
            // NUM_SLOTS leaves a per-slot slack of tokenSlotSize_ % hiddenBytes_ bytes
            // whenever tokenArea isn't a multiple of numRanks_*NUM_SLOTS*hiddenBytes_;
            // reading contiguously then mixes never-written window memory into recvGrad
            // and misaligns every following slot (grad rows zero/shifted while indices
            // stay correct). Only batch when there is no slack; otherwise read per slot.
            bool noSlotSlack =
                (tokenSlotSize_ == static_cast<uint64_t>(maxTokensPerSlot_) * static_cast<uint64_t>(hiddenBytes_));
            if (noSlotSlack && batchSlots < remainingChunks) {
                uint32_t totalTokens = batchSlots * maxTokensPerSlot_;
                uint32_t firstSegIdx = localReadCnt % slotsPerSender;
                uint32_t firstSlotIdx = si * slotsPerSender + firstSegIdx;
                uint64_t firstSlotOffset = slotAreaBase + static_cast<uint64_t>(firstSlotIdx) * tokenSlotSize_;

                LocalCopySlice(recvGradGM_ + (senderBase + senderReceived) * hiddenBytes_,
                               localWinBase + firstSlotOffset, static_cast<uint64_t>(totalTokens) * hiddenBytes_);
                slotsRead = batchSlots;
                tokensRead = totalTokens;
            } else {
                while (slotsRead < batchSlots) {
                    uint32_t segIdx = (localReadCnt + slotsRead) % slotsPerSender;
                    uint32_t slotGlobalIdx = si * slotsPerSender + segIdx;
                    uint64_t slotOffset = slotAreaBase + static_cast<uint64_t>(slotGlobalIdx) * tokenSlotSize_;

                    uint32_t thisTokens = maxTokensPerSlot_;
                    if (tokensRead + thisTokens > remaining) {
                        thisTokens = remaining - tokensRead;
                    }

                    LocalCopySlice(recvGradGM_ + (senderBase + senderReceived + tokensRead) * hiddenBytes_,
                                   localWinBase + slotOffset, static_cast<uint64_t>(thisTokens) * hiddenBytes_);

                    tokensRead += thisTokens;
                    slotsRead++;
                }
            }

            localReadCnt += slotsRead;
            senderReceived += tokensRead;
            WriteRemoteCounter(srcRank, tokenReadOffset_, static_cast<int32_t>(localReadCnt), si);
        }
    }
}

__aicore__ inline void EngramFetchGradArch35::InitCompanion(uint32_t numRecv)
{
    uint32_t chunk = (numRecv + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t start = aivId_ * chunk;
    uint32_t end = start + chunk;
    if (end > numRecv) {
        end = numRecv;
    }

    LocalTensor<int32_t> idxLocal = indicesBuf_.Get<int32_t>();
    GlobalTensor<int32_t> compGM;
    compGM.SetGlobalBuffer((__gm__ int32_t *)sortCompanionGM_);

    uint32_t cur = start;
    while (cur < end) {
        uint32_t batchLen = end - cur;
        if (batchLen > ENTRY_BATCH_CAP) {
            batchLen = ENTRY_BATCH_CAP;
        }
        for (uint32_t i = 0; i < batchLen; i++) {
            idxLocal.SetValue(i, static_cast<int32_t>(cur + i));
        }
        EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
        DataCopyParams cp = {1U, static_cast<uint16_t>(batchLen * sizeof(int32_t)), 0U, 0U};
        DataCopyPad(compGM[cur], idxLocal, cp);
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
        cur += batchLen;
    }
}

__aicore__ inline void EngramFetchGradArch35::RunSort(uint32_t numRecv)
{
    // 乘积上限由 Host 侧 rankSize*numEntriesPerRank <= INT32_MAX 校验保证，int64 中间量双保险
    int64_t offset = static_cast<int64_t>(rankId_) * static_cast<int64_t>(numEntriesPerRank_);
    sorter_.SetValueOffset(static_cast<int32_t>(offset));
    sorter_.SetMaxValue(static_cast<uint32_t>(numEntriesPerRank_));
    sorter_.Process(numRecv, *tpipe_);
}

__aicore__ inline void EngramFetchGradArch35::InitFlagsAndDispls()
{
    if (aivId_ != 0) {
        return;
    }

    LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> flagInit;
    flagInit.SetGlobalBuffer((__gm__ int32_t *)flagScratchGM_);
    DataCopy(flagInit, flagLocal, STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();

    LocalTensor<int32_t> countsBuf = indicesBuf_.Get<int32_t>();
    GlobalTensor<int32_t> recvCountsGM;
    recvCountsGM.SetGlobalBuffer((__gm__ int32_t *)recvCountsOutGM_);
    DataCopyExtParams rcParams{1U, numRanks_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> i32Pad{false, 0, 0, 0};
    DataCopyPad(countsBuf, recvCountsGM, rcParams, i32Pad);

    uint32_t sendUbIdx = numRanks_;
    GlobalTensor<int32_t> sendCountsGM;
    sendCountsGM.SetGlobalBuffer((__gm__ int32_t *)sendCountsOutGM_);
    DataCopyExtParams scParams{1U, numRanks_ * UB_ALIGN, 0U, 0U, 0U};
    DataCopyPad(countsBuf[sendUbIdx * SENDCOUNT_STRIDE_RATIO], sendCountsGM, scParams, i32Pad);
    EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

    int64_t sAccum = 0;
    int64_t rAccum = 0;
    // Batch: build all per-rank 32B displ slots in UB (value in [0,8B), tail ignored by
    // readers), then one MTE3 per array — replaces numRanks*2 serialized MTE3 round trips.
    LocalTensor<int64_t> sdisplUb = tempBuf_.Get<int64_t>();
    LocalTensor<int64_t> rdisplUb = sdisplUb[numRanks_ * (UB_ALIGN / sizeof(int64_t))];
    for (uint32_t r = 0; r < numRanks_; r++) {
        int32_t sCount = countsBuf.GetValue(sendUbIdx * SENDCOUNT_STRIDE_RATIO + r * SENDCOUNT_STRIDE_RATIO);
        int32_t rCount = countsBuf.GetValue(r);
        sdisplUb.SetValue(r * (UB_ALIGN / sizeof(int64_t)), sAccum);
        rdisplUb.SetValue(r * (UB_ALIGN / sizeof(int64_t)), rAccum);
        sAccum += sCount;
        rAccum += rCount;
    }
    EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
    DataCopyParams displParams = {1U, static_cast<uint16_t>(numRanks_ * UB_ALIGN), 0U, 0U};
    GlobalTensor<int64_t> sdGM;
    sdGM.SetGlobalBuffer((__gm__ int64_t *)sdisplsGM_);
    DataCopyPad(sdGM, sdisplUb, displParams);
    GlobalTensor<int64_t> rdGM;
    rdGM.SetGlobalBuffer((__gm__ int64_t *)rdisplsGM_);
    DataCopyPad(rdGM, rdisplUb, displParams);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::ClearWinCounters()
{
    if (aivId_ != 0) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t cleanElements = numRanks_ * sendersPerRank_ * STATE_OFFSET / sizeof(int32_t);
    uint32_t totalCleanElements = 2U * cleanElements;
    LocalTensor<int32_t> cleanLocal = entryBuf_.Get<int32_t>(totalCleanElements);
    Duplicate<int32_t>(cleanLocal, 0, totalCleanElements);
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> countersGM;
    countersGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenWriteOffset_));
    DataCopy(countersGM, cleanLocal, totalCleanElements);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

// First half of the cross-rank barrier: issue all barrier flag writes WITHOUT waiting
// for completion (no drain, no sync). Split from CrossRankBarrierWait so the cross-rank
// flag flight overlaps with local work (UnsortGrad) instead of blocking.
__aicore__ inline void EngramFetchGradArch35::CrossRankBarrierIssue()
{
    if (isSender_) {
        uint32_t dstRank = aivId_ % numRanks_;
        uint32_t senderIdx = aivId_ / numRanks_;
        if (senderIdx == 0U && dstRank != rankId_) {
            uint64_t handle = GetCommHandle(dstRank, 0U);
            GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, barrierFlagOffset_) + rankId_ * STATE_OFFSET;
            WriteNbiChecked(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
        }
    }
    if (aivId_ == 0) {
        GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
        GM_ADDR localFlagAddr = localWinBase + barrierFlagOffset_ + rankId_ * STATE_OFFSET;
        LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
        Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
        EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
        GlobalTensor<int32_t> flagGM;
        flagGM.SetGlobalBuffer((__gm__ int32_t *)localFlagAddr);
        DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
    }
}

// Second half of the cross-rank barrier: drain the flag writes issued by
// CrossRankBarrierIssue (the drain latency was overlapped with UnsortGrad), then wait for
// every rank's flag. The trailing SyncAll also orders every core's UnsortGrad output
// before SendGradToPeers reads gradSortedGM_.
__aicore__ inline void EngramFetchGradArch35::CrossRankBarrierWait()
{
    if (isSender_) {
        uint32_t dstRank = aivId_ % numRanks_;
        uint32_t senderIdx = aivId_ / numRanks_;
        if (senderIdx == 0U && dstRank != rankId_) {
            uint64_t handle = GetCommHandle(dstRank, 0U);
            DrainChecked(handle);
        }
    }
    SyncAll<true>();

    if (isFlagCore_) {
        GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
        GM_ADDR localFlagBase = localWinBase + barrierFlagOffset_;
        WaitAllStatusFlags(localFlagBase, numRanks_);
        ClearStatusFlags(localFlagBase);
    }
    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradArch35::Process()
{
    if ASCEND_IS_AIV {
        InitFlagsAndDispls();
        ClearWinCounters();
        SyncAll<true>();

        // Barrier split: issue the cross-rank flag writes, then run the local 10MB
        // UnsortGrad while the flags fly; the wait phase drains + spins, and its trailing
        // SyncAll doubles as the gradSorted-completion barrier before SendGradToPeers.
        CrossRankBarrierIssue();
        UnsortGrad();
        CrossRankBarrierWait();

        SendGradToPeers();

        // RecvGradFromPeers 必须先于 entry 阶段的 SyncAll 执行：其内部的 tokenRead 计数器推进是
        // 发送核流控自旋（SendGradRemote）唯一的解锁来源；若延后到 sort 之后的 FinishExchangeGrad，
        // 发送核会因无法到达 SyncAll 而与接收核形成跨 rank 进度依赖环（超窗口场景死锁）。
        RecvGradFromPeers();

        // PR10464 flow: receivers pull grad right after senders finish issuing, so the
        // flow-control tokenRead advance never depends on a full-core barrier; the
        // entry-only phases (initCompanion/sort/count/zero) then run on the received
        // entries, and scatterCast is the only consumer of recvGradGM_.
        GlobalTensor<int32_t> numRecvGM;
        numRecvGM.SetGlobalBuffer((__gm__ int32_t *)numRecvOutGM_);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(numRecvGM);
        uint32_t numRecv = static_cast<uint32_t>(numRecvGM.GetValue(0));
        if (numRecv == 0) {
            FinishExchangeGrad();
            uniqueScatter_.WriteNumUniqueZero(numUniqueOutGM_);
        } else {
            InitCompanion(numRecv);
            SyncAll<true>();

            sortPool_.Reset();
            uint32_t maxSortCount = static_cast<uint32_t>(totalRecv_);
            sorter_.Init(maxSortCount, totalBlocks_, recvLocalEntryOutGM_, sortCompanionGM_, sortWorkspaceGm_, *tpipe_,
                         sortPool_);
            RunSort(numRecv);

            sortPool_.Reset();
            sortPool_.InitBuffer(entryBuf_, Mc2Kernel::ENTRY_BUF_BYTES);
            sortPool_.InitBuffer(gradBuf_, Mc2Kernel::GRAD_BUF_BYTES);
            uint32_t maxByPong = MaxGradRowsPerPing(static_cast<uint32_t>(hiddenBytes_), gradSubBatch_);
            if (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) {
                sortPool_.InitBuffer(castFp32Buf_, CastBufBytes(hiddenDim_, maxByPong));
            }
            sortPool_.InitBuffer(accumBuf_, AccumBufBytes(hiddenDim_));
            uniqueScatter_.SetCastBuf(castFp32Buf_);
            uniqueScatter_.SetAccumBuf(accumBuf_);
            uniqueScatter_.SetGradSubBatch(gradSubBatch_);

            uniqueScatter_.CountUniquesParallel(numRecv, recvLocalEntryOutGM_, coreStartGM_, segCountGM_);
            uniqueScatter_.ZeroGradUnique(numRecv, gradUniqueOutGM_);
            FinishExchangeGrad();
            uniqueScatter_.RunScatterCast(numRecv, recvLocalEntryOutGM_, uniqueLocalEntryOutGM_, numUniqueOutGM_,
                                          gradUniqueOutGM_, recvGradGM_, coreStartGM_, segCountGM_, sortCompanionGM_);
        }
    }
}

#endif // defined(ENABLE_ENGRAM_FETCH_GRAD_KERNEL)

} // namespace Mc2Kernel

#endif
