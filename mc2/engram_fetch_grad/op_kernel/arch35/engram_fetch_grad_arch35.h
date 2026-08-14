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
    __aicore__ inline void ExchangeGrad();
    __aicore__ inline void SendGradToPeers();
    __aicore__ inline void SendGradLocal(int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase);
    __aicore__ inline void SendGradRemote(uint32_t dstRank, int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase);
    __aicore__ inline void RecvGradFromPeers();
    __aicore__ inline void UniqueScatterAdd();
    __aicore__ inline void InitCompanion(uint32_t numRecv);
    __aicore__ inline void RunSort(uint32_t numRecv);
    __aicore__ inline void InitFlagsAndDispls();
    __aicore__ inline void ClearWinCounters();
    __aicore__ inline void CrossRankBarrier();

    __aicore__ inline void WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount);
    __aicore__ inline void ClearStatusFlags(GM_ADDR statusWinBase);
    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset);
    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank);
    __aicore__ inline int32_t ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset, uint32_t srcRank);
    __aicore__ inline void WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset, int32_t value);
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

    uint64_t barrierFlagOffset_{0};
    uint64_t tokenFlagOffset_{0};
    uint64_t tokenWriteOffset_{0};
    uint64_t tokenReadOffset_{0};
    uint64_t tokenDataOffset_{0};
    uint64_t tokenSlotSize_{0};
    uint32_t maxTokensPerSlot_{0};

    uint32_t numSendCores_{0};
    uint32_t numRecvCores_{0};
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
    GM_ADDR gradUniqueFp32GM_{0};

    TBuf<> pingBuf_;
    TBuf<> pongBuf_;
    int32_t ppEvtMte2ToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToMte2_[2] = {0, 0};
    TBuf<> hcommBuf_;
    TBuf<> statusBuf_;
    TBuf<> tempBuf_;
    TBuf<> indicesBuf_;
    TBuf<> gradSumBuf_;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    EngramFetchGradSort::EngramFetchGradSort sorter_;
    EngramFetchGradUnique::EngramFetchGradUnique uniqueScatter_;
};

__aicore__ inline void EngramFetchGradArch35::WriteNbiChecked(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    int32_t ret = hcomm_.WriteNbi(handle, dst, src, len);
    ascendc_assert(ret == 0, "WriteNbi failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
}

__aicore__ inline void EngramFetchGradArch35::DrainChecked(uint64_t handle)
{
    int32_t ret = hcomm_.Drain(handle);
}

__aicore__ inline void EngramFetchGradArch35::TimeoutCheck(uint64_t startTime)
{
    uint64_t nowUs = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
    ascendc_assert((nowUs - startTime) < ENGRAM_GRAD_TIMEOUT_US,
                   "timeout, rankId=%u, aivId=%u, elapsed=%llu us", rankId_, aivId_, nowUs - startTime);
}

__aicore__ inline GM_ADDR EngramFetchGradArch35::GetRemoteWinAddr(uint32_t dstRank, uint64_t offset)
{
    return (GM_ADDR)ctxPtr_->commBuffer[dstRank] + offset;
}

__aicore__ inline uint64_t EngramFetchGradArch35::GetCommHandle(uint32_t dstRank)
{
    uint32_t channelIdx = isSender_ ? SENDER_CHANNEL_IDX : RECEIVER_CHANNEL_IDX;
    return ctxPtr_->hcommHandle[dstRank * channelsPerRank_ + channelIdx];
}

__aicore__ inline int32_t EngramFetchGradArch35::ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset,
                                                                  uint32_t srcRank)
{
    GM_ADDR counterAddr = winBase + counterOffset + srcRank * STATE_OFFSET;
    GlobalTensor<int32_t> counterGM;
    counterGM.SetGlobalBuffer((__gm__ int32_t *)counterAddr);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counterGM);
    return counterGM.GetValue(0);
}

__aicore__ inline void EngramFetchGradArch35::WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset,
                                                                 int32_t value)
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

    uint64_t handle = GetCommHandle(dstRank);
    GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, counterOffset) + rankId_ * STATE_OFFSET;
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
    LocalTensor<uint8_t> tmp0 = pingBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> tmp1 = pongBuf_.Get<uint8_t>();
    GlobalTensor<uint8_t> srcGm;
    GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);
    uint32_t tileLen = TILE_BYTES;
    uint64_t off = 0;
    uint32_t tileIdx = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > TILE_BYTES) ? tileLen : (len - off);
        uint32_t bufIdx = tileIdx % 2U;
        LocalTensor<uint8_t> tmp = (bufIdx == 0U) ? tmp0 : tmp1;
        if (tileIdx >= 2U) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        }
        DataCopyExtParams mte2Params{1U, static_cast<uint32_t>(thisLen), 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> mte2Pad{false, 0, 0, 0};
        DataCopyPad(tmp, srcGm[off], mte2Params, mte2Pad);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        DataCopyParams mte3Params{1U, static_cast<uint16_t>(thisLen), 0U, 0U};
        DataCopyPad(dstGm[off], tmp, mte3Params);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        off += thisLen;
        tileIdx++;
    }
    if (tileIdx >= 1U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[(tileIdx - 1U) % 2U]);
    }
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[tileIdx % 2U]);
    }
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

    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;
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

    winSize_ = tilingData->commBufferSize;

    uint64_t totalRanksOffset = numRanks_ * STATE_OFFSET;
    barrierFlagOffset_ = 0;
    tokenFlagOffset_ = totalRanksOffset;
    tokenWriteOffset_ = tokenFlagOffset_ + totalRanksOffset;
    tokenReadOffset_ = tokenWriteOffset_ + totalRanksOffset;
    tokenDataOffset_ = 4U * numRanks_ * STATE_OFFSET;

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

    numSendCores_ = totalBlocks_ / 2U;
    if (numSendCores_ == 0U) {
        numSendCores_ = 1U;
    }
    numRecvCores_ = totalBlocks_ - numSendCores_;
    isSender_ = (aivId_ < numSendCores_) || (totalBlocks_ <= 1U);
    isReceiver_ = (aivId_ >= numSendCores_ && aivId_ < totalBlocks_ - 1U) || (totalBlocks_ <= 1U);
    isFlagCore_ = (aivId_ == totalBlocks_ - 1U);

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(pingBuf_, TILE_BYTES);
    tpipe_->InitBuffer(pongBuf_, TILE_BYTES);
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
    uint64_t sortWorkspaceSize = EngramFetchGradSort::EngramFetchGradSort::GetWorkspaceSize(
        maxSortCount, totalBlocks_);

    sorter_.Init(maxSortCount, totalBlocks_, recvLocalEntryOutGM_, sortCompanionGM_,
                 workspaceGM_ + wsOffset, *tpipe_);
    wsOffset += sortWorkspaceSize;

    gradUniqueFp32GM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(totalRecv_) * static_cast<uint64_t>(hiddenDim_) * sizeof(float);

    uint32_t statusBufSize = Ceil(numRanks_ * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    uint32_t tempBufSize = statusBufSize;
    uint32_t entryBatchBytes = ENTRY_BATCH_CAP * sizeof(int32_t);
    if (tempBufSize < entryBatchBytes) {
        tempBufSize = entryBatchBytes;
    }
    uint32_t coreArrayBytes = Ceil(static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    if (tempBufSize < coreArrayBytes) {
        tempBufSize = coreArrayBytes;
    }
    tpipe_->InitBuffer(tempBuf_, tempBufSize);
    uint32_t gradSumBufSize = Ceil(static_cast<uint32_t>(hiddenDim_) * sizeof(float) * GRAD_SUB_BATCH,
                                   UB_ALIGN) *
                              UB_ALIGN;
    tpipe_->InitBuffer(gradSumBuf_, gradSumBufSize);
    uint32_t indicesBufSize = 4U * 1024U;
    if (indicesBufSize < statusBufSize) {
        indicesBufSize = statusBufSize;
    }
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);

    uniqueScatter_.Init(aivId_, totalBlocks_, rankId_, numRanks_, numEntriesPerRank_,
                        hiddenDim_, hiddenBytes_, inputDtype_, outputDtype_, tpipe_,
                        pingBuf_, pongBuf_, indicesBuf_, tempBuf_, statusBuf_, gradSumBuf_);
}

__aicore__ inline void EngramFetchGradArch35::UnsortGrad()
{
    GlobalTensor<int32_t> permGM;
    permGM.SetGlobalBuffer((__gm__ int32_t *)permOutGM_);

    for (int64_t i = static_cast<int64_t>(aivId_); i < numTokens_; i += static_cast<int64_t>(totalBlocks_)) {
        int32_t origIdx = permGM.GetValue(static_cast<uint32_t>(i));
        GM_ADDR src = gradFetchedGM_ + static_cast<uint64_t>(origIdx) * static_cast<uint64_t>(hiddenBytes_);
        GM_ADDR dst = gradSortedGM_ + static_cast<uint64_t>(i) * static_cast<uint64_t>(hiddenBytes_);
        LocalCopySlice(dst, src, static_cast<uint64_t>(hiddenBytes_));
    }

    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradArch35::ExchangeGrad()
{
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];

    SendGradToPeers();
    RecvGradFromPeers();

    if (isFlagCore_) {
        GM_ADDR localFlagBase = localWinBase + tokenFlagOffset_;
        WaitAllStatusFlags(localFlagBase, numRanks_);
        ClearStatusFlags(localFlagBase);
    }

    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradArch35::SendGradToPeers()
{
    if (!isSender_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];

    for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
        GM_ADDR sSlot = sendCountsGM_ + dstRank * UB_ALIGN;
        GlobalTensor<int32_t> sSlotGM;
        sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
        int32_t sendCount = sSlotGM.GetValue(0);

        GM_ADDR sDisplSlot = sdisplsGM_ + dstRank * UB_ALIGN;
        GlobalTensor<int64_t> sDisplSlotGM;
        sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
        int64_t sdispl = sDisplSlotGM.GetValue(0);

        if (dstRank == rankId_) {
            SendGradLocal(sendCount, sdispl, localWinBase);
            continue;
        }
        SendGradRemote(dstRank, sendCount, sdispl, localWinBase);
    }
}

__aicore__ inline void EngramFetchGradArch35::SendGradLocal(int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase)
{
    if (sendCount > 0) {
        GM_ADDR rDisplSlot = rdisplsGM_ + rankId_ * UB_ALIGN;
        GlobalTensor<int64_t> rDisplSlotGM;
        rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
        int64_t rdispl = rDisplSlotGM.GetValue(0);

        LocalCopySlice(recvGradGM_ + rdispl * hiddenBytes_, gradSortedGM_ + sdispl * hiddenBytes_,
                       static_cast<uint64_t>(sendCount) * hiddenBytes_);
    }
    GM_ADDR localFlagAddr = localWinBase + tokenFlagOffset_ + rankId_ * STATE_OFFSET;
    LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> flagGM;
    flagGM.SetGlobalBuffer((__gm__ int32_t *)localFlagAddr);
    DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::SendGradRemote(uint32_t dstRank, int32_t sendCount, int64_t sdispl,
                                                             GM_ADDR localWinBase)
{
    uint64_t handle = GetCommHandle(dstRank);
    uint32_t totalSent = 0;
    uint32_t localWriteCnt = 0;

    while (totalSent < static_cast<uint32_t>(sendCount)) {
        if (totalBlocks_ > 1U) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
            int32_t remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank);
            while (localWriteCnt >= static_cast<uint32_t>(remoteReadCnt) &&
                   localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= NUM_SLOTS) {
                remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank);
                TimeoutCheck(startTime);
            }
        }

        uint32_t remaining = static_cast<uint32_t>(sendCount) - totalSent;
        uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;
        ascendc_assert(chunkLen != 0U, "ExchangeGrad chunkLen is 0");

        uint64_t slotOffset =
            tokenDataOffset_ + rankId_ * NUM_SLOTS * tokenSlotSize_ + (localWriteCnt % NUM_SLOTS) * tokenSlotSize_;
        GM_ADDR remoteSlotAddr = GetRemoteWinAddr(dstRank, slotOffset);
        GM_ADDR srcAddr = gradSortedGM_ + (sdispl + totalSent) * hiddenBytes_;
        uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

        GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, tokenWriteOffset_) + rankId_ * STATE_OFFSET;
        int32_t ret = hcomm_.WriteWithNotifyNbi(handle, remoteSlotAddr, srcAddr, dataBytes, remoteCounterAddr,
                                                static_cast<uint64_t>(localWriteCnt + 1));
        ascendc_assert(ret == 0, "WriteWithNotifyNbi failed, ret=%d, tag=ExTok_data, rankId=%u, dstRank=%u", ret,
                       rankId_, dstRank);
        DrainChecked(handle);

        localWriteCnt++;
        totalSent += chunkLen;
    }

    GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, tokenFlagOffset_) + rankId_ * STATE_OFFSET;
    WriteNbiChecked(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
    DrainChecked(handle);
}

__aicore__ inline void EngramFetchGradArch35::RecvGradFromPeers()
{
    if (!isReceiver_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t recvIdx = aivId_ - numSendCores_;

    for (uint32_t srcRank = recvIdx; srcRank < numRanks_; srcRank += numRecvCores_) {
        if (srcRank == rankId_) {
            continue;
        }

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

        uint32_t totalReceived = 0;
        uint32_t localReadCnt = 0;

        while (totalReceived < static_cast<uint32_t>(recvCount)) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
            int32_t remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank);
            while (remoteWriteCnt <= 0 || static_cast<uint32_t>(remoteWriteCnt) <= localReadCnt) {
                remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank);
                TimeoutCheck(startTime);
            }

            uint32_t remaining = static_cast<uint32_t>(recvCount) - totalReceived;
            uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;

            uint64_t slotOffset =
                tokenDataOffset_ + srcRank * NUM_SLOTS * tokenSlotSize_ + (localReadCnt % NUM_SLOTS) * tokenSlotSize_;
            GM_ADDR localSlotAddr = localWinBase + slotOffset;
            uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

            LocalCopySlice(recvGradGM_ + (rdispl + totalReceived) * hiddenBytes_, localSlotAddr, dataBytes);

            localReadCnt++;
            totalReceived += chunkLen;
            WriteRemoteCounter(srcRank, tokenReadOffset_, static_cast<int32_t>(localReadCnt));
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
    sorter_.SetMaxValue(static_cast<uint32_t>(numRanks_) * static_cast<uint32_t>(numEntriesPerRank_));
    sorter_.Process(numRecv, *tpipe_);
}

__aicore__ inline void EngramFetchGradArch35::UniqueScatterAdd()
{
    GlobalTensor<int32_t> numRecvGM;
    numRecvGM.SetGlobalBuffer((__gm__ int32_t *)numRecvOutGM_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(numRecvGM);
    uint32_t numRecv = static_cast<uint32_t>(numRecvGM.GetValue(0));
    if (numRecv == 0) {
        uniqueScatter_.WriteNumUniqueZero(numUniqueOutGM_);
        return;
    }

    InitCompanion(numRecv);
    SyncAll<true>();

    RunSort(numRecv);
    SyncAll<true>();

    uniqueScatter_.Run(numRecv, recvLocalEntryOutGM_, uniqueLocalEntryOutGM_, numUniqueOutGM_,
                       gradUniqueOutGM_, gradUniqueFp32GM_, recvGradGM_, coreStartGM_, segCountGM_,
                       sortCompanionGM_);
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

    GlobalTensor<int32_t> sendCountsOutGM;
    sendCountsOutGM.SetGlobalBuffer((__gm__ int32_t *)sendCountsOutGM_);
    GlobalTensor<int32_t> recvCountsOutGM;
    recvCountsOutGM.SetGlobalBuffer((__gm__ int32_t *)recvCountsOutGM_);

    int64_t sAccum = 0;
    int64_t rAccum = 0;
    for (uint32_t r = 0; r < numRanks_; r++) {
        LocalTensor<int64_t> displVal = tempBuf_.Get<int64_t>();
        displVal.SetValue(0, sAccum);
        EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int64_t> sdGM;
        sdGM.SetGlobalBuffer((__gm__ int64_t *)(sdisplsGM_ + r * UB_ALIGN));
        DataCopyParams displParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
        DataCopyPad(sdGM, displVal, displParams);
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
        int32_t sCount = sendCountsOutGM.GetValue(r * SENDCOUNT_STRIDE_RATIO);
        sAccum += sCount;

        displVal.SetValue(0, rAccum);
        EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int64_t> rdGM;
        rdGM.SetGlobalBuffer((__gm__ int64_t *)(rdisplsGM_ + r * UB_ALIGN));
        DataCopyPad(rdGM, displVal, displParams);
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
        int32_t rCount = recvCountsOutGM.GetValue(r);
        rAccum += rCount;
    }
}

__aicore__ inline void EngramFetchGradArch35::ClearWinCounters()
{
    if (aivId_ != 0) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    LocalTensor<int32_t> cleanLocal = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(cleanLocal, 0, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    uint64_t cleanSize = numRanks_ * STATE_OFFSET / sizeof(int32_t);
    GlobalTensor<int32_t> flagGM;
    GlobalTensor<int32_t> writeGM;
    GlobalTensor<int32_t> readGM;
    flagGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenFlagOffset_));
    writeGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenWriteOffset_));
    readGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenReadOffset_));
    DataCopy(flagGM, cleanLocal, cleanSize);
    DataCopy(writeGM, cleanLocal, cleanSize);
    DataCopy(readGM, cleanLocal, cleanSize);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::CrossRankBarrier()
{
    if (isSender_) {
        for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
            if (dstRank == rankId_) {
                continue;
            }
            uint64_t handle = GetCommHandle(dstRank);
            GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, barrierFlagOffset_) + rankId_ * STATE_OFFSET;
            WriteNbiChecked(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
            DrainChecked(handle);
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
        SyncAll<true>();

        ClearWinCounters();
        SyncAll<true>();

        CrossRankBarrier();

        UnsortGrad();
        ExchangeGrad();
        UniqueScatterAdd();
    }
}

#endif

} // namespace Mc2Kernel

#endif
