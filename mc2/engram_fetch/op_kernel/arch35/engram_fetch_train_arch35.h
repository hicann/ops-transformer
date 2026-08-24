/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_FETCH_TRAIN_ARCH35_H
#define ENGRAM_FETCH_TRAIN_ARCH35_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_ENGRAM_FETCH_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_tiling_data.h"
#include "../engram_fetch_utils.h"
#include "adv_api/hccl/hccl.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif

namespace Mc2Kernel {

#if defined(ENABLE_ENGRAM_FETCH_KERNEL)

using namespace AscendC;

template <AscendC::HardEvent event>
__aicore__ inline void EngramFetchTrainSyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

constexpr uint32_t ENGRAM_TIMEOUT_US = 5U * 1000U * 1000U;
constexpr uint32_t ENGRAM_CYCLES_PER_US = 1000U;

class EngramFetchTrainArch35 {
public:
    __aicore__ inline EngramFetchTrainArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR permOut,
                                GM_ADDR sendCountsOut, GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
                                GM_ADDR numRecvOut, GM_ADDR workspaceGM, GM_ADDR localStorageAddr, TPipe *pipe,
                                const EngramFetchTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteNbiChecked(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void DrainChecked(uint64_t handle);
    __aicore__ inline void TimeoutCheck(uint64_t startTime);
    __aicore__ inline void InitMembers(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR permOut,
                                       GM_ADDR sendCountsOut, GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
                                       GM_ADDR numRecvOut, GM_ADDR workspaceGM, GM_ADDR localStorageAddr, TPipe *pipe,
                                       const EngramFetchTilingData *tilingData);
    __aicore__ inline void InitWinLayout();
    __aicore__ inline void InitCoreRoles();
    __aicore__ inline void InitHcommAndPipe();
    __aicore__ inline void InitWorkspaceLayout(const EngramFetchTilingData *tilingData);
    __aicore__ inline void InitUbBuffers(const EngramFetchTilingData *tilingData);
    __aicore__ inline void InitFlagsAndCounters();

    __aicore__ inline void CountAndScatterPhase();
    __aicore__ inline void ComputeSdispls();
    __aicore__ inline void GatherAndSortPhase();

    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen);
    __aicore__ inline void SendCountPhase();
    __aicore__ inline void SendCountToPeers();
    __aicore__ inline void GatherRecvCounts();
    __aicore__ inline void ExchangeIndices();
    __aicore__ inline void SendIndicesToPeers();
    __aicore__ inline void SendIndicesLocal(int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase);
    __aicore__ inline void SendIndicesRemote(uint32_t dstRank, int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase);
    __aicore__ inline void RecvIndicesFromPeers();
    __aicore__ inline void LocalReadTable();
    __aicore__ inline void ExchangeToken();
    __aicore__ inline void SendTokensToPeers();
    __aicore__ inline void SendTokensLocal(int32_t sendCount, int64_t rdispl, GM_ADDR localWinBase);
    __aicore__ inline void SendTokensRemote(uint32_t dstRank, int32_t sendCount, int64_t rdispl, GM_ADDR localWinBase);
    __aicore__ inline void RecvTokensFromPeers();
    __aicore__ inline void ReorderAndSaveCtx();
    __aicore__ inline void WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount);
    __aicore__ inline void ClearStatusFlags(GM_ADDR statusWinBase);
    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset);
    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank);
    __aicore__ inline int32_t ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset, uint32_t srcRank);
    __aicore__ inline void WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset, int32_t value);

    TPipe *tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    GM_ADDR permOutGM_{nullptr};
    GM_ADDR sendCountsOutGM_{nullptr};
    GM_ADDR recvCountsOutGM_{nullptr};
    GM_ADDR recvLocalEntryOutGM_{nullptr};
    GM_ADDR numRecvOutGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    uint32_t channelsPerRank_{1};
    int32_t numEntriesPerRank_{0};
    uint32_t numTokens_{0};
    int64_t hiddenBytes_{0};
    int64_t hiddenDim_{0};
    uint64_t winSize_{0};
    uint64_t maxTokensPerRank_{0};
    uint64_t localStorageAddr_{0};

    uint64_t countFlagOffset_{0};
    uint64_t indicesFlagOffset_{0};
    uint64_t tokenFlagOffset_{0};
    uint64_t indicesWriteOffset_{0};
    uint64_t indicesReadOffset_{0};
    uint64_t tokenWriteOffset_{0};
    uint64_t tokenReadOffset_{0};
    uint64_t sendCountOffset_{0};
    uint64_t indicesDataOffset_{0};
    uint64_t tokenDataOffset_{0};

    uint64_t indicesSlotSize_{0};
    uint64_t tokenSlotSize_{0};
    uint32_t maxIndicesPerSlot_{0};
    uint32_t maxTokensPerSlot_{0};

    uint32_t numSendCores_{0};
    uint32_t numRecvCores_{0};
    bool isSender_{false};
    bool isReceiver_{false};
    bool isFlagCore_{false};

    uint64_t ubSize_{0};

    GM_ADDR sdisplsGM_{0};
    GM_ADDR rdisplsGM_{0};
    GM_ADDR sortedIndicesGM_{0};
    GM_ADDR localDataGM_{0};
    GM_ADDR recvDataGM_{0};
    GM_ADDR counterScratchGM_{0};
    GM_ADDR flagScratchGM_{0};

    TBuf<> pingBuf_;
    TBuf<> pongBuf_;
    int32_t ppEvtMte2ToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToMte2_[2] = {0, 0};
    TBuf<> indicesBuf_;
    TBuf<> hcommBuf_;
    TBuf<> rankCountsBuf_;
    TBuf<> tokenIdxInRankBuf_;
    TBuf<> rankIDsBuf_;
    TBuf<> positionsBuf_;
    TBuf<> divisorBuf_;
    TBuf<> maskBuf_;
    TBuf<> statusBuf_;
    TBuf<> tempBuf_;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    uint32_t indicesBatchSize_{0};
    uint32_t compareCntMax_{0};
};

__aicore__ inline void EngramFetchTrainArch35::WriteNbiChecked(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    int32_t ret = hcomm_.WriteNbi(handle, dst, src, len);
    ascendc_assert(ret == 0, "WriteNbi failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
}

__aicore__ inline void EngramFetchTrainArch35::DrainChecked(uint64_t handle)
{
    int32_t ret = hcomm_.Drain(handle);
}

__aicore__ inline void EngramFetchTrainArch35::TimeoutCheck(uint64_t startTime)
{
    uint64_t nowUs = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
    ascendc_assert((nowUs - startTime) < ENGRAM_TIMEOUT_US, "timeout, rankId=%u, aivId=%u, elapsed=%llu us", rankId_,
                   aivId_, nowUs - startTime);
}

__aicore__ inline GM_ADDR EngramFetchTrainArch35::GetRemoteWinAddr(uint32_t dstRank, uint64_t offset)
{
    return (GM_ADDR)ctxPtr_->commBuffer[dstRank] + offset;
}

__aicore__ inline uint64_t EngramFetchTrainArch35::GetCommHandle(uint32_t dstRank)
{
    uint32_t channelIdx = aivId_ / numRanks_ + 1U;
    if (channelIdx >= channelsPerRank_) {
        channelIdx = 1U;
    }
    return ctxPtr_->hcommHandle[dstRank * channelsPerRank_ + channelIdx];
}

__aicore__ inline void EngramFetchTrainArch35::Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
                                                    GM_ADDR permOut, GM_ADDR sendCountsOut, GM_ADDR recvCountsOut,
                                                    GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut, GM_ADDR workspaceGM,
                                                    GM_ADDR localStorageAddr, TPipe *pipe,
                                                    const EngramFetchTilingData *tilingData)
{
    InitMembers(commContext, indices, fetched, permOut, sendCountsOut, recvCountsOut, recvLocalEntryOut, numRecvOut,
                workspaceGM, localStorageAddr, pipe, tilingData);
    InitWinLayout();
    InitCoreRoles();
    InitHcommAndPipe();
    InitWorkspaceLayout(tilingData);
    InitUbBuffers(tilingData);
}

__aicore__ inline void EngramFetchTrainArch35::InitMembers(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
                                                           GM_ADDR permOut, GM_ADDR sendCountsOut,
                                                           GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
                                                           GM_ADDR numRecvOut, GM_ADDR workspaceGM,
                                                           GM_ADDR localStorageAddr, TPipe *pipe,
                                                           const EngramFetchTilingData *tilingData)
{
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    permOutGM_ = permOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCountsOutGM_ = recvCountsOut;
    recvLocalEntryOutGM_ = recvLocalEntryOut;
    numRecvOutGM_ = numRecvOut;
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
    numTokens_ = static_cast<uint32_t>(tilingData->numTokens);
    hiddenBytes_ = tilingData->hiddenBytes;
    hiddenDim_ = tilingData->hiddenDim;
    maxTokensPerRank_ = tilingData->numMaxTokensPerRank;
    ubSize_ = tilingData->ubSize;
    winSize_ = tilingData->commBufferSize;

    AscendC::GlobalTensor<int64_t> localStorageTensor;
    localStorageTensor.SetGlobalBuffer((__gm__ int64_t *)localStorageAddr);
    localStorageAddr_ = static_cast<uint64_t>(localStorageTensor.GetValue(0));
}

__aicore__ inline void EngramFetchTrainArch35::InitWinLayout()
{
    uint64_t fixedSize = WIN_REGION_COUNT * numRanks_ * STATE_OFFSET;
    uint64_t remaining = (winSize_ > fixedSize) ? (winSize_ - fixedSize) : 0U;

    uint64_t totalRanksOffset = numRanks_ * STATE_OFFSET;
    countFlagOffset_ = 0;
    indicesFlagOffset_ = totalRanksOffset;
    tokenFlagOffset_ = indicesFlagOffset_ + totalRanksOffset;
    indicesWriteOffset_ = tokenFlagOffset_ + totalRanksOffset;
    indicesReadOffset_ = indicesWriteOffset_ + totalRanksOffset;
    tokenWriteOffset_ = indicesReadOffset_ + totalRanksOffset;
    tokenReadOffset_ = tokenWriteOffset_ + totalRanksOffset;
    sendCountOffset_ = tokenReadOffset_ + totalRanksOffset;

    uint64_t indicesAreaRaw = remaining / INDICES_RATIO;
    uint64_t indicesGranularity = static_cast<uint64_t>(numRanks_) * NUM_SLOTS * UB_ALIGN;
    uint64_t indicesArea = Ceil(indicesAreaRaw, indicesGranularity) * indicesGranularity;
    uint64_t tokenAreaRaw = (remaining > indicesArea) ? (remaining - indicesArea) : 0U;
    uint64_t tokenGranularity = static_cast<uint64_t>(numRanks_) * NUM_SLOTS * static_cast<uint64_t>(hiddenBytes_);
    uint64_t tokenArea =
        (tokenGranularity > 0U) ? Ceil(tokenAreaRaw, tokenGranularity) * tokenGranularity : tokenAreaRaw;
    if (tokenArea > tokenAreaRaw) {
        tokenArea = (tokenAreaRaw / tokenGranularity) * tokenGranularity;
    }

    indicesDataOffset_ = sendCountOffset_ + numRanks_ * STATE_OFFSET;
    tokenDataOffset_ = indicesDataOffset_ + indicesArea;

    indicesSlotSize_ = indicesArea / numRanks_ / NUM_SLOTS;
    tokenSlotSize_ = tokenArea / numRanks_ / NUM_SLOTS;
    maxIndicesPerSlot_ = static_cast<uint32_t>(indicesSlotSize_ / sizeof(int32_t));
    maxTokensPerSlot_ = static_cast<uint64_t>(tokenSlotSize_) >= static_cast<uint64_t>(hiddenBytes_) ?
                            static_cast<uint32_t>(tokenSlotSize_ / static_cast<uint64_t>(hiddenBytes_)) :
                            0U;
}

__aicore__ inline void EngramFetchTrainArch35::InitCoreRoles()
{
    ascendc_assert(maxIndicesPerSlot_ != 0U && maxTokensPerSlot_ != 0U, "slot too small");

    numSendCores_ = totalBlocks_ / 2U;
    if (numSendCores_ == 0U) {
        numSendCores_ = 1U;
    }
    if (totalBlocks_ > 1U) {
        numRecvCores_ = totalBlocks_ - numSendCores_ - 1U;
    } else {
        numRecvCores_ = 1U;
    }
    if (numRecvCores_ == 0U) {
        numRecvCores_ = 1U;
    }
    isSender_ = (aivId_ < numSendCores_) || (totalBlocks_ <= 1U);
    isReceiver_ = (aivId_ >= numSendCores_ && aivId_ < totalBlocks_ - 1U) || (totalBlocks_ <= 1U);
    isFlagCore_ = (aivId_ == totalBlocks_ - 1U);
}

__aicore__ inline void EngramFetchTrainArch35::InitHcommAndPipe()
{
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(pingBuf_, TILE_BYTES);
    tpipe_->InitBuffer(pongBuf_, TILE_BYTES);
    ppEvtMte2ToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte2ToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte3ToMte2_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    ppEvtMte3ToMte2_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
}

__aicore__ inline void EngramFetchTrainArch35::InitWorkspaceLayout(const EngramFetchTilingData *tilingData)
{
    uint64_t wsOffset = 0;
    sdisplsGM_ = workspaceGM_ + wsOffset;
    wsOffset += Ceil(numRanks_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    rdisplsGM_ = workspaceGM_ + wsOffset;
    wsOffset += Ceil(numRanks_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    sortedIndicesGM_ = workspaceGM_ + wsOffset;
    wsOffset += Ceil(static_cast<uint64_t>(numTokens_) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    int64_t totalRecv = tilingData->totalRecv;
    localDataGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(totalRecv) * static_cast<uint64_t>(hiddenBytes_);
    recvDataGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(numTokens_) * static_cast<uint64_t>(hiddenBytes_);
    counterScratchGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * UB_ALIGN;
    flagScratchGM_ = workspaceGM_ + wsOffset;
}

__aicore__ inline void EngramFetchTrainArch35::InitUbBuffers(const EngramFetchTilingData *tilingData)
{
    uint32_t countsBufSize = Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankCountsBuf_, countsBufSize);
    uint32_t statusBufSize = Ceil(numRanks_ * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    uint32_t tempBufSize = Ceil(numRanks_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tempBuf_, tempBufSize);

    uint32_t bytesPerIndice = sizeof(int32_t) + sizeof(uint32_t) + sizeof(int32_t) * 3U + 1U;
    uint64_t usedUb = HCOMM_INIT_SIZE + TILE_BYTES * 2U + countsBufSize + statusBufSize + tempBufSize;
    uint64_t availableUb = (ubSize_ > usedUb + UB_RESERVED_SIZE) ? (ubSize_ - usedUb - UB_RESERVED_SIZE) : 0U;
    uint32_t maxBatchSize = static_cast<uint32_t>(availableUb / bytesPerIndice);
    indicesBatchSize_ = (numTokens_ <= maxBatchSize) ? numTokens_ : maxBatchSize;
    if (indicesBatchSize_ == 0U) {
        indicesBatchSize_ = 1U;
    }

    uint32_t indicesBufSize = Ceil(indicesBatchSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    uint32_t tokenIdxInRankBufSize = Ceil(indicesBatchSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tokenIdxInRankBuf_, tokenIdxInRankBufSize);

    uint32_t compareCntMax =
        Ceil(indicesBatchSize_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    compareCntMax_ = compareCntMax;
    uint32_t int32BufSize = compareCntMax * sizeof(int32_t);
    tpipe_->InitBuffer(rankIDsBuf_, int32BufSize);
    tpipe_->InitBuffer(positionsBuf_, int32BufSize);
    tpipe_->InitBuffer(divisorBuf_, int32BufSize);
    uint32_t maskBufSize = Ceil(Ceil(compareCntMax, BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(maskBuf_, maskBufSize);
}

__aicore__ inline void EngramFetchTrainArch35::InitFlagsAndCounters()
{
    if (aivId_ == 0) {
        LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
        Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
        EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
        GlobalTensor<int32_t> flagInit;
        flagInit.SetGlobalBuffer((__gm__ int32_t *)flagScratchGM_);
        DataCopy(flagInit, flagLocal, STATE_OFFSET / sizeof(int32_t));
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

        Duplicate<int32_t>(flagLocal, 0, numRanks_ * STATE_OFFSET / sizeof(int32_t));
        EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();

        uint64_t cleanSize = numRanks_ * STATE_OFFSET / sizeof(int32_t);
        GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
        GlobalTensor<int32_t> writeGM;
        GlobalTensor<int32_t> readGM;
        writeGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + indicesWriteOffset_));
        readGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + indicesReadOffset_));
        DataCopy(writeGM, flagLocal, cleanSize);
        DataCopy(readGM, flagLocal, cleanSize);
        GlobalTensor<int32_t> tWriteGM;
        GlobalTensor<int32_t> tReadGM;
        tWriteGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenWriteOffset_));
        tReadGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenReadOffset_));
        DataCopy(tWriteGM, flagLocal, cleanSize);
        DataCopy(tReadGM, flagLocal, cleanSize);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
    }
    SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainArch35::CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen)
{
    GlobalTensor<int32_t> indicesGlobal;
    indicesGlobal.SetGlobalBuffer((__gm__ int32_t *)indicesGM_);
    LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    DataCopyExtParams params{1U, indicesBatchLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(indicesLocal, indicesGlobal[indicesBatchStart], params, pad);
}

__aicore__ inline void EngramFetchTrainArch35::WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount)
{
    int32_t sumOfFlag = 0;
    int32_t compareFlag = static_cast<int32_t>(expectCount);
    uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
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

__aicore__ inline void EngramFetchTrainArch35::ClearStatusFlags(GM_ADDR statusWinBase)
{
    LocalTensor<int32_t> cleanTensor = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(cleanTensor, 0, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();

    GlobalTensor<int32_t> statusGM;
    statusGM.SetGlobalBuffer((__gm__ int32_t *)statusWinBase);
    DataCopy(statusGM, cleanTensor, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::SendCountPhase()
{
    SendCountToPeers();
    SyncAll<true>();
    GatherRecvCounts();
    SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainArch35::SendCountToPeers()
{
    for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += totalBlocks_) {
        GM_ADDR sSlot = sendCountsOutGM_ + dstRank * UB_ALIGN;
        GlobalTensor<int32_t> sSlotGM;
        sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
        int32_t countVal = sSlotGM.GetValue(0);

        GM_ADDR remoteSendCountAddr = GetRemoteWinAddr(dstRank, sendCountOffset_) + rankId_ * STATE_OFFSET;
        GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, countFlagOffset_) + rankId_ * STATE_OFFSET;

        if (dstRank == rankId_) {
            LocalTensor<int32_t> valLocal = statusBuf_.Get<int32_t>();
            valLocal.SetValue(0, countVal);
            EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
            GlobalTensor<int32_t> sendCountGM;
            sendCountGM.SetGlobalBuffer((__gm__ int32_t *)remoteSendCountAddr);
            DataCopyParams valParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
            DataCopyPad(sendCountGM, valLocal, valParams);
            EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

            LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
            Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
            EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
            GlobalTensor<int32_t> flagGM;
            flagGM.SetGlobalBuffer((__gm__ int32_t *)remoteFlagAddr);
            DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
            EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
        } else {
            uint64_t handle = GetCommHandle(dstRank);
            GM_ADDR countSrcAddr = sendCountsOutGM_ + dstRank * UB_ALIGN;

            int32_t ret = hcomm_.WriteWithNotifyNbi(handle, remoteSendCountAddr, countSrcAddr, sizeof(int32_t),
                                                    remoteFlagAddr, 1);
            ascendc_assert(ret == 0, "WriteWithNotifyNbi failed, ret=%d, tag=SendCount, rankId=%u, dstRank=%u", ret,
                           rankId_, dstRank);
        }
    }
}

__aicore__ inline void EngramFetchTrainArch35::GatherRecvCounts()
{
    if (!isFlagCore_) {
        return;
    }

    GM_ADDR localFlagBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_] + countFlagOffset_;
    WaitAllStatusFlags(localFlagBase, numRanks_);

    LocalTensor<int32_t> recvCountsUb = rankCountsBuf_.Get<int32_t>();
    LocalTensor<int64_t> rdisplsUb = tempBuf_.Get<int64_t>();

    int64_t rSum = 0;
    for (uint32_t r = 0; r < numRanks_; r++) {
        GM_ADDR countAddr = (GM_ADDR)ctxPtr_->commBuffer[rankId_] + sendCountOffset_ + r * STATE_OFFSET;
        GlobalTensor<int32_t> countGM;
        countGM.SetGlobalBuffer((__gm__ int32_t *)countAddr);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(countGM);
        int32_t recvCount = countGM.GetValue(0);

        recvCountsUb.SetValue(r, recvCount);
        rdisplsUb.SetValue(r, rSum);
        rSum += static_cast<int64_t>(recvCount);
    }

    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> recvCountsOutGM;
    recvCountsOutGM.SetGlobalBuffer((__gm__ int32_t *)recvCountsOutGM_);
    DataCopyParams recvParams = {1U, static_cast<uint16_t>(numRanks_ * sizeof(int32_t)), 0U, 0U};
    DataCopyPad(recvCountsOutGM, recvCountsUb, recvParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int64_t> rdisplsOutGM;
    rdisplsOutGM.SetGlobalBuffer((__gm__ int64_t *)rdisplsGM_);
    DataCopyParams rdisplParams = {1U, static_cast<uint16_t>(numRanks_ * sizeof(int64_t)), 0U, 0U};
    DataCopyPad(rdisplsOutGM, rdisplsUb, rdisplParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

    ClearStatusFlags(localFlagBase);
}

__aicore__ inline int32_t EngramFetchTrainArch35::ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset,
                                                                   uint32_t srcRank)
{
    GM_ADDR counterAddr = winBase + counterOffset + srcRank * STATE_OFFSET;
    GlobalTensor<int32_t> counterGM;
    counterGM.SetGlobalBuffer((__gm__ int32_t *)counterAddr);
    LocalTensor<int32_t> counterLocal = statusBuf_.Get<int32_t>();
    DataCopy(counterLocal, counterGM, UB_ALIGN / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();
    return counterLocal.GetValue(0);
}

__aicore__ inline void EngramFetchTrainArch35::WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset,
                                                                  int32_t value)
{
    if (dstRank == rankId_)
        return;

    LocalTensor<int32_t> valLocal = statusBuf_.Get<int32_t>();
    valLocal.SetValue(0, value);
    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();

    GM_ADDR srcAddr = counterScratchGM_ + static_cast<uint64_t>(aivId_) * UB_ALIGN;
    GlobalTensor<int32_t> srcGM;
    srcGM.SetGlobalBuffer((__gm__ int32_t *)srcAddr);
    DataCopyParams valParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
    DataCopyPad(srcGM, valLocal, valParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

    uint64_t handle = GetCommHandle(dstRank);
    GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, counterOffset) + rankId_ * STATE_OFFSET;
    WriteNbiChecked(handle, remoteCounterAddr, srcAddr, sizeof(int32_t));
    DrainChecked(handle);
}

__aicore__ inline void EngramFetchTrainArch35::ExchangeIndices()
{
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];

    SendIndicesToPeers();
    RecvIndicesFromPeers();

    if (isFlagCore_) {
        GM_ADDR localFlagBase = localWinBase + indicesFlagOffset_;
        WaitAllStatusFlags(localFlagBase, numRanks_);
        ClearStatusFlags(localFlagBase);
    }

    SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainArch35::SendIndicesToPeers()
{
    if (!isSender_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];

    for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
        GM_ADDR sSlot = sendCountsOutGM_ + dstRank * UB_ALIGN;
        GlobalTensor<int32_t> sSlotGM;
        sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
        int32_t sendCount = sSlotGM.GetValue(0);

        GM_ADDR sDisplSlot = sdisplsGM_ + dstRank * sizeof(int64_t);
        GlobalTensor<int64_t> sDisplSlotGM;
        sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
        int64_t sdispl = sDisplSlotGM.GetValue(0);

        if (dstRank == rankId_) {
            SendIndicesLocal(sendCount, sdispl, localWinBase);
            continue;
        }
        SendIndicesRemote(dstRank, sendCount, sdispl, localWinBase);
    }
}

__aicore__ inline void EngramFetchTrainArch35::SendIndicesLocal(int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase)
{
    if (sendCount > 0) {
        GM_ADDR rDisplSlot = rdisplsGM_ + rankId_ * sizeof(int64_t);
        GlobalTensor<int64_t> rDisplSlotGM;
        rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
        int64_t rdispl = rDisplSlotGM.GetValue(0);
        LocalCopySlice(recvLocalEntryOutGM_ + static_cast<uint64_t>(rdispl) * sizeof(int32_t),
                       sortedIndicesGM_ + static_cast<uint64_t>(sdispl) * sizeof(int32_t),
                       static_cast<uint64_t>(sendCount) * sizeof(int32_t));
    }
    GM_ADDR localFlagAddr = localWinBase + indicesFlagOffset_ + rankId_ * STATE_OFFSET;
    LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> flagGM;
    flagGM.SetGlobalBuffer((__gm__ int32_t *)localFlagAddr);
    DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::SendIndicesRemote(uint32_t dstRank, int32_t sendCount, int64_t sdispl,
                                                                 GM_ADDR localWinBase)
{
    uint64_t handle = GetCommHandle(dstRank);
    uint32_t totalSent = 0;
    uint32_t localWriteCnt = 0;

    {
        LocalTensor<int32_t> zeroLocal = statusBuf_.Get<int32_t>();
        zeroLocal.SetValue(0, 0);
        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
        GM_ADDR readCntAddr = localWinBase + indicesReadOffset_ + dstRank * STATE_OFFSET;
        GlobalTensor<int32_t> readCntGM;
        readCntGM.SetGlobalBuffer((__gm__ int32_t *)readCntAddr);
        DataCopyParams zeroParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
        DataCopyPad(readCntGM, zeroLocal, zeroParams);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
    }

    while (totalSent < static_cast<uint32_t>(sendCount)) {
        if (totalBlocks_ > 1U) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
            int32_t remoteReadCnt = ReadLocalCounter(localWinBase, indicesReadOffset_, dstRank);
            while (localWriteCnt >= static_cast<uint32_t>(remoteReadCnt) &&
                   localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= NUM_SLOTS) {
                remoteReadCnt = ReadLocalCounter(localWinBase, indicesReadOffset_, dstRank);
                TimeoutCheck(startTime);
            }
        }

        uint32_t remaining = static_cast<uint32_t>(sendCount) - totalSent;
        uint32_t chunkLen = (remaining > maxIndicesPerSlot_) ? maxIndicesPerSlot_ : remaining;
        ascendc_assert(chunkLen != 0U, "ExchangeIndices chunkLen is 0");

        uint64_t slotOffset = indicesDataOffset_ + rankId_ * NUM_SLOTS * indicesSlotSize_ +
                              (localWriteCnt % NUM_SLOTS) * indicesSlotSize_;
        GM_ADDR remoteSlotAddr = GetRemoteWinAddr(dstRank, slotOffset);
        GM_ADDR srcAddr = sortedIndicesGM_ + (sdispl + totalSent) * sizeof(int32_t);

        GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, indicesWriteOffset_) + rankId_ * STATE_OFFSET;
        int32_t ret = hcomm_.WriteWithNotifyNbi(handle, remoteSlotAddr, srcAddr, chunkLen * sizeof(int32_t),
                                                remoteCounterAddr, static_cast<uint64_t>(localWriteCnt + 1));
        ascendc_assert(ret == 0, "WriteWithNotifyNbi failed, ret=%d, tag=ExIdx_data, rankId=%u, dstRank=%u", ret,
                       rankId_, dstRank);

        localWriteCnt++;
        totalSent += chunkLen;
    }

    GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, indicesFlagOffset_) + rankId_ * STATE_OFFSET;
    WriteNbiChecked(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
    DrainChecked(handle);
}

__aicore__ inline void EngramFetchTrainArch35::RecvIndicesFromPeers()
{
    if (!isReceiver_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t recvIdx = aivId_ - numSendCores_;

    for (uint32_t srcRank = recvIdx; srcRank < numRanks_; srcRank += numRecvCores_) {
        if (srcRank == rankId_)
            continue;

        GM_ADDR rSlot = recvCountsOutGM_ + srcRank * sizeof(int32_t);
        GlobalTensor<int32_t> rSlotGM;
        rSlotGM.SetGlobalBuffer((__gm__ int32_t *)rSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rSlotGM);
        int32_t recvCount = rSlotGM.GetValue(0);

        GM_ADDR rDisplSlot = rdisplsGM_ + srcRank * sizeof(int64_t);
        GlobalTensor<int64_t> rDisplSlotGM;
        rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
        int64_t rdispl = rDisplSlotGM.GetValue(0);

        uint32_t totalReceived = 0;
        uint32_t localReadCnt = 0;

        while (totalReceived < static_cast<uint32_t>(recvCount)) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
            int32_t remoteWriteCnt = ReadLocalCounter(localWinBase, indicesWriteOffset_, srcRank);
            while (remoteWriteCnt <= 0 || static_cast<uint32_t>(remoteWriteCnt) <= localReadCnt) {
                remoteWriteCnt = ReadLocalCounter(localWinBase, indicesWriteOffset_, srcRank);
                TimeoutCheck(startTime);
            }

            uint32_t remaining = static_cast<uint32_t>(recvCount) - totalReceived;
            uint32_t chunkLen = (remaining > maxIndicesPerSlot_) ? maxIndicesPerSlot_ : remaining;

            uint64_t slotOffset = indicesDataOffset_ + srcRank * NUM_SLOTS * indicesSlotSize_ +
                                  (localReadCnt % NUM_SLOTS) * indicesSlotSize_;
            GM_ADDR localSlotAddr = localWinBase + slotOffset;
            uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * sizeof(int32_t);

            LocalCopySlice(recvLocalEntryOutGM_ + static_cast<uint64_t>(rdispl + totalReceived) * sizeof(int32_t),
                           localSlotAddr, dataBytes);

            localReadCnt++;
            totalReceived += chunkLen;
            WriteRemoteCounter(srcRank, indicesReadOffset_, static_cast<int32_t>(localReadCnt));
        }
    }
}

__aicore__ inline void EngramFetchTrainArch35::ExchangeToken()
{
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];

    SendTokensToPeers();
    RecvTokensFromPeers();

    if (isFlagCore_) {
        GM_ADDR localFlagBase = localWinBase + tokenFlagOffset_;
        WaitAllStatusFlags(localFlagBase, numRanks_);
        ClearStatusFlags(localFlagBase);
    }

    SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainArch35::SendTokensToPeers()
{
    if (!isSender_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];

    for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
        GM_ADDR rSlot = recvCountsOutGM_ + dstRank * sizeof(int32_t);
        GlobalTensor<int32_t> rSlotGM;
        rSlotGM.SetGlobalBuffer((__gm__ int32_t *)rSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rSlotGM);
        int32_t sendCount = rSlotGM.GetValue(0);

        GM_ADDR rDisplSlot = rdisplsGM_ + dstRank * sizeof(int64_t);
        GlobalTensor<int64_t> rDisplSlotGM;
        rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
        int64_t rdispl = rDisplSlotGM.GetValue(0);

        if (dstRank == rankId_) {
            SendTokensLocal(sendCount, rdispl, localWinBase);
            continue;
        }
        SendTokensRemote(dstRank, sendCount, rdispl, localWinBase);
    }
}

__aicore__ inline void EngramFetchTrainArch35::SendTokensLocal(int32_t sendCount, int64_t rdispl, GM_ADDR localWinBase)
{
    if (sendCount > 0) {
        GM_ADDR sDisplSlot = sdisplsGM_ + rankId_ * sizeof(int64_t);
        GlobalTensor<int64_t> sDisplSlotGM;
        sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
        int64_t sdispl = sDisplSlotGM.GetValue(0);

        uint64_t totalBytes = static_cast<uint64_t>(sendCount) * static_cast<uint64_t>(hiddenBytes_);
        LocalCopySlice(recvDataGM_ + sdispl * hiddenBytes_, localDataGM_ + rdispl * hiddenBytes_, totalBytes);
    }
    GM_ADDR localFlagAddr = localWinBase + tokenFlagOffset_ + rankId_ * STATE_OFFSET;
    LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> flagGM;
    flagGM.SetGlobalBuffer((__gm__ int32_t *)localFlagAddr);
    DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::SendTokensRemote(uint32_t dstRank, int32_t sendCount, int64_t rdispl,
                                                                GM_ADDR localWinBase)
{
    uint64_t handle = GetCommHandle(dstRank);
    uint32_t totalSent = 0;
    uint32_t localWriteCnt = 0;

    {
        LocalTensor<int32_t> zeroLocal = statusBuf_.Get<int32_t>();
        zeroLocal.SetValue(0, 0);
        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
        GM_ADDR readCntAddr = localWinBase + tokenReadOffset_ + dstRank * STATE_OFFSET;
        GlobalTensor<int32_t> readCntGM;
        readCntGM.SetGlobalBuffer((__gm__ int32_t *)readCntAddr);
        DataCopyParams zeroParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
        DataCopyPad(readCntGM, zeroLocal, zeroParams);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
    }

    while (totalSent < static_cast<uint32_t>(sendCount)) {
        if (totalBlocks_ > 1U) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
            int32_t remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank);
            while (localWriteCnt >= static_cast<uint32_t>(remoteReadCnt) &&
                   localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= NUM_SLOTS) {
                remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank);
                TimeoutCheck(startTime);
            }
        }

        uint32_t remaining = static_cast<uint32_t>(sendCount) - totalSent;
        uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;
        ascendc_assert(chunkLen != 0U, "ExchangeToken chunkLen is 0");

        uint64_t slotOffset =
            tokenDataOffset_ + rankId_ * NUM_SLOTS * tokenSlotSize_ + (localWriteCnt % NUM_SLOTS) * tokenSlotSize_;
        GM_ADDR remoteSlotAddr = GetRemoteWinAddr(dstRank, slotOffset);
        GM_ADDR srcAddr = localDataGM_ + (rdispl + totalSent) * hiddenBytes_;
        uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

        GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, tokenWriteOffset_) + rankId_ * STATE_OFFSET;
        int32_t ret = hcomm_.WriteWithNotifyNbi(handle, remoteSlotAddr, srcAddr, dataBytes, remoteCounterAddr,
                                                static_cast<uint64_t>(localWriteCnt + 1));
        ascendc_assert(ret == 0, "WriteWithNotifyNbi failed, ret=%d, tag=ExTok_data, rankId=%u, dstRank=%u", ret,
                       rankId_, dstRank);

        localWriteCnt++;
        totalSent += chunkLen;
    }

    GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, tokenFlagOffset_) + rankId_ * STATE_OFFSET;
    WriteNbiChecked(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
    DrainChecked(handle);
}

__aicore__ inline void EngramFetchTrainArch35::RecvTokensFromPeers()
{
    if (!isReceiver_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t recvIdx = aivId_ - numSendCores_;

    for (uint32_t srcRank = recvIdx; srcRank < numRanks_; srcRank += numRecvCores_) {
        if (srcRank == rankId_)
            continue;

        GM_ADDR sSlot = sendCountsOutGM_ + srcRank * UB_ALIGN;
        GlobalTensor<int32_t> sSlotGM;
        sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
        int32_t recvCount = sSlotGM.GetValue(0);

        GM_ADDR sDisplSlot = sdisplsGM_ + srcRank * sizeof(int64_t);
        GlobalTensor<int64_t> sDisplSlotGM;
        sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
        int64_t sdispl = sDisplSlotGM.GetValue(0);

        uint32_t totalReceived = 0;
        uint32_t localReadCnt = 0;

        while (totalReceived < static_cast<uint32_t>(recvCount)) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
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

            LocalCopySlice(recvDataGM_ + (sdispl + totalReceived) * hiddenBytes_, localSlotAddr, dataBytes);

            localReadCnt++;
            totalReceived += chunkLen;
            WriteRemoteCounter(srcRank, tokenReadOffset_, static_cast<int32_t>(localReadCnt));
        }
    }
}

__aicore__ inline void EngramFetchTrainArch35::ReorderAndSaveCtx()
{
    uint32_t totalPerCore = (numTokens_ + totalBlocks_ - 1) / totalBlocks_;
    uint32_t start = aivId_ * totalPerCore;
    uint32_t end = start + totalPerCore;
    if (end > numTokens_) {
        end = numTokens_;
    }

    LocalTensor<uint32_t> tokenIdxUb = tokenIdxInRankBuf_.Get<uint32_t>();
    uint32_t tokenIdxBufCap = indicesBatchSize_;

    uint32_t cur = start;
    while (cur < end) {
        uint32_t batchLen = end - cur;
        if (batchLen > tokenIdxBufCap) {
            batchLen = tokenIdxBufCap;
        }
        GlobalTensor<uint32_t> tokenIdxBatchGM;
        tokenIdxBatchGM.SetGlobalBuffer((__gm__ uint32_t *)permOutGM_);
        DataCopyExtParams cpParams{1U, batchLen * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<uint32_t> cpPad{false, 0, 0, 0};
        DataCopyPad(tokenIdxUb, tokenIdxBatchGM[cur], cpParams, cpPad);
        EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

        for (uint32_t j = 0; j < batchLen; j++) {
            uint32_t originalIdx = tokenIdxUb.GetValue(j);
            if (originalIdx >= numTokens_) {
                continue;
            }
            uint32_t gmIdx = cur + j;
            GM_ADDR src = recvDataGM_ + static_cast<uint64_t>(gmIdx) * static_cast<uint64_t>(hiddenBytes_);
            GM_ADDR dst = fetchedGM_ + static_cast<uint64_t>(originalIdx) * static_cast<uint64_t>(hiddenBytes_);
            LocalCopySlice(dst, src, static_cast<uint64_t>(hiddenBytes_));
        }
        cur += batchLen;
    }

    SyncAll<true>();

    if (aivId_ == 0) {
        LocalTensor<int32_t> tmp32 = indicesBuf_.Get<int32_t>();

        GM_ADDR lastRDisplSlot = rdisplsGM_ + (numRanks_ - 1) * sizeof(int64_t);
        GlobalTensor<int64_t> lastRDisplSlotGM;
        lastRDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)lastRDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lastRDisplSlotGM);
        GM_ADDR lastRecvSlot = recvCountsOutGM_ + (numRanks_ - 1) * sizeof(int32_t);
        GlobalTensor<int32_t> lastRecvSlotGM;
        lastRecvSlotGM.SetGlobalBuffer((__gm__ int32_t *)lastRecvSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lastRecvSlotGM);
        int32_t totalRecv = static_cast<int32_t>(lastRDisplSlotGM.GetValue(0) + lastRecvSlotGM.GetValue(0));

        tmp32.SetValue(0, totalRecv);
        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
        DataCopyParams gmParamsScalar = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
        GlobalTensor<int32_t> numRecvOutGM;
        numRecvOutGM.SetGlobalBuffer((__gm__ int32_t *)numRecvOutGM_);
        DataCopyPad(numRecvOutGM, tmp32, gmParamsScalar);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
    }
}

__aicore__ inline void EngramFetchTrainArch35::LocalReadTable()
{
    if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
        return;
    }

    GM_ADDR lastRDisplSlot = rdisplsGM_ + (numRanks_ - 1) * sizeof(int64_t);
    GlobalTensor<int64_t> lastRDisplSlotGM;
    lastRDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)lastRDisplSlot);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lastRDisplSlotGM);
    GM_ADDR lastRecvSlot = recvCountsOutGM_ + (numRanks_ - 1) * sizeof(int32_t);
    GlobalTensor<int32_t> lastRecvSlotGM;
    lastRecvSlotGM.SetGlobalBuffer((__gm__ int32_t *)lastRecvSlot);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lastRecvSlotGM);
    int64_t totalRecv = lastRDisplSlotGM.GetValue(0) + lastRecvSlotGM.GetValue(0);

    int64_t totalPerCore = (totalRecv + static_cast<int64_t>(totalBlocks_) - 1) / static_cast<int64_t>(totalBlocks_);
    int64_t start = static_cast<int64_t>(aivId_) * totalPerCore;
    int64_t end = start + totalPerCore;
    if (end > totalRecv) {
        end = totalRecv;
    }

    LocalTensor<int32_t> indicesUb = indicesBuf_.Get<int32_t>();
    uint32_t indicesBufCap = indicesBatchSize_;

    int64_t cur = start;
    while (cur < end) {
        int64_t batchLen = end - cur;
        if (batchLen > static_cast<int64_t>(indicesBufCap)) {
            batchLen = static_cast<int64_t>(indicesBufCap);
        }
        GlobalTensor<int32_t> recvIndicesBatchGM;
        recvIndicesBatchGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);
        DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen) * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U,
                                   0U};
        DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        DataCopyPad(indicesUb, recvIndicesBatchGM[static_cast<uint32_t>(cur)], cpParams, cpPad);
        EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

        for (int64_t j = 0; j < batchLen; j++) {
            int32_t globalIdx = indicesUb.GetValue(static_cast<uint32_t>(j));
            if (globalIdx < 0) {
                continue;
            }
            uint32_t localEntryIdx =
                static_cast<uint32_t>(static_cast<int64_t>(globalIdx) -
                                      static_cast<int64_t>(rankId_) * static_cast<int64_t>(numEntriesPerRank_));
            if (localEntryIdx >= static_cast<uint32_t>(numEntriesPerRank_)) {
                continue;
            }
            int64_t gmIdx = cur + j;
            GM_ADDR src =
                (GM_ADDR)localStorageAddr_ + static_cast<uint64_t>(localEntryIdx) * static_cast<uint64_t>(hiddenBytes_);
            GM_ADDR dst = localDataGM_ + static_cast<uint64_t>(gmIdx) * static_cast<uint64_t>(hiddenBytes_);
            LocalCopySlice(dst, src, static_cast<uint64_t>(hiddenBytes_));
        }
        cur += batchLen;
    }

    SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
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

__aicore__ inline void EngramFetchTrainArch35::CountAndScatterPhase()
{
    LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();
    LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>();
    LocalTensor<int32_t> divisor = divisorBuf_.Get<int32_t>();
    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

    Duplicate<int32_t>(divisor, numEntriesPerRank_, compareCntMax_);

    for (uint32_t ownerRank = aivId_; ownerRank < numRanks_; ownerRank += totalBlocks_) {
        uint32_t myCount = 0;

        uint32_t indicesBatchStart = 0;
        while (indicesBatchStart < numTokens_) {
            uint32_t indicesBatchLen = indicesBatchSize_;
            if (indicesBatchStart + indicesBatchLen > numTokens_) {
                indicesBatchLen = numTokens_ - indicesBatchStart;
            }
            CopyIndicesToUb(indicesBatchStart, indicesBatchLen);
            EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

            uint32_t batchCompareCnt =
                Ceil(indicesBatchLen * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);

            ArithProgression<int32_t>(positions, 0, 1, indicesBatchLen);
            PipeBarrier<PIPE_V>();
            Div<int32_t>(rankIDs, indicesLocal, divisor, indicesBatchLen);
            PipeBarrier<PIPE_V>();
            EngramFetchTrainSyncFunc<HardEvent::V_S>();

            CompareScalar(mask, rankIDs, static_cast<int32_t>(ownerRank), AscendC::CMPMODE::EQ, batchCompareCnt);
            PipeBarrier<PIPE_V>();

            uint64_t rsvdCnt = 0;
            GatherMask(positions, positions, mask.ReinterpretCast<uint32_t>(), true, indicesBatchLen, {1, 1, 0, 0},
                       rsvdCnt);
            PipeBarrier<PIPE_V>();
            EngramFetchTrainSyncFunc<HardEvent::V_S>();

            myCount += static_cast<uint32_t>(rsvdCnt);
            indicesBatchStart += indicesBatchLen;
        }

        LocalTensor<int32_t> countLocal = rankIDsBuf_.Get<int32_t>();
        countLocal.SetValue(0, static_cast<int32_t>(myCount));
        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
        GM_ADDR sendCountSlot = sendCountsOutGM_ + ownerRank * UB_ALIGN;
        GlobalTensor<int32_t> sendCountsSlotGM;
        sendCountsSlotGM.SetGlobalBuffer((__gm__ int32_t *)sendCountSlot);
        DataCopyParams countParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
        DataCopyPad(sendCountsSlotGM, countLocal, countParams);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
    }
    SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainArch35::ComputeSdispls()
{
    if (aivId_ == 0) {
        LocalTensor<int32_t> countsLocal = rankIDsBuf_.Get<int32_t>();

        GlobalTensor<int32_t> sendCountsGM;
        sendCountsGM.SetGlobalBuffer((__gm__ int32_t *)sendCountsOutGM_);
        DataCopyExtParams cpParams{1U, static_cast<uint32_t>(numRanks_ * UB_ALIGN), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        DataCopyPad(countsLocal, sendCountsGM, cpParams, cpPad);
        EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

        LocalTensor<int64_t> sdisplsUb = tempBuf_.Get<int64_t>();

        int64_t sdisplsAccum = 0;
        for (uint32_t r = 0; r < numRanks_; r++) {
            int32_t sCount = countsLocal.GetValue(r * UB_ALIGN / sizeof(int32_t));
            sdisplsUb.SetValue(r, sdisplsAccum);
            sdisplsAccum += static_cast<int64_t>(sCount);
        }

        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int64_t> sdisplsOutGM;
        sdisplsOutGM.SetGlobalBuffer((__gm__ int64_t *)sdisplsGM_);
        DataCopyParams sdisplParams = {1U, static_cast<uint16_t>(numRanks_ * sizeof(int64_t)), 0U, 0U};
        DataCopyPad(sdisplsOutGM, sdisplsUb, sdisplParams);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
    }
    SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainArch35::GatherAndSortPhase()
{
    LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();
    LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>();
    LocalTensor<int32_t> divisor = divisorBuf_.Get<int32_t>();
    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

    Duplicate<int32_t>(divisor, numEntriesPerRank_, compareCntMax_);

    for (uint32_t ownerRank = aivId_; ownerRank < numRanks_; ownerRank += totalBlocks_) {
        GM_ADDR sdisplSlot = sdisplsGM_ + ownerRank * sizeof(int64_t);
        GlobalTensor<int64_t> sdisplSlotGM;
        sdisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sdisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sdisplSlotGM);
        int64_t sdispl = sdisplSlotGM.GetValue(0);

        uint32_t prevAccum = 0;
        uint32_t indicesBatchStart = 0;
        while (indicesBatchStart < numTokens_) {
            uint32_t indicesBatchLen = indicesBatchSize_;
            if (indicesBatchStart + indicesBatchLen > numTokens_) {
                indicesBatchLen = numTokens_ - indicesBatchStart;
            }
            CopyIndicesToUb(indicesBatchStart, indicesBatchLen);
            EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

            uint32_t batchCompareCnt =
                Ceil(indicesBatchLen * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);

            ArithProgression<int32_t>(positions, 0, 1, indicesBatchLen);
            PipeBarrier<PIPE_V>();
            Div<int32_t>(rankIDs, indicesLocal, divisor, indicesBatchLen);
            PipeBarrier<PIPE_V>();
            EngramFetchTrainSyncFunc<HardEvent::V_S>();

            CompareScalar(mask, rankIDs, static_cast<int32_t>(ownerRank), AscendC::CMPMODE::EQ, batchCompareCnt);
            PipeBarrier<PIPE_V>();

            uint64_t rsvdCnt = 0;
            GatherMask(positions, positions, mask.ReinterpretCast<uint32_t>(), true, indicesBatchLen, {1, 1, 0, 0},
                       rsvdCnt);
            PipeBarrier<PIPE_V>();
            EngramFetchTrainSyncFunc<HardEvent::V_S>();

            uint32_t batchCount = static_cast<uint32_t>(rsvdCnt);
            if (batchCount > 0) {
                uint32_t gmWriteOffset = static_cast<uint32_t>(sdispl) + prevAccum;

                LocalTensor<uint32_t> tokenIdxLocal = rankIDsBuf_.Get<uint32_t>();
                LocalTensor<int32_t> sortedLocal = positionsBuf_.Get<int32_t>();
                for (uint32_t i = 0; i < batchCount; i++) {
                    uint32_t origPos = positions.GetValue(i);
                    tokenIdxLocal.SetValue(i, origPos + indicesBatchStart);
                    sortedLocal.SetValue(i, indicesLocal.GetValue(origPos));
                }

                EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
                GlobalTensor<uint32_t> tokenIdxInRankGM;
                tokenIdxInRankGM.SetGlobalBuffer((__gm__ uint32_t *)permOutGM_);
                DataCopyParams tokenIdxGmParams = {1U, static_cast<uint16_t>(batchCount * sizeof(uint32_t)), 0U, 0U};
                DataCopyPad(tokenIdxInRankGM[gmWriteOffset], tokenIdxLocal, tokenIdxGmParams);

                GlobalTensor<int32_t> sortedIndicesGM;
                sortedIndicesGM.SetGlobalBuffer((__gm__ int32_t *)sortedIndicesGM_);
                DataCopyParams sortedGmParams = {1U, static_cast<uint16_t>(batchCount * sizeof(int32_t)), 0U, 0U};
                DataCopyPad(sortedIndicesGM[gmWriteOffset], sortedLocal, sortedGmParams);
                EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

                prevAccum += batchCount;
            }

            indicesBatchStart += indicesBatchLen;
        }
    }
    SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainArch35::Process()
{
    if ASCEND_IS_AIV {
        if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
            return;
        }

        InitFlagsAndCounters();

        CountAndScatterPhase();

        ComputeSdispls();

        GatherAndSortPhase();

        SendCountPhase();
        ExchangeIndices();
        LocalReadTable();
        ExchangeToken();
        ReorderAndSaveCtx();
    }
}

#endif

} // namespace Mc2Kernel

#endif
