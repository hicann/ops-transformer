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

#include "../../../engram_fetch_grad/op_kernel/arch35/sortlib/sort_lib.h"

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

constexpr uint32_t ENGRAM_TIMEOUT_US = 60U * 1000U * 1000U;
constexpr uint32_t ENGRAM_CYCLES_PER_US = 1000U;
constexpr uint32_t ENGRAM_LOCAL_PIPE_DEPTH = 16U;
constexpr int STATUS_FLAGS_TIME_CHECK = 1;
constexpr int SEND_INDICES_REMOTE_TIME_CHECK = 2;
constexpr int RECV_INDICES_FROM_PEER_TIME_CHECK = 3;
constexpr int RECV_TOKEN_CHUNK_TIME_CHECK = 4;
constexpr int WAIT_INDICES_READY_FLAG_TIME_CHECK = 5;
constexpr int LOCAL_READ_TABLE_SEND_REMOTE_TIME_CHECK = 6;
constexpr int RETIRE_CREDIT_COUNTER_TIME_CHECK = 7;
constexpr AscendC::UrmaWqeEntry URMA_NO_CQE_FENCE_CFG = {
    .odr = 6,
    .fence = 1,
    .se = 0,
    .cqe = 0,
    .inlineEn = 0,
};

constexpr int32_t CREDIT_READ_SENTINEL = -1;

class EngramFetchTrainArch35 {
public:
    __aicore__ inline EngramFetchTrainArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR permOut,
                                GM_ADDR sendCountsOut, GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
                                GM_ADDR numRecvOut, GM_ADDR workspaceGM, GM_ADDR localStorageAddr, TPipe *pipe,
                                const EngramFetchTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void TimeoutCheck(uint64_t startTime, int tag);
    __aicore__ inline void InitMembers(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR permOut,
                                       GM_ADDR sendCountsOut, GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
                                       GM_ADDR numRecvOut, GM_ADDR workspaceGM, GM_ADDR localStorageAddr, TPipe *pipe,
                                       const EngramFetchTilingData *tilingData);
    __aicore__ inline void InitWinLayout();
    __aicore__ inline void InitCoreRoles();
    __aicore__ inline void InitHcommAndPipe();
    __aicore__ inline void InitWorkspaceLayout(const EngramFetchTilingData *tilingData);
    __aicore__ inline void InitUbBuffers();
    __aicore__ inline void InitFlagsAndCounters();

    __aicore__ inline void CountAndSortPhase();
    __aicore__ inline void SortInputIndices();
    __aicore__ inline void CountSendCountsFromSorted();
    __aicore__ inline void WriteSendCount(uint32_t ownerRank, int32_t myCount);
    __aicore__ inline void ComputeSdisplsLocal();

    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void LocalReadTablePerToken(int64_t batchLen);
    __aicore__ inline void LocalReadTablePerTokenSlow(int64_t batchLen);
    __aicore__ inline void LocalReadTablePerTokenPipe(int64_t batchLen, uint32_t hiddenBytesU32);
    __aicore__ inline bool GetLocalRowAddr(const LocalTensor<int32_t> &indicesUb,
                                           const LocalTensor<uint32_t> &tokenIdxUb, int64_t j, GM_ADDR &src,
                                           GM_ADDR &dst);
    __aicore__ inline uint32_t CalcTokensPerTile() const;
    __aicore__ inline void SendCountPhase();
    __aicore__ inline void SendCountToPeers();
    __aicore__ inline void GatherRecvCounts();
    __aicore__ inline void ExchangeIndices();
    __aicore__ inline void SendIndicesToPeers();
    __aicore__ inline void SendIndicesLocal(int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase);
    __aicore__ inline void SendIndicesRemote(uint32_t dstRank, int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase);
    __aicore__ inline void RecvIndicesFromPeers();
    __aicore__ inline void RecvIndicesFromPeer(GM_ADDR localWinBase, uint32_t srcRank, int32_t recvCount,
                                               int64_t rdispl);
    __aicore__ inline void LocalReadTableAndSend();
    __aicore__ inline void LocalReadTableAndSendLocal(uint32_t subIdx, int32_t sendCount, uint32_t indicesBufCap);
    __aicore__ inline void LocalReadTableAndSendRemote(uint32_t subIdx, uint32_t dstRank, int32_t sendCount,
                                                       int64_t rdispl, uint32_t indicesBufCap);
    __aicore__ inline GM_ADDR GetTableRowSrc(int32_t globalIdx, uint32_t numEntriesU32);
    __aicore__ inline void GatherRowsToStaging(const LocalTensor<int32_t> &indicesUb, int64_t stagingRowBase,
                                               uint32_t chunkLen);
    __aicore__ inline void ExchangeTokenWithLocalRead();
    __aicore__ inline void RecvTokensFromPeers();
    __aicore__ inline uint32_t RecvTokenChunk(GM_ADDR localWinBase, uint32_t srcRank, uint32_t subIdx, uint32_t myCount,
                                              int64_t sdispl, int64_t myStart, uint32_t totalReceived);
    __aicore__ inline void WriteNumRecv();
    __aicore__ inline void ScatterSlotToOutput(GM_ADDR srcBase, uint32_t gatherBase, uint32_t rowCount);
    __aicore__ inline void ScatterRowsPerToken(GM_ADDR srcBase, uint32_t batchLen,
                                               const LocalTensor<uint32_t> &tokenIdxUb);
    __aicore__ inline void ScatterRowsBatched(GM_ADDR srcBase, uint32_t batchLen, uint32_t tokensPerTile,
                                              const LocalTensor<uint32_t> &tokenIdxUb);
    __aicore__ inline void EnsureCachedTotalRecv();
    __aicore__ inline uint32_t CalcMyShareCount(uint32_t subIdx, uint32_t totalU32, uint32_t &myStart);
    __aicore__ inline void WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount);
    __aicore__ inline void ClearStatusFlags(GM_ADDR statusWinBase);
    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset);
    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank);
    __aicore__ inline int32_t ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset, uint32_t srcRank,
                                               uint32_t subIdx = 0U, bool useGroupSize = true);
    __aicore__ inline void WriteLocalCounter(GM_ADDR winBase, uint64_t counterOffset, uint32_t peerRank,
                                             uint32_t subIdx, bool useGroupSize, int32_t value);
    __aicore__ inline void PrefetchCreditCounter(uint32_t dstRank, uint64_t counterOffset, uint32_t subIdx,
                                                 bool useGroupSize);
    __aicore__ inline int32_t CompleteCreditCounter(uint64_t startTime, int tag);
    __aicore__ inline void RetireCreditCounter();
    __aicore__ inline void LoadSendCountsToUb();
    __aicore__ inline void LoadRecvCountsToUb();
    __aicore__ inline void LoadInt64ArrayToUb(GM_ADDR gmAddr);

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
    uint64_t winSize_{0};
    uint64_t localStorageAddr_{0};

    uint64_t countFlagOffset_{0};
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
    uint32_t groupSize_{1};
    uint32_t totalSlots_{NUM_SLOTS};
    bool isSender_{false};
    bool isReceiver_{false};
    bool isFlagCore_{false};

    uint64_t ubSize_{0};

    GM_ADDR sdisplsGM_{0};
    GM_ADDR rdisplsGM_{0};
    GM_ADDR sortedIndicesGM_{0};
    GM_ADDR creditScratchGM_{0};
    GM_ADDR partialCountsGM_{0};
    GM_ADDR indicesReadyFlagGM_{0};
    GM_ADDR tokenStagingGM_{0};
    int64_t stagingRows_{0};
    GM_ADDR sortWorkspaceGm_{0};
    uint32_t sortNumTileData_{0U};
    uint32_t sortTmpUbSize_{0U};

    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, RELAY_BUFFER_NUM> relayQue_;
    int32_t mte2SEvt_{0};
    int32_t mte2SCntEvt_{0};
    TBuf<> indicesBuf_;
    TBuf<> hcommBuf_;
    TBuf<> rankCountsBuf_;
    TBuf<> tokenIdxInRankBuf_;
    TBuf<> rankIDsBuf_;
    TBuf<> positionsBuf_;
    TBuf<> divisorBuf_;
    TBuf<> gatherTmpBuf_;
    TBuf<> maskBuf_;
    TBuf<> statusBuf_;
    TBuf<> tempBuf_;
    TBuf<> partialCountsBuf_;
    TBuf<> creditBuf_;

    int64_t cachedTotalRecv_{-1};

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    bool creditReadInFlight_{false};
    uint32_t indicesBatchSize_{0};
    uint32_t compareCntMax_{0};
    uint32_t tileBytes_{TILE_BYTES};
};

__aicore__ inline void EngramFetchTrainArch35::TimeoutCheck(uint64_t startTime, int tag)
{
    uint64_t nowUs = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
    ascendc_assert((nowUs - startTime) < ENGRAM_TIMEOUT_US, "timeout, tag=%d, rankId=%u, aivId=%u, elapsed=%llu us\n",
                   tag, rankId_, aivId_, nowUs - startTime);
}

__aicore__ inline GM_ADDR EngramFetchTrainArch35::GetRemoteWinAddr(uint32_t dstRank, uint64_t offset)
{
    return (GM_ADDR)ctxPtr_->commBuffer[dstRank] + offset;
}

__aicore__ inline uint64_t EngramFetchTrainArch35::GetCommHandle(uint32_t dstRank)
{
    uint32_t channelIdx = aivId_ / numRanks_;
    if (channelIdx >= groupSize_) {
        channelIdx = 0U;
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
    InitUbBuffers();
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
    ubSize_ = tilingData->ubSize;
    winSize_ = tilingData->commBufferSize;
    sortNumTileData_ = tilingData->sortNumTileData;
    sortTmpUbSize_ = tilingData->sortTmpUbSize;

    AscendC::GlobalTensor<int64_t> localStorageTensor;
    localStorageTensor.SetGlobalBuffer((__gm__ int64_t *)localStorageAddr);
    localStorageAddr_ = static_cast<uint64_t>(localStorageTensor.GetValue(0));

    numSendCores_ = totalBlocks_ / 2U;
    if (numSendCores_ == 0U) {
        numSendCores_ = 1U;
    }
    if (numSendCores_ >= numRanks_) {
        numSendCores_ = (numSendCores_ / numRanks_) * numRanks_;
    }
    groupSize_ = numSendCores_ / numRanks_;
    if (groupSize_ == 0U) {
        groupSize_ = 1U;
    }
    if (groupSize_ > channelsPerRank_) {
        groupSize_ = channelsPerRank_;
    }
    totalSlots_ = groupSize_;
}

__aicore__ inline void EngramFetchTrainArch35::InitWinLayout()
{
    uint64_t indicesCounterSize = static_cast<uint64_t>(numRanks_) * STATE_OFFSET;
    uint64_t tokenCounterSize = static_cast<uint64_t>(numRanks_) * STATE_OFFSET * groupSize_;
    uint64_t fixedSize = static_cast<uint64_t>(WIN_REGION_COUNT - 2) * indicesCounterSize +
                         static_cast<uint64_t>(WIN_REGION_COUNT - 4) * tokenCounterSize;
    uint64_t remaining = (winSize_ > fixedSize) ? (winSize_ - fixedSize) : 0U;

    uint64_t offset = 0;
    countFlagOffset_ = offset;
    offset += indicesCounterSize;
    indicesWriteOffset_ = offset;
    offset += indicesCounterSize;
    indicesReadOffset_ = offset;
    offset += indicesCounterSize;
    tokenWriteOffset_ = offset;
    offset += tokenCounterSize;
    tokenReadOffset_ = offset;
    offset += tokenCounterSize;
    sendCountOffset_ = offset;
    offset += indicesCounterSize;

    uint64_t indicesAreaRaw = remaining / INDICES_RATIO;
    uint64_t indicesGranularity = static_cast<uint64_t>(numRanks_) * NUM_SLOTS * UB_ALIGN;
    uint64_t indicesArea = Ceil(indicesAreaRaw, indicesGranularity) * indicesGranularity;
    uint64_t tokenAreaRaw = (remaining > indicesArea) ? (remaining - indicesArea) : 0U;
    uint64_t tokenGranularity = static_cast<uint64_t>(numRanks_) * totalSlots_ * static_cast<uint64_t>(hiddenBytes_);
    uint64_t tokenArea =
        (tokenGranularity > 0U) ? Ceil(tokenAreaRaw, tokenGranularity) * tokenGranularity : tokenAreaRaw;
    if (tokenArea > tokenAreaRaw) {
        tokenArea = (tokenAreaRaw / tokenGranularity) * tokenGranularity;
    }

    indicesDataOffset_ = offset;
    tokenDataOffset_ = indicesDataOffset_ + indicesArea;

    indicesSlotSize_ = indicesArea / numRanks_ / NUM_SLOTS;
    tokenSlotSize_ = tokenArea / numRanks_ / totalSlots_;
    maxIndicesPerSlot_ = static_cast<uint32_t>(indicesSlotSize_ / sizeof(int32_t));
    maxTokensPerSlot_ = static_cast<uint64_t>(tokenSlotSize_) >= static_cast<uint64_t>(hiddenBytes_) ?
                            static_cast<uint32_t>(tokenSlotSize_ / static_cast<uint64_t>(hiddenBytes_)) :
                            0U;
}

__aicore__ inline void EngramFetchTrainArch35::InitCoreRoles()
{
    ascendc_assert(maxIndicesPerSlot_ != 0U && maxTokensPerSlot_ != 0U, "slot too small");

    if (totalBlocks_ > 1U) {
        numRecvCores_ = totalBlocks_ - numSendCores_;
    } else {
        numRecvCores_ = 1U;
    }
    if (numRecvCores_ == 0U) {
        numRecvCores_ = 1U;
    }
    isSender_ = (aivId_ < numSendCores_) || (totalBlocks_ <= 1U);
    isReceiver_ = (aivId_ >= numSendCores_) || (totalBlocks_ <= 1U);
    isFlagCore_ = (aivId_ == totalBlocks_ - 1U) && (totalBlocks_ > 1U);
}

__aicore__ inline void EngramFetchTrainArch35::InitHcommAndPipe()
{
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(relayQue_, RELAY_BUFFER_NUM, tileBytes_);
    mte2SEvt_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    mte2SCntEvt_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
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

    creditScratchGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * UB_ALIGN;
    partialCountsGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * static_cast<uint64_t>(numRanks_) * sizeof(int32_t);
    indicesReadyFlagGM_ = workspaceGM_ + wsOffset;
    wsOffset += Ceil(static_cast<uint64_t>(numRanks_) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;

    stagingRows_ = tilingData->totalRecv;
    tokenStagingGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(stagingRows_) * static_cast<uint64_t>(hiddenBytes_);

    // SortLib 排序区（六段 workspace），与 host 侧 SetWorkSpace 逐段一致；
    wsOffset = Ceil(wsOffset, UB_ALIGN) * UB_ALIGN;
    constexpr uint64_t kRadixRounds = 4U;
    uint64_t slTileCount = static_cast<uint64_t>(tilingData->sortTileCount);
    uint64_t seg0 = Ceil(256U * kRadixRounds * 4U, UB_ALIGN) * UB_ALIGN;
    uint64_t seg1 = Ceil(slTileCount * 256U * kRadixRounds * 4U, UB_ALIGN) * UB_ALIGN;
    uint64_t seg2 = Ceil(static_cast<uint64_t>(numTokens_) * 4U, UB_ALIGN) * UB_ALIGN;
    uint64_t seg3 = 2U * Ceil(slTileCount * 256U * 2U, UB_ALIGN) * UB_ALIGN;
    uint64_t seg4 = Ceil(slTileCount * static_cast<uint64_t>(sortNumTileData_), UB_ALIGN) * UB_ALIGN;
    uint64_t seg5 = Ceil(static_cast<uint64_t>(numTokens_) * 4U, UB_ALIGN) * UB_ALIGN;
    sortWorkspaceGm_ = workspaceGM_ + wsOffset;
    wsOffset += seg0 + seg1 + seg2 + seg3 + seg4 + seg5;
}

__aicore__ inline void EngramFetchTrainArch35::InitUbBuffers()
{
    uint32_t countsBufSize = Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankCountsBuf_, countsBufSize);
    uint32_t statusBufSize = Ceil(numRanks_ * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    uint32_t tempBufSize = Ceil(numRanks_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tempBuf_, tempBufSize);
    uint32_t partialCountsSize = Ceil(totalBlocks_ * numRanks_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(partialCountsBuf_, partialCountsSize);
    tpipe_->InitBuffer(creditBuf_, UB_ALIGN);

    uint32_t bytesPerIndice = sizeof(int32_t) + sizeof(uint32_t) + sizeof(int32_t) * 3U + 1U + sizeof(int32_t);
    tileBytes_ = TILE_BYTES;
    uint64_t fixedUb =
        HCOMM_INIT_SIZE + countsBufSize + statusBufSize + tempBufSize + partialCountsSize + UB_ALIGN + UB_RESERVED_SIZE;
    uint64_t relayUb = static_cast<uint64_t>(RELAY_BUFFER_NUM) * tileBytes_;
    uint64_t availableUb = (ubSize_ > fixedUb + relayUb) ? (ubSize_ - fixedUb - relayUb) : 0U;

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
    tpipe_->InitBuffer(gatherTmpBuf_, int32BufSize);
    uint32_t maskBufSize = Ceil(Ceil(compareCntMax, BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(maskBuf_, maskBufSize);
}

__aicore__ inline void EngramFetchTrainArch35::InitFlagsAndCounters()
{
    if (aivId_ == 0) {
        LocalTensor<int32_t> zeroLocal = statusBuf_.Get<int32_t>();
        uint32_t flagCount = Ceil(numRanks_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
        Duplicate<int32_t>(zeroLocal, 0, flagCount);
        EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
        GlobalTensor<int32_t> indicesFlagInit;
        indicesFlagInit.SetGlobalBuffer((__gm__ int32_t *)indicesReadyFlagGM_);
        DataCopy(indicesFlagInit, zeroLocal, flagCount);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

        uint32_t counterInts = numRanks_ * STATE_OFFSET / sizeof(int32_t);
        Duplicate<int32_t>(zeroLocal, 0, counterInts);
        EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
        GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
        GlobalTensor<int32_t> counterGM;
        counterGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + indicesWriteOffset_));
        DataCopy(counterGM, zeroLocal, counterInts);
        counterGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + indicesReadOffset_));
        DataCopy(counterGM, zeroLocal, counterInts);
        for (uint32_t g = 0; g < groupSize_; g++) {
            uint64_t off = static_cast<uint64_t>(g) * static_cast<uint64_t>(counterInts) * sizeof(int32_t);
            counterGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenWriteOffset_ + off));
            DataCopy(counterGM, zeroLocal, counterInts);
            counterGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenReadOffset_ + off));
            DataCopy(counterGM, zeroLocal, counterInts);
        }
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
    }
}

__aicore__ inline void EngramFetchTrainArch35::WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount)
{
    int32_t compareFlag = static_cast<int32_t>(expectCount);
    uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
    LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
    uint32_t flagCount = numRanks_ * STATE_OFFSET / sizeof(int32_t);

    GlobalTensor<int32_t> flagGM;
    flagGM.SetGlobalBuffer((__gm__ int32_t *)statusWinBase);

    int32_t sumOfFlag = 0;
    while (sumOfFlag != compareFlag) {
        DataCopy(flagLocal, flagGM, flagCount);
        EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

        sumOfFlag = 0;
        for (uint32_t i = 0; i < numRanks_; i++) {
            sumOfFlag += flagLocal.GetValue(i * STATE_OFFSET / sizeof(int32_t));
        }
        TimeoutCheck(startTime, STATUS_FLAGS_TIME_CHECK);
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
    GatherRecvCounts();
    SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainArch35::SendCountToPeers()
{
    if (!isSender_) {
        return;
    }
    LoadSendCountsToUb();
    LocalTensor<int32_t> sendCountsUb = statusBuf_.Get<int32_t>();
    for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
        int32_t countVal = sendCountsUb.GetValue(dstRank * UB_ALIGN / sizeof(int32_t));

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

            LocalTensor<int32_t> flagLocal = rankIDsBuf_.Get<int32_t>();
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

    LocalTensor<int32_t> countAllLocal = statusBuf_.Get<int32_t>();
    LocalTensor<int32_t> recvCountsUb = rankCountsBuf_.Get<int32_t>();
    LocalTensor<int64_t> rdisplsUb = tempBuf_.Get<int64_t>();

    GM_ADDR countBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_] + sendCountOffset_;
    GlobalTensor<int32_t> countAllGM;
    countAllGM.SetGlobalBuffer((__gm__ int32_t *)countBase);
    DataCopy(countAllLocal, countAllGM, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

    int64_t rSum = 0;
    for (uint32_t r = 0; r < numRanks_; r++) {
        int32_t recvCount = countAllLocal.GetValue(r * STATE_OFFSET / sizeof(int32_t));
        recvCountsUb.SetValue(r, recvCount);
        rdisplsUb.SetValue(r, rSum);
        rSum += static_cast<int64_t>(recvCount);
    }
    cachedTotalRecv_ = rSum;

    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> recvCountsOutGM;
    recvCountsOutGM.SetGlobalBuffer((__gm__ int32_t *)recvCountsOutGM_);
    DataCopyParams recvParams = {1U, static_cast<uint16_t>(numRanks_ * sizeof(int32_t)), 0U, 0U};
    DataCopyPad(recvCountsOutGM, recvCountsUb, recvParams);

    GlobalTensor<int64_t> rdisplsOutGM;
    rdisplsOutGM.SetGlobalBuffer((__gm__ int64_t *)rdisplsGM_);
    DataCopyParams rdisplParams = {1U, static_cast<uint16_t>(numRanks_ * sizeof(int64_t)), 0U, 0U};
    DataCopyPad(rdisplsOutGM, rdisplsUb, rdisplParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

    ClearStatusFlags(localFlagBase);
}

__aicore__ inline int32_t EngramFetchTrainArch35::ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset,
                                                                   uint32_t srcRank, uint32_t subIdx, bool useGroupSize)
{
    uint64_t rankStride = useGroupSize ? (STATE_OFFSET * groupSize_) : STATE_OFFSET;
    GM_ADDR counterAddr = winBase + counterOffset + srcRank * rankStride + subIdx * STATE_OFFSET;
    GlobalTensor<int32_t> counterGM;
    counterGM.SetGlobalBuffer((__gm__ int32_t *)counterAddr);
    LocalTensor<int32_t> counterLocal = statusBuf_.Get<int32_t>();
    DataCopy(counterLocal, counterGM, UB_ALIGN / sizeof(int32_t));
    AscendC::SetFlag<HardEvent::MTE2_S>(mte2SCntEvt_);
    AscendC::WaitFlag<HardEvent::MTE2_S>(mte2SCntEvt_);
    return counterLocal.GetValue(0);
}

__aicore__ inline void EngramFetchTrainArch35::WriteLocalCounter(GM_ADDR winBase, uint64_t counterOffset,
                                                                 uint32_t peerRank, uint32_t subIdx, bool useGroupSize,
                                                                 int32_t value)
{
    LocalTensor<int32_t> valLocal = creditBuf_.Get<int32_t>();
    valLocal.SetValue(0, value);
    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();

    uint64_t rankStride = useGroupSize ? (STATE_OFFSET * groupSize_) : STATE_OFFSET;
    GM_ADDR counterAddr = winBase + counterOffset + peerRank * rankStride + subIdx * STATE_OFFSET;
    GlobalTensor<int32_t> counterGM;
    counterGM.SetGlobalBuffer((__gm__ int32_t *)counterAddr);
    DataCopyParams valParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
    DataCopyPad(counterGM, valLocal, valParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::PrefetchCreditCounter(uint32_t dstRank, uint64_t counterOffset,
                                                                     uint32_t subIdx, bool useGroupSize)
{
    if (creditReadInFlight_) {
        return;
    }

    GM_ADDR scratchAddr = creditScratchGM_ + static_cast<uint64_t>(aivId_) * UB_ALIGN;
    LocalTensor<int32_t> sentinelLocal = creditBuf_.Get<int32_t>();
    sentinelLocal.SetValue(0, CREDIT_READ_SENTINEL);
    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> scratchGM;
    scratchGM.SetGlobalBuffer((__gm__ int32_t *)scratchAddr);
    DataCopyParams sentinelParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
    DataCopyPad(scratchGM, sentinelLocal, sentinelParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

    uint64_t rankStride = useGroupSize ? (STATE_OFFSET * groupSize_) : STATE_OFFSET;
    GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, counterOffset) + rankId_ * rankStride + subIdx * STATE_OFFSET;
    uint64_t handle = GetCommHandle(dstRank);
    int32_t ret = hcomm_.ReadNbi(handle, scratchAddr, remoteCounterAddr, sizeof(int32_t));
    ascendc_assert(ret == 0, "CreditRead launch failed, ret=%d, rankId=%u, dstRank=%u", ret, rankId_, dstRank);
    creditReadInFlight_ = true;
}

__aicore__ inline int32_t EngramFetchTrainArch35::CompleteCreditCounter(uint64_t startTime, int tag)
{
    GM_ADDR scratchAddr = creditScratchGM_ + static_cast<uint64_t>(aivId_) * UB_ALIGN;
    GlobalTensor<int32_t> scratchGM;
    scratchGM.SetGlobalBuffer((__gm__ int32_t *)scratchAddr);
    LocalTensor<int32_t> creditLocal = creditBuf_.Get<int32_t>();
    int32_t value = CREDIT_READ_SENTINEL;
    while (value == CREDIT_READ_SENTINEL) {
        DataCopyExtParams cpParams{1U, static_cast<uint32_t>(UB_ALIGN), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        DataCopyPad(creditLocal, scratchGM, cpParams, cpPad);
        AscendC::SetFlag<HardEvent::MTE2_S>(mte2SCntEvt_);
        AscendC::WaitFlag<HardEvent::MTE2_S>(mte2SCntEvt_);
        value = creditLocal.GetValue(0);
        TimeoutCheck(startTime, tag);
    }
    creditReadInFlight_ = false;
    return value;
}

__aicore__ inline void EngramFetchTrainArch35::RetireCreditCounter()
{
    if (!creditReadInFlight_) {
        return;
    }
    uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
    (void)CompleteCreditCounter(startTime, RETIRE_CREDIT_COUNTER_TIME_CHECK);
}

__aicore__ inline void EngramFetchTrainArch35::LoadSendCountsToUb()
{
    LocalTensor<int32_t> ub = statusBuf_.Get<int32_t>();
    GlobalTensor<int32_t> gm;
    gm.SetGlobalBuffer((__gm__ int32_t *)sendCountsOutGM_);
    DataCopy(ub, gm, numRanks_ * UB_ALIGN / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();
}

__aicore__ inline void EngramFetchTrainArch35::LoadRecvCountsToUb()
{
    LocalTensor<int32_t> ub = rankCountsBuf_.Get<int32_t>();
    GlobalTensor<int32_t> gm;
    gm.SetGlobalBuffer((__gm__ int32_t *)recvCountsOutGM_);
    DataCopyExtParams cpParams{1U, numRanks_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    DataCopyPad(ub, gm, cpParams, cpPad);
    EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();
}

__aicore__ inline void EngramFetchTrainArch35::LoadInt64ArrayToUb(GM_ADDR gmAddr)
{
    LocalTensor<int64_t> ub = tempBuf_.Get<int64_t>();
    GlobalTensor<int64_t> gm;
    gm.SetGlobalBuffer((__gm__ int64_t *)gmAddr);
    DataCopyExtParams cpParams{1U, numRanks_ * static_cast<uint32_t>(sizeof(int64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int64_t> cpPad{false, 0, 0, 0};
    DataCopyPad(ub, gm, cpParams, cpPad);
    EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();
}

__aicore__ inline void EngramFetchTrainArch35::ExchangeIndices()
{
    SendIndicesToPeers();
    RecvIndicesFromPeers();
}

__aicore__ inline void EngramFetchTrainArch35::SendIndicesToPeers()
{
    if (!isSender_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    LoadSendCountsToUb();
    LocalTensor<int32_t> sendCountsUb = statusBuf_.Get<int32_t>();
    LoadInt64ArrayToUb(sdisplsGM_);
    LocalTensor<int64_t> sdisplsUb = tempBuf_.Get<int64_t>();

    for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
        int32_t sendCount = sendCountsUb.GetValue(dstRank * UB_ALIGN / sizeof(int32_t));
        int64_t sdispl = sdisplsUb.GetValue(dstRank);

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
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

        LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
        flagLocal.SetValue(0, 1);
        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
        GM_ADDR flagAddr = indicesReadyFlagGM_ + static_cast<uint64_t>(rankId_) * sizeof(int32_t);
        GlobalTensor<int32_t> flagGM;
        flagGM.SetGlobalBuffer((__gm__ int32_t *)flagAddr);
        DataCopyParams flagParams = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
        DataCopyPad(flagGM, flagLocal, flagParams);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
    }
}

__aicore__ inline void EngramFetchTrainArch35::SendIndicesRemote(uint32_t dstRank, int32_t sendCount, int64_t sdispl,
                                                                 GM_ADDR localWinBase)
{
    uint64_t handle = GetCommHandle(dstRank);
    uint32_t totalSent = 0;
    uint32_t localWriteCnt = 0;
    int32_t remoteReadCnt = 0;

    while (totalSent < static_cast<uint32_t>(sendCount)) {
        if (totalBlocks_ > 1U) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
            if (!creditReadInFlight_ && localWriteCnt + 1U >= static_cast<uint32_t>(remoteReadCnt) + NUM_SLOTS) {
                PrefetchCreditCounter(dstRank, indicesReadOffset_, 0U, false);
            }
            while (localWriteCnt >= static_cast<uint32_t>(remoteReadCnt) &&
                   localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= NUM_SLOTS) {
                PrefetchCreditCounter(dstRank, indicesReadOffset_, 0U, false);
                remoteReadCnt = CompleteCreditCounter(startTime, SEND_INDICES_REMOTE_TIME_CHECK);
                TimeoutCheck(startTime, SEND_INDICES_REMOTE_TIME_CHECK);
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
        int32_t ret = hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, URMA_NO_CQE_FENCE_CFG>(
            handle, remoteSlotAddr, srcAddr, chunkLen * sizeof(int32_t), remoteCounterAddr,
            static_cast<uint64_t>(localWriteCnt + 1));
        ascendc_assert(ret == 0, "WriteWithNotifyNbi failed, ret=%d, tag=ExIdx_data, rankId=%u, dstRank=%u", ret,
                       rankId_, dstRank);

        localWriteCnt++;
        totalSent += chunkLen;
    }
    RetireCreditCounter();
}

__aicore__ inline void EngramFetchTrainArch35::RecvIndicesFromPeer(GM_ADDR localWinBase, uint32_t srcRank,
                                                                   int32_t recvCount, int64_t rdispl)
{
    uint32_t totalReceived = 0;
    uint32_t localReadCnt = 0;
    while (totalReceived < static_cast<uint32_t>(recvCount)) {
        uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
        int32_t remoteWriteCnt = ReadLocalCounter(localWinBase, indicesWriteOffset_, srcRank, 0U, false);
        while (remoteWriteCnt <= 0 || static_cast<uint32_t>(remoteWriteCnt) <= localReadCnt) {
            remoteWriteCnt = ReadLocalCounter(localWinBase, indicesWriteOffset_, srcRank, 0U, false);
            TimeoutCheck(startTime, RECV_INDICES_FROM_PEER_TIME_CHECK);
        }
        uint32_t availableSlots = static_cast<uint32_t>(remoteWriteCnt) - localReadCnt;
        if (availableSlots > NUM_SLOTS) {
            availableSlots = NUM_SLOTS;
        }
        uint32_t slotIdxStart = localReadCnt % NUM_SLOTS;
        uint32_t contiguousSlots = NUM_SLOTS - slotIdxStart;
        if (contiguousSlots > availableSlots) {
            contiguousSlots = availableSlots;
        }
        uint32_t remaining = static_cast<uint32_t>(recvCount) - totalReceived;
        uint32_t availableItems = contiguousSlots * maxIndicesPerSlot_;
        uint32_t readItems = (remaining < availableItems) ? remaining : availableItems;
        uint64_t slotOffset =
            indicesDataOffset_ + srcRank * NUM_SLOTS * indicesSlotSize_ + slotIdxStart * indicesSlotSize_;
        GM_ADDR localSlotAddr = localWinBase + slotOffset;
        uint64_t dataBytes = static_cast<uint64_t>(readItems) * sizeof(int32_t);
        LocalCopySlice(recvLocalEntryOutGM_ + static_cast<uint64_t>(rdispl + totalReceived) * sizeof(int32_t),
                       localSlotAddr, dataBytes);
        uint32_t readSlots = (readItems + maxIndicesPerSlot_ - 1U) / maxIndicesPerSlot_;
        localReadCnt += readSlots;
        totalReceived += readItems;
        WriteLocalCounter(localWinBase, indicesReadOffset_, srcRank, 0U, false, static_cast<int32_t>(localReadCnt));
    }
}

__aicore__ inline void EngramFetchTrainArch35::RecvIndicesFromPeers()
{
    if (!isReceiver_) {
        return;
    }
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t recvIdx = aivId_ - numSendCores_;
    LoadRecvCountsToUb();
    LocalTensor<int32_t> recvCountsUb = rankCountsBuf_.Get<int32_t>();
    LoadInt64ArrayToUb(rdisplsGM_);
    LocalTensor<int64_t> rdisplsUb = tempBuf_.Get<int64_t>();
    for (uint32_t srcRank = recvIdx; srcRank < numRanks_; srcRank += numRecvCores_) {
        if (srcRank == rankId_)
            continue;
        int32_t recvCount = recvCountsUb.GetValue(srcRank);
        int64_t rdispl = rdisplsUb.GetValue(srcRank);
        if (recvCount <= 0) {
            continue;
        }
        RecvIndicesFromPeer(localWinBase, srcRank, recvCount, rdispl);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
        LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
        flagLocal.SetValue(0, 1);
        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
        GM_ADDR flagAddr = indicesReadyFlagGM_ + static_cast<uint64_t>(srcRank) * sizeof(int32_t);
        GlobalTensor<int32_t> flagGM;
        flagGM.SetGlobalBuffer((__gm__ int32_t *)flagAddr);
        DataCopyParams flagParams = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
        DataCopyPad(flagGM, flagLocal, flagParams);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
    }
}

__aicore__ inline void EngramFetchTrainArch35::ExchangeTokenWithLocalRead()
{
    if (isSender_) {
        LocalReadTableAndSend();
    }
    if (isReceiver_) {
        RecvTokensFromPeers();
    }
    SyncAll<true>();
}

__aicore__ inline uint32_t EngramFetchTrainArch35::RecvTokenChunk(GM_ADDR localWinBase, uint32_t srcRank,
                                                                  uint32_t subIdx, uint32_t myCount, int64_t sdispl,
                                                                  int64_t myStart, uint32_t totalReceived)
{
    uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
    uint64_t subIdxAreaBase = tokenDataOffset_ + srcRank * totalSlots_ * tokenSlotSize_ + subIdx * tokenSlotSize_;
    GM_ADDR localSubIdxBase = localWinBase + subIdxAreaBase;
    while (totalReceived < myCount) {
        int32_t remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank, subIdx);
        while (remoteWriteCnt <= 0 || static_cast<uint32_t>(remoteWriteCnt) <= totalReceived) {
            remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank, subIdx);
            TimeoutCheck(startTime, RECV_TOKEN_CHUNK_TIME_CHECK);
        }
        uint32_t available = static_cast<uint32_t>(remoteWriteCnt) - totalReceived;
        uint32_t remaining = myCount - totalReceived;
        uint32_t readItems = (available < remaining) ? available : remaining;
        uint32_t spaceToEnd = maxTokensPerSlot_ - (totalReceived % maxTokensPerSlot_);
        if (readItems > spaceToEnd) {
            readItems = spaceToEnd;
        }
        ascendc_assert(readItems != 0U, "RecvTokens readItems is 0");
        GM_ADDR localSlotAddr =
            localSubIdxBase + static_cast<uint64_t>(totalReceived % maxTokensPerSlot_) * hiddenBytes_;
        ScatterSlotToOutput(localSlotAddr, static_cast<uint32_t>(sdispl) + myStart + totalReceived, readItems);
        totalReceived += readItems;
        WriteLocalCounter(localWinBase, tokenReadOffset_, srcRank, subIdx, true, static_cast<int32_t>(totalReceived));
    }
    return totalReceived;
}

__aicore__ inline void EngramFetchTrainArch35::RecvTokensFromPeers()
{
    if (!isReceiver_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t recvIdx = aivId_ - numSendCores_;
    uint32_t subIdx = recvIdx / numRanks_;
    if (subIdx >= groupSize_) {
        return;
    }

    LoadSendCountsToUb();
    LocalTensor<int32_t> sendCountsUb = statusBuf_.Get<int32_t>();
    LoadInt64ArrayToUb(sdisplsGM_);
    LocalTensor<int64_t> sdisplsUb = tempBuf_.Get<int64_t>();

    for (uint32_t srcRank = recvIdx % numRanks_; srcRank < numRanks_; srcRank += numRecvCores_) {
        if (srcRank == rankId_) {
            continue;
        }
        int32_t recvCount = sendCountsUb.GetValue(srcRank * UB_ALIGN / sizeof(int32_t));
        int64_t sdispl = sdisplsUb.GetValue(srcRank);
        if (recvCount <= 0) {
            continue;
        }
        uint32_t myStart;
        uint32_t myCount = CalcMyShareCount(subIdx, static_cast<uint32_t>(recvCount), myStart);
        if (myCount == 0U) {
            continue;
        }
        RecvTokenChunk(localWinBase, srcRank, subIdx, myCount, sdispl, static_cast<int64_t>(myStart), 0U);
    }
}

__aicore__ inline void EngramFetchTrainArch35::ScatterRowsPerToken(GM_ADDR srcBase, uint32_t batchLen,
                                                                   const LocalTensor<uint32_t> &tokenIdxUb)
{
    for (uint32_t j = 0; j < batchLen; j++) {
        uint32_t originalIdx = tokenIdxUb.GetValue(j);
        if (originalIdx >= numTokens_) {
            continue;
        }
        GM_ADDR src = srcBase + static_cast<uint64_t>(j) * static_cast<uint64_t>(hiddenBytes_);
        GM_ADDR dst = fetchedGM_ + static_cast<uint64_t>(originalIdx) * static_cast<uint64_t>(hiddenBytes_);
        LocalCopySlice(dst, src, static_cast<uint64_t>(hiddenBytes_));
    }
}

__aicore__ inline void EngramFetchTrainArch35::ScatterRowsBatched(GM_ADDR srcBase, uint32_t batchLen,
                                                                  uint32_t tokensPerTile,
                                                                  const LocalTensor<uint32_t> &tokenIdxUb)
{
    uint32_t hiddenBytesU32 = static_cast<uint32_t>(hiddenBytes_);
    uint32_t j = 0;
    while (j < batchLen) {
        uint32_t groupSize = batchLen - j;
        if (groupSize > tokensPerTile) {
            groupSize = tokensPerTile;
        }
        LocalTensor<uint8_t> batchBuf = relayQue_.AllocTensor<uint8_t>();
        GM_ADDR src = srcBase + static_cast<uint64_t>(j) * static_cast<uint64_t>(hiddenBytes_);
        GlobalTensor<uint8_t> srcGm;
        srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
        uint32_t totalBytes = groupSize * hiddenBytesU32;
        DataCopyExtParams mte2Params{1U, totalBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> mte2Pad{false, 0, 0, 0};
        DataCopyPad(batchBuf, srcGm, mte2Params, mte2Pad);
        relayQue_.EnQue<uint8_t>(batchBuf);
        batchBuf = relayQue_.DeQue<uint8_t>();
        for (uint32_t k = 0; k < groupSize; k++) {
            uint32_t originalIdx = tokenIdxUb.GetValue(j + k);
            if (originalIdx >= numTokens_) {
                continue;
            }
            GM_ADDR dst = fetchedGM_ + static_cast<uint64_t>(originalIdx) * static_cast<uint64_t>(hiddenBytes_);
            GlobalTensor<uint8_t> dstGm;
            dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);
            DataCopyParams mte3Params{1U, static_cast<uint16_t>(hiddenBytesU32), 0U, 0U};
            DataCopyPad(dstGm, batchBuf[k * hiddenBytesU32], mte3Params);
        }
        relayQue_.FreeTensor<uint8_t>(batchBuf);
        j += groupSize;
    }
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::ScatterSlotToOutput(GM_ADDR srcBase, uint32_t gatherBase,
                                                                   uint32_t rowCount)
{
    LocalTensor<uint32_t> tokenIdxUb = tokenIdxInRankBuf_.Get<uint32_t>();
    uint32_t tokenIdxBufCap = indicesBatchSize_;
    uint32_t tokensPerTile = CalcTokensPerTile();

    GlobalTensor<uint32_t> tokenIdxBatchGM;
    tokenIdxBatchGM.SetGlobalBuffer((__gm__ uint32_t *)permOutGM_);
    uint32_t done = 0;
    while (done < rowCount) {
        uint32_t batchLen = rowCount - done;
        if (batchLen > tokenIdxBufCap) {
            batchLen = tokenIdxBufCap;
        }
        DataCopyExtParams cpParams{1U, batchLen * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<uint32_t> cpPad{false, 0, 0, 0};
        DataCopyPad(tokenIdxUb, tokenIdxBatchGM[gatherBase + done], cpParams, cpPad);
        EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

        GM_ADDR batchSrc = srcBase + static_cast<uint64_t>(done) * static_cast<uint64_t>(hiddenBytes_);
        if (tokensPerTile == 0U) {
            ScatterRowsPerToken(batchSrc, batchLen, tokenIdxUb);
        } else {
            ScatterRowsBatched(batchSrc, batchLen, tokensPerTile, tokenIdxUb);
        }
        done += batchLen;
    }
}

__aicore__ inline void EngramFetchTrainArch35::EnsureCachedTotalRecv()
{
    if (cachedTotalRecv_ >= 0) {
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
    cachedTotalRecv_ = lastRDisplSlotGM.GetValue(0) + lastRecvSlotGM.GetValue(0);
}

__aicore__ inline uint32_t EngramFetchTrainArch35::CalcMyShareCount(uint32_t subIdx, uint32_t totalU32,
                                                                    uint32_t &myStart)
{
    uint32_t myShare = (totalU32 + groupSize_ - 1U) / groupSize_;
    myStart = subIdx * myShare;
    if (myStart >= totalU32) {
        return 0U;
    }
    uint32_t myEnd = myStart + myShare;
    if (subIdx == groupSize_ - 1U || myEnd > totalU32) {
        myEnd = totalU32;
    }
    return myEnd - myStart;
}

__aicore__ inline void EngramFetchTrainArch35::WriteNumRecv()
{
    EnsureCachedTotalRecv();
    LocalTensor<int32_t> tmp32 = indicesBuf_.Get<int32_t>();
    tmp32.SetValue(0, static_cast<int32_t>(cachedTotalRecv_));
    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
    DataCopyParams gmParamsScalar = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    GlobalTensor<int32_t> numRecvOutGM;
    numRecvOutGM.SetGlobalBuffer((__gm__ int32_t *)numRecvOutGM_);
    DataCopyPad(numRecvOutGM, tmp32, gmParamsScalar);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline uint32_t EngramFetchTrainArch35::CalcTokensPerTile() const
{
    uint32_t hiddenBytesU32 = static_cast<uint32_t>(hiddenBytes_);
    if (hiddenBytesU32 == 0U) {
        return 0U;
    }

    if (hiddenBytesU32 % UB_ALIGN != 0U) {
        return 0U;
    }
    return tileBytes_ / hiddenBytesU32;
}

__aicore__ inline bool EngramFetchTrainArch35::GetLocalRowAddr(const LocalTensor<int32_t> &indicesUb,
                                                               const LocalTensor<uint32_t> &tokenIdxUb, int64_t j,
                                                               GM_ADDR &src, GM_ADDR &dst)
{
    int32_t globalIdx = indicesUb.GetValue(static_cast<uint32_t>(j));
    uint32_t localEntryIdx = static_cast<uint32_t>(
        static_cast<int64_t>(globalIdx) - static_cast<int64_t>(rankId_) * static_cast<int64_t>(numEntriesPerRank_));
    if (localEntryIdx >= static_cast<uint32_t>(numEntriesPerRank_)) {
        return false;
    }
    src = (GM_ADDR)localStorageAddr_ + static_cast<uint64_t>(localEntryIdx) * static_cast<uint64_t>(hiddenBytes_);
    dst = fetchedGM_ +
          static_cast<uint64_t>(tokenIdxUb.GetValue(static_cast<uint32_t>(j))) * static_cast<uint64_t>(hiddenBytes_);
    return true;
}

__aicore__ inline void EngramFetchTrainArch35::LocalReadTablePerToken(int64_t batchLen)
{
    AscendC::WaitFlag<HardEvent::MTE2_S>(mte2SEvt_);
    uint32_t hiddenBytesU32 = static_cast<uint32_t>(hiddenBytes_);
    if (hiddenBytesU32 == 0U || (hiddenBytesU32 % UB_ALIGN) != 0U || hiddenBytesU32 > tileBytes_) {
        LocalReadTablePerTokenSlow(batchLen);
    } else {
        LocalReadTablePerTokenPipe(batchLen, hiddenBytesU32);
    }
}

__aicore__ inline void EngramFetchTrainArch35::LocalReadTablePerTokenSlow(int64_t batchLen)
{
    LocalTensor<int32_t> indicesUb = indicesBuf_.Get<int32_t>();
    LocalTensor<uint32_t> tokenIdxUb = tokenIdxInRankBuf_.Get<uint32_t>();

    for (int64_t j = 0; j < batchLen; j++) {
        GM_ADDR src;
        GM_ADDR dst;
        if (!GetLocalRowAddr(indicesUb, tokenIdxUb, j, src, dst)) {
            continue;
        }
        LocalCopySlice(dst, src, static_cast<uint64_t>(hiddenBytes_));
    }
}

__aicore__ inline void EngramFetchTrainArch35::LocalReadTablePerTokenPipe(int64_t batchLen, uint32_t hiddenBytesU32)
{
    LocalTensor<int32_t> indicesUb = indicesBuf_.Get<int32_t>();
    LocalTensor<uint32_t> tokenIdxUb = tokenIdxInRankBuf_.Get<uint32_t>();
    LocalTensor<uint64_t> dstAddrUb = positionsBuf_.Get<uint64_t>();

    uint32_t rowsPerBuf = tileBytes_ / hiddenBytesU32;
    uint32_t depth = (rowsPerBuf < ENGRAM_LOCAL_PIPE_DEPTH) ? rowsPerBuf : ENGRAM_LOCAL_PIPE_DEPTH;

    int64_t chunkStart = 0;
    while (chunkStart < batchLen) {
        LocalTensor<uint8_t> slotBase = relayQue_.AllocTensor<uint8_t>();

        int64_t jj = chunkStart;
        uint32_t n = 0;
        while (jj < batchLen && n < depth) {
            GM_ADDR src;
            GM_ADDR dst;
            if (!GetLocalRowAddr(indicesUb, tokenIdxUb, jj, src, dst)) {
                jj++;
                continue;
            }
            dstAddrUb.SetValue(n, reinterpret_cast<uint64_t>(dst));
            GlobalTensor<uint8_t> srcGm;
            srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
            DataCopy(slotBase[n * hiddenBytesU32], srcGm, hiddenBytesU32);
            n++;
            jj++;
        }
        relayQue_.EnQue<uint8_t>(slotBase);
        slotBase = relayQue_.DeQue<uint8_t>();

        for (uint32_t m = 0; m < n; m++) {
            GlobalTensor<uint8_t> dstGm;
            dstGm.SetGlobalBuffer((__gm__ uint8_t *)dstAddrUb.GetValue(m));
            DataCopy(dstGm, slotBase[m * hiddenBytesU32], hiddenBytesU32);
        }
        relayQue_.FreeTensor<uint8_t>(slotBase);

        chunkStart = jj;
    }
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::LocalReadTableAndSendLocal(uint32_t subIdx, int32_t sendCount,
                                                                          uint32_t indicesBufCap)
{
    if (sendCount <= 0) {
        return;
    }
    uint32_t myStart;
    uint32_t myCount = CalcMyShareCount(subIdx, static_cast<uint32_t>(sendCount), myStart);
    if (myCount == 0U) {
        return;
    }
    GM_ADDR sDisplSlot = sdisplsGM_ + rankId_ * sizeof(int64_t);
    GlobalTensor<int64_t> sDisplSlotGM;
    sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
    int64_t sdispl = sDisplSlotGM.GetValue(0);

    LocalTensor<int32_t> indicesUb = indicesBuf_.Get<int32_t>();
    LocalTensor<uint32_t> tokenIdxUb = tokenIdxInRankBuf_.Get<uint32_t>();
    GlobalTensor<int32_t> sortedGM;
    sortedGM.SetGlobalBuffer((__gm__ int32_t *)sortedIndicesGM_);
    GlobalTensor<uint32_t> tokenIdxBatchGM;
    tokenIdxBatchGM.SetGlobalBuffer((__gm__ uint32_t *)permOutGM_);

    uint32_t totalSent = 0;
    while (totalSent < myCount) {
        uint32_t remaining = myCount - totalSent;
        uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;
        if (chunkLen > indicesBufCap) {
            chunkLen = indicesBufCap;
        }
        int64_t cur = sdispl + myStart + totalSent;
        DataCopyExtParams idxParams{1U, chunkLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> idxPad{false, 0, 0, 0};
        DataCopyPad(indicesUb, sortedGM[static_cast<uint32_t>(cur)], idxParams, idxPad);
        AscendC::SetFlag<HardEvent::MTE2_S>(mte2SEvt_);
        DataCopyExtParams permParams{1U, chunkLen * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<uint32_t> permPad{false, 0, 0, 0};
        DataCopyPad(tokenIdxUb, tokenIdxBatchGM[static_cast<uint32_t>(cur)], permParams, permPad);
        EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();
        LocalReadTablePerToken(static_cast<int64_t>(chunkLen));
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
        totalSent += chunkLen;
    }
}

__aicore__ inline GM_ADDR EngramFetchTrainArch35::GetTableRowSrc(int32_t globalIdx, uint32_t numEntriesU32)
{
    GM_ADDR rowSrc = reinterpret_cast<GM_ADDR>(localStorageAddr_);
    if (globalIdx >= 0) {
        uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - rankId_ * numEntriesU32;
        if (localEntryIdx < numEntriesU32) {
            rowSrc = reinterpret_cast<GM_ADDR>(localStorageAddr_) +
                     static_cast<uint64_t>(localEntryIdx) * static_cast<uint64_t>(hiddenBytes_);
        }
    }
    return rowSrc;
}

__aicore__ inline void EngramFetchTrainArch35::GatherRowsToStaging(const LocalTensor<int32_t> &indicesUb,
                                                                   int64_t stagingRowBase, uint32_t chunkLen)
{
    uint32_t hiddenBytesU32 = static_cast<uint32_t>(hiddenBytes_);
    uint32_t numEntriesU32 = static_cast<uint32_t>(numEntriesPerRank_);
    uint64_t hiddenBytesU64 = static_cast<uint64_t>(hiddenBytes_);
    GM_ADDR stagingBase = tokenStagingGM_ + static_cast<uint64_t>(stagingRowBase) * hiddenBytesU64;

    if (hiddenBytesU32 == 0U || (hiddenBytesU32 % UB_ALIGN) != 0U || hiddenBytesU32 > tileBytes_) {
        for (uint32_t k = 0U; k < chunkLen; k++) {
            GM_ADDR rowSrc = GetTableRowSrc(indicesUb.GetValue(k), numEntriesU32);
            LocalCopySlice(stagingBase + static_cast<uint64_t>(k) * hiddenBytesU64, rowSrc, hiddenBytesU64);
        }
        return;
    }

    uint32_t rowsPerTile = tileBytes_ / hiddenBytesU32;
    uint32_t k = 0U;
    while (k < chunkLen) {
        uint32_t batchLen = chunkLen - k;
        if (batchLen > rowsPerTile) {
            batchLen = rowsPerTile;
        }
        LocalTensor<uint8_t> tileBuf = relayQue_.AllocTensor<uint8_t>();
        for (uint32_t i = 0U; i < batchLen; i++) {
            GM_ADDR rowSrc = GetTableRowSrc(indicesUb.GetValue(k + i), numEntriesU32);
            GlobalTensor<uint8_t> srcGm;
            srcGm.SetGlobalBuffer((__gm__ uint8_t *)rowSrc);
            DataCopy(tileBuf[i * hiddenBytesU32], srcGm, hiddenBytesU32);
        }
        relayQue_.EnQue<uint8_t>(tileBuf);
        tileBuf = relayQue_.DeQue<uint8_t>();

        GlobalTensor<uint8_t> dstGm;
        dstGm.SetGlobalBuffer((__gm__ uint8_t *)(stagingBase + static_cast<uint64_t>(k) * hiddenBytesU64));
        DataCopy(dstGm, tileBuf, batchLen * hiddenBytesU32);
        relayQue_.FreeTensor<uint8_t>(tileBuf);
        k += batchLen;
    }
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::LocalReadTableAndSendRemote(uint32_t subIdx, uint32_t dstRank,
                                                                           int32_t sendCount, int64_t rdispl,
                                                                           uint32_t indicesBufCap)
{
    if (sendCount <= 0) {
        return;
    }
    uint32_t myStart;
    uint32_t myCount = CalcMyShareCount(subIdx, static_cast<uint32_t>(sendCount), myStart);
    if (myCount == 0U) {
        return;
    }
    uint64_t handle = GetCommHandle(dstRank);
    GM_ADDR remoteCounterAddr =
        GetRemoteWinAddr(dstRank, tokenWriteOffset_) + rankId_ * STATE_OFFSET * groupSize_ + subIdx * STATE_OFFSET;
    GM_ADDR remoteSubIdxBase =
        GetRemoteWinAddr(dstRank, tokenDataOffset_) + rankId_ * totalSlots_ * tokenSlotSize_ + subIdx * tokenSlotSize_;
    uint64_t hiddenBytesU64 = static_cast<uint64_t>(hiddenBytes_);
    LocalTensor<int32_t> indicesUb = indicesBuf_.Get<int32_t>();
    GlobalTensor<int32_t> recvGM;
    recvGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);
    DataCopyPadExtParams<int32_t> idxPad{false, 0, 0, 0};

    uint32_t totalSent = 0;
    int32_t remoteReadCnt = 0;
    while (totalSent < myCount) {
        if (totalBlocks_ > 1U) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
            if (!creditReadInFlight_ &&
                totalSent + indicesBufCap >= static_cast<uint32_t>(remoteReadCnt) + maxTokensPerSlot_) {
                PrefetchCreditCounter(dstRank, tokenReadOffset_, subIdx, true);
            }
            while (totalSent >= static_cast<uint32_t>(remoteReadCnt) + maxTokensPerSlot_) {
                PrefetchCreditCounter(dstRank, tokenReadOffset_, subIdx, true);
                remoteReadCnt = CompleteCreditCounter(startTime, LOCAL_READ_TABLE_SEND_REMOTE_TIME_CHECK);
                TimeoutCheck(startTime, LOCAL_READ_TABLE_SEND_REMOTE_TIME_CHECK);
            }
        }
        uint32_t remaining = myCount - totalSent;
        uint32_t chunkLen = (remaining > indicesBufCap) ? indicesBufCap : remaining;
        uint32_t spaceToEnd = maxTokensPerSlot_ - (totalSent % maxTokensPerSlot_);
        if (chunkLen > spaceToEnd) {
            chunkLen = spaceToEnd;
        }
        ascendc_assert(chunkLen != 0U, "LocalReadTableAndSend chunkLen is 0");
        int64_t cur = static_cast<int64_t>(rdispl + myStart + totalSent);
        DataCopyExtParams idxParams{1U, chunkLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(indicesUb, recvGM[static_cast<uint32_t>(cur)], idxParams, idxPad);
        EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

        GatherRowsToStaging(indicesUb, cur, chunkLen);

        GM_ADDR stagingSrc = tokenStagingGM_ + static_cast<uint64_t>(cur) * hiddenBytesU64;
        GM_ADDR rowDst = remoteSubIdxBase + static_cast<uint64_t>(totalSent % maxTokensPerSlot_) * hiddenBytesU64;
        uint64_t sendBytes = static_cast<uint64_t>(chunkLen) * hiddenBytesU64;
        int32_t ret = hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, URMA_NO_CQE_FENCE_CFG>(
            handle, rowDst, stagingSrc, sendBytes, remoteCounterAddr,
            static_cast<uint64_t>(totalSent) + static_cast<uint64_t>(chunkLen));
        ascendc_assert(ret == 0, "WriteWithNotifyNbi failed, ret=%d, tag=ExToken_data, rankId=%u, dstRank=%u", ret,
                       rankId_, dstRank);
        totalSent += chunkLen;
    }
    RetireCreditCounter();
}

__aicore__ inline void EngramFetchTrainArch35::LocalReadTableAndSend()
{
    if (!isSender_ || numEntriesPerRank_ == 0 || numTokens_ == 0) {
        return;
    }
    EnsureCachedTotalRecv();
    uint32_t subIdx = aivId_ / numRanks_;
    LoadRecvCountsToUb();
    LocalTensor<int32_t> recvCountsUb = rankCountsBuf_.Get<int32_t>();
    LoadInt64ArrayToUb(rdisplsGM_);
    LocalTensor<int64_t> rdisplsUb = tempBuf_.Get<int64_t>();
    if (subIdx >= groupSize_) {
        return;
    }
    uint32_t indicesBufCap = indicesBatchSize_;
    for (uint32_t dstRank = aivId_ % numRanks_; dstRank < numRanks_; dstRank += numSendCores_) {
        int32_t sendCount = recvCountsUb.GetValue(dstRank);
        int64_t rdispl = rdisplsUb.GetValue(dstRank);
        if (dstRank == rankId_) {
            LocalReadTableAndSendLocal(subIdx, sendCount, indicesBufCap);
        } else {
            LocalReadTableAndSendRemote(subIdx, dstRank, sendCount, rdispl, indicesBufCap);
        }
    }
}

__aicore__ inline void EngramFetchTrainArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    GlobalTensor<uint8_t> srcGm;
    GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);

    uint64_t off = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > tileBytes_) ? tileBytes_ : (len - off);
        LocalTensor<uint8_t> tmp = relayQue_.AllocTensor<uint8_t>();
        DataCopyExtParams mte2Params{1U, static_cast<uint32_t>(thisLen), 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> mte2Pad{false, 0, 0, 0};
        DataCopyPad(tmp, srcGm[off], mte2Params, mte2Pad);
        relayQue_.EnQue<uint8_t>(tmp);
        tmp = relayQue_.DeQue<uint8_t>();
        DataCopyParams mte3Params{1U, static_cast<uint16_t>(thisLen), 0U, 0U};
        DataCopyPad(dstGm[off], tmp, mte3Params);
        relayQue_.FreeTensor<uint8_t>(tmp);
        off += thisLen;
    }
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::SortInputIndices()
{
    tpipe_->ReleaseEventID<HardEvent::MTE2_S>(mte2SEvt_);
    tpipe_->ReleaseEventID<HardEvent::MTE2_S>(mte2SCntEvt_);
    tpipe_->Reset();

    SortLib::SortParams p;
    p.numTileData = sortNumTileData_;
    p.tileCount = (numTokens_ + p.numTileData - 1U) / p.numTileData;
    p.activeCores = (p.tileCount < totalBlocks_) ? p.tileCount : totalBlocks_;
    p.tmpUbSize = sortTmpUbSize_;
    p.totalElements = static_cast<int64_t>(numTokens_);
    p.isSingleCore = 0;
    SortLib::SortInvoke<int32_t, int32_t, uint32_t, false>(
        tpipe_, (__gm__ int32_t *)indicesGM_, (__gm__ int32_t *)sortedIndicesGM_, (__gm__ int32_t *)permOutGM_,
        (__gm__ char *)sortWorkspaceGm_, p);

    tpipe_->Reset();
    InitHcommAndPipe();
    InitUbBuffers();
}

__aicore__ inline void EngramFetchTrainArch35::CountSendCountsFromSorted()
{
    LocalTensor<int32_t> cntUb = rankCountsBuf_.Get<int32_t>();
    Duplicate<int32_t>(cntUb, 0, numRanks_);
    EngramFetchTrainSyncFunc<HardEvent::V_S>();

    uint32_t perCore = (numTokens_ + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t lo = aivId_ * perCore;
    uint32_t hi = lo + perCore;
    if (hi > numTokens_) {
        hi = numTokens_;
    }
    if (lo < hi) {
        LocalTensor<int32_t> idsUb = indicesBuf_.Get<int32_t>();
        LocalTensor<int32_t> ownerUb = rankIDsBuf_.Get<int32_t>();
        LocalTensor<int32_t> divisorUb = divisorBuf_.Get<int32_t>();
        LocalTensor<int32_t> compactUb = gatherTmpBuf_.Get<int32_t>();
        LocalTensor<uint8_t> maskUb = maskBuf_.Get<uint8_t>();
        Duplicate<int32_t>(divisorUb, numEntriesPerRank_, compareCntMax_);
        EngramFetchTrainSyncFunc<HardEvent::V_S>();

        GlobalTensor<int32_t> sortedGM;
        sortedGM.SetGlobalBuffer((__gm__ int32_t *)sortedIndicesGM_);
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        uint32_t done = 0U;
        while (done < hi - lo) {
            uint32_t batch = (hi - lo) - done;
            if (batch > indicesBatchSize_) {
                batch = indicesBatchSize_;
            }
            DataCopyExtParams cpParams{1U, batch * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPad(idsUb, sortedGM[lo + done], cpParams, pad);
            EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

            int32_t ownerFirst = idsUb.GetValue(0) / numEntriesPerRank_;
            int32_t ownerLast = idsUb.GetValue(batch - 1U) / numEntriesPerRank_;
            if (ownerFirst == ownerLast) {
                uint32_t slot = (ownerFirst >= 0 && ownerFirst < static_cast<int32_t>(numRanks_)) ?
                                    static_cast<uint32_t>(ownerFirst) :
                                    0U;
                cntUb.SetValue(slot, cntUb.GetValue(slot) + static_cast<int32_t>(batch));
                done += batch;
                continue;
            }

            Div<int32_t>(ownerUb, idsUb, divisorUb, batch);
            PipeBarrier<PIPE_V>();
            uint32_t batchCompareCnt = Ceil(batch * static_cast<uint32_t>(sizeof(int32_t)), ALIGNED_LEN_256) *
                                       ALIGNED_LEN_256 / static_cast<uint32_t>(sizeof(int32_t));
            int32_t matched = 0;
            for (uint32_t r = 0; r < numRanks_; r++) {
                CompareScalar(maskUb, ownerUb, static_cast<int32_t>(r), AscendC::CMPMODE::EQ, batchCompareCnt);
                uint64_t rsvdCnt = 0;
                GatherMask(compactUb, compactUb, maskUb.ReinterpretCast<uint32_t>(), true, batch, {1, 1, 0, 0},
                           rsvdCnt);
                EngramFetchTrainSyncFunc<HardEvent::V_S>();
                cntUb.SetValue(r, cntUb.GetValue(r) + static_cast<int32_t>(rsvdCnt));
                matched += static_cast<int32_t>(rsvdCnt);
            }
            if (matched < static_cast<int32_t>(batch)) {
                cntUb.SetValue(0, cntUb.GetValue(0) + (static_cast<int32_t>(batch) - matched));
            }
            done += batch;
        }
    }

    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> partialGM;
    partialGM.SetGlobalBuffer(
        (__gm__ int32_t *)(partialCountsGM_ + static_cast<uint64_t>(aivId_) * numRanks_ * sizeof(int32_t)));
    DataCopyParams cntParams = {1U, static_cast<uint16_t>(numRanks_ * sizeof(int32_t)), 0U, 0U};
    DataCopyPad(partialGM, cntUb, cntParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::WriteSendCount(uint32_t ownerRank, int32_t myCount)
{
    LocalTensor<int32_t> countLocal = rankIDsBuf_.Get<int32_t>();
    countLocal.SetValue(0, myCount);
    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
    GM_ADDR sendCountSlot = sendCountsOutGM_ + ownerRank * UB_ALIGN;
    GlobalTensor<int32_t> sendCountsSlotGM;
    sendCountsSlotGM.SetGlobalBuffer((__gm__ int32_t *)sendCountSlot);
    DataCopyParams countParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
    DataCopyPad(sendCountsSlotGM, countLocal, countParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::ComputeSdisplsLocal()
{
    LocalTensor<int32_t> partialUb = partialCountsBuf_.Get<int32_t>();
    GlobalTensor<int32_t> partialGM;
    partialGM.SetGlobalBuffer((__gm__ int32_t *)partialCountsGM_);
    uint32_t totalInts = totalBlocks_ * numRanks_;
    DataCopyExtParams cpParams{1U, totalInts * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    DataCopyPad(partialUb, partialGM, cpParams, cpPad);
    EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

    LocalTensor<int32_t> countsLocal = rankCountsBuf_.Get<int32_t>();
    for (uint32_t r = 0; r < numRanks_; r++) {
        int32_t sum = 0;
        for (uint32_t c = 0; c < totalBlocks_; c++) {
            sum += partialUb.GetValue(c * numRanks_ + r);
        }
        countsLocal.SetValue(r, sum);
    }

    if (aivId_ == 0) {
        for (uint32_t r = 0; r < numRanks_; r++) {
            WriteSendCount(r, countsLocal.GetValue(r));
        }
    }

    LocalTensor<int64_t> sdisplsUb = tempBuf_.Get<int64_t>();
    int64_t sdisplsAccum = 0;
    for (uint32_t r = 0; r < numRanks_; r++) {
        int32_t sCount = countsLocal.GetValue(r);
        sdisplsUb.SetValue(r, sdisplsAccum);
        sdisplsAccum += static_cast<int64_t>(sCount);
    }

    if (aivId_ == 0) {
        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int64_t> sdisplsOutGM;
        sdisplsOutGM.SetGlobalBuffer((__gm__ int64_t *)sdisplsGM_);
        DataCopyParams sdisplParams = {1U, static_cast<uint16_t>(numRanks_ * sizeof(int64_t)), 0U, 0U};
        DataCopyPad(sdisplsOutGM, sdisplsUb, sdisplParams);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
    }
}

__aicore__ inline void EngramFetchTrainArch35::CountAndSortPhase()
{
    SortInputIndices();
    SyncAll<true>();
    CountSendCountsFromSorted();
    SyncAll<true>();
    ComputeSdisplsLocal();
    SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainArch35::Process()
{
    if ASCEND_IS_AIV {
        if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
            return;
        }

        InitFlagsAndCounters();
        CountAndSortPhase();
        SendCountPhase();
        ExchangeIndices();
        SyncAll<true>();
        ExchangeTokenWithLocalRead();
        if (aivId_ == 0) {
            WriteNumRecv();
        }
    }
}

#endif

} // namespace Mc2Kernel

#endif
