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

constexpr uint32_t ENGRAM_TIMEOUT_US = 60U * 1000U * 1000U;
constexpr uint32_t ENGRAM_CYCLES_PER_US = 1000U;

constexpr uint32_t HCOMM_BATCH_CAPACITY = 128U;
constexpr uint32_t HCOMM_PLAIN_WRITE_WQE_BYTES = 64U;
constexpr uint32_t HCOMM_BATCH_BUFFER_BYTES = HCOMM_BATCH_CAPACITY * HCOMM_PLAIN_WRITE_WQE_BYTES;

constexpr AscendC::UrmaWqeEntry URMA_NO_CQE_CFG = {
    .odr = 5,
    .fence = 1,
    .se = 0,
    .cqe = 0,
    .inlineEn = 0,
};

constexpr AscendC::UrmaWqeEntry URMA_NO_CQE_FENCE_CFG = {
    .odr = 6,
    .fence = 1,
    .se = 0,
    .cqe = 0,
    .inlineEn = 0,
};

constexpr AscendC::UrmaWqeEntry URMA_CQE_FENCE_CFG = {
    .odr = 6,
    .fence = 1,
    .se = 0,
    .cqe = 1,
    .inlineEn = 0,
};

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
    template <auto const &config>
    __aicore__ inline void PrepareWrite(uint64_t commHandle, GM_ADDR dstBase, GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void FlushPreparedWrites();
    __aicore__ inline void FlushAndDrain();
    __aicore__ inline void TimeoutCheck(uint64_t startTime, int tag);
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

    __aicore__ inline void CountGatherAndSortPhase();
    __aicore__ inline void CountGatherToTemp();
    __aicore__ inline uint32_t GatherBatchToTemp(uint32_t ownerRank, uint32_t indicesBatchStart,
                                                 uint32_t indicesBatchLen, uint64_t slotByteOffset, uint32_t cursor);
    __aicore__ inline void WriteSendCount(uint32_t ownerRank, int32_t myCount);
    __aicore__ inline void WritePartialCountsToGM();
    __aicore__ inline void ComputeSdisplsLocal();
    __aicore__ inline void RelocateFromTemp();

    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen);
    __aicore__ inline void CopyRecvIndicesBatch(int64_t cur, int64_t batchLen);
    __aicore__ inline void LocalReadTablePerToken(int64_t cur, int64_t batchLen);
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
    __aicore__ inline void WaitIndicesReadyFlag(uint32_t dstRank);
    __aicore__ inline void LocalReadTableAndSendLocal(uint32_t subIdx, int32_t sendCount, int64_t rdispl,
                                                      uint32_t indicesBufCap);
    __aicore__ inline void LocalReadTableAndSendRemote(uint32_t subIdx, uint32_t dstRank, int32_t sendCount,
                                                       int64_t rdispl, uint32_t indicesBufCap);
    __aicore__ inline void SendTokenBatchRemote(uint32_t chunkLen, uint32_t totalSent, uint32_t dstRank,
                                                uint64_t handle, GM_ADDR rowDstBase, GM_ADDR remoteCounterAddr,
                                                GM_ADDR notifyScratchAddr, const LocalTensor<int32_t> &indicesUb,
                                                uint32_t numEntriesU32);
    __aicore__ inline void ExchangeTokenWithLocalRead();
    __aicore__ inline void RecvTokensFromPeers();
    __aicore__ inline uint32_t RecvTokenChunk(GM_ADDR localWinBase, uint32_t srcRank, uint32_t subIdx, uint32_t myCount,
                                              int64_t sdispl, int64_t myStart, uint32_t totalReceived);
    __aicore__ inline void ReorderAndSaveCtx();
    __aicore__ inline void ReorderPerToken(uint32_t cur, uint32_t batchLen, LocalTensor<uint32_t> &tokenIdxUb);
    __aicore__ inline void ReorderBatched(uint32_t cur, uint32_t batchLen, uint32_t tokensPerTile,
                                          LocalTensor<uint32_t> &tokenIdxUb);
    __aicore__ inline void WriteNumRecv();
    __aicore__ inline void EnsureCachedTotalRecv();
    __aicore__ inline uint32_t CalcMyShareCount(uint32_t subIdx, uint32_t totalU32, uint32_t &myStart);
    __aicore__ inline void WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount);
    __aicore__ inline void ClearStatusFlags(GM_ADDR statusWinBase);
    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset);
    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank);
    __aicore__ inline int32_t ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset, uint32_t srcRank,
                                               uint32_t subIdx = 0U, bool useGroupSize = true);
    __aicore__ inline void WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset, int32_t value,
                                              uint32_t subIdx = 0U, bool useGroupSize = true);
    __aicore__ inline void LoadSendCountsToUb();
    __aicore__ inline void LoadRecvCountsToUb();
    __aicore__ inline void LoadInt64ArrayToUb(GM_ADDR gmAddr);
    __aicore__ inline void ClearLocalReadCounter(GM_ADDR localWinBase, uint64_t readOffset, uint32_t dstRank,
                                                 uint32_t subIdx, bool useGroupSize);

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

    uint32_t rankCores_{0};
    uint32_t tokenGroups_{1};
    uint32_t myOwnerRank_{0};
    uint32_t myTokenGroup_{0};

    uint64_t ubSize_{0};

    GM_ADDR sdisplsGM_{0};
    GM_ADDR rdisplsGM_{0};
    GM_ADDR sortedIndicesGM_{0};
    GM_ADDR sortedIndicesTempGM_{0};
    GM_ADDR permOutTempGM_{0};
    uint64_t perCoreTempSize_{0};
    uint64_t slotSize_{0};
    GM_ADDR localDataGM_{0};
    GM_ADDR recvDataGM_{0};
    GM_ADDR counterScratchGM_{0};
    GM_ADDR flagScratchGM_{0};
    GM_ADDR partialCountsGM_{0};
    GM_ADDR indicesReadyFlagGM_{0};

    TBuf<> pingBuf_;
    TBuf<> pongBuf_;
    int32_t ppEvtMte2ToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToMte2_[2] = {0, 0};
    int32_t mte2SEvt_{0};
    int32_t mte2SCntEvt_{0};
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
    TBuf<> partialCountsBuf_;

    int64_t cachedTotalRecv_{-1};

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;
    TBuf<> hcommBatchBuf_;
    HcommBatchHandle activeBatchHandle_{};
    uint64_t activeBatchChannel_{0};
    uint32_t preparedWriteCount_{0};
    uint32_t indicesBatchSize_{0};
    uint32_t compareCntMax_{0};
    uint32_t tileBytes_{TILE_BYTES};
};

__aicore__ inline void EngramFetchTrainArch35::WriteNbiChecked(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    int32_t ret = hcomm_.WriteNbi(handle, dst, src, len);
    ascendc_assert(ret == 0, "WriteNbi failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
}

__aicore__ inline void EngramFetchTrainArch35::DrainChecked(uint64_t handle)
{
    int32_t ret = hcomm_.Drain(handle);
    ascendc_assert(ret == 0, "Drain failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
}

__aicore__ inline void EngramFetchTrainArch35::TimeoutCheck(uint64_t startTime, int tag)
{
    uint64_t nowUs = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
    ascendc_assert((nowUs - startTime) < ENGRAM_TIMEOUT_US, "timeout, tag=%d, rankId=%u, aivId=%u, elapsed=%llu us\n",
                   tag, rankId_, aivId_, nowUs - startTime);
}

template <auto const &config>
__aicore__ inline void EngramFetchTrainArch35::PrepareWrite(uint64_t commHandle, GM_ADDR dstBase, GM_ADDR dst,
                                                            GM_ADDR src, uint64_t len)
{
    if (preparedWriteCount_ != 0U && activeBatchChannel_ != commHandle) {
        FlushPreparedWrites();
    }
    if (preparedWriteCount_ == HCOMM_BATCH_CAPACITY) {
        FlushPreparedWrites();
    }
    if (preparedWriteCount_ == 0U) {
        LocalTensor<uint8_t> hcommBatchTensor = hcommBatchBuf_.Get<uint8_t>();
        activeBatchHandle_ = hcomm_.MakeBatchHandle(commHandle, hcommBatchTensor, HCOMM_BATCH_BUFFER_BYTES, dstBase);
        activeBatchChannel_ = commHandle;
    }
    int32_t ret = hcomm_.WriteNbi<config>(activeBatchHandle_, dst, src, static_cast<uint32_t>(len));
    ascendc_assert(ret == 0, "batch WriteNbi failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
    ++preparedWriteCount_;
}

__aicore__ inline void EngramFetchTrainArch35::FlushPreparedWrites()
{
    if (preparedWriteCount_ == 0U) {
        return;
    }
    int32_t ret = hcomm_.BatchCommit(activeBatchHandle_);
    ascendc_assert(ret == 0, "BatchCommit failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
    activeBatchHandle_ = {};
    activeBatchChannel_ = 0;
    preparedWriteCount_ = 0U;
}

__aicore__ inline void EngramFetchTrainArch35::FlushAndDrain()
{
    if (preparedWriteCount_ == 0U) {
        return;
    }
    int32_t ret = hcomm_.BatchCommit(activeBatchHandle_);
    ascendc_assert(ret == 0, "BatchCommit failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
    int32_t drainRet = hcomm_.Drain<PIPE_MTE3>(activeBatchHandle_);
    ascendc_assert(drainRet == 0, "BatchDrain failed, ret=%d, rankId=%u, aivId=%u", drainRet, rankId_, aivId_);
    activeBatchHandle_ = {};
    activeBatchChannel_ = 0;
    preparedWriteCount_ = 0U;
}

__aicore__ inline GM_ADDR EngramFetchTrainArch35::GetRemoteWinAddr(uint32_t dstRank, uint64_t offset)
{
    return (GM_ADDR)ctxPtr_->commBuffer[dstRank] + offset;
}

__aicore__ inline uint64_t EngramFetchTrainArch35::GetCommHandle(uint32_t dstRank)
{
    uint32_t channelIdx;
    if (isSender_) {
        channelIdx = aivId_ / numRanks_;
        if (channelIdx >= groupSize_) {
            channelIdx = 0U;
        }
    } else {
        uint32_t recvIdx = aivId_ - numSendCores_;
        channelIdx = groupSize_ + recvIdx / numRanks_;
        if (channelIdx >= groupSize_ * 2U) {
            channelIdx = groupSize_;
        }
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

    rankCores_ = (totalBlocks_ < numRanks_) ? totalBlocks_ : numRanks_;
    tokenGroups_ = (totalBlocks_ + rankCores_ - 1U) / rankCores_;
    myOwnerRank_ = aivId_ % rankCores_;
    myTokenGroup_ = aivId_ / rankCores_;
}

__aicore__ inline void EngramFetchTrainArch35::InitHcommAndPipe()
{
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(hcommBatchBuf_, HCOMM_BATCH_BUFFER_BYTES);
    LocalTensor<uint8_t> hcommBatchTensor = hcommBatchBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(hcommBatchTensor, 0U, HCOMM_BATCH_BUFFER_BYTES);
    SyncFunc<AscendC::HardEvent::V_S>();

    tpipe_->InitBuffer(pingBuf_, tileBytes_);
    tpipe_->InitBuffer(pongBuf_, tileBytes_);
    ppEvtMte2ToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte2ToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte3ToMte2_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    ppEvtMte3ToMte2_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
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

    slotSize_ = Ceil(static_cast<uint64_t>(maxTokensPerRank_) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    if (slotSize_ == 0U) {
        slotSize_ = UB_ALIGN;
    }
    uint32_t numOwnerRanksMax = (numRanks_ + rankCores_ - 1U) / rankCores_;
    if (numOwnerRanksMax == 0U) {
        numOwnerRanksMax = 1U;
    }
    perCoreTempSize_ = static_cast<uint64_t>(numOwnerRanksMax) * slotSize_;
    sortedIndicesTempGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * perCoreTempSize_;
    permOutTempGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * perCoreTempSize_;

    int64_t totalRecv = tilingData->totalRecv;
    localDataGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(totalRecv) * static_cast<uint64_t>(hiddenBytes_);
    recvDataGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(numTokens_) * static_cast<uint64_t>(hiddenBytes_);
    counterScratchGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * UB_ALIGN;
    partialCountsGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * static_cast<uint64_t>(numRanks_) * sizeof(int32_t);
    flagScratchGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * UB_ALIGN;
    indicesReadyFlagGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(numRanks_) * sizeof(int32_t);
}

__aicore__ inline void EngramFetchTrainArch35::InitUbBuffers(const EngramFetchTilingData *tilingData)
{
    uint32_t countsBufSize = Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankCountsBuf_, countsBufSize);
    uint32_t statusBufSize = Ceil(numRanks_ * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    uint32_t tempBufSize = Ceil(numRanks_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tempBuf_, tempBufSize);
    uint32_t partialCountsSize = Ceil(totalBlocks_ * numRanks_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(partialCountsBuf_, partialCountsSize);

    uint32_t bytesPerIndice = sizeof(int32_t) + sizeof(uint32_t) + sizeof(int32_t) * 3U + 1U;
    tileBytes_ = TILE_BYTES;
    uint64_t fixedUb =
        HCOMM_INIT_SIZE + countsBufSize + statusBufSize + tempBufSize + partialCountsSize + UB_RESERVED_SIZE;
    uint64_t pingPongUb = 2U * tileBytes_;
    uint64_t availableUb = (ubSize_ > fixedUb + pingPongUb) ? (ubSize_ - fixedUb - pingPongUb) : 0U;

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
        LocalTensor<int32_t> zeroLocal = statusBuf_.Get<int32_t>();
        Duplicate<int32_t>(zeroLocal, 0, STATE_OFFSET / sizeof(int32_t));
        EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
        GlobalTensor<int32_t> flagInit;
        flagInit.SetGlobalBuffer((__gm__ int32_t *)flagScratchGM_);
        DataCopy(flagInit, zeroLocal, STATE_OFFSET / sizeof(int32_t));
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

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
        TimeoutCheck(startTime, 1);
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
    LoadSendCountsToUb();
    LocalTensor<int32_t> sendCountsUb = statusBuf_.Get<int32_t>();
    for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += totalBlocks_) {
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

__aicore__ inline void EngramFetchTrainArch35::WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset,
                                                                  int32_t value, uint32_t subIdx, bool useGroupSize)
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

    uint64_t rankStride = useGroupSize ? (STATE_OFFSET * groupSize_) : STATE_OFFSET;
    uint64_t handle = GetCommHandle(dstRank);
    GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, counterOffset) + rankId_ * rankStride + subIdx * STATE_OFFSET;
    WriteNbiChecked(handle, remoteCounterAddr, srcAddr, sizeof(int32_t));
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

__aicore__ inline void EngramFetchTrainArch35::ClearLocalReadCounter(GM_ADDR localWinBase, uint64_t readOffset,
                                                                     uint32_t dstRank, uint32_t subIdx,
                                                                     bool useGroupSize)
{
    LocalTensor<int32_t> zeroLocal = statusBuf_.Get<int32_t>();
    zeroLocal.SetValue(0, 0);
    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
    uint64_t rankStride = useGroupSize ? (STATE_OFFSET * groupSize_) : STATE_OFFSET;
    GM_ADDR readCntAddr = localWinBase + readOffset + dstRank * rankStride + subIdx * STATE_OFFSET;
    GlobalTensor<int32_t> readCntGM;
    readCntGM.SetGlobalBuffer((__gm__ int32_t *)readCntAddr);
    DataCopyParams zeroParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
    DataCopyPad(readCntGM, zeroLocal, zeroParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
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
    ClearLocalReadCounter(localWinBase, indicesReadOffset_, dstRank, 0U, false);

    while (totalSent < static_cast<uint32_t>(sendCount)) {
        if (totalBlocks_ > 1U) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
            int32_t remoteReadCnt = ReadLocalCounter(localWinBase, indicesReadOffset_, dstRank, 0U, false);
            while (localWriteCnt >= static_cast<uint32_t>(remoteReadCnt) &&
                   localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= NUM_SLOTS) {
                remoteReadCnt = ReadLocalCounter(localWinBase, indicesReadOffset_, dstRank, 0U, false);
                TimeoutCheck(startTime, 2);
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
            TimeoutCheck(startTime, 3);
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
        WriteRemoteCounter(srcRank, indicesReadOffset_, static_cast<int32_t>(localReadCnt), 0U, false);
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
            TimeoutCheck(startTime, 4);
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
        uint64_t dataBytes = static_cast<uint64_t>(readItems) * hiddenBytes_;
        LocalCopySlice(recvDataGM_ + (sdispl + myStart + totalReceived) * hiddenBytes_, localSlotAddr, dataBytes);
        totalReceived += readItems;
        WriteRemoteCounter(srcRank, tokenReadOffset_, static_cast<int32_t>(totalReceived), subIdx);
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

__aicore__ inline void EngramFetchTrainArch35::ReorderPerToken(uint32_t cur, uint32_t batchLen,
                                                               LocalTensor<uint32_t> &tokenIdxUb)
{
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
}

__aicore__ inline void EngramFetchTrainArch35::ReorderBatched(uint32_t cur, uint32_t batchLen, uint32_t tokensPerTile,
                                                              LocalTensor<uint32_t> &tokenIdxUb)
{
    uint32_t hiddenBytesU32 = static_cast<uint32_t>(hiddenBytes_);
    LocalTensor<uint8_t> buf0 = pingBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> buf1 = pongBuf_.Get<uint8_t>();
    uint32_t j = 0;
    uint32_t tileIdx = 0;
    while (j < batchLen) {
        uint32_t groupSize = batchLen - j;
        if (groupSize > tokensPerTile) {
            groupSize = tokensPerTile;
        }
        uint32_t bufIdx = tileIdx % 2U;
        LocalTensor<uint8_t> batchBuf = (bufIdx == 0U) ? buf0 : buf1;
        if (tileIdx >= 2U) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        }
        GM_ADDR srcBase = recvDataGM_ + static_cast<uint64_t>(cur + j) * static_cast<uint64_t>(hiddenBytes_);
        GlobalTensor<uint8_t> srcGm;
        srcGm.SetGlobalBuffer((__gm__ uint8_t *)srcBase);
        uint32_t totalBytes = groupSize * hiddenBytesU32;
        DataCopyExtParams mte2Params{1U, totalBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> mte2Pad{false, 0, 0, 0};
        DataCopyPad(batchBuf, srcGm, mte2Params, mte2Pad);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
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
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        j += groupSize;
        tileIdx++;
    }
    if (tileIdx >= 1U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[(tileIdx - 1U) % 2U]);
    }
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[tileIdx % 2U]);
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
    uint32_t tokensPerTile = CalcTokensPerTile();

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

        if (tokensPerTile == 0U) {
            ReorderPerToken(cur, batchLen, tokenIdxUb);
        } else {
            ReorderBatched(cur, batchLen, tokensPerTile, tokenIdxUb);
        }
        cur += batchLen;
    }

    if (aivId_ == 0) {
        WriteNumRecv();
    }
}

__aicore__ inline uint32_t EngramFetchTrainArch35::CalcTokensPerTile() const
{
    uint32_t hiddenBytesU32 = static_cast<uint32_t>(hiddenBytes_);
    if (hiddenBytesU32 == 0U) {
        return 0U;
    }
    return tileBytes_ / hiddenBytesU32;
}

__aicore__ inline void EngramFetchTrainArch35::CopyRecvIndicesBatch(int64_t cur, int64_t batchLen)
{
    LocalTensor<int32_t> indicesUb = indicesBuf_.Get<int32_t>();
    GlobalTensor<int32_t> recvIndicesBatchGM;
    recvIndicesBatchGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);
    DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen) * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U,
                               0U};
    DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    DataCopyPad(indicesUb, recvIndicesBatchGM[static_cast<uint32_t>(cur)], cpParams, cpPad);
    AscendC::SetFlag<HardEvent::MTE2_S>(mte2SEvt_);
}

__aicore__ inline void EngramFetchTrainArch35::LocalReadTablePerToken(int64_t cur, int64_t batchLen)
{
    AscendC::WaitFlag<HardEvent::MTE2_S>(mte2SEvt_);
    LocalTensor<int32_t> indicesUb = indicesBuf_.Get<int32_t>();
    uint32_t hiddenBytesU32 = static_cast<uint32_t>(hiddenBytes_);
    LocalTensor<uint8_t> buf0 = pingBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> buf1 = pongBuf_.Get<uint8_t>();

    uint32_t tileIdx = 0;
    for (int64_t j = 0; j < batchLen; j++) {
        int32_t globalIdx = indicesUb.GetValue(static_cast<uint32_t>(j));
        if (globalIdx < 0) {
            continue;
        }
        uint32_t localEntryIdx = static_cast<uint32_t>(
            static_cast<int64_t>(globalIdx) - static_cast<int64_t>(rankId_) * static_cast<int64_t>(numEntriesPerRank_));
        if (localEntryIdx >= static_cast<uint32_t>(numEntriesPerRank_)) {
            continue;
        }
        int64_t gmIdx = cur + j;
        GM_ADDR src =
            (GM_ADDR)localStorageAddr_ + static_cast<uint64_t>(localEntryIdx) * static_cast<uint64_t>(hiddenBytes_);
        GM_ADDR dst = localDataGM_ + static_cast<uint64_t>(gmIdx) * static_cast<uint64_t>(hiddenBytes_);

        uint32_t bufIdx = tileIdx % 2U;
        LocalTensor<uint8_t> tmp = (bufIdx == 0U) ? buf0 : buf1;

        if (tileIdx >= 2U) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        }

        GlobalTensor<uint8_t> srcGm;
        srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
        DataCopy(tmp, srcGm, hiddenBytesU32);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);

        GlobalTensor<uint8_t> dstGm;
        dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        DataCopy(dstGm, tmp, hiddenBytesU32);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);

        tileIdx++;
    }

    if (tileIdx >= 1U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[(tileIdx - 1U) % 2U]);
    }
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[tileIdx % 2U]);
    }
}

__aicore__ inline void EngramFetchTrainArch35::WaitIndicesReadyFlag(uint32_t dstRank)
{
    GM_ADDR flagAddr = indicesReadyFlagGM_ + static_cast<uint64_t>(dstRank) * sizeof(int32_t);
    GlobalTensor<int32_t> flagGM;
    flagGM.SetGlobalBuffer((__gm__ int32_t *)flagAddr);
    LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
    uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
    int32_t flagVal = 0;
    while (flagVal != 1) {
        DataCopyExtParams cpParams{1U, sizeof(int32_t), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        DataCopyPad(flagLocal, flagGM, cpParams, cpPad);
        EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();
        flagVal = flagLocal.GetValue(0);
        TimeoutCheck(startTime, 5);
    }
}

__aicore__ inline void EngramFetchTrainArch35::LocalReadTableAndSendLocal(uint32_t subIdx, int32_t sendCount,
                                                                          int64_t rdispl, uint32_t indicesBufCap)
{
    if (sendCount <= 0) {
        return;
    }
    uint32_t myStart;
    uint32_t myCount = CalcMyShareCount(subIdx, static_cast<uint32_t>(sendCount), myStart);
    if (myCount == 0U) {
        return;
    }
    uint32_t totalSent = 0;
    while (totalSent < myCount) {
        uint32_t remaining = myCount - totalSent;
        uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;
        if (chunkLen > indicesBufCap) {
            chunkLen = indicesBufCap;
        }
        int64_t cur = static_cast<int64_t>(rdispl + myStart + totalSent);
        CopyRecvIndicesBatch(cur, static_cast<int64_t>(chunkLen));
        LocalReadTablePerToken(cur, static_cast<int64_t>(chunkLen));
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
        totalSent += chunkLen;
    }
    GM_ADDR sDisplSlot = sdisplsGM_ + rankId_ * sizeof(int64_t);
    GlobalTensor<int64_t> sDisplSlotGM;
    sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
    int64_t sdispl = sDisplSlotGM.GetValue(0);
    uint64_t myBytes = static_cast<uint64_t>(myCount) * static_cast<uint64_t>(hiddenBytes_);
    LocalCopySlice(recvDataGM_ + (sdispl + myStart) * hiddenBytes_, localDataGM_ + (rdispl + myStart) * hiddenBytes_,
                   myBytes);
}

__aicore__ inline void EngramFetchTrainArch35::SendTokenBatchRemote(
    uint32_t chunkLen, uint32_t totalSent, uint32_t dstRank, uint64_t handle, GM_ADDR rowDstBase,
    GM_ADDR remoteCounterAddr, GM_ADDR notifyScratchAddr, const LocalTensor<int32_t> &indicesUb, uint32_t numEntriesU32)
{
    uint64_t hiddenBytesU64 = static_cast<uint64_t>(hiddenBytes_);
    GlobalTensor<int32_t> notifyScratchGM;
    notifyScratchGM.SetGlobalBuffer((__gm__ int32_t *)notifyScratchAddr);
    constexpr uint32_t DATA_PER_NOTIFY = 64U;
    uint32_t j = 0U;
    while (j < chunkLen) {
        uint32_t batchLen = chunkLen - j;
        if (batchLen > DATA_PER_NOTIFY) {
            batchLen = DATA_PER_NOTIFY;
        }
        activeBatchHandle_ = {};
        activeBatchChannel_ = 0;
        preparedWriteCount_ = 0U;
        for (uint32_t k = 0U; k < batchLen; k++) {
            int32_t globalIdx = indicesUb.GetValue(j + k);
            GM_ADDR rowSrc = reinterpret_cast<GM_ADDR>(localStorageAddr_);
            if (globalIdx >= 0) {
                uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - rankId_ * numEntriesU32;
                if (localEntryIdx < numEntriesU32) {
                    rowSrc = reinterpret_cast<GM_ADDR>(localStorageAddr_) +
                             static_cast<uint64_t>(localEntryIdx) * hiddenBytesU64;
                }
            }
            GM_ADDR rowDst = rowDstBase + static_cast<uint64_t>(j + k) * hiddenBytesU64;
            PrepareWrite<URMA_NO_CQE_CFG>(handle, GetRemoteWinAddr(dstRank, 0), rowDst, rowSrc, hiddenBytes_);
        }
        LocalTensor<int32_t> notifyLocal = statusBuf_.Get<int32_t>();
        notifyLocal.SetValue(0, static_cast<int32_t>(totalSent + j + batchLen));
        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
        DataCopyParams notifyParams = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
        DataCopyPad(notifyScratchGM, notifyLocal, notifyParams);
        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
        PrepareWrite<URMA_CQE_FENCE_CFG>(handle, GetRemoteWinAddr(dstRank, 0), remoteCounterAddr, notifyScratchAddr,
                                         sizeof(int32_t));
        FlushAndDrain();
        j += batchLen;
    }
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
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint64_t handle = GetCommHandle(dstRank);
    GM_ADDR remoteCounterAddr =
        GetRemoteWinAddr(dstRank, tokenWriteOffset_) + rankId_ * STATE_OFFSET * groupSize_ + subIdx * STATE_OFFSET;
    GM_ADDR remoteSubIdxBase =
        GetRemoteWinAddr(dstRank, tokenDataOffset_) + rankId_ * totalSlots_ * tokenSlotSize_ + subIdx * tokenSlotSize_;
    uint32_t numEntriesU32 = static_cast<uint32_t>(numEntriesPerRank_);
    uint64_t hiddenBytesU64 = static_cast<uint64_t>(hiddenBytes_);
    GM_ADDR notifyScratchAddr = counterScratchGM_ + static_cast<uint64_t>(aivId_) * UB_ALIGN;
    ClearLocalReadCounter(localWinBase, tokenReadOffset_, dstRank, subIdx, true);
    uint32_t totalSent = 0;
    while (totalSent < myCount) {
        if (totalBlocks_ > 1U) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_CYCLES_PER_US;
            int32_t remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank, subIdx);
            while (totalSent >= static_cast<uint32_t>(remoteReadCnt) + maxTokensPerSlot_) {
                remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank, subIdx);
                TimeoutCheck(startTime, 6);
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
        CopyRecvIndicesBatch(cur, static_cast<int64_t>(chunkLen));
        AscendC::WaitFlag<HardEvent::MTE2_S>(mte2SEvt_);
        LocalTensor<int32_t> indicesUb = indicesBuf_.Get<int32_t>();
        GM_ADDR rowDstBase = remoteSubIdxBase + static_cast<uint64_t>(totalSent % maxTokensPerSlot_) * hiddenBytesU64;
        activeBatchHandle_ = {};
        activeBatchChannel_ = 0;
        preparedWriteCount_ = 0U;
        SendTokenBatchRemote(chunkLen, totalSent, dstRank, handle, rowDstBase, remoteCounterAddr, notifyScratchAddr,
                             indicesUb, numEntriesU32);
        totalSent += chunkLen;
    }
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
        if (sendCount > 0) {
            WaitIndicesReadyFlag(dstRank);
        }
        if (dstRank == rankId_) {
            LocalReadTableAndSendLocal(subIdx, sendCount, rdispl, indicesBufCap);
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

    LocalTensor<uint8_t> buf0 = pingBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> buf1 = pongBuf_.Get<uint8_t>();

    uint32_t tileLen = tileBytes_;
    uint64_t off = 0;
    uint32_t tileIdx = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > tileBytes_) ? tileLen : (len - off);
        uint32_t bufIdx = tileIdx % 2U;
        LocalTensor<uint8_t> tmp = (bufIdx == 0U) ? buf0 : buf1;

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

__aicore__ inline uint32_t EngramFetchTrainArch35::GatherBatchToTemp(uint32_t ownerRank, uint32_t indicesBatchStart,
                                                                     uint32_t indicesBatchLen, uint64_t slotByteOffset,
                                                                     uint32_t cursor)
{
    LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();
    LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>();
    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

    uint32_t batchCompareCnt =
        Ceil(indicesBatchLen * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);

    CompareScalar(mask, rankIDs, static_cast<int32_t>(ownerRank), AscendC::CMPMODE::EQ, batchCompareCnt);

    uint64_t rsvdCnt = 0;
    GatherMask(positions, positions, mask.ReinterpretCast<uint32_t>(), true, indicesBatchLen, {1, 1, 0, 0}, rsvdCnt);
    EngramFetchTrainSyncFunc<HardEvent::V_S>();

    uint32_t batchCount = static_cast<uint32_t>(rsvdCnt);
    if (batchCount == 0) {
        return 0;
    }

    LocalTensor<uint32_t> tokenIdxLocal = tokenIdxInRankBuf_.Get<uint32_t>();
    Adds<int32_t>(tokenIdxLocal.ReinterpretCast<int32_t>(), positions, static_cast<int32_t>(indicesBatchStart),
                  batchCount);
    EngramFetchTrainSyncFunc<HardEvent::V_S>();

    for (uint32_t i = 0; i < batchCount; i++) {
        uint32_t origPos = positions.GetValue(i);
        positions.SetValue(i, indicesLocal.GetValue(origPos));
    }

    GM_ADDR sortedTempBase = sortedIndicesTempGM_ + static_cast<uint64_t>(aivId_) * perCoreTempSize_;
    GM_ADDR permTempBase = permOutTempGM_ + static_cast<uint64_t>(aivId_) * perCoreTempSize_;
    uint64_t writeByteOffset = slotByteOffset + static_cast<uint64_t>(cursor) * sizeof(int32_t);

    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> sortedTempGM;
    sortedTempGM.SetGlobalBuffer((__gm__ int32_t *)(sortedTempBase + writeByteOffset));
    DataCopyParams sortedParams = {1U, static_cast<uint16_t>(batchCount * sizeof(int32_t)), 0U, 0U};
    DataCopyPad(sortedTempGM, positions, sortedParams);

    GlobalTensor<uint32_t> permTempGM;
    permTempGM.SetGlobalBuffer((__gm__ uint32_t *)(permTempBase + writeByteOffset));
    DataCopyParams permParams = {1U, static_cast<uint16_t>(batchCount * sizeof(uint32_t)), 0U, 0U};
    DataCopyPad(permTempGM, tokenIdxLocal, permParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

    return batchCount;
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

__aicore__ inline void EngramFetchTrainArch35::WritePartialCountsToGM()
{
    LocalTensor<int32_t> partialUb = partialCountsBuf_.Get<int32_t>();
    LocalTensor<uint32_t> cursors = rankCountsBuf_.Get<uint32_t>();

    Duplicate<int32_t>(partialUb, 0, numRanks_);
    EngramFetchTrainSyncFunc<HardEvent::V_S>();

    uint32_t slotIdx = 0;
    for (uint32_t ownerRank = myOwnerRank_; ownerRank < numRanks_; ownerRank += rankCores_, slotIdx++) {
        partialUb.SetValue(ownerRank, static_cast<int32_t>(cursors.GetValue(slotIdx)));
    }

    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
    GM_ADDR rowAddr = partialCountsGM_ + static_cast<uint64_t>(aivId_) * numRanks_ * sizeof(int32_t);
    GlobalTensor<int32_t> partialGM;
    partialGM.SetGlobalBuffer((__gm__ int32_t *)rowAddr);
    DataCopyParams cpParams = {1U, static_cast<uint16_t>(numRanks_ * sizeof(int32_t)), 0U, 0U};
    DataCopyPad(partialGM, partialUb, cpParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainArch35::CountGatherToTemp()
{
    LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();
    LocalTensor<int32_t> divisor = divisorBuf_.Get<int32_t>();
    LocalTensor<uint32_t> cursors = rankCountsBuf_.Get<uint32_t>();

    Duplicate<int32_t>(divisor, numEntriesPerRank_, compareCntMax_);

    uint32_t numOwnerRanks = (numRanks_ - myOwnerRank_ + rankCores_ - 1U) / rankCores_;
    Duplicate<uint32_t>(cursors, 0U, numOwnerRanks);
    EngramFetchTrainSyncFunc<HardEvent::V_S>();

    uint32_t tokensPerGroup = (numTokens_ + tokenGroups_ - 1U) / tokenGroups_;
    uint32_t myTokenStart = myTokenGroup_ * tokensPerGroup;
    uint32_t myTokenEnd = (myTokenStart + tokensPerGroup > numTokens_) ? numTokens_ : (myTokenStart + tokensPerGroup);

    uint32_t indicesBatchStart = myTokenStart;
    while (indicesBatchStart < myTokenEnd) {
        uint32_t indicesBatchLen = indicesBatchSize_;
        if (indicesBatchStart + indicesBatchLen > myTokenEnd) {
            indicesBatchLen = myTokenEnd - indicesBatchStart;
        }
        CopyIndicesToUb(indicesBatchStart, indicesBatchLen);
        EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

        LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>();
        ArithProgression<int32_t>(positions, 0, 1, indicesBatchLen);
        Div<int32_t>(rankIDs, indicesLocal, divisor, indicesBatchLen);
        PipeBarrier<PIPE_V>();
        EngramFetchTrainSyncFunc<HardEvent::V_S>();

        uint32_t slotIdx = 0;
        for (uint32_t ownerRank = myOwnerRank_; ownerRank < numRanks_; ownerRank += rankCores_, slotIdx++) {
            uint32_t cursor = cursors.GetValue(slotIdx);
            uint64_t slotByteOffset = static_cast<uint64_t>(slotIdx) * slotSize_;
            uint32_t batchCount =
                GatherBatchToTemp(ownerRank, indicesBatchStart, indicesBatchLen, slotByteOffset, cursor);
            cursors.SetValue(slotIdx, cursor + batchCount);
        }

        indicesBatchStart += indicesBatchLen;
    }

    WritePartialCountsToGM();
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

    if (myTokenGroup_ == 0U) {
        for (uint32_t ownerRank = myOwnerRank_; ownerRank < numRanks_; ownerRank += rankCores_) {
            WriteSendCount(ownerRank, countsLocal.GetValue(ownerRank));
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

__aicore__ inline void EngramFetchTrainArch35::RelocateFromTemp()
{
    GM_ADDR sortedTempBase = sortedIndicesTempGM_ + static_cast<uint64_t>(aivId_) * perCoreTempSize_;
    GM_ADDR permTempBase = permOutTempGM_ + static_cast<uint64_t>(aivId_) * perCoreTempSize_;

    LoadSendCountsToUb();
    LocalTensor<int32_t> sendCountsUb = statusBuf_.Get<int32_t>();
    LoadInt64ArrayToUb(sdisplsGM_);
    LocalTensor<int64_t> sdisplsUb = tempBuf_.Get<int64_t>();

    LocalTensor<int32_t> partialUb = partialCountsBuf_.Get<int32_t>();
    GlobalTensor<int32_t> partialGM;
    partialGM.SetGlobalBuffer((__gm__ int32_t *)partialCountsGM_);
    uint32_t totalInts = totalBlocks_ * numRanks_;
    DataCopyExtParams cpParams{1U, totalInts * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    DataCopyPad(partialUb, partialGM, cpParams, cpPad);
    EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

    uint32_t slotIdx = 0;
    for (uint32_t ownerRank = myOwnerRank_; ownerRank < numRanks_; ownerRank += rankCores_, slotIdx++) {
        int64_t sdispl = sdisplsUb.GetValue(ownerRank);
        uint32_t totalCount = static_cast<uint32_t>(sendCountsUb.GetValue(ownerRank * UB_ALIGN / sizeof(int32_t)));
        if (totalCount == 0U) {
            continue;
        }

        uint32_t groupOffset = 0;
        for (uint32_t g = 0; g < myTokenGroup_; g++) {
            uint32_t coreId = g * rankCores_ + myOwnerRank_;
            groupOffset += static_cast<uint32_t>(partialUb.GetValue(coreId * numRanks_ + ownerRank));
        }
        uint32_t myCount = static_cast<uint32_t>(partialUb.GetValue(aivId_ * numRanks_ + ownerRank));
        if (myCount == 0U) {
            continue;
        }

        uint64_t slotByteOffset = static_cast<uint64_t>(slotIdx) * slotSize_;
        uint64_t dstOffset = static_cast<uint64_t>(sdispl + static_cast<int64_t>(groupOffset));

        GM_ADDR srcSorted = sortedTempBase + slotByteOffset;
        GM_ADDR dstSorted = sortedIndicesGM_ + dstOffset * sizeof(int32_t);
        LocalCopySlice(dstSorted, srcSorted, static_cast<uint64_t>(myCount) * sizeof(int32_t));

        GM_ADDR srcPerm = permTempBase + slotByteOffset;
        GM_ADDR dstPerm = permOutGM_ + dstOffset * sizeof(uint32_t);
        LocalCopySlice(dstPerm, srcPerm, static_cast<uint64_t>(myCount) * sizeof(uint32_t));
    }
}

__aicore__ inline void EngramFetchTrainArch35::CountGatherAndSortPhase()
{
    CountGatherToTemp();
    SyncAll<true>();

    ComputeSdisplsLocal();
    SyncAll<true>();

    RelocateFromTemp();
}

__aicore__ inline void EngramFetchTrainArch35::Process()
{
    if ASCEND_IS_AIV {
        if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
            return;
        }

        InitFlagsAndCounters();
        CountGatherAndSortPhase();
        SendCountPhase();
        ExchangeIndices();
        ExchangeTokenWithLocalRead();
        ReorderAndSaveCtx();
    }
}

#endif

} // namespace Mc2Kernel

#endif
