/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License");
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_fetch_rail_arch35.h
 * \brief engram_fetch_rail算子arch35 kernel实现
 */

#ifndef ENGRAM_FETCH_RAIL_ARCH35_H
#define ENGRAM_FETCH_RAIL_ARCH35_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
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
#include "adv_api/index/arithprogression.h"

namespace Mc2Kernel {

#if defined(ENABLE_ENGRAM_FETCH_KERNEL)

class EngramFetchRailArch35 {
public:
    __aicore__ inline EngramFetchRailArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR sfTable,
                                GM_ADDR fetchedSf, GM_ADDR workspaceGM, AscendC::TPipe* pipe,
                                const EngramFetchTilingData* tilingData);

    template <bool HasSf>
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyContextToUb();
    __aicore__ inline void CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen);
    __aicore__ inline void ScatterByRank(uint32_t batchLen);
    __aicore__ inline void GatherRankTokens(uint32_t ownerRank, uint32_t batchLen, uint32_t compareCntMax,
                                            uint32_t& runningOffset);
    template <bool HasSf>
    __aicore__ inline void FetchByRank(uint32_t indicesBatchStart);
    __aicore__ inline uint32_t OutSuperMachineIdx(uint32_t p);
    template <bool HasSf>
    __aicore__ inline void LocalFetchRank(uint32_t ownerRank, uint32_t indicesBatchStart, uint32_t tokenOffset,
                                          uint32_t tokenStride);
    template <bool HasSf>
    __aicore__ inline void FetchOutSuperMachine(uint32_t nodeIdx, uint32_t indicesBatchStart);
    template <bool HasSf>
    __aicore__ inline void FetchOwnerRows(uint32_t ownerRank, uint64_t channelHandle, GM_ADDR remoteBase,
                                          uint32_t indicesBatchStart, uint32_t channelIdxInRank, uint32_t channelCount);
    template <auto const& config>
    __aicore__ inline void PrepareRead(uint64_t commHandle, GM_ADDR remoteBase, GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void FlushPreparedReads();
    __aicore__ inline void FetchSfAll();
    __aicore__ inline void StageSfGroup(uint32_t groupIdx, uint32_t segTokenCnt, uint32_t rowsPerTile, uint32_t sfSlot,
                                        uint64_t sfBytes, AscendC::LocalTensor<int32_t>& idxUb);
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);

    AscendC::TPipe* tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    __gm__ EngramCommContext* ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    uint32_t totalBlocks_{0};
    int32_t numEntriesPerRank_{0};
    uint32_t channelsPerRank_{1};
    uint32_t ranksPerMachine_{0};
    uint32_t machineStartRank_{0};
    uint32_t machineCnt_{1};
    uint32_t ranksPerSupernode_{0};
    uint32_t supernodeStartRank_{0};
    uint32_t superStartMachine_{0};
    uint32_t machinesPerSupernode_{1};
    uint32_t outSuperMachineCnt_{0};
    uint64_t rankSlotStride_{0};
    uint64_t ubSize_{0};
    uint32_t indicesBatchSize_{0};
    uint32_t tileBytes_{TILE_BYTES_INFER_MAX};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};

    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, RELAY_BUFFER_NUM_INFER> relayQue_;
    AscendC::TBuf<> indicesBuf_;
    AscendC::TBuf<> hcommBuf_;
    AscendC::TBuf<> rankCountsBuf_;
    AscendC::TBuf<> rankOffsetsBuf_;
    AscendC::TBuf<> tokenIdxInRankBuf_;
    AscendC::TBuf<> commBufferBuf_;
    AscendC::TBuf<> hcommHandleBuf_;
    AscendC::TBuf<> rankIDsBuf_;
    AscendC::TBuf<> positionsBuf_;
    AscendC::TBuf<> divisorBuf_;
    AscendC::TBuf<> maskBuf_;
    AscendC::TBuf<> hcommBatchBuf_;
    AscendC::TBuf<> rankReadCntBuf_;

    AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> hcomm_;
    using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;
    HcommBatchHandle activeBatchHandle_{};
    uint32_t preparedReadCount_{0};

    __gm__ uint8_t* sfTableGM_{nullptr};
    GM_ADDR fetchedSfGM_{nullptr};
    int64_t numSfPacks_{0};
    int64_t sfElemSize_{0};
};
__aicore__ inline void EngramFetchRailArch35::Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
                                                   GM_ADDR sfTable, GM_ADDR fetchedSf, GM_ADDR workspaceGM,
                                                   AscendC::TPipe* pipe, const EngramFetchTilingData* tilingData)
{
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    fetchedSfGM_ = fetchedSf;
    sfTableGM_ = reinterpret_cast<__gm__ uint8_t*>(sfTable);
    aivId_ = AscendC::GetBlockIdx();
    (void)workspaceGM;

    ctxPtr_ = (__gm__ EngramCommContext*)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;
    channelsPerRank_ = ctxPtr_->channelsPerRank;
    ranksPerMachine_ = (ctxPtr_->ranksPerMachine != 0U) ? ctxPtr_->ranksPerMachine : numRanks_;
    machineStartRank_ = (rankId_ / ranksPerMachine_) * ranksPerMachine_;
    machineCnt_ = numRanks_ / ranksPerMachine_;
    ranksPerSupernode_ = (ctxPtr_->ranksPerSupernode != 0U) ? ctxPtr_->ranksPerSupernode : numRanks_;
    supernodeStartRank_ = (rankId_ / ranksPerSupernode_) * ranksPerSupernode_;
    superStartMachine_ = supernodeStartRank_ / ranksPerMachine_;
    machinesPerSupernode_ = ranksPerSupernode_ / ranksPerMachine_;
    outSuperMachineCnt_ = machineCnt_ - machinesPerSupernode_;
    rankSlotStride_ = ctxPtr_->rankSlotStride;
    if (channelsPerRank_ == 0U) {
        channelsPerRank_ = 1U;
    }

    numEntriesPerRank_ = tilingData->numEntriesPerRank;
    numTokens_ = tilingData->numTokens;
    hiddenBytes_ = tilingData->hiddenBytes;
    ubSize_ = tilingData->ubSize;

    numSfPacks_ = tilingData->numSfPacks;
    sfElemSize_ = tilingData->sfElemSize;

    const uint64_t alignedHiddenBytes = (static_cast<uint64_t>(hiddenBytes_) + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
    tileBytes_ =
        alignedHiddenBytes > TILE_BYTES_INFER_MAX ? TILE_BYTES_INFER_MAX : static_cast<uint32_t>(alignedHiddenBytes);
    uint64_t sfBytes = 0U;
    if (numSfPacks_ > 0 && sfElemSize_ > 0) {
        sfBytes = static_cast<uint64_t>(numSfPacks_) * static_cast<uint64_t>(sfElemSize_);
    }
    const uint64_t sfSlot = (sfBytes + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
    if (tileBytes_ > TILE_BYTES_INFER_MAX) {
        tileBytes_ = TILE_BYTES_INFER_MAX;
    }

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    AscendC::LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(relayQue_, RELAY_BUFFER_NUM_INFER, tileBytes_);
    tpipe_->InitBuffer(hcommBatchBuf_, ENGRAM_BATCH_BUFFER_BYTES);
    AscendC::LocalTensor<uint8_t> hcommBatchTensor = hcommBatchBuf_.Get<uint8_t>();
    AscendC::Duplicate<uint8_t>(hcommBatchTensor, 0U, ENGRAM_BATCH_BUFFER_BYTES);
    SyncFunc<AscendC::HardEvent::V_S>();

    uint32_t countsBufSize = AscendC::Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankCountsBuf_, countsBufSize);
    uint32_t offsetsBufSize = AscendC::Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankOffsetsBuf_, offsetsBufSize);
    uint32_t commBufferBufSize =
        AscendC::Ceil((ranksPerSupernode_ + outSuperMachineCnt_) * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(commBufferBuf_, commBufferBufSize);
    uint32_t hcommHandleBufSize =
        AscendC::Ceil((ranksPerSupernode_ + outSuperMachineCnt_) * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(hcommHandleBuf_, hcommHandleBufSize);

    totalBlocks_ = AscendC::GetBlockNum();
    uint32_t rankReadCntSlots = (totalBlocks_ >= numRanks_) ? 1U : AscendC::Ceil(numRanks_, totalBlocks_);
    uint32_t rankReadCntBufSize = AscendC::Ceil(rankReadCntSlots * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankReadCntBuf_, rankReadCntBufSize);
    AscendC::Duplicate<uint32_t>(rankReadCntBuf_.Get<uint32_t>(), 0U, rankReadCntSlots);
    SyncFunc<AscendC::HardEvent::V_S>();

    constexpr uint64_t ubReserved = 8U * 1024U;
    // ScatterByRank 需要：indices + tokenIdxInRank + rankIDs + positions + divisor (各4字节) + mask(每元素1字节近似)
    uint32_t bytesPerIndice = sizeof(int32_t) + sizeof(uint32_t) + sizeof(int32_t) * 3U + 1U;
    uint64_t usedUb = HCOMM_INIT_SIZE + tileBytes_ * RELAY_BUFFER_NUM_INFER + ENGRAM_BATCH_BUFFER_BYTES +
                      countsBufSize + offsetsBufSize + commBufferBufSize + hcommHandleBufSize + rankReadCntBufSize;
    uint64_t availableUb = (ubSize_ > usedUb + ubReserved) ? (ubSize_ - usedUb - ubReserved) : 0U;
    uint32_t maxBatchSize = static_cast<uint32_t>(availableUb / bytesPerIndice);
    uint32_t numTokens = static_cast<uint32_t>(numTokens_);
    indicesBatchSize_ = (numTokens <= maxBatchSize) ? numTokens : maxBatchSize;
    if (indicesBatchSize_ == 0U) {
        indicesBatchSize_ = 1U;
    }

    uint32_t indicesBufSize = AscendC::Ceil(indicesBatchSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    uint32_t tokenIdxInRankBufSize = AscendC::Ceil(indicesBatchSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tokenIdxInRankBuf_, tokenIdxInRankBufSize);

    uint32_t compareCntMax =
        AscendC::Ceil(indicesBatchSize_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    uint32_t int32BufSize = compareCntMax * sizeof(int32_t);
    tpipe_->InitBuffer(rankIDsBuf_, int32BufSize);
    tpipe_->InitBuffer(positionsBuf_, int32BufSize);
    tpipe_->InitBuffer(divisorBuf_, int32BufSize);
    uint32_t maskBufSize = AscendC::Ceil(AscendC::Ceil(compareCntMax, BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(maskBuf_, maskBufSize);
}

__aicore__ inline void EngramFetchRailArch35::CopyContextToUb()
{
    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    AscendC::GlobalTensor<uint64_t> commBufferGm;
    commBufferGm.SetGlobalBuffer((__gm__ uint64_t*)&ctxPtr_->commBuffer[0]);
    AscendC::DataCopyExtParams cpComm{
        1U, (ranksPerSupernode_ + outSuperMachineCnt_) * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint64_t> padComm{false, 0, 0, 0};
    AscendC::DataCopyPad(commBufferLocal, commBufferGm, cpComm, padComm);

    AscendC::LocalTensor<uint64_t> hcommHandleLocal = hcommHandleBuf_.Get<uint64_t>();
    AscendC::GlobalTensor<uint64_t> hcommHandleGm;
    hcommHandleGm.SetGlobalBuffer((__gm__ uint64_t*)&ctxPtr_->hcommHandle[0]);
    AscendC::DataCopyExtParams cpHandle{
        1U, (ranksPerSupernode_ + outSuperMachineCnt_) * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint64_t> padHandle{false, 0, 0, 0};
    AscendC::DataCopyPad(hcommHandleLocal, hcommHandleGm, cpHandle, padHandle);
}

__aicore__ inline void EngramFetchRailArch35::CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen)
{
    AscendC::GlobalTensor<int32_t> indicesGlobal;
    indicesGlobal.SetGlobalBuffer((__gm__ int32_t*)indicesGM_);
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    AscendC::DataCopyExtParams params{1U, indicesBatchLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(indicesLocal, indicesGlobal[indicesBatchStart], params, pad);
}

__aicore__ inline void EngramFetchRailArch35::ScatterByRank(uint32_t batchLen)
{
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();

    uint32_t compareCntMax =
        AscendC::Ceil(batchLen * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);

    AscendC::LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();
    AscendC::LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>();
    AscendC::LocalTensor<int32_t> divisor = divisorBuf_.Get<int32_t>();

    AscendC::ArithProgression<int32_t>(positions, 0, 1, batchLen);
    AscendC::Duplicate<int32_t>(divisor, numEntriesPerRank_, compareCntMax);

    AscendC::Duplicate<int32_t>(rankIDs, static_cast<int32_t>(numRanks_), compareCntMax);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Div<int32_t>(rankIDs, indicesLocal, divisor, batchLen);
    AscendC::PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_S>();

    uint32_t runningOffset = 0;
    // 超外每个machine视作一个节点，超内每个rank视作一个节点，从0节点遍历所有节点，生产GatherRankTokensk分组数据
    for (uint32_t p = aivId_; p < ranksPerSupernode_ + outSuperMachineCnt_; p += totalBlocks_) {
        if (p < superStartMachine_) {
            for (uint32_t slot = 0; slot < ranksPerMachine_; ++slot) {
                GatherRankTokens(p * ranksPerMachine_ + slot, batchLen, compareCntMax, runningOffset);
            }
        } else if (p < superStartMachine_ + ranksPerSupernode_) {
            GatherRankTokens(supernodeStartRank_ + p - superStartMachine_, batchLen, compareCntMax, runningOffset);
        } else {
            uint32_t machineIdx = OutSuperMachineIdx(p);
            for (uint32_t slot = 0; slot < ranksPerMachine_; ++slot) {
                GatherRankTokens(machineIdx * ranksPerMachine_ + slot, batchLen, compareCntMax, runningOffset);
            }
        }
    }
}

__aicore__ inline void EngramFetchRailArch35::GatherRankTokens(uint32_t ownerRank, uint32_t batchLen,
                                                               uint32_t compareCntMax, uint32_t& runningOffset)
{
    AscendC::LocalTensor<uint32_t> rankCounts = rankCountsBuf_.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> rankOffsets = rankOffsetsBuf_.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> tokenIdxInRank = tokenIdxInRankBuf_.Get<uint32_t>();
    AscendC::LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();
    AscendC::LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>();
    AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

    AscendC::CompareScalar(mask, rankIDs, static_cast<int32_t>(ownerRank), AscendC::CMPMODE::EQ, compareCntMax);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::LocalTensor<int32_t> dstRegion = tokenIdxInRank[runningOffset].ReinterpretCast<int32_t>();
    uint64_t rsvdCnt = 0;
    AscendC::GatherMask(dstRegion, positions, mask.ReinterpretCast<uint32_t>(), true, batchLen, {1, 1, 0, 0}, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_S>();

    uint32_t count = static_cast<uint32_t>(rsvdCnt);
    rankOffsets.SetValue(ownerRank, runningOffset);
    rankCounts.SetValue(ownerRank, count);
    runningOffset += count;
}

template <bool HasSf>
__aicore__ inline void EngramFetchRailArch35::FetchByRank(uint32_t indicesBatchStart)
{
    AscendC::LocalTensor<uint64_t> hcommHandleLocal = hcommHandleBuf_.Get<uint64_t>();
    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    // 超外每个machine视作一个节点，超内每个rank视作一个节点，从0节点遍历所有节点，消费ScatterByRank分组数据
    for (uint32_t p = aivId_; p < ranksPerSupernode_ + outSuperMachineCnt_; p += totalBlocks_) {
        if (p < superStartMachine_) {
            FetchOutSuperMachine<HasSf>(p, indicesBatchStart);
        } else if (p < superStartMachine_ + ranksPerSupernode_) {
            uint32_t ownerRank = supernodeStartRank_ + p - superStartMachine_;
            if (ownerRank >= machineStartRank_ && ownerRank < machineStartRank_ + ranksPerMachine_) {
                LocalFetchRank<HasSf>(ownerRank, indicesBatchStart, 0, 1);
            } else {
                uint64_t channelHandle = hcommHandleLocal(p);
                GM_ADDR remoteBase = (GM_ADDR)commBufferLocal(p) + (ownerRank % ranksPerMachine_) * rankSlotStride_;
                FetchOwnerRows<HasSf>(ownerRank, channelHandle, remoteBase, indicesBatchStart, 0, 1);
            }
        } else {
            FetchOutSuperMachine<HasSf>(p, indicesBatchStart);
        }
    }
}

__aicore__ inline uint32_t EngramFetchRailArch35::OutSuperMachineIdx(uint32_t p)
{
    return (p < superStartMachine_) ? p : p - (ranksPerSupernode_ - machinesPerSupernode_);
}

template <bool HasSf>
__aicore__ inline void EngramFetchRailArch35::LocalFetchRank(uint32_t ownerRank, uint32_t indicesBatchStart,
                                                             uint32_t tokenOffset, uint32_t tokenStride)
{
    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    AscendC::LocalTensor<uint32_t> rankCounts = rankCountsBuf_.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> rankOffsets = rankOffsetsBuf_.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> tokenIdxInRank = tokenIdxInRankBuf_.Get<uint32_t>();

    const uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
    uint32_t numEntriesPerRank = static_cast<uint32_t>(numEntriesPerRank_);
    uint32_t ownerIdxStart = ownerRank * numEntriesPerRank;
    uint32_t rankStart = rankOffsets(ownerRank);
    uint32_t cnt = rankCounts(ownerRank);
    const int64_t totalEntries = static_cast<int64_t>(numEntriesPerRank_) * numRanks_;

    for (uint32_t tokenPos = tokenOffset; tokenPos < cnt; tokenPos += tokenStride) {
        uint32_t i = tokenIdxInRank(rankStart + tokenPos);
        int32_t globalIdx = indicesLocal(i);
        if (globalIdx < 0 || static_cast<int64_t>(globalIdx) >= totalEntries) {
            continue;
        }
        uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - ownerIdxStart;
        uint64_t globalTokenIdx = indicesBatchStart + i;
        LocalCopySlice(fetchedGM_ + globalTokenIdx * hiddenBytes,
                       (GM_ADDR)commBufferLocal(superStartMachine_ + ownerRank - supernodeStartRank_) +
                           static_cast<uint64_t>(localEntryIdx) * hiddenBytes,
                       hiddenBytes);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <bool HasSf>
__aicore__ inline void EngramFetchRailArch35::FetchOutSuperMachine(uint32_t nodeIdx, uint32_t indicesBatchStart)
{
    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    AscendC::LocalTensor<uint64_t> hcommHandleLocal = hcommHandleBuf_.Get<uint64_t>();
    const uint32_t machineIdx = OutSuperMachineIdx(nodeIdx);
    const uint64_t channelHandle = hcommHandleLocal(nodeIdx);
    GM_ADDR machineBase = (GM_ADDR)commBufferLocal(nodeIdx);
    for (uint32_t s = 0; s < ranksPerMachine_; ++s) {
        uint32_t ownerRank = machineIdx * ranksPerMachine_ + s;
        GM_ADDR remoteBase = machineBase + static_cast<uint64_t>(s) * rankSlotStride_;
        FetchOwnerRows<HasSf>(ownerRank, channelHandle, remoteBase, indicesBatchStart, 0, 1);
    }
}

template <bool HasSf>
__aicore__ inline void EngramFetchRailArch35::FetchOwnerRows(uint32_t ownerRank, uint64_t channelHandle,
                                                             GM_ADDR remoteBase, uint32_t indicesBatchStart,
                                                             uint32_t channelIdxInRank, uint32_t channelCount)
{
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    AscendC::LocalTensor<uint32_t> rankCounts = rankCountsBuf_.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> rankOffsets = rankOffsetsBuf_.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> tokenIdxInRank = tokenIdxInRankBuf_.Get<uint32_t>();

    uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
    uint32_t numEntriesPerRank = static_cast<uint32_t>(numEntriesPerRank_);

    uint32_t idxStart = ownerRank * numEntriesPerRank;
    uint32_t rankStart = rankOffsets(ownerRank);
    uint32_t rankTokenCount = rankCounts(ownerRank);
    if (rankTokenCount == 0) {
        return;
    }

    uint32_t rankReadIdx = (totalBlocks_ >= numRanks_) ? 0U : (ownerRank / totalBlocks_);
    AscendC::LocalTensor<uint32_t> rankReadCnt = rankReadCntBuf_.Get<uint32_t>();
    uint32_t readCnt = rankReadCnt.GetValue(rankReadIdx);
    const int64_t totalEntries = static_cast<int64_t>(numEntriesPerRank_) * numRanks_;

    uint32_t tokenPos = channelIdxInRank;
    if (tokenPos < rankTokenCount) {
        uint32_t i = tokenIdxInRank(rankStart + tokenPos);
        int32_t globalIdx = indicesLocal(i);
        while (true) {
            bool isLast = (tokenPos + channelCount >= rankTokenCount);
            uint32_t nextI = 0U;
            int32_t nextGlobalIdx = 0;
            if (!isLast) {
                nextI = tokenIdxInRank(rankStart + tokenPos + channelCount);
                nextGlobalIdx = indicesLocal(nextI);
            }
            if (globalIdx < 0 || static_cast<int64_t>(globalIdx) >= totalEntries) {
            } else {
                uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - idxStart;
                uint64_t globalTokenIdx = indicesBatchStart + i;
                GM_ADDR dst = fetchedGM_ + globalTokenIdx * hiddenBytes;
                GM_ADDR remoteSrcAddr = remoteBase + static_cast<uint64_t>(localEntryIdx) * hiddenBytes;
                bool needDrain = (readCnt + 1U >= HCOMM_SQ_MAX_PENDING);
                if (isLast || needDrain) {
                    PrepareRead<URMA_CQE_CFG>(channelHandle, remoteBase, dst, remoteSrcAddr, hiddenBytes);
                } else {
                    PrepareRead<URMA_UNORDERED_CFG>(channelHandle, remoteBase, dst, remoteSrcAddr, hiddenBytes);
                }
                ++readCnt;
                if (needDrain) {
                    FlushPreparedReads();
                    int32_t drainRet = hcomm_.Drain(static_cast<AscendC::ChannelHandle>(channelHandle));
                    ascendc_assert(drainRet == 0, "mid-stream Drain failed, ret=%d\n", drainRet);
                    readCnt = 0;
                }
            }
            if (isLast) {
                break;
            }
            tokenPos += channelCount;
            i = nextI;
            globalIdx = nextGlobalIdx;
        }
    }
    rankReadCnt.SetValue(rankReadIdx, readCnt);
    if (preparedReadCount_ > 0U) {
        FlushPreparedReads();
    }
}

template <auto const& config>
__aicore__ inline void EngramFetchRailArch35::PrepareRead(uint64_t commHandle, GM_ADDR remoteBase, GM_ADDR dst,
                                                          GM_ADDR src, uint64_t len)
{
    if (preparedReadCount_ == ENGRAM_BATCH_CAPACITY) {
        FlushPreparedReads();
    }
    if (preparedReadCount_ == 0U) {
        AscendC::LocalTensor<uint8_t> hcommBatchTensor = hcommBatchBuf_.Get<uint8_t>();
        activeBatchHandle_ =
            hcomm_.MakeBatchHandle(commHandle, hcommBatchTensor, ENGRAM_BATCH_BUFFER_BYTES, remoteBase);
    }
    (void)hcomm_.ReadNbi<config>(activeBatchHandle_, dst, src, static_cast<uint32_t>(len));
    ++preparedReadCount_;
}

__aicore__ inline void EngramFetchRailArch35::FlushPreparedReads()
{
    (void)hcomm_.BatchCommit(activeBatchHandle_);
    preparedReadCount_ = 0U;
}

__aicore__ inline void EngramFetchRailArch35::FetchSfAll()
{
    const uint64_t sfBytes = static_cast<uint64_t>(numSfPacks_) * static_cast<uint64_t>(sfElemSize_);
    const uint32_t sfSlot = static_cast<uint32_t>((sfBytes + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN);
    uint32_t rowsPerTile = tileBytes_ / sfSlot;
    if (rowsPerTile == 0U) {
        rowsPerTile = 1U;
    }
    uint32_t numTokens = static_cast<uint32_t>(numTokens_);
    uint32_t chunkCap = (numTokens + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t chunkStart = aivId_ * chunkCap;
    if (chunkStart >= numTokens) {
        return;
    }
    uint32_t chunkEnd = (chunkStart + chunkCap > numTokens) ? numTokens : chunkStart + chunkCap;
    const int64_t totalEntries = static_cast<int64_t>(numEntriesPerRank_) * numRanks_;

    AscendC::LocalTensor<int32_t> idxUb = indicesBuf_.Get<int32_t>();
    AscendC::GlobalTensor<int32_t> indicesGm;
    indicesGm.SetGlobalBuffer((__gm__ int32_t*)indicesGM_);
    AscendC::GlobalTensor<uint8_t> sfDstGm;
    sfDstGm.SetGlobalBuffer((__gm__ uint8_t*)fetchedSfGM_);
    AscendC::DataCopyPadExtParams<int32_t> idxPad{false, 0, 0, 0};
    AscendC::DataCopyExtParams sfStoreParams{1U, static_cast<uint32_t>(sfBytes), 0U, 0U, 0U};

    uint32_t chunkDoneCnt = 0U;
    while (chunkStart + chunkDoneCnt < chunkEnd) {
        uint32_t segTokenCnt = chunkEnd - chunkStart - chunkDoneCnt;
        if (segTokenCnt > indicesBatchSize_) {
            segTokenCnt = indicesBatchSize_;
        }
        uint32_t segTokenBase = chunkStart + chunkDoneCnt;
        AscendC::DataCopyExtParams idxParams{1U, segTokenCnt * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(idxUb, indicesGm[segTokenBase], idxParams, idxPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        uint32_t totalGroups = (segTokenCnt + rowsPerTile - 1U) / rowsPerTile;
        for (uint32_t g = 0U; g < totalGroups; ++g) {
            StageSfGroup(g, segTokenCnt, rowsPerTile, sfSlot, sfBytes, idxUb);
            AscendC::LocalTensor<uint8_t> batchBuf = relayQue_.DeQue<uint8_t>();
            const uint32_t kStart = g * rowsPerTile;
            uint32_t count = segTokenCnt - kStart;
            if (count > rowsPerTile) {
                count = rowsPerTile;
            }
            for (uint32_t k = 0U; k < count; ++k) {
                int32_t globalIdx = idxUb.GetValue(kStart + k);
                if (globalIdx < 0 || globalIdx >= totalEntries) {
                    continue;
                }
                AscendC::DataCopyPad(sfDstGm[static_cast<uint64_t>(segTokenBase + kStart + k) * sfBytes],
                                     batchBuf[static_cast<uint64_t>(k) * sfSlot], sfStoreParams);
            }
            relayQue_.FreeTensor<uint8_t>(batchBuf);
        }
        chunkDoneCnt += segTokenCnt;
    }
}

__aicore__ inline void EngramFetchRailArch35::StageSfGroup(uint32_t groupIdx, uint32_t segTokenCnt,
                                                           uint32_t rowsPerTile, uint32_t sfSlot, uint64_t sfBytes,
                                                           AscendC::LocalTensor<int32_t>& idxUb)
{
    const uint32_t kStart = groupIdx * rowsPerTile;
    uint32_t count = segTokenCnt - kStart;
    if (count > rowsPerTile) {
        count = rowsPerTile;
    }
    const int64_t totalEntries = static_cast<int64_t>(numEntriesPerRank_) * numRanks_;
    AscendC::GlobalTensor<uint8_t> sfSrcGm;
    sfSrcGm.SetGlobalBuffer(sfTableGM_);
    AscendC::DataCopyExtParams sfLoadParams{1U, static_cast<uint32_t>(sfBytes), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> sfPad{false, 0, 0, 0};
    AscendC::LocalTensor<uint8_t> batchBuf = relayQue_.AllocTensor<uint8_t>();
    for (uint32_t k = 0U; k < count; ++k) {
        int32_t globalIdx = idxUb.GetValue(kStart + k);
        if (globalIdx < 0 || globalIdx >= totalEntries) {
            continue;
        }
        AscendC::DataCopyPad(batchBuf[static_cast<uint64_t>(k) * sfSlot],
                             sfSrcGm[static_cast<uint64_t>(globalIdx) * sfBytes], sfLoadParams, sfPad);
    }
    relayQue_.EnQue<uint8_t>(batchBuf);
}

__aicore__ inline void EngramFetchRailArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    AscendC::GlobalTensor<uint8_t> srcGm;
    AscendC::GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t*)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t*)dst);

    uint32_t tileLen = tileBytes_;
    uint64_t off = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > tileBytes_) ? tileLen : (len - off);

        AscendC::LocalTensor<uint8_t> tmp = relayQue_.AllocTensor<uint8_t>();
        AscendC::DataCopyExtParams mte2Params{1U, static_cast<uint32_t>(thisLen), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<uint8_t> mte2Pad{false, 0, 0, 0};
        AscendC::DataCopyPad(tmp, srcGm[off], mte2Params, mte2Pad);
        relayQue_.EnQue<uint8_t>(tmp);
        tmp = relayQue_.DeQue<uint8_t>();
        AscendC::DataCopyParams mte3Params{1U, static_cast<uint16_t>(thisLen), 0U, 0U};
        AscendC::DataCopyPad(dstGm[off], tmp, mte3Params);
        relayQue_.FreeTensor<uint8_t>(tmp);

        off += thisLen;
    }
}

template <bool HasSf>
__aicore__ inline void EngramFetchRailArch35::Process()
{
    if ASCEND_IS_AIV {
        if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
            return;
        }
        CopyContextToUb();
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint32_t numTokens = static_cast<uint32_t>(numTokens_);
        uint32_t indicesBatchStart = 0;
        while (indicesBatchStart < numTokens) {
            uint32_t indicesBatchLen = indicesBatchSize_;
            if (indicesBatchStart + indicesBatchLen > numTokens) {
                indicesBatchLen = numTokens - indicesBatchStart;
            }
            CopyIndicesToUb(indicesBatchStart, indicesBatchLen);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            ScatterByRank(indicesBatchLen);
            FetchByRank<HasSf>(indicesBatchStart);
            indicesBatchStart += indicesBatchLen;
        }
        if constexpr (HasSf) {
            FetchSfAll();
        }
    }
}

#endif

} // namespace Mc2Kernel

#endif // ENGRAM_FETCH_RAIL_ARCH35_H
