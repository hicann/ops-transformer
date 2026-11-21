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
 * \file engram_fetch_arch35.h
 * \brief engram_fetch算子arch35 kernel实现
 */

#ifndef ENGRAM_FETCH_ARCH35_H
#define ENGRAM_FETCH_ARCH35_H

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

// 统一校验 indices 中的条目是否落在合法表范围 [0, totalEntries)
__aicore__ inline bool IsValidEntry(int32_t idx, int64_t totalEntries)
{
    return idx >= 0 && static_cast<int64_t>(idx) < totalEntries;
}

class EngramFetchArch35 {
public:
    __aicore__ inline EngramFetchArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR sfTable,
                                GM_ADDR fetchedSf, GM_ADDR workspaceGM, AscendC::TPipe* pipe,
                                const EngramFetchTilingData* tilingData);

    template <bool HasSf>
    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void CopyLocalToken(const AscendC::LocalTensor<int32_t>& indicesLocal,
                                          const AscendC::LocalTensor<uint32_t>& tokenIdxInRank,
                                          const AscendC::LocalTensor<uint64_t>& commBufferLocal, uint32_t rankStart,
                                          uint32_t tokenPos, uint32_t indicesBatchStart, uint32_t localIdxStart,
                                          uint64_t hiddenBytes, int64_t totalEntries);
    __aicore__ inline void FetchSfAll();
    __aicore__ inline void StageSfGroup(uint32_t groupIdx, uint32_t chunkLen, uint32_t rowsPerTile, uint32_t sfSlot,
                                        uint64_t sfBytes, const AscendC::LocalTensor<int32_t>& idxUb);
    __aicore__ inline void GatherSf(int32_t globalIdx, uint64_t globalTokenIdx);
    template <bool HasSf>
    __aicore__ inline void GatherWave(const AscendC::LocalTensor<int32_t>& indicesLocal, uint32_t batchOffset,
                                      uint32_t waveLen, uint32_t tokenBase);
    __aicore__ inline void GatherLocalWave(const AscendC::LocalTensor<int32_t>& indicesLocal,
                                           const AscendC::LocalTensor<uint32_t>& tokenIdxInRank,
                                           uint32_t indicesBatchStart, uint32_t rankStart, uint32_t tokenPosBase,
                                           uint32_t tokenStride, uint32_t waveLen);
    template <bool HasSf>
    __aicore__ inline void LocalFetchRange(uint32_t indicesBatchStart, uint32_t tokenStart, uint32_t tokenEnd);
    __aicore__ inline void CopyContextToUb();
    __aicore__ inline void CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen);
    __aicore__ inline void ScatterByRank(uint32_t batchLen);
    __aicore__ inline void FetchByRank(uint32_t indicesBatchStart);
    __aicore__ inline void LocalFetchTokens(uint32_t indicesBatchStart, uint32_t tokenOffset, uint32_t tokenStride);
    __aicore__ inline void RemoteFetchRank(uint32_t ownerRank, uint32_t indicesBatchStart, uint32_t channelIdxInRank,
                                           uint32_t channelCount);
    __aicore__ inline void GatherRankTokens(uint32_t ownerRank, uint32_t batchLen, uint32_t compareCntMax,
                                            uint32_t& runningOffset);
    AscendC::TPipe* tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    // 兜底 WQE(fence=1+cqe=1) 写目标: 仅作 barrier/CQE 落地, 不承载有效数据
    GM_ADDR workspaceGM_{nullptr};
    __gm__ EngramCommContext* ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    uint32_t totalBlocks_{0};
    int32_t numEntriesPerRank_{0};
    uint32_t channelsPerRank_{1};
    uint64_t ubSize_{0};
    uint32_t indicesBatchSize_{0};
    uint32_t tileBytes_{TILE_BYTES_INFER_MAX};
    bool localWaveEnabled_{false};
    uint32_t localRowsPerWave_{1U};
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
    AscendC::TBuf<> rankReadCntBuf_;
    AscendC::TBuf<> rankIDsBuf_;
    AscendC::TBuf<> positionsBuf_;
    AscendC::TBuf<> divisorBuf_;
    AscendC::TBuf<> maskBuf_;
    AscendC::TBuf<> hcommBatchBuf_;

    AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> hcomm_;
    using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;
    HcommBatchHandle activeBatchHandle_{};
    uint32_t preparedReadCount_{0};

    __gm__ uint8_t* sfTableGM_{nullptr};
    GM_ADDR fetchedSfGM_{nullptr};
    int64_t numSfPacks_{0};
    int64_t sfElemSize_{0};

    template <auto const& config>
    __aicore__ inline void PrepareRead(uint64_t commHandle, GM_ADDR remoteBase, GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void FlushPreparedReads();
};

__aicore__ inline void EngramFetchArch35::Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR sfTable,
                                               GM_ADDR fetchedSf, GM_ADDR workspaceGM, AscendC::TPipe* pipe,
                                               const EngramFetchTilingData* tilingData)
{
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    fetchedSfGM_ = fetchedSf;
    sfTableGM_ = reinterpret_cast<__gm__ uint8_t*>(sfTable);
    workspaceGM_ = workspaceGM;
    aivId_ = AscendC::GetBlockIdx();

    ctxPtr_ = (__gm__ EngramCommContext*)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;
    channelsPerRank_ = ctxPtr_->channelsPerRank;
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
    uint64_t rowBytes = static_cast<uint64_t>(hiddenBytes_) + sfSlot;
    localWaveEnabled_ = false;
    localRowsPerWave_ = 1U;
    // wave 打包要求行宽按 32B 对齐；不满足时保留逐行 fallback。
    // sliced(numRanks>1) 模式本地 token 同样启用 wave: 行宽只含 data(SF 由 FetchSfAll 统一后置处理)。
    const uint64_t waveRowBytes = (numRanks_ == 1U) ? rowBytes : static_cast<uint64_t>(hiddenBytes_);
    if (hiddenBytes_ > 0 && static_cast<uint64_t>(hiddenBytes_) % UB_ALIGN == 0U &&
        waveRowBytes <= TILE_BYTES_INFER_MAX) {
        uint32_t rows = static_cast<uint32_t>(static_cast<uint64_t>(TILE_BYTES_INFER_MAX) / waveRowBytes);
        constexpr uint32_t minRowsPerWave = 8U;
        uint64_t waveFixedUb = HCOMM_INIT_SIZE + ENGRAM_BATCH_BUFFER_BYTES + UB_RESERVED_SIZE + UB_ALIGN * 5U;
        uint64_t waveRelayUb = static_cast<uint64_t>(TILE_BYTES_INFER_MAX) * RELAY_BUFFER_NUM_INFER;
        if (rows >= minRowsPerWave && waveFixedUb + waveRelayUb <= ubSize_) {
            localWaveEnabled_ = true;
            localRowsPerWave_ = rows;
            tileBytes_ = TILE_BYTES_INFER_MAX;
        }
    }
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
    uint32_t commBufferBufSize = AscendC::Ceil(numRanks_ * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(commBufferBuf_, commBufferBufSize);
    uint32_t hcommHandleBufSize = AscendC::Ceil(numRanks_ * channelsPerRank_ * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(hcommHandleBuf_, hcommHandleBufSize);

    totalBlocks_ = AscendC::GetBlockNum();
    uint32_t rankReadCntSlots = (totalBlocks_ >= numRanks_) ? 1U : AscendC::Ceil(numRanks_, totalBlocks_);
    uint32_t rankReadCntBufSize = AscendC::Ceil(rankReadCntSlots * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankReadCntBuf_, rankReadCntBufSize);
    AscendC::Duplicate<uint32_t>(rankReadCntBuf_.Get<uint32_t>(), 0U, rankReadCntSlots);
    SyncFunc<AscendC::HardEvent::V_S>();

    constexpr uint64_t ubReserved = 8U * 1024U;
    // ScatterByRank 需要: indices + tokenIdxInRank + rankIDs + positions + divisor (各4字节) + mask(每元素1字节近似)
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

    // positions 复用: [0, indicesBatchSize) 常量序列, Init 生成一次, ScatterByRank 直接复用
    // (batchLen <= indicesBatchSize_, 前 batchLen 个即 [0, batchLen) 的 GatherMask 序列)
    AscendC::ArithProgression<int32_t>(positionsBuf_.Get<int32_t>(), 0, 1, indicesBatchSize_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

__aicore__ inline void EngramFetchArch35::CopyContextToUb()
{
    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    AscendC::GlobalTensor<uint64_t> commBufferGm;
    commBufferGm.SetGlobalBuffer((__gm__ uint64_t*)&ctxPtr_->commBuffer[0]);
    AscendC::DataCopyExtParams cpComm{1U, numRanks_ * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint64_t> padComm{false, 0, 0, 0};
    AscendC::DataCopyPad(commBufferLocal, commBufferGm, cpComm, padComm);

    AscendC::LocalTensor<uint64_t> hcommHandleLocal = hcommHandleBuf_.Get<uint64_t>();
    AscendC::GlobalTensor<uint64_t> hcommHandleGm;
    hcommHandleGm.SetGlobalBuffer((__gm__ uint64_t*)&ctxPtr_->hcommHandle[0]);
    AscendC::DataCopyExtParams cpHandle{1U, numRanks_ * channelsPerRank_ * static_cast<uint32_t>(sizeof(uint64_t)), 0U,
                                        0U, 0U};
    AscendC::DataCopyPadExtParams<uint64_t> padHandle{false, 0, 0, 0};
    AscendC::DataCopyPad(hcommHandleLocal, hcommHandleGm, cpHandle, padHandle);
}

__aicore__ inline void EngramFetchArch35::CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen)
{
    AscendC::GlobalTensor<int32_t> indicesGlobal;
    indicesGlobal.SetGlobalBuffer((__gm__ int32_t*)indicesGM_);
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    AscendC::DataCopyExtParams params{1U, indicesBatchLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(indicesLocal, indicesGlobal[indicesBatchStart], params, pad);
}

__aicore__ inline void EngramFetchArch35::GatherRankTokens(uint32_t ownerRank, uint32_t batchLen,
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

__aicore__ inline void EngramFetchArch35::ScatterByRank(uint32_t batchLen)
{
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();

    uint32_t compareCntMax =
        AscendC::Ceil(batchLen * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);

    AscendC::LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();
    AscendC::LocalTensor<int32_t> divisor = divisorBuf_.Get<int32_t>();

    AscendC::Duplicate<int32_t>(divisor, numEntriesPerRank_, compareCntMax);

    AscendC::Duplicate<int32_t>(rankIDs, static_cast<int32_t>(numRanks_), compareCntMax);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Div<int32_t>(rankIDs, indicesLocal, divisor, batchLen);
    AscendC::PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_S>();

    uint32_t runningOffset = 0;
    if (totalBlocks_ >= numRanks_) {
        auto coreAssign = GetCoreAssignment(totalBlocks_, aivId_, numRanks_, rankId_, channelsPerRank_);
        GatherRankTokens(coreAssign.assignedRank, batchLen, compareCntMax, runningOffset);
    } else {
        for (uint32_t ownerRank = aivId_; ownerRank < numRanks_; ownerRank += totalBlocks_) {
            GatherRankTokens(ownerRank, batchLen, compareCntMax, runningOffset);
        }
    }
}

template <bool HasSf>
__aicore__ inline void EngramFetchArch35::Process()
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
            if (numRanks_ == 1U) {
                auto range = GetTokenRangeAssignment(totalBlocks_, aivId_, numTokens);
                uint32_t batchEnd = indicesBatchStart + indicesBatchLen;
                uint32_t tokenStart = (range.start > indicesBatchStart) ? range.start : indicesBatchStart;
                uint32_t tokenStop = range.start + range.count;
                tokenStop = (tokenStop > batchEnd) ? batchEnd : tokenStop;
                if (tokenStart < tokenStop) {
                    LocalFetchRange<HasSf>(indicesBatchStart, tokenStart, tokenStop);
                }
            } else {
                ScatterByRank(indicesBatchLen);
                FetchByRank(indicesBatchStart);
            }
            indicesBatchStart += indicesBatchLen;
        }
        if constexpr (HasSf) {
            if (numRanks_ > 1U) {
                // SF统一后置：所有WQE投递完成后再执行。最后一条WQE最早下发（不被各批SF推迟），
                // SF的MTE时间与仍在进行的远端传输重叠（不同引擎），wait尾延迟随之缩短
                FetchSfAll();
            }
        }
    }
}

__aicore__ inline void EngramFetchArch35::LocalFetchTokens(uint32_t indicesBatchStart, uint32_t tokenOffset,
                                                           uint32_t tokenStride)
{
    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    AscendC::LocalTensor<uint32_t> rankCounts = rankCountsBuf_.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> rankOffsets = rankOffsetsBuf_.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> tokenIdxInRank = tokenIdxInRankBuf_.Get<uint32_t>();

    uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
    uint32_t numEntriesPerRank = static_cast<uint32_t>(numEntriesPerRank_);
    uint32_t localIdxStart = rankId_ * numEntriesPerRank;
    uint32_t rankStart = rankOffsets(rankId_);
    uint32_t cnt = rankCounts(rankId_);
    const int64_t totalEntries = static_cast<int64_t>(numEntriesPerRank_) * numRanks_;
    if (localWaveEnabled_ && tokenStride > 0U) {
        // sliced 本地路径 wave 批量化: 本核按 tokenStride 连续的一组 token 打包进同一 relay
        // buffer, MTE2 批量装载一次同步, MTE3 按全局 token 序散射写出; 索引非法的 wave 回退逐行。
        uint32_t pos = tokenOffset;
        while (pos < cnt) {
            uint32_t waveLen = (cnt - pos + tokenStride - 1U) / tokenStride;
            if (waveLen > localRowsPerWave_) {
                waveLen = localRowsPerWave_;
            }
            bool hasInvalid = false;
            for (uint32_t row = 0; row < waveLen; ++row) {
                int32_t globalIdx = indicesLocal(tokenIdxInRank(rankStart + pos + row * tokenStride));
                if (!IsValidEntry(globalIdx, totalEntries)) {
                    hasInvalid = true;
                    break;
                }
            }
            if (!hasInvalid) {
                GatherLocalWave(indicesLocal, tokenIdxInRank, indicesBatchStart, rankStart, pos, tokenStride, waveLen);
            } else {
                for (uint32_t row = 0; row < waveLen; ++row) {
                    uint32_t tokenPos = pos + row * tokenStride;
                    CopyLocalToken(indicesLocal, tokenIdxInRank, commBufferLocal, rankStart, tokenPos,
                                   indicesBatchStart, localIdxStart, hiddenBytes, totalEntries);
                }
            }
            pos += waveLen * tokenStride;
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        return;
    }
    for (uint32_t tokenPos = tokenOffset; tokenPos < cnt; tokenPos += tokenStride) {
        CopyLocalToken(indicesLocal, tokenIdxInRank, commBufferLocal, rankStart, tokenPos, indicesBatchStart,
                       localIdxStart, hiddenBytes, totalEntries);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchArch35::GatherLocalWave(const AscendC::LocalTensor<int32_t>& indicesLocal,
                                                          const AscendC::LocalTensor<uint32_t>& tokenIdxInRank,
                                                          uint32_t indicesBatchStart, uint32_t rankStart,
                                                          uint32_t tokenPosBase, uint32_t tokenStride, uint32_t waveLen)
{
    // sliced 本地 wave: waveLen 行源数据(经 tokenIdxInRank 间接寻址)批量搬入 relay buffer,
    // 一次 EnQue/DeQue 同步后按全局 token 序散射写回 fetchedGM; 索引有效性由调用方预检。
    const uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    GM_ADDR tableBase = (GM_ADDR)commBufferLocal(rankId_);
    AscendC::GlobalTensor<uint8_t> srcGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t*)tableBase);
    AscendC::LocalTensor<uint8_t> relayBuffer = relayQue_.AllocTensor<uint8_t>();
    // hiddenBytes已32B整除(localWaveEnabled_前提), 裸DataCopy的blockLen单位=32B块(dav_3510)
    AscendC::DataCopyParams loadParams{1U, static_cast<uint16_t>(hiddenBytes / UB_ALIGN), 0U, 0U};
    const uint32_t localIdxStart = rankId_ * static_cast<uint32_t>(numEntriesPerRank_);
    for (uint32_t row = 0; row < waveLen; ++row) {
        uint32_t i = tokenIdxInRank(rankStart + tokenPosBase + row * tokenStride);
        int32_t globalIdx = indicesLocal(i);
        uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - localIdxStart;
        AscendC::DataCopy(relayBuffer[row * hiddenBytes], srcGm[static_cast<uint64_t>(localEntryIdx) * hiddenBytes],
                          loadParams);
    }
    relayQue_.EnQue<uint8_t>(relayBuffer);
    relayBuffer = relayQue_.DeQue<uint8_t>();
    AscendC::GlobalTensor<uint8_t> dstGm;
    dstGm.SetGlobalBuffer((__gm__ uint8_t*)fetchedGM_);
    AscendC::DataCopyParams storeParams{1U, static_cast<uint16_t>(hiddenBytes), 0U, 0U};
    for (uint32_t row = 0; row < waveLen; ++row) {
        uint32_t i = tokenIdxInRank(rankStart + tokenPosBase + row * tokenStride);
        uint64_t globalTokenIdx = static_cast<uint64_t>(indicesBatchStart) + i;
        AscendC::DataCopyPad(dstGm[globalTokenIdx * hiddenBytes], relayBuffer[row * hiddenBytes], storeParams);
    }
    relayQue_.FreeTensor<uint8_t>(relayBuffer);
}

template <bool HasSf>
__aicore__ inline void EngramFetchArch35::GatherWave(const AscendC::LocalTensor<int32_t>& indicesLocal,
                                                     uint32_t batchOffset, uint32_t waveLen, uint32_t tokenBase)
{
    // 将同一 wave 的 data 与 SF 打包到 relay buffer，减少搬运次数。
    const uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
    uint64_t sfBytes = 0U;
    if constexpr (HasSf) {
        sfBytes = static_cast<uint64_t>(numSfPacks_) * static_cast<uint64_t>(sfElemSize_);
    }
    const uint32_t sfSlot = static_cast<uint32_t>((sfBytes + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN);
    const uint32_t dataBytes = static_cast<uint32_t>(waveLen) * static_cast<uint32_t>(hiddenBytes_);

    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    GM_ADDR tableBase = (GM_ADDR)commBufferLocal(rankId_);
    AscendC::GlobalTensor<uint8_t> srcGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t*)tableBase);
    AscendC::LocalTensor<uint8_t> relayBuffer = relayQue_.AllocTensor<uint8_t>();
    // hiddenBytes已32B整除(localWaveEnabled_前提), 裸DataCopy的blockLen单位=32B块(dav_3510)
    AscendC::DataCopyParams dataLoadParams{1U, static_cast<uint16_t>(hiddenBytes / UB_ALIGN), 0U, 0U};
    for (uint32_t row = 0; row < waveLen; ++row) {
        int32_t entry = indicesLocal.GetValue(batchOffset + row);
        AscendC::DataCopy(relayBuffer[row * hiddenBytes], srcGm[static_cast<uint64_t>(entry) * hiddenBytes],
                          dataLoadParams);
    }
    if constexpr (HasSf) {
        AscendC::GlobalTensor<uint8_t> sfSrcGm;
        sfSrcGm.SetGlobalBuffer(sfTableGM_);
        AscendC::DataCopyExtParams sfLoadParams{1U, static_cast<uint32_t>(sfBytes), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<uint8_t> sfLoadPad{false, 0, 0, 0};
        for (uint32_t row = 0; row < waveLen; ++row) {
            int32_t entry = indicesLocal.GetValue(batchOffset + row);
            AscendC::DataCopyPad(relayBuffer[dataBytes + row * sfSlot], sfSrcGm[static_cast<uint64_t>(entry) * sfBytes],
                                 sfLoadParams, sfLoadPad);
        }
    }
    relayQue_.EnQue<uint8_t>(relayBuffer);
    relayBuffer = relayQue_.DeQue<uint8_t>();

    AscendC::GlobalTensor<uint8_t> dstGm;
    dstGm.SetGlobalBuffer((__gm__ uint8_t*)fetchedGM_);
    AscendC::DataCopyParams dataStoreParams{1U, static_cast<uint16_t>(dataBytes), 0U, 0U};
    AscendC::DataCopyPad(dstGm[static_cast<uint64_t>(tokenBase) * hiddenBytes], relayBuffer, dataStoreParams);
    if constexpr (HasSf) {
        AscendC::GlobalTensor<uint8_t> sfDstGm;
        sfDstGm.SetGlobalBuffer((__gm__ uint8_t*)fetchedSfGM_);
        AscendC::DataCopyParams sfStoreParams{1U, static_cast<uint16_t>(sfBytes), 0U, 0U};
        for (uint32_t row = 0; row < waveLen; ++row) {
            AscendC::DataCopyPad(sfDstGm[static_cast<uint64_t>(tokenBase + row) * sfBytes],
                                 relayBuffer[dataBytes + row * sfSlot], sfStoreParams);
        }
    }
    relayQue_.FreeTensor<uint8_t>(relayBuffer);
}

template <bool HasSf>
__aicore__ inline void EngramFetchArch35::LocalFetchRange(uint32_t indicesBatchStart, uint32_t tokenStart,
                                                          uint32_t tokenEnd)
{
    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    GM_ADDR tableBase = (GM_ADDR)commBufferLocal(rankId_);
    const uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
    const int64_t totalEntries = static_cast<int64_t>(numEntriesPerRank_) * static_cast<int64_t>(numRanks_);
    uint32_t current = tokenStart;
    while (current < tokenEnd) {
        uint32_t waveLen = tokenEnd - current;
        if (!localWaveEnabled_) {
            waveLen = tokenEnd - current;
        } else if (waveLen > localRowsPerWave_) {
            waveLen = localRowsPerWave_;
        }
        bool hasInvalid = false;
        if (localWaveEnabled_) {
            for (uint32_t offset = 0; offset < waveLen; ++offset) {
                int32_t entry = indicesLocal.GetValue(current - indicesBatchStart + offset);
                if (!IsValidEntry(entry, totalEntries)) {
                    hasInvalid = true;
                    break;
                }
            }
        }
        if (localWaveEnabled_ && !hasInvalid) {
            GatherWave<HasSf>(indicesLocal, current - indicesBatchStart, waveLen, current);
            current += waveLen;
            continue;
        }
        for (uint32_t offset = 0; offset < waveLen; ++offset) {
            uint32_t token = current + offset;
            int32_t entry = indicesLocal.GetValue(token - indicesBatchStart);
            if (!IsValidEntry(entry, totalEntries)) {
                continue;
            }
            GM_ADDR dst = fetchedGM_ + static_cast<uint64_t>(token) * hiddenBytes;
            GM_ADDR src = tableBase + static_cast<uint64_t>(entry) * hiddenBytes;
            LocalCopySlice(dst, src, hiddenBytes);
            if constexpr (HasSf) {
                GatherSf(entry, token);
            }
        }
        current += waveLen;
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <auto const& config>
__aicore__ inline void EngramFetchArch35::PrepareRead(uint64_t commHandle, GM_ADDR remoteBase, GM_ADDR dst, GM_ADDR src,
                                                      uint64_t len)
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

__aicore__ inline void EngramFetchArch35::FlushPreparedReads()
{
    (void)hcomm_.BatchCommit(activeBatchHandle_);
    preparedReadCount_ = 0U;
}

__aicore__ inline void EngramFetchArch35::RemoteFetchRank(uint32_t ownerRank, uint32_t indicesBatchStart,
                                                          uint32_t channelIdxInRank, uint32_t channelCount)
{
    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    AscendC::LocalTensor<uint64_t> hcommHandleLocal = hcommHandleBuf_.Get<uint64_t>();
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    AscendC::LocalTensor<uint32_t> rankCounts = rankCountsBuf_.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> rankOffsets = rankOffsetsBuf_.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> tokenIdxInRank = tokenIdxInRankBuf_.Get<uint32_t>();

    uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
    uint32_t numEntriesPerRank = static_cast<uint32_t>(numEntriesPerRank_);
    const int64_t totalEntries = static_cast<int64_t>(numEntriesPerRank_) * static_cast<int64_t>(numRanks_);

    uint32_t idxStart = ownerRank * numEntriesPerRank;
    uint32_t rankStart = rankOffsets(ownerRank);
    uint32_t rankTokenCount = rankCounts(ownerRank);
    if (rankTokenCount == 0) {
        return;
    }
    uint64_t channelHandle = hcommHandleLocal(ownerRank * channelsPerRank_ + channelIdxInRank);
    uint32_t rankReadIdx = (totalBlocks_ >= numRanks_) ? 0U : (ownerRank / totalBlocks_);
    GM_ADDR remoteBase = (GM_ADDR)commBufferLocal(ownerRank);
    AscendC::LocalTensor<uint32_t> rankReadCnt = rankReadCntBuf_.Get<uint32_t>();
    // 在途计数放标量寄存器：循环内免UB读改写，函数出口统一写回
    uint32_t readCnt = rankReadCnt.GetValue(rankReadIdx);

    uint32_t tokenPos = channelIdxInRank;
    if (tokenPos < rankTokenCount) {
        uint32_t i = tokenIdxInRank(rankStart + tokenPos);
        int32_t globalIdx = indicesLocal(i);
        while (true) {
            bool isLast = (tokenPos + channelCount >= rankTokenCount);
            // 预取下一token的(i, globalIdx)：与本次WQE投递重叠，消除 i→indices 两次串行UB读的阻塞。
            // nextGlobalIdx 的合法性不在此处校验，下一轮 globalIdx=nextGlobalIdx 后经 IsValidEntry 覆盖。
            uint32_t nextI = 0U;
            int32_t nextGlobalIdx = 0;
            if (!isLast) {
                nextI = tokenIdxInRank(rankStart + tokenPos + channelCount);
                nextGlobalIdx = indicesLocal(nextI);
            }
            if (IsValidEntry(globalIdx, totalEntries)) {
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
            } else if (isLast) {
                // isLast 的 token 索引非法：本批 cqe=0 乱序读既无 fence 屏障又无 CQE 完成标记，
                // 补一条 fence=1+cqe=1 兜底 WQE（读 remoteBase 安全地址、写 workspace scratch），
                // 不触碰非法 token 的输出槽，避免把表首行数据写进本应保持原值的槽位。
                PrepareRead<URMA_CQE_CFG>(channelHandle, remoteBase, workspaceGM_, remoteBase, hiddenBytes);
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

__aicore__ inline void EngramFetchArch35::FetchByRank(uint32_t indicesBatchStart)
{
    if (totalBlocks_ >= numRanks_) {
        auto coreAssign = GetCoreAssignment(totalBlocks_, aivId_, numRanks_, rankId_, channelsPerRank_);
        if (coreAssign.assignedRank != rankId_) {
            RemoteFetchRank(coreAssign.assignedRank, indicesBatchStart, coreAssign.idxInRankGroup,
                            coreAssign.rankGroupSize);
        } else {
            LocalFetchTokens(indicesBatchStart, coreAssign.idxInRankGroup, coreAssign.rankGroupSize);
        }
    } else {
        uint32_t startRank = (aivId_ < numRanks_) ? aivId_ : rankId_;
        for (uint32_t ownerRank = startRank; ownerRank < numRanks_; ownerRank += totalBlocks_) {
            if (ownerRank == rankId_) {
                LocalFetchTokens(indicesBatchStart, 0, 1);
            } else {
                RemoteFetchRank(ownerRank, indicesBatchStart, 0, 1);
            }
        }
    }
}

__aicore__ inline void EngramFetchArch35::GatherSf(int32_t globalIdx, uint64_t globalTokenIdx)
{
    // tiling 已保证单行 SF 不超过 relay tile，因此这里可一次拷贝完整 SF 行。
    const uint64_t sfBytes = static_cast<uint64_t>(numSfPacks_) * static_cast<uint64_t>(sfElemSize_);
    GM_ADDR sfSrc = sfTableGM_ + static_cast<uint64_t>(globalIdx) * sfBytes;
    GM_ADDR sfDst = fetchedSfGM_ + globalTokenIdx * sfBytes;
    AscendC::GlobalTensor<uint8_t> sfSrcGm;
    AscendC::GlobalTensor<uint8_t> sfDstGm;
    sfSrcGm.SetGlobalBuffer((__gm__ uint8_t*)sfSrc);
    sfDstGm.SetGlobalBuffer((__gm__ uint8_t*)sfDst);

    const uint32_t sfCopyBytes = static_cast<uint32_t>(sfBytes);
    AscendC::LocalTensor<uint8_t> sfTmp = relayQue_.AllocTensor<uint8_t>();
    AscendC::DataCopyExtParams sfCopyIn{1U, sfCopyBytes, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> sfPadIn{false, 0, 0, 0};
    AscendC::DataCopyPad(sfTmp, sfSrcGm, sfCopyIn, sfPadIn);
    relayQue_.EnQue<uint8_t>(sfTmp);
    sfTmp = relayQue_.DeQue<uint8_t>();
    AscendC::DataCopyExtParams sfCopyOut{1U, sfCopyBytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(sfDstGm, sfTmp, sfCopyOut);
    relayQue_.FreeTensor<uint8_t>(sfTmp);
}

__aicore__ inline void EngramFetchArch35::FetchSfAll()
{
    // SF统一后置：SF与rank切片彻底解耦（sfTable/fetchedSf均为本地内存），按连续chunk均分到所有AIV核，
    // chunk indices一次性载入UB后攒UB批拷，relayQue_深度做组间流水。
    const uint64_t sfBytes = static_cast<uint64_t>(numSfPacks_) * static_cast<uint64_t>(sfElemSize_);
    if (sfBytes == 0U) {
        return;
    }
    const uint32_t sfSlot = static_cast<uint32_t>((sfBytes + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN);
    uint32_t rowsPerTile = tileBytes_ / sfSlot;
    if (rowsPerTile == 0U) {
        rowsPerTile = 1U;
    }
    uint32_t numTokens = static_cast<uint32_t>(numTokens_);
    uint32_t chunkCap = (numTokens + totalBlocks_ - 1U) / totalBlocks_;
    if (chunkCap == 0U) {
        return;
    }
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

    uint32_t done = 0U;
    while (chunkStart + done < chunkEnd) {
        uint32_t curBatchLen = chunkEnd - chunkStart - done;
        if (curBatchLen > indicesBatchSize_) {
            curBatchLen = indicesBatchSize_;
        }
        uint32_t batchTokenBase = chunkStart + done;
        AscendC::DataCopyExtParams idxParams{1U, curBatchLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(idxUb, indicesGm[batchTokenBase], idxParams, idxPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        uint32_t totalGroups = (curBatchLen + rowsPerTile - 1U) / rowsPerTile;
        constexpr uint32_t PIPE_DEPTH = RELAY_BUFFER_NUM_INFER;
        uint32_t nextStage = 0U;
        const uint32_t prologue = (totalGroups < PIPE_DEPTH - 1U) ? totalGroups : (PIPE_DEPTH - 1U);
        for (uint32_t g = 0U; g < prologue; ++g) {
            StageSfGroup(g, curBatchLen, rowsPerTile, sfSlot, sfBytes, idxUb);
            ++nextStage;
        }
        for (uint32_t g = 0U; g < totalGroups; ++g) {
            if (nextStage < totalGroups) {
                StageSfGroup(nextStage, curBatchLen, rowsPerTile, sfSlot, sfBytes, idxUb);
                ++nextStage;
            }
            AscendC::LocalTensor<uint8_t> batchBuf = relayQue_.DeQue<uint8_t>();
            const uint32_t kStart = g * rowsPerTile;
            uint32_t count = curBatchLen - kStart;
            if (count > rowsPerTile) {
                count = rowsPerTile;
            }
            // 拷出：UB批量结果逐行落到fetchedSf（token连续）；无效globalIdx跳过
            for (uint32_t k = 0U; k < count; ++k) {
                int32_t globalIdx = idxUb.GetValue(kStart + k);
                if (!IsValidEntry(globalIdx, totalEntries)) {
                    continue;
                }
                AscendC::DataCopyPad(sfDstGm[static_cast<uint64_t>(batchTokenBase + kStart + k) * sfBytes],
                                     batchBuf[static_cast<uint64_t>(k) * sfSlot], sfStoreParams);
            }
            relayQue_.FreeTensor<uint8_t>(batchBuf);
        }
        done += curBatchLen;
    }
}

__aicore__ inline void EngramFetchArch35::StageSfGroup(uint32_t groupIdx, uint32_t chunkLen, uint32_t rowsPerTile,
                                                       uint32_t sfSlot, uint64_t sfBytes,
                                                       const AscendC::LocalTensor<int32_t>& idxUb)
{
    // 攒一组：组内SF行按globalIdx散布在sfTable，逐行搬入UB后EnQue（不等待，由流水统一消费）
    const uint32_t kStart = groupIdx * rowsPerTile;
    uint32_t count = chunkLen - kStart;
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
        if (!IsValidEntry(globalIdx, totalEntries)) {
            continue;
        }
        AscendC::DataCopyPad(batchBuf[static_cast<uint64_t>(k) * sfSlot],
                             sfSrcGm[static_cast<uint64_t>(globalIdx) * sfBytes], sfLoadParams, sfPad);
    }
    relayQue_.EnQue<uint8_t>(batchBuf);
}

__aicore__ inline void EngramFetchArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
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

__aicore__ inline void EngramFetchArch35::CopyLocalToken(const AscendC::LocalTensor<int32_t>& indicesLocal,
                                                         const AscendC::LocalTensor<uint32_t>& tokenIdxInRank,
                                                         const AscendC::LocalTensor<uint64_t>& commBufferLocal,
                                                         uint32_t rankStart, uint32_t tokenPos,
                                                         uint32_t indicesBatchStart, uint32_t localIdxStart,
                                                         uint64_t hiddenBytes, int64_t totalEntries)
{
    uint32_t i = tokenIdxInRank(rankStart + tokenPos);
    int32_t globalIdx = indicesLocal(i);
    if (!IsValidEntry(globalIdx, totalEntries)) {
        return;
    }
    uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - localIdxStart;
    uint64_t globalTokenIdx = static_cast<uint64_t>(indicesBatchStart) + i;
    GM_ADDR dst = fetchedGM_ + globalTokenIdx * hiddenBytes;
    GM_ADDR src = (GM_ADDR)commBufferLocal(rankId_) + static_cast<uint64_t>(localEntryIdx) * hiddenBytes;
    LocalCopySlice(dst, src, hiddenBytes);
}
#endif

} // namespace Mc2Kernel

#endif
