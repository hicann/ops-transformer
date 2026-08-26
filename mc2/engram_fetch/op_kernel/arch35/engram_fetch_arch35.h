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
#include "adv_api/index/arithprogression.h"

namespace Mc2Kernel {

#if defined(ENABLE_ENGRAM_FETCH_KERNEL)

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

class EngramFetchArch35 {
public:
    __aicore__ inline EngramFetchArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspaceGM,
                                AscendC::TPipe *pipe, const EngramFetchTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void CopyContextToUb();
    __aicore__ inline void CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen);
    __aicore__ inline void ScatterByRank(uint32_t batchLen);
    __aicore__ inline void FetchByRank(uint32_t indicesBatchStart);
    __aicore__ inline void LocalFetchTokens(uint32_t indicesBatchStart, uint32_t tokenOffset, uint32_t tokenStride);
    __aicore__ inline void RemoteFetchRank(uint32_t ownerRank, uint32_t indicesBatchStart, uint32_t channelIdxInRank,
                                           uint32_t channelCount);
    __aicore__ inline void GatherRankTokens(uint32_t ownerRank, uint32_t batchLen, uint32_t compareCntMax,
                                            uint32_t &runningOffset);
    AscendC::TPipe *tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    int32_t numEntriesPerRank_{0};
    uint32_t channelsPerRank_{1};
    uint64_t ubSize_{0};
    uint32_t indicesBatchSize_{0};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};

    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, RELAY_BUFFER_NUM> relayQue_;
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

    AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> hcomm_;
};

__aicore__ inline void EngramFetchArch35::Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
                                               GM_ADDR workspaceGM, AscendC::TPipe *pipe,
                                               const EngramFetchTilingData *tilingData)
{
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    aivId_ = AscendC::GetBlockIdx();
    (void)workspaceGM;

    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;
    channelsPerRank_ = (HANDLE_ARRAY_SIZE + numRanks_ - 1U) / numRanks_;
    if (channelsPerRank_ == 0U) {
        channelsPerRank_ = 1U;
    }

    numEntriesPerRank_ = tilingData->numEntriesPerRank;
    numTokens_ = tilingData->numTokens;
    hiddenBytes_ = tilingData->hiddenBytes;
    ubSize_ = tilingData->ubSize;

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    AscendC::LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(relayQue_, RELAY_BUFFER_NUM, TILE_BYTES);

    uint32_t countsBufSize = AscendC::Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankCountsBuf_, countsBufSize);
    uint32_t offsetsBufSize = AscendC::Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankOffsetsBuf_, offsetsBufSize);
    uint32_t commBufferBufSize = AscendC::Ceil(numRanks_ * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(commBufferBuf_, commBufferBufSize);
    uint32_t hcommHandleBufSize = AscendC::Ceil(numRanks_ * channelsPerRank_ * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(hcommHandleBuf_, hcommHandleBufSize);

    constexpr uint64_t ubReserved = 8U * 1024U;
    // ScatterByRank 需要: indices + tokenIdxInRank + rankIDs + positions + divisor (各4字节) + mask(每元素1字节近似)
    uint32_t bytesPerIndice = sizeof(int32_t) + sizeof(uint32_t) + sizeof(int32_t) * 3U + 1U;
    uint64_t usedUb = HCOMM_INIT_SIZE + TILE_BYTES * RELAY_BUFFER_NUM + countsBufSize + offsetsBufSize +
                      commBufferBufSize + hcommHandleBufSize;
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

__aicore__ inline void EngramFetchArch35::CopyContextToUb()
{
    AscendC::LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    AscendC::GlobalTensor<uint64_t> commBufferGm;
    commBufferGm.SetGlobalBuffer((__gm__ uint64_t *)&ctxPtr_->commBuffer[0]);
    AscendC::DataCopyExtParams cpComm{1U, numRanks_ * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint64_t> padComm{false, 0, 0, 0};
    AscendC::DataCopyPad(commBufferLocal, commBufferGm, cpComm, padComm);

    AscendC::LocalTensor<uint64_t> hcommHandleLocal = hcommHandleBuf_.Get<uint64_t>();
    AscendC::GlobalTensor<uint64_t> hcommHandleGm;
    hcommHandleGm.SetGlobalBuffer((__gm__ uint64_t *)&ctxPtr_->hcommHandle[0]);
    AscendC::DataCopyExtParams cpHandle{1U, numRanks_ * channelsPerRank_ * static_cast<uint32_t>(sizeof(uint64_t)), 0U,
                                        0U, 0U};
    AscendC::DataCopyPadExtParams<uint64_t> padHandle{false, 0, 0, 0};
    AscendC::DataCopyPad(hcommHandleLocal, hcommHandleGm, cpHandle, padHandle);
}

__aicore__ inline void EngramFetchArch35::CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen)
{
    AscendC::GlobalTensor<int32_t> indicesGlobal;
    indicesGlobal.SetGlobalBuffer((__gm__ int32_t *)indicesGM_);
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    AscendC::DataCopyExtParams params{1U, indicesBatchLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(indicesLocal, indicesGlobal[indicesBatchStart], params, pad);
}

__aicore__ inline void EngramFetchArch35::GatherRankTokens(uint32_t ownerRank, uint32_t batchLen,
                                                           uint32_t compareCntMax, uint32_t &runningOffset)
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
    uint32_t totalBlocks = AscendC::GetBlockNum();
    if (totalBlocks >= numRanks_) {
        auto coreAssign = GetCoreAssignment(totalBlocks, aivId_, numRanks_);
        GatherRankTokens(coreAssign.assignedRank, batchLen, compareCntMax, runningOffset);
    } else {
        for (uint32_t ownerRank = aivId_; ownerRank < numRanks_; ownerRank += totalBlocks) {
            GatherRankTokens(ownerRank, batchLen, compareCntMax, runningOffset);
        }
    }
}

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
            ScatterByRank(indicesBatchLen);
            FetchByRank(indicesBatchStart);
            indicesBatchStart += indicesBatchLen;
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
    for (uint32_t tokenPos = tokenOffset; tokenPos < cnt; tokenPos += tokenStride) {
        uint32_t i = tokenIdxInRank(rankStart + tokenPos);
        int32_t globalIdx = indicesLocal(i);
        uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - localIdxStart;
        uint64_t globalTokenIdx = indicesBatchStart + i;
        GM_ADDR dst = fetchedGM_ + globalTokenIdx * hiddenBytes;
        GM_ADDR src = (GM_ADDR)commBufferLocal(rankId_) + static_cast<uint64_t>(localEntryIdx) * hiddenBytes;
        LocalCopySlice(dst, src, hiddenBytes);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
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

    uint32_t idxStart = ownerRank * numEntriesPerRank;
    uint32_t rankStart = rankOffsets(ownerRank);
    uint32_t cnt = rankCounts(ownerRank);
    if (cnt == 0) {
        return;
    }
    uint64_t channelHandle = hcommHandleLocal(ownerRank * channelsPerRank_ + channelIdxInRank);
    GM_ADDR remoteBase = (GM_ADDR)commBufferLocal(ownerRank);
    uint32_t pendingReadCount = 0;
    uint32_t threshold = READ_COMMIT_FIRST_THRESHOLD;

    for (uint32_t tokenPos = channelIdxInRank; tokenPos < cnt; tokenPos += channelCount) {
        uint32_t i = tokenIdxInRank(rankStart + tokenPos);
        int32_t globalIdx = indicesLocal(i);
        uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - idxStart;
        uint64_t globalTokenIdx = indicesBatchStart + i;
        GM_ADDR dst = fetchedGM_ + globalTokenIdx * hiddenBytes;
        GM_ADDR remoteSrcAddr = remoteBase + static_cast<uint64_t>(localEntryIdx) * hiddenBytes;
        int32_t ret = hcomm_.ReadNbi<false>(channelHandle, dst, remoteSrcAddr, hiddenBytes);
        ascendc_assert(ret == 0, "Urma readNbi failed, ret=%d, ownerRank=%u", ret, ownerRank);
        pendingReadCount++;
        if (pendingReadCount >= threshold) {
            ret = hcomm_.Commit(channelHandle);
            ascendc_assert(ret == 0, "Urma commit failed, ret=%d, ownerRank=%u", ret, ownerRank);
            pendingReadCount = 0;
            threshold = READ_COMMIT_THRESHOLD;
        }
    }
    if (pendingReadCount > 0) {
        int32_t ret = hcomm_.Commit(channelHandle);
        ascendc_assert(ret == 0, "Urma commit failed, ret=%d, ownerRank=%u", ret, ownerRank);
    }
}

__aicore__ inline void EngramFetchArch35::FetchByRank(uint32_t indicesBatchStart)
{
    uint32_t totalBlocks = AscendC::GetBlockNum();
    if (totalBlocks >= numRanks_) {
        auto coreAssign = GetCoreAssignment(totalBlocks, aivId_, numRanks_);
        if (coreAssign.assignedRank == rankId_) {
            LocalFetchTokens(indicesBatchStart, coreAssign.idxInRankGroup, coreAssign.rankGroupSize);
        } else if (coreAssign.idxInRankGroup < channelsPerRank_) {
            uint32_t effectiveStride =
                (coreAssign.rankGroupSize < channelsPerRank_) ? coreAssign.rankGroupSize : channelsPerRank_;
            RemoteFetchRank(coreAssign.assignedRank, indicesBatchStart, coreAssign.idxInRankGroup, effectiveStride);
        }
    } else {
        uint32_t startRank = (aivId_ < numRanks_) ? aivId_ : rankId_;
        for (uint32_t ownerRank = startRank; ownerRank < numRanks_; ownerRank += totalBlocks) {
            if (ownerRank == rankId_) {
                LocalFetchTokens(indicesBatchStart, 0, 1);
            } else {
                RemoteFetchRank(ownerRank, indicesBatchStart, 0, 1);
            }
        }
    }
}

__aicore__ inline void EngramFetchArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    AscendC::GlobalTensor<uint8_t> srcGm;
    AscendC::GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);

    uint32_t tileLen = TILE_BYTES;
    uint64_t off = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > TILE_BYTES) ? tileLen : (len - off);

        AscendC::LocalTensor<uint8_t> tmp = relayQue_.AllocTensor<uint8_t>();
        AscendC::DataCopy(tmp, srcGm[off], thisLen);
        relayQue_.EnQue<uint8_t>(tmp);
        tmp = relayQue_.DeQue<uint8_t>();
        AscendC::DataCopy(dstGm[off], tmp, thisLen);
        relayQue_.FreeTensor<uint8_t>(tmp);

        off += thisLen;
    }
}
#endif

} // namespace Mc2Kernel

#endif
