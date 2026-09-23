/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_FETCH_GRAD_UNIQUE_H
#define ENGRAM_FETCH_GRAD_UNIQUE_H

#include "kernel_operator.h"
#include "adv_api/reduce/reduce.h"
#include "../engram_fetch_grad_utils.h"

namespace EngramFetchGradUnique {

constexpr uint32_t ENTRY_BATCH_CAP_HALF = 2U;
constexpr uint32_t RUN_START_UB_OFFSET = 2U;
constexpr uint32_t RUN_LEN_UB_OFFSET = 2U;
constexpr uint32_t ACCUM_LIST_UB_OFFSET = 4U;
constexpr uint32_t COMPACT_INDEX_UB_OFFSET = 3U;
constexpr uint32_t DIRECT_FLAG_UB_OFFSET = 4U;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc(AscendC::TPipe &pipe)
{
    int32_t eventID = static_cast<int32_t>(pipe.FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

static __aicore__ inline uint32_t GetDtypeSize(int32_t dtype)
{
    if (dtype == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
        return sizeof(bfloat16_t);
    }
    if (dtype == Mc2Kernel::ENGRAM_DT_FLOAT16) {
        return sizeof(half);
    }
    return sizeof(float);
}

class EngramFetchGradUnique {
public:
    __aicore__ inline EngramFetchGradUnique() = default;

    __aicore__ inline void Init(uint32_t aivId, uint32_t totalBlocks, uint32_t rankId, uint32_t numRanks,
                                int32_t numEntriesPerRank, int64_t hiddenDim, int64_t hiddenBytes, int32_t inputDtype,
                                int32_t outputDtype, AscendC::TPipe *pipe, AscendC::TBuf<> &entryBuf,
                                AscendC::TBuf<> &gradBuf, AscendC::TBuf<> &indicesBuf, AscendC::TBuf<> &tempBuf,
                                AscendC::TBuf<> &statusBuf, AscendC::TBuf<> &castBuf, AscendC::TBuf<> &accumBuf);

    __aicore__ inline void ZeroGradUnique(uint32_t numRecv, GM_ADDR gradUniqueOutGM);

    __aicore__ inline void CountUniquesParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR coreStartGM,
                                                GM_ADDR segCountGM);
    __aicore__ inline void RunScatterCast(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM,
                                          GM_ADDR numUniqueOutGM, GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM,
                                          GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM);

    __aicore__ inline void WriteNumUniqueZero(GM_ADDR numUniqueOutGM);

    __aicore__ inline void SetCastBuf(AscendC::TBuf<> &castBuf)
    {
        castBuf_ = &castBuf;
    }
    __aicore__ inline void SetAccumBuf(AscendC::TBuf<> &accumBuf)
    {
        accumBuf_ = &accumBuf;
    }
    __aicore__ inline void SetGradSubBatch(uint32_t batch)
    {
        gradSubBatch_ = batch;
    }
    __aicore__ inline void SetEntryBufBytes(uint32_t bytes)
    {
        entryBufBytes_ = bytes;
    }
    __aicore__ inline void SetChunkElems(uint32_t elems)
    {
        chunkElems_ = elems;
        if (chunkElems_ > 0U) {
            uint32_t inSize = GetDtypeSize(inputDtype_);
            uint32_t outSize = GetDtypeSize(outputDtype_);
            chunkInStride_ =
                (chunkElems_ * inSize + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
            chunkOutStride_ =
                (chunkElems_ * outSize + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
            chunkFp32Stride_ =
                (chunkElems_ * sizeof(float) + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
            numChunks_ = (static_cast<uint32_t>(hiddenDim_) + chunkElems_ - 1U) / chunkElems_;
            accumFloats_ = chunkFp32Stride_ / sizeof(float);
        } else {
            numChunks_ = 0U;
        }
    }

    // entryBuf_ int32 slot layout: one ENTRY_BATCH_CAP-sized slot per array.
    __aicore__ inline AscendC::LocalTensor<int32_t> CompUb()
    {
        return entryBuf_->Get<int32_t>();
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> UniqueUb()
    {
        return entryBuf_->Get<int32_t>()[Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> CompactIdxUb()
    {
        return entryBuf_->Get<int32_t>()[COMPACT_INDEX_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> DirectFlagUb()
    {
        return entryBuf_->Get<int32_t>()[DIRECT_FLAG_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> RunStartUb()
    {
        return entryBuf_->Get<int32_t>()[RUN_START_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> RunLenUb()
    {
        return entryBuf_->Get<int32_t>()[RUN_LEN_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP +
                                         Mc2Kernel::ENTRY_BATCH_CAP / ENTRY_BATCH_CAP_HALF];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> AccumListUb()
    {
        return entryBuf_->Get<int32_t>()[ACCUM_LIST_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    }

private:
    __aicore__ inline void ScatterAccumulateParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                     GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
                                                     GM_ADDR recvGradGM, GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                     GM_ADDR sortCompanionGM);
    __aicore__ inline void WriteNumUnique(GM_ADDR segCountGM, GM_ADDR numUniqueOutGM);
    __aicore__ inline void LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, GM_ADDR segCountGM, uint32_t &start,
                                         uint32_t &end, int32_t &preCoreOffset);
    __aicore__ inline uint32_t ProcessScatterBatch(uint32_t cur, uint32_t end, int32_t &runningOffset,
                                                   int32_t &runningUniqueOffset, int32_t &prevEntry,
                                                   bool &isFirstElement, GM_ADDR recvLocalEntryOutGM,
                                                   GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
                                                   GM_ADDR recvGradGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void ChunkedScatterRange(uint32_t start, uint32_t end, int32_t preCoreOffset,
                                               GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM,
                                               GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void ChunkColumnPass(uint32_t chunkIdx, uint32_t start, uint32_t end, int32_t preCoreOffset,
                                           GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM,
                                           GM_ADDR recvGradGM, GM_ADDR sortCompanionGM, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void ProcessChunkTile(uint32_t subStart, uint32_t subLen, uint32_t chunkIdx,
                                            uint32_t chunkSrcElems, AscendC::LocalTensor<uint8_t> &gradRaw,
                                            uint32_t &accumCursor, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void ProcessAccumRowChunk(int32_t pos, uint32_t subStart, uint32_t chunkIdx,
                                                uint32_t chunkSrcElems, AscendC::LocalTensor<uint8_t> &gradRaw,
                                                AscendC::LocalTensor<float> &castChunk, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void FlushAccumChunk(uint32_t chunkIdx, GM_ADDR gradUniqueOutGM);
    __aicore__ inline uint32_t ProcessBatchUnique(uint32_t cur, uint32_t batchLen, int32_t runningOffset,
                                                  int32_t &prevEntry, bool &isFirstElement, int32_t &inclusiveSum,
                                                  GM_ADDR recvLocalEntryOutGM, GM_ADDR sortCompanionGM, bool emitLists);
    __aicore__ inline void WriteBatchUnique(uint32_t tileUniqueCnt, int32_t &runningUniqueOffset,
                                            GM_ADDR uniqueLocalEntryOutGM);

    __aicore__ inline void FlushAccum(GM_ADDR gradUniqueOutGM);
    __aicore__ inline void FlushDirect(AscendC::LocalTensor<uint8_t> &gradRaw, uint32_t elemIdx, int32_t compactIdx,
                                       GM_ADDR gradUniqueOutGM);
    __aicore__ inline void AccumulateSubBatch(AscendC::LocalTensor<float> &gradFp32, uint32_t rowStrideFloats,
                                              uint32_t subStart, uint32_t subLen, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void AccumulateDirectSubBatch(AscendC::LocalTensor<uint8_t> &gradRaw, uint32_t subStart,
                                                    uint32_t subLen, AscendC::LocalTensor<float> &castRow,
                                                    GM_ADDR gradUniqueOutGM);
    __aicore__ inline void ProcessDirectSubBatchLoop(uint32_t batchLen, uint32_t maxGradPerBatch,
                                                     AscendC::LocalTensor<uint8_t> &gradBase, GM_ADDR recvGradGM,
                                                     GM_ADDR gradUniqueOutGM);
    __aicore__ inline void ProcessNonDirectRow(int32_t pos, uint32_t subStart, AscendC::LocalTensor<uint8_t> &gradRaw,
                                               AscendC::LocalTensor<float> &castRow, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void AccumulateDupListWalk(uint32_t subStart, uint32_t subLen, uint32_t &cursor,
                                                 AscendC::LocalTensor<uint8_t> &gradRaw,
                                                 AscendC::LocalTensor<float> &castRow, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void EmitRunsFromDupList(uint32_t subStart, uint32_t subLen, uint32_t &cursor,
                                               AscendC::LocalTensor<uint8_t> &gradRaw, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void ProcessDirectSubBatchLoopV2(uint32_t batchLen, uint32_t maxGradPerBatch,
                                                       AscendC::LocalTensor<uint8_t> &gradBase, GM_ADDR recvGradGM,
                                                       GM_ADDR gradUniqueOutGM);
    __aicore__ inline void FlushDirectRun(AscendC::LocalTensor<uint8_t> &gradRaw, uint32_t runStart, uint32_t runLen,
                                          int32_t compactFirst, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void CastToFP32(AscendC::LocalTensor<float> outT, AscendC::LocalTensor<uint8_t> gradRaw,
                                      uint32_t count);

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    int32_t numEntriesPerRank_{0};
    int64_t hiddenDim_{0};
    int64_t hiddenBytes_{0};
    uint32_t inRowStride_{0};   // GM 行字节数向 32B 上取整（UB 内行排布 stride）
    uint32_t fp32RowStride_{0}; // fp32 行字节数向 32B 上取整
    uint32_t chunkElems_{0};
    uint32_t chunkInStride_{0};
    uint32_t chunkOutStride_{0};
    uint32_t chunkFp32Stride_{0};
    uint32_t numChunks_{0};
    uint32_t numRecv_{0};
    int32_t inputDtype_{0};
    int32_t outputDtype_{0};
    AscendC::TPipe *pipe_{nullptr};
    AscendC::TBuf<> *entryBuf_{nullptr};
    AscendC::TBuf<> *gradBuf_{nullptr};
    AscendC::TBuf<> *indicesBuf_{nullptr};
    AscendC::TBuf<> *tempBuf_{nullptr};
    AscendC::TBuf<> *statusBuf_{nullptr};
    AscendC::TBuf<> *castBuf_{nullptr};
    AscendC::TBuf<> *accumBuf_{nullptr};
    uint32_t gradSubBatch_{Mc2Kernel::GRAD_SUB_BATCH};
    uint32_t entryBufBytes_{Mc2Kernel::ENTRY_BUF_BYTES};

    int32_t myPreCoreOffset_{0};
    int32_t mySegCount_{0};

    int32_t accumCompactIdx_{-1};
    bool accumDirty_{false};
    uint32_t accumFloats_{0};
    uint32_t accumIdx_{0};
    bool flushPending_[2]{false, false};
    int32_t flushEvtVMte3Arr_[2]{0, 0};
    int32_t flushEvtMte3VArr_[2]{0, 0};
    uint32_t runCnt_{0};      // 本批 direct-run 数
    uint32_t accumCnt_{0};    // 本批 accum 行数(≤ENTRY_BATCH_CAP)
    uint32_t accumRowCnt_{0}; // 当前 accum 组已累加行数
};

__aicore__ inline void EngramFetchGradUnique::Init(uint32_t aivId, uint32_t totalBlocks, uint32_t rankId,
                                                   uint32_t numRanks, int32_t numEntriesPerRank, int64_t hiddenDim,
                                                   int64_t hiddenBytes, int32_t inputDtype, int32_t outputDtype,
                                                   AscendC::TPipe *pipe, AscendC::TBuf<> &entryBuf,
                                                   AscendC::TBuf<> &gradBuf, AscendC::TBuf<> &indicesBuf,
                                                   AscendC::TBuf<> &tempBuf, AscendC::TBuf<> &statusBuf,
                                                   AscendC::TBuf<> &castBuf, AscendC::TBuf<> &accumBuf)
{
    aivId_ = aivId;
    totalBlocks_ = totalBlocks;
    rankId_ = rankId;
    numRanks_ = numRanks;
    numEntriesPerRank_ = numEntriesPerRank;
    hiddenDim_ = hiddenDim;
    hiddenBytes_ = hiddenBytes;
    inRowStride_ =
        (static_cast<uint32_t>(hiddenBytes) + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
    fp32RowStride_ = (static_cast<uint32_t>(hiddenDim) * sizeof(float) + Mc2Kernel::UB_ALIGN - 1U) /
                     Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
    inputDtype_ = inputDtype;
    outputDtype_ = outputDtype;
    pipe_ = pipe;
    entryBuf_ = &entryBuf;
    gradBuf_ = &gradBuf;
    indicesBuf_ = &indicesBuf;
    tempBuf_ = &tempBuf;
    statusBuf_ = &statusBuf;
    castBuf_ = &castBuf;
    accumBuf_ = &accumBuf;
    accumFloats_ = (static_cast<uint32_t>(hiddenDim) * sizeof(float) + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN *
                   Mc2Kernel::UB_ALIGN / sizeof(float);
}

__aicore__ inline void EngramFetchGradUnique::RunScatterCast(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                             GM_ADDR uniqueLocalEntryOutGM, GM_ADDR numUniqueOutGM,
                                                             GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM,
                                                             GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                             GM_ADDR sortCompanionGM)
{
    flushEvtVMte3Arr_[0] = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE3));
    flushEvtVMte3Arr_[1] = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE3));
    flushEvtMte3VArr_[0] = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_V));
    flushEvtMte3VArr_[1] = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_V));
    numRecv_ = numRecv;
    accumIdx_ = 0;
    flushPending_[0] = false;
    flushPending_[1] = false;

    ScatterAccumulateParallel(numRecv, recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM, recvGradGM,
                              coreStartGM, segCountGM, sortCompanionGM);

    if (accumDirty_) {
        FlushAccum(gradUniqueOutGM);
        accumDirty_ = false;
    }

    for (uint32_t i = 0; i < 2U; i++) {
        if (flushPending_[i]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[i]);
            flushPending_[i] = false;
        }
    }

    WriteNumUnique(segCountGM, numUniqueOutGM);

    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[0]);
    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[1]);
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[0]);
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[1]);
}

__aicore__ inline void EngramFetchGradUnique::WriteNumUniqueZero(GM_ADDR numUniqueOutGM)
{
    if (aivId_ != 0) {
        AscendC::SyncAll<true>();
        return;
    }
    AscendC::GlobalTensor<int32_t> numUniqueGM;
    numUniqueGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM);
    AscendC::LocalTensor<int32_t> tmp = statusBuf_->Get<int32_t>();
    tmp.SetValue(0, 0);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::DataCopyParams params = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(numUniqueGM, tmp, params);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    AscendC::SyncAll<true>();
}

__aicore__ inline void EngramFetchGradUnique::ZeroGradUnique(uint32_t numRecv, GM_ADDR gradUniqueOutGM)
{
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint64_t totalBytes = static_cast<uint64_t>(numRecv) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize;
    uint64_t chunk = (totalBytes + totalBlocks_ - 1U) / totalBlocks_;
    uint64_t start = static_cast<uint64_t>(aivId_) * chunk;
    uint64_t end = start + chunk;
    if (end > totalBytes) {
        end = totalBytes;
    }

    if (start < end) {
        AscendC::LocalTensor<float> zeroBuf = gradBuf_->Get<float>();
        uint32_t tileFloats = Mc2Kernel::TILE_BYTES / sizeof(float);
        constexpr uint32_t maxBlockFloats = Mc2Kernel::MAX_BLOCK_BYTES / sizeof(float);
        if (tileFloats > maxBlockFloats) {
            tileFloats = maxBlockFloats;
        }
        AscendC::Duplicate<float>(zeroBuf, 0.0f, tileFloats);
        SyncFunc<AscendC::HardEvent::V_MTE3>(*pipe_);

        // Chunk stride must equal the zeroed span (tileFloats*4 <= 65532): DataCopyParams
        // blockLen is uint16_t, and TILE_BYTES(64KB) as a full chunk wraps to 0 in the
        // uint16 cast -> the zero pass silently wrote nothing for every full 64KB chunk
        // (regression from the TILE_BYTES 32K->64K change; masked only by pre-zeroed
        // output buffers, exposed by generalization cases).
        uint32_t chunkBytes = tileFloats * sizeof(float);
        AscendC::GlobalTensor<uint8_t> dstGM;
        dstGM.SetGlobalBuffer((__gm__ uint8_t *)gradUniqueOutGM);
        for (uint64_t off = start; off < end; off += chunkBytes) {
            uint64_t thisLen = end - off;
            if (thisLen > chunkBytes) {
                thisLen = chunkBytes;
            }
            AscendC::DataCopyParams params{1U, static_cast<uint16_t>(thisLen), 0U, 0U};
            AscendC::DataCopyPad(dstGM[off], zeroBuf.ReinterpretCast<uint8_t>(), params);
        }
    }
}

__aicore__ inline void EngramFetchGradUnique::CountUniquesParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                                   GM_ADDR coreStartGM, GM_ADDR segCountGM)
{
    uint32_t chunk = (numRecv + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t rawStart = aivId_ * chunk;
    uint32_t rawEnd = rawStart + chunk;
    if (rawEnd > numRecv) {
        rawEnd = numRecv;
    }

    AscendC::GlobalTensor<int32_t> sortedEntryGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM);

    uint32_t start = rawStart;
    int32_t boundaryEntry = 0;
    if (rawStart > 0 && rawStart < rawEnd) {
        AscendC::LocalTensor<int32_t> probeUb = tempBuf_->Get<int32_t>();
        AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        uint32_t probeBase = rawStart - 1U;
        start = rawEnd;
        while (probeBase < rawEnd) {
            uint32_t probeLen = rawEnd - probeBase;
            if (probeLen > Mc2Kernel::ENTRY_BATCH_CAP) {
                probeLen = Mc2Kernel::ENTRY_BATCH_CAP;
            }
            AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(probeLen * sizeof(int32_t)), 0U, 0U, 0U};
            AscendC::DataCopyPad(probeUb, sortedEntryGM[probeBase], cpParams, cpPad);
            SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

            boundaryEntry = probeUb.GetValue(0);
            uint32_t p = 1U;
            while (p < probeLen && probeUb.GetValue(p) == boundaryEntry) {
                p++;
            }
            if (p < probeLen) {
                start = probeBase + p;
                break;
            }
            if (probeBase + probeLen >= rawEnd) {
                break;
            }
            probeBase += probeLen - 1U;
        }
    }

    uint32_t localUniqueCount = 0;
    if (start < rawEnd) {
        AscendC::LocalTensor<int32_t> xUb = gradBuf_->Get<int32_t>();
        AscendC::LocalTensor<int32_t> yUb = xUb[Mc2Kernel::ENTRY_BATCH_CAP];
        AscendC::LocalTensor<int32_t> fUb = xUb[2 * Mc2Kernel::ENTRY_BATCH_CAP];
        AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        uint32_t cur = start;
        while (cur < rawEnd) {
            uint32_t batchLen = rawEnd - cur;
            if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
                batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
            }
            AscendC::DataCopyExtParams yParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
            AscendC::DataCopyPad(yUb, sortedEntryGM[cur], yParams, cpPad);
            bool isFirstBatch = (cur == 0U);
            uint32_t xBase = isFirstBatch ? (cur + 1U) : (cur - 1U);
            AscendC::DataCopyExtParams cpX{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
            AscendC::DataCopyPad(xUb, sortedEntryGM[xBase], cpX, cpPad);
            SyncFunc<AscendC::HardEvent::MTE2_V>(*pipe_);

            uint32_t cntLen = isFirstBatch ? (batchLen - 1U) : batchLen;
            if (cntLen > 0U) {
                if (isFirstBatch) {
                    AscendC::Sub<int32_t>(fUb, xUb, yUb, static_cast<int32_t>(cntLen));
                } else {
                    AscendC::Sub<int32_t>(fUb, yUb, xUb, static_cast<int32_t>(cntLen));
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins<int32_t>(fUb, fUb, 1, static_cast<int32_t>(cntLen));
                AscendC::PipeBarrier<PIPE_V>();

                const uint32_t rshape[] = {1U, cntLen};
                AscendC::LocalTensor<uint8_t> rsTmp = tempBuf_->Get<uint8_t>();
                AscendC::ReduceSum<int32_t, AscendC::Pattern::Reduce::AR>(xUb, fUb, rsTmp, rshape, false);
                SyncFunc<AscendC::HardEvent::V_S>(*pipe_);
                localUniqueCount += static_cast<uint32_t>(xUb.GetValue(0));
            }
            if (isFirstBatch) {
                localUniqueCount += 1U;
            }
            cur += batchLen;
        }
    }

    AscendC::LocalTensor<int32_t> cntUb = statusBuf_->Get<int32_t>();
    cntUb.SetValue(0, static_cast<int32_t>(start));
    cntUb.SetValue(Mc2Kernel::STATE_OFFSET / sizeof(int32_t), static_cast<int32_t>(localUniqueCount));
    mySegCount_ = static_cast<int32_t>(localUniqueCount);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
    AscendC::DataCopyParams cntParams{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(coreStartGMT[aivId_], cntUb, cntParams);
    AscendC::DataCopyPad(segCountGMT[aivId_], cntUb[Mc2Kernel::STATE_OFFSET / sizeof(int32_t)], cntParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
}
__aicore__ inline void EngramFetchGradUnique::ChunkedScatterRange(uint32_t start, uint32_t end, int32_t preCoreOffset,
                                                                  GM_ADDR recvLocalEntryOutGM,
                                                                  GM_ADDR uniqueLocalEntryOutGM,
                                                                  GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM,
                                                                  GM_ADDR sortCompanionGM)
{
    for (uint32_t c = 0U; c < numChunks_; c++) {
        accumCompactIdx_ = -1;
        accumDirty_ = false;
        accumRowCnt_ = 0U;
        ChunkColumnPass(c, start, end, preCoreOffset, recvLocalEntryOutGM, uniqueLocalEntryOutGM, recvGradGM,
                        sortCompanionGM, gradUniqueOutGM);
        if (accumDirty_) {
            FlushAccumChunk(c, gradUniqueOutGM);
            accumDirty_ = false;
        }
    }
}
__aicore__ inline void EngramFetchGradUnique::ChunkColumnPass(uint32_t chunkIdx, uint32_t start, uint32_t end,
                                                              int32_t preCoreOffset, GM_ADDR recvLocalEntryOutGM,
                                                              GM_ADDR uniqueLocalEntryOutGM, GM_ADDR recvGradGM,
                                                              GM_ADDR sortCompanionGM, GM_ADDR gradUniqueOutGM)
{
    uint32_t inDtypeSize = GetDtypeSize(inputDtype_);
    uint32_t chunkSrcElems =
        (chunkIdx + 1U == numChunks_) ? (static_cast<uint32_t>(hiddenDim_) - chunkIdx * chunkElems_) : chunkElems_;
    uint32_t chunkSrcBytes = chunkSrcElems * inDtypeSize;
    uint64_t srcByteOff = static_cast<uint64_t>(chunkIdx) * chunkElems_ * inDtypeSize;
    constexpr uint32_t kBufs = 4U;
    event_t evtMte2V[kBufs];
    event_t evtVMte2[kBufs];
    event_t evtMte2Mte3[kBufs];
    event_t evtMte3Mte2[kBufs];
    for (uint32_t b = 0U; b < kBufs; b++) {
        evtMte2V[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        evtVMte2[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
        evtMte2Mte3[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
        evtMte3Mte2[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    }
    AscendC::LocalTensor<uint8_t> gradBase = gradBuf_->Get<uint8_t>();
    constexpr uint32_t kBufBytes = Mc2Kernel::GRAD_PING_BYTES / 2U;
    uint32_t bufRows = kBufBytes / chunkInStride_;
    if (bufRows < 1U) {
        bufRows = 1U;
    }
    uint32_t maxRows = bufRows;
    if (maxRows > gradSubBatch_) {
        maxRows = gradSubBatch_;
    }
    if (maxRows < 1U) {
        maxRows = 1U;
    }
    AscendC::DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};
    int32_t runningOffset = preCoreOffset;
    int32_t runningUniqueOffset = preCoreOffset;
    int32_t prevEntry = 0;
    bool isFirstElement = true;
    if (start > 0U) {
        AscendC::GlobalTensor<int32_t> sortedEntryGM;
        sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM);
        AscendC::LocalTensor<int32_t> prevUb = tempBuf_->Get<int32_t>();
        AscendC::DataCopyPadExtParams<int32_t> prevPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams prevParams{1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(prevUb, sortedEntryGM[start - 1U], prevParams, prevPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
        prevEntry = prevUb.GetValue(0);
        isFirstElement = false;
    }
    uint32_t tileIdx = 0U;
    uint32_t cur = start;
    while (cur < end) {
        uint32_t batchLen = end - cur;
        if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
            batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
        }
        int32_t inclusiveSum = 0;
        uint32_t tileUniqueCnt = ProcessBatchUnique(cur, batchLen, runningOffset, prevEntry, isFirstElement,
                                                    inclusiveSum, recvLocalEntryOutGM, sortCompanionGM, true);
        if (chunkIdx == 0U && tileUniqueCnt > 0) {
            int32_t uniqOff = runningUniqueOffset;
            WriteBatchUnique(tileUniqueCnt, uniqOff, uniqueLocalEntryOutGM);
            runningUniqueOffset = uniqOff;
        }
        runningOffset += inclusiveSum;
        uint32_t accumCursor = 0U;
        for (uint32_t subStart = 0U; subStart < batchLen; subStart += maxRows) {
            uint32_t subLen = batchLen - subStart;
            if (subLen > maxRows) {
                subLen = maxRows;
            }
            uint32_t b = tileIdx % kBufs;
            AscendC::LocalTensor<uint8_t> gradRaw = gradBase[b * kBufBytes];
            if (tileIdx >= kBufs) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
            }
            for (uint32_t j = 0U; j < subLen; j++) {
                int32_t recvIdx = CompUb().GetValue(subStart + j);
                if (recvIdx < 0 || static_cast<uint32_t>(recvIdx) >= numRecv_) {
                    recvIdx = 0;
                }
                GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(recvIdx) * hiddenBytes_ + srcByteOff;
                AscendC::GlobalTensor<uint8_t> gradSrcGM;
                gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
                AscendC::DataCopyExtParams gradParams{1U, chunkSrcBytes, 0U, 0U, 0U};
                AscendC::DataCopyPad(gradRaw[static_cast<uint64_t>(j) * chunkInStride_], gradSrcGM, gradParams,
                                     gradPad);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V[b]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[b]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V[b]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[b]);
            ProcessChunkTile(subStart, subLen, chunkIdx, chunkSrcElems, gradRaw, accumCursor, gradUniqueOutGM);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
            tileIdx++;
        }
        cur += batchLen;
    }
    uint32_t drainFrom = (tileIdx > kBufs) ? (tileIdx - kBufs) : 0U;
    for (uint32_t t = drainFrom; t < tileIdx; t++) {
        uint32_t b = t % kBufs;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
    }
    for (uint32_t b = 0U; b < kBufs; b++) {
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evtMte2V[b]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[b]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
    }
}
__aicore__ inline void EngramFetchGradUnique::ProcessChunkTile(uint32_t subStart, uint32_t subLen, uint32_t chunkIdx,
                                                               uint32_t chunkSrcElems,
                                                               AscendC::LocalTensor<uint8_t> &gradRaw,
                                                               uint32_t &accumCursor, GM_ADDR gradUniqueOutGM)
{
    bool directAllowed = (inputDtype_ == outputDtype_);
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint32_t inDtypeSize = GetDtypeSize(inputDtype_);
    uint32_t chunkSrcBytes = chunkSrcElems * inDtypeSize;
    uint64_t outByteOff = static_cast<uint64_t>(chunkIdx) * chunkElems_ * outDtypeSize;
    AscendC::LocalTensor<float> castChunk = castBuf_->Get<float>();
    for (uint32_t j = 0U; j < subLen; j++) {
        uint32_t pos = subStart + j;
        bool isAccum;
        if (!directAllowed) {
            isAccum = true;
        } else {
            while (accumCursor < accumCnt_ && static_cast<uint32_t>(AccumListUb().GetValue(accumCursor)) < pos) {
                accumCursor++;
            }
            isAccum = (accumCursor < accumCnt_ && static_cast<uint32_t>(AccumListUb().GetValue(accumCursor)) == pos);
        }
        if (isAccum) {
            ProcessAccumRowChunk(static_cast<int32_t>(pos), subStart, chunkIdx, chunkSrcElems, gradRaw, castChunk,
                                 gradUniqueOutGM);
        } else {
            int32_t compactIdx = CompactIdxUb().GetValue(pos);
            if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
                continue;
            }
            uint64_t gmByteOffset =
                static_cast<uint64_t>(compactIdx) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize + outByteOff;
            AscendC::GlobalTensor<uint8_t> dstGM;
            dstGM.SetGlobalBuffer((__gm__ uint8_t *)(gradUniqueOutGM + gmByteOffset));
            AscendC::DataCopyParams params{1U, static_cast<uint16_t>(chunkSrcBytes), 0U, 0U};
            AscendC::DataCopyPad(dstGM, gradRaw[j * chunkInStride_], params);
        }
    }
}
__aicore__ inline void EngramFetchGradUnique::ProcessAccumRowChunk(int32_t pos, uint32_t subStart, uint32_t chunkIdx,
                                                                   uint32_t chunkSrcElems,
                                                                   AscendC::LocalTensor<uint8_t> &gradRaw,
                                                                   AscendC::LocalTensor<float> &castChunk,
                                                                   GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();
    int32_t compactIdx = CompactIdxUb().GetValue(pos);
    if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
        return;
    }
    if (compactIdx != accumCompactIdx_) {
        if (accumDirty_) {
            FlushAccumChunk(chunkIdx, gradUniqueOutGM);
            accumIdx_ ^= 1U;
            if (flushPending_[accumIdx_]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[accumIdx_]);
                flushPending_[accumIdx_] = false;
            }
        }
        accumCompactIdx_ = compactIdx;
        accumRowCnt_ = 0;
    }
    AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
    uint32_t rowOff = static_cast<uint32_t>(pos - static_cast<int32_t>(subStart)) * chunkInStride_;
    if (accumRowCnt_ == 0U) {
        if (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) {
            CastToFP32(accum, gradRaw[rowOff], chunkSrcElems);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::LocalTensor<float> gradFp32 = gradRaw[rowOff].ReinterpretCast<float>();
            AscendC::Adds<float>(accum, gradFp32, 0.0f, chunkSrcElems);
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else if (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) {
        CastToFP32(castChunk, gradRaw[rowOff], chunkSrcElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(accum, accum, castChunk, chunkSrcElems);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        AscendC::LocalTensor<float> gradFp32 = gradRaw[rowOff].ReinterpretCast<float>();
        AscendC::Add<float>(accum, accum, gradFp32, chunkSrcElems);
        AscendC::PipeBarrier<PIPE_V>();
    }
    accumRowCnt_++;
    accumDirty_ = true;
}
__aicore__ inline void EngramFetchGradUnique::FlushAccumChunk(uint32_t chunkIdx, GM_ADDR gradUniqueOutGM)
{
    uint32_t bufIdx = accumIdx_;
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();
    AscendC::LocalTensor<float> accum = accumBase[bufIdx * accumFloats_];
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint32_t chunkSrcElems =
        (chunkIdx + 1U == numChunks_) ? (static_cast<uint32_t>(hiddenDim_) - chunkIdx * chunkElems_) : chunkElems_;
    uint32_t chunkOutBytes = chunkSrcElems * outDtypeSize;
    if (accumCompactIdx_ < 0 || accumCompactIdx_ >= numEntriesPerRank_) {
        return;
    }
    uint64_t gmByteOffset = static_cast<uint64_t>(accumCompactIdx_) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize +
                            static_cast<uint64_t>(chunkIdx) * chunkElems_ * outDtypeSize;
    if (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
        AscendC::GlobalTensor<float> dstGM;
        dstGM.SetGlobalBuffer((__gm__ float *)(gradUniqueOutGM + gmByteOffset));
        AscendC::DataCopyParams params{1U, static_cast<uint16_t>(chunkOutBytes), 0U, 0U};
        AscendC::DataCopyPad(dstGM, accum, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
    } else {
        AscendC::LocalTensor<uint8_t> entryRaw = entryBuf_->Get<uint8_t>();
        uint32_t flushCastOffset = Mc2Kernel::FLUSH_CAST_HEAD_BYTES;
        uint32_t castTailBytes = flushCastOffset + 2U * chunkOutStride_;
        ascendc_assert(castTailBytes <= entryBufBytes_,
                       "FlushAccumChunk staging overflow: need %u bytes, entryBuf=%u bytes", castTailBytes,
                       entryBufBytes_);
        AscendC::LocalTensor<uint8_t> flushCastBuf = entryRaw[flushCastOffset + bufIdx * chunkOutStride_];
        if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
            AscendC::LocalTensor<bfloat16_t> outT = flushCastBuf.ReinterpretCast<bfloat16_t>();
            AscendC::Cast(outT, accum, AscendC::RoundMode::CAST_RINT, chunkSrcElems);
        } else {
            AscendC::LocalTensor<half> outT = flushCastBuf.ReinterpretCast<half>();
            AscendC::Cast(outT, accum, AscendC::RoundMode::CAST_RINT, chunkSrcElems);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
        AscendC::GlobalTensor<uint8_t> dstGM;
        dstGM.SetGlobalBuffer((__gm__ uint8_t *)(gradUniqueOutGM + gmByteOffset));
        AscendC::DataCopyParams params{1U, static_cast<uint16_t>(chunkOutBytes), 0U, 0U};
        AscendC::DataCopyPad(dstGM, flushCastBuf, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
    }
    flushPending_[bufIdx] = false;
}

__aicore__ inline void EngramFetchGradUnique::LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                            uint32_t &start, uint32_t &end, int32_t &preCoreOffset)
{
    AscendC::GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);

    AscendC::LocalTensor<int32_t> ub = tempBuf_->Get<int32_t>();
    uint32_t totalBytes = totalBlocks_ * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t segOffElems = (totalBytes + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN *
                           (Mc2Kernel::UB_ALIGN / static_cast<uint32_t>(sizeof(int32_t)));
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyExtParams params{1U, totalBytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(ub, coreStartGMT, params, pad);
    AscendC::LocalTensor<int32_t> segUb = ub[segOffElems];
    AscendC::DataCopyPad(segUb, segCountGMT, params, pad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    int32_t nextValidStart = -1;
    preCoreOffset = 0;
    for (int32_t core = static_cast<int32_t>(totalBlocks_) - 1; core >= 0; core--) {
        int32_t segCnt = segUb.GetValue(core);
        if (segCnt == 0) {
            if (nextValidStart >= 0) {
                ub.SetValue(core, nextValidStart);
            } else {
                ub.SetValue(core, static_cast<int32_t>(numRecv));
            }
        } else {
            nextValidStart = ub.GetValue(core);
        }
        if (static_cast<uint32_t>(core) < aivId_) {
            preCoreOffset += segCnt;
        }
    }

    start = static_cast<uint32_t>(ub.GetValue(aivId_));
    if (aivId_ == totalBlocks_ - 1U) {
        end = numRecv;
    } else {
        end = static_cast<uint32_t>(ub.GetValue(aivId_ + 1U));
    }

    if (end > numRecv || start > numRecv || start > end) {
        if (end > numRecv) {
            end = numRecv;
        }
        if (start > numRecv) {
            start = numRecv;
        }
        if (start > end) {
            start = end;
        }
    }
    myPreCoreOffset_ = preCoreOffset;
}

__aicore__ inline void EngramFetchGradUnique::FlushAccum(GM_ADDR gradUniqueOutGM)
{
    uint32_t bufIdx = accumIdx_;
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();
    AscendC::LocalTensor<float> accum = accumBase[bufIdx * accumFloats_];
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint64_t gmByteOffset = static_cast<uint64_t>(accumCompactIdx_) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize;
    if (accumCompactIdx_ < 0 || accumCompactIdx_ >= numEntriesPerRank_) {
        return;
    }

    if (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);

        AscendC::GlobalTensor<float> dstGM;
        dstGM.SetGlobalBuffer((__gm__ float *)(gradUniqueOutGM + gmByteOffset));
        uint32_t totalBytes = static_cast<uint32_t>(hiddenDim_) * sizeof(float);
        uint32_t numBlocks = (totalBytes + Mc2Kernel::MAX_BLOCK_BYTES - 1U) / Mc2Kernel::MAX_BLOCK_BYTES;
        for (uint32_t b = 0; b < numBlocks; b++) {
            uint32_t byteOffset = b * Mc2Kernel::MAX_BLOCK_BYTES;
            uint32_t elemOffset = byteOffset / sizeof(float);
            uint32_t remaining = totalBytes - byteOffset;
            uint16_t blkBytes =
                static_cast<uint16_t>(remaining > Mc2Kernel::MAX_BLOCK_BYTES ? Mc2Kernel::MAX_BLOCK_BYTES : remaining);
            AscendC::DataCopyParams params{1U, blkBytes, 0U, 0U};
            AscendC::DataCopyPad(dstGM[elemOffset], accum[elemOffset], params);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
    } else {
        // flushCastBuf MUST NOT live in castBuf_: castBuf_ is the fp32 grad staging
        // (castPingF/castPongF) for the in-flight sub-batch; casting the accumulated row
        // here overwrites gradFp32 row 0 and FlushAccum fires mid-AccumulateSubBatch
        // (at compactIdx switches, before Add reads gradFp32[0]) -> corrupted accum.
        // Use the unused tail of entryBuf_ (pingInt32 only occupies the first
        // 5*ENTRY_BATCH_CAP*4 bytes), as the pre-merge baseline did.
        AscendC::LocalTensor<uint8_t> entryRaw = entryBuf_->Get<uint8_t>();
        uint32_t flushCastOffset = Mc2Kernel::FLUSH_CAST_HEAD_BYTES;
        uint32_t castHalfBytes = (static_cast<uint32_t>(hiddenDim_) * outDtypeSize + Mc2Kernel::UB_ALIGN - 1U) /
                                 Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
        // 双缓冲借用区必须完整落在 entryBuf_ 尾部内（Host 侧已按 hiddenDim 上界拒绝超限 shape，此处兜底）
        uint32_t castTailBytes = flushCastOffset + 2U * castHalfBytes;
        ascendc_assert(castTailBytes <= entryBufBytes_,
                       "FlushAccum cast staging overflow: need %u bytes, entryBuf=%u bytes, hiddenDim=%u",
                       castTailBytes, entryBufBytes_, static_cast<uint32_t>(hiddenDim_));
        AscendC::LocalTensor<uint8_t> flushCastBuf = entryRaw[flushCastOffset + bufIdx * castHalfBytes];
        uint32_t castCount = static_cast<uint32_t>(hiddenDim_);

        if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
            AscendC::LocalTensor<bfloat16_t> outT = flushCastBuf.ReinterpretCast<bfloat16_t>();
            AscendC::Cast(outT, accum, AscendC::RoundMode::CAST_RINT, castCount);
        } else {
            AscendC::LocalTensor<half> outT = flushCastBuf.ReinterpretCast<half>();
            AscendC::Cast(outT, accum, AscendC::RoundMode::CAST_RINT, castCount);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);

        if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
            AscendC::GlobalTensor<bfloat16_t> dstGM;
            dstGM.SetGlobalBuffer((__gm__ bfloat16_t *)(gradUniqueOutGM + gmByteOffset));
            uint32_t totalBytes = static_cast<uint32_t>(hiddenDim_) * sizeof(bfloat16_t);
            uint32_t numBlocks = (totalBytes + Mc2Kernel::MAX_BLOCK_BYTES - 1U) / Mc2Kernel::MAX_BLOCK_BYTES;
            AscendC::LocalTensor<bfloat16_t> outT = flushCastBuf.ReinterpretCast<bfloat16_t>();
            for (uint32_t b = 0; b < numBlocks; b++) {
                uint32_t byteOffset = b * Mc2Kernel::MAX_BLOCK_BYTES;
                uint32_t elemOffset = byteOffset / sizeof(bfloat16_t);
                uint32_t remaining = totalBytes - byteOffset;
                uint16_t blkBytes = static_cast<uint16_t>(
                    remaining > Mc2Kernel::MAX_BLOCK_BYTES ? Mc2Kernel::MAX_BLOCK_BYTES : remaining);
                AscendC::DataCopyParams params{1U, blkBytes, 0U, 0U};
                AscendC::DataCopyPad(dstGM[elemOffset], outT[elemOffset], params);
            }
        } else {
            AscendC::GlobalTensor<half> dstGM;
            dstGM.SetGlobalBuffer((__gm__ half *)(gradUniqueOutGM + gmByteOffset));
            uint32_t totalBytes = static_cast<uint32_t>(hiddenDim_) * sizeof(half);
            uint32_t numBlocks = (totalBytes + Mc2Kernel::MAX_BLOCK_BYTES - 1U) / Mc2Kernel::MAX_BLOCK_BYTES;
            AscendC::LocalTensor<half> outT = flushCastBuf.ReinterpretCast<half>();
            for (uint32_t b = 0; b < numBlocks; b++) {
                uint32_t byteOffset = b * Mc2Kernel::MAX_BLOCK_BYTES;
                uint32_t elemOffset = byteOffset / sizeof(half);
                uint32_t remaining = totalBytes - byteOffset;
                uint16_t blkBytes = static_cast<uint16_t>(
                    remaining > Mc2Kernel::MAX_BLOCK_BYTES ? Mc2Kernel::MAX_BLOCK_BYTES : remaining);
                AscendC::DataCopyParams params{1U, blkBytes, 0U, 0U};
                AscendC::DataCopyPad(dstGM[elemOffset], outT[elemOffset], params);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
    }
    flushPending_[bufIdx] = true;
}

__aicore__ inline void EngramFetchGradUnique::FlushDirect(AscendC::LocalTensor<uint8_t> &gradRaw, uint32_t elemIdx,
                                                          int32_t compactIdx, GM_ADDR gradUniqueOutGM)
{
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint64_t gmByteOffset = static_cast<uint64_t>(compactIdx) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize;

    AscendC::GlobalTensor<uint8_t> dstGM;
    dstGM.SetGlobalBuffer((__gm__ uint8_t *)(gradUniqueOutGM + gmByteOffset));
    uint32_t srcOffset = elemIdx * inRowStride_;
    uint32_t totalBytes = static_cast<uint32_t>(hiddenBytes_);
    uint32_t numBlocks = (totalBytes + Mc2Kernel::MAX_BLOCK_BYTES - 1U) / Mc2Kernel::MAX_BLOCK_BYTES;
    for (uint32_t b = 0; b < numBlocks; b++) {
        uint32_t byteOffset = b * Mc2Kernel::MAX_BLOCK_BYTES;
        uint32_t remaining = totalBytes - byteOffset;
        uint16_t blkBytes =
            static_cast<uint16_t>(remaining > Mc2Kernel::MAX_BLOCK_BYTES ? Mc2Kernel::MAX_BLOCK_BYTES : remaining);
        AscendC::DataCopyParams params{1U, blkBytes, 0U, 0U};
        AscendC::DataCopyPad(dstGM[byteOffset], gradRaw[srcOffset + byteOffset], params);
    }
}

__aicore__ inline void EngramFetchGradUnique::CastToFP32(AscendC::LocalTensor<float> outT,
                                                         AscendC::LocalTensor<uint8_t> gradRaw, uint32_t count)
{
    if (inputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
        AscendC::LocalTensor<bfloat16_t> inT = gradRaw.ReinterpretCast<bfloat16_t>();
        AscendC::Cast(outT, inT, AscendC::RoundMode::CAST_NONE, count);
    } else {
        AscendC::LocalTensor<half> inT = gradRaw.ReinterpretCast<half>();
        AscendC::Cast(outT, inT, AscendC::RoundMode::CAST_NONE, count);
    }
}

__aicore__ inline void EngramFetchGradUnique::AccumulateSubBatch(AscendC::LocalTensor<float> &gradFp32,
                                                                 uint32_t rowStrideFloats, uint32_t subStart,
                                                                 uint32_t subLen, GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();

    bool canDirectCopy = (inputDtype_ == outputDtype_);

    for (uint32_t j = 0; j < subLen; j++) {
        if (canDirectCopy && DirectFlagUb().GetValue(subStart + j) != 0) {
            continue;
        }
        int32_t compactIdx = CompactIdxUb().GetValue(subStart + j);
        if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
            continue;
        }
        if (compactIdx != accumCompactIdx_) {
            if (accumDirty_) {
                FlushAccum(gradUniqueOutGM);
                accumIdx_ ^= 1U;
                if (flushPending_[accumIdx_]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[accumIdx_]);
                    flushPending_[accumIdx_] = false;
                }
            }
            AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
            AscendC::Duplicate<float>(accum, 0.0f, static_cast<uint32_t>(hiddenDim_));
            AscendC::PipeBarrier<PIPE_V>();
            accumCompactIdx_ = compactIdx;
        }
        AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
        AscendC::Add<float>(accum, accum, gradFp32[j * rowStrideFloats], static_cast<uint32_t>(hiddenDim_));
        AscendC::PipeBarrier<PIPE_V>();
        accumDirty_ = true;
    }
}

__aicore__ inline void EngramFetchGradUnique::AccumulateDirectSubBatch(AscendC::LocalTensor<uint8_t> &gradRaw,
                                                                       uint32_t subStart, uint32_t subLen,
                                                                       AscendC::LocalTensor<float> &castRow,
                                                                       GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();
    uint32_t hiddenDim = static_cast<uint32_t>(hiddenDim_);

    for (uint32_t j = 0; j < subLen; j++) {
        if (DirectFlagUb().GetValue(subStart + j) != 0) {
            continue;
        }
        int32_t compactIdx = CompactIdxUb().GetValue(subStart + j);
        if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
            continue;
        }
        if (compactIdx != accumCompactIdx_) {
            if (accumDirty_) {
                FlushAccum(gradUniqueOutGM);
                accumIdx_ ^= 1U;
                if (flushPending_[accumIdx_]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[accumIdx_]);
                    flushPending_[accumIdx_] = false;
                }
            }
            AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
            AscendC::Duplicate<float>(accum, 0.0f, hiddenDim);
            AscendC::PipeBarrier<PIPE_V>();
            accumCompactIdx_ = compactIdx;
        }
        AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
        CastToFP32(castRow, gradRaw[j * inRowStride_], hiddenDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(accum, accum, castRow, hiddenDim);
        AscendC::PipeBarrier<PIPE_V>();
        accumDirty_ = true;
    }
}

__aicore__ inline void EngramFetchGradUnique::FlushDirectRun(AscendC::LocalTensor<uint8_t> &gradRaw, uint32_t runStart,
                                                             uint32_t runLen, int32_t compactFirst,
                                                             GM_ADDR gradUniqueOutGM)
{
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint64_t gmByteOffset = static_cast<uint64_t>(compactFirst) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize;
    AscendC::GlobalTensor<uint8_t> dstGM;
    dstGM.SetGlobalBuffer((__gm__ uint8_t *)(gradUniqueOutGM + gmByteOffset));
    uint32_t totalBytes = runLen * static_cast<uint32_t>(hiddenBytes_);
    AscendC::DataCopyParams params{1U, static_cast<uint16_t>(totalBytes), 0U, 0U};
    AscendC::DataCopyPad(dstGM, gradRaw[runStart * inRowStride_], params);
}
__aicore__ inline void EngramFetchGradUnique::ProcessDirectSubBatchLoop(uint32_t batchLen, uint32_t maxGradPerBatch,
                                                                        AscendC::LocalTensor<uint8_t> &gradBase,
                                                                        GM_ADDR recvGradGM, GM_ADDR gradUniqueOutGM)
{
    constexpr uint32_t kBufs = 4U;
    event_t evtMte2V[kBufs];
    event_t evtVMte2[kBufs];
    event_t evtMte2Mte3[kBufs];
    event_t evtMte3Mte2[kBufs];
    for (uint32_t b = 0; b < kBufs; b++) {
        evtMte2V[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        evtVMte2[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
        evtMte2Mte3[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
        evtMte3Mte2[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    }
    bool needCast = (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT);
    bool packed = (inRowStride_ == static_cast<uint32_t>(hiddenBytes_));
    constexpr uint32_t kBufBytes = Mc2Kernel::GRAD_PING_BYTES / 2U;
    uint32_t bufRows = kBufBytes / inRowStride_;
    if (bufRows < 1U) {
        bufRows = 1U;
    }
    if (bufRows < maxGradPerBatch) {
        maxGradPerBatch = bufRows;
    }
    AscendC::LocalTensor<float> castRow = castBuf_->Get<float>();
    AscendC::DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};

    uint32_t tileIdx = 0;
    for (uint32_t subStart = 0; subStart < batchLen; subStart += maxGradPerBatch) {
        uint32_t subLen = batchLen - subStart;
        if (subLen > maxGradPerBatch) {
            subLen = maxGradPerBatch;
        }
        uint32_t b = tileIdx % kBufs;
        AscendC::LocalTensor<uint8_t> gradRaw = gradBase[b * kBufBytes];

        if (tileIdx >= kBufs) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
        }

        for (uint32_t j = 0; j < subLen; j++) {
            int32_t recvIdx = CompUb().GetValue(subStart + j);
            if (recvIdx < 0 || static_cast<uint32_t>(recvIdx) >= numRecv_) {
                recvIdx = 0;
            }
            GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(recvIdx) * hiddenBytes_;
            AscendC::DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
            AscendC::GlobalTensor<uint8_t> gradSrcGM;
            gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
            AscendC::DataCopyPad(gradRaw[static_cast<uint64_t>(j) * inRowStride_], gradSrcGM, gradParams, gradPad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V[b]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[b]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V[b]);
        if (needCast) {
            AccumulateDirectSubBatch(gradRaw, subStart, subLen, castRow, gradUniqueOutGM);
        } else {
            AscendC::LocalTensor<float> gradFp32 = gradRaw.ReinterpretCast<float>();
            AccumulateSubBatch(gradFp32, inRowStride_ / sizeof(float), subStart, subLen, gradUniqueOutGM);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[b]);
        if (packed) {
            uint32_t j = 0;
            while (j < subLen) {
                if (DirectFlagUb().GetValue(subStart + j) != 0) {
                    uint32_t runStart = j;
                    j++;
                    while (j < subLen && DirectFlagUb().GetValue(subStart + j) != 0) {
                        j++;
                    }
                    int32_t compactFirst = CompactIdxUb().GetValue(subStart + runStart);
                    if (compactFirst >= 0 && compactFirst < numEntriesPerRank_) {
                        FlushDirectRun(gradRaw, runStart, j - runStart, compactFirst, gradUniqueOutGM);
                    }
                } else {
                    j++;
                }
            }
        } else {
            for (uint32_t j = 0; j < subLen; j++) {
                if (DirectFlagUb().GetValue(subStart + j) != 0) {
                    int32_t compactIdx = CompactIdxUb().GetValue(subStart + j);
                    if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
                        continue;
                    }
                    FlushDirect(gradRaw, j, compactIdx, gradUniqueOutGM);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
        tileIdx++;
    }

    uint32_t drainFrom = (tileIdx > kBufs) ? (tileIdx - kBufs) : 0U;
    for (uint32_t t = drainFrom; t < tileIdx; t++) {
        uint32_t b = t % kBufs;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
    }
    for (uint32_t b = 0; b < kBufs; b++) {
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evtMte2V[b]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[b]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
    }
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessBatchUnique(uint32_t cur, uint32_t batchLen,
                                                                     int32_t runningOffset, int32_t &prevEntry,
                                                                     bool &isFirstElement, int32_t &inclusiveSum,
                                                                     GM_ADDR recvLocalEntryOutGM,
                                                                     GM_ADDR sortCompanionGM, bool emitLists)
{
    AscendC::LocalTensor<int32_t> entryUb = indicesBuf_->Get<int32_t>();

    AscendC::GlobalTensor<int32_t> sortedEntryGM;
    AscendC::GlobalTensor<int32_t> compGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM);
    compGM.SetGlobalBuffer((__gm__ int32_t *)sortCompanionGM);
    if (cur >= numRecv_) {
        return 0;
    }
    if (cur + batchLen > numRecv_) {
        batchLen = numRecv_ - cur;
    }
    AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    AscendC::DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
    AscendC::DataCopyPad(CompUb(), compGM[cur], cpParams, cpPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    inclusiveSum = 0;
    uint32_t tileUniqueCnt = 0;
    if (emitLists) {
        AscendC::LocalTensor<int32_t> uniqueT = UniqueUb();
        AscendC::LocalTensor<int32_t> compactT = CompactIdxUb();
        AscendC::LocalTensor<int32_t> runStartT = RunStartUb();
        AscendC::LocalTensor<int32_t> runLenT = RunLenUb();
        AscendC::LocalTensor<int32_t> accumT = AccumListUb();
        int32_t rankBase = static_cast<int32_t>(static_cast<int64_t>(rankId_) * numEntriesPerRank_);
        bool prevIsNewUnique = false;
        int32_t prevCompact = 0;
        uint32_t runCnt = 0;
        uint32_t accumCnt = 0;
        bool inRun = false;
        uint32_t runStartPos = 0;
        for (uint32_t i = 0; i < batchLen; i++) {
            int32_t entry = entryUb.GetValue(i);
            bool isNewUnique = isFirstElement || (entry != prevEntry);
            isFirstElement = false;
            prevEntry = entry;
            if (isNewUnique) {
                inclusiveSum++;
                uniqueT.SetValue(tileUniqueCnt, entry - rankBase);
                tileUniqueCnt++;
            }
            int32_t compactIdx = runningOffset + inclusiveSum - 1;
            compactT.SetValue(i, compactIdx);
            if (i > 0) {
                bool dPrev = prevIsNewUnique && prevCompact != compactIdx;
                if (dPrev) {
                    if (!inRun) {
                        inRun = true;
                        runStartPos = i - 1U;
                    }
                } else {
                    if (inRun) {
                        runStartT.SetValue(runCnt, static_cast<int32_t>(runStartPos));
                        runLenT.SetValue(runCnt, static_cast<int32_t>(i - 1U - runStartPos));
                        runCnt++;
                        inRun = false;
                    }
                    accumT.SetValue(accumCnt, static_cast<int32_t>(i - 1U));
                    accumCnt++;
                }
            }
            prevIsNewUnique = isNewUnique;
            prevCompact = compactIdx;
        }
        if (batchLen > 0) {
            if (inRun) {
                runStartT.SetValue(runCnt, static_cast<int32_t>(runStartPos));
                runLenT.SetValue(runCnt, static_cast<int32_t>(batchLen - 1U - runStartPos));
                runCnt++;
                inRun = false;
            }
            accumT.SetValue(accumCnt, static_cast<int32_t>(batchLen - 1U));
            accumCnt++;
        }
        runCnt_ = runCnt;
        accumCnt_ = accumCnt;
        return tileUniqueCnt;
    }
    bool canDirectCopy = (inputDtype_ == outputDtype_);
    // flag[i] = isNewUnique[i] && i+1<batchLen && compact[i]!=compact[i+1]；compact 仅在 isNewUnique 时
    // 递增，故 compact[i]!=compact[i+1] ⟺ isNewUnique[i+1]。延迟一拍在主循环内直接生成最终 flag，
    // 消除原本逐元素遍历的第二遍循环（PERF-1）
    bool prevIsNewUnique = false;
    int32_t prevCompact = 0;
    for (uint32_t i = 0; i < batchLen; i++) {
        int32_t entry = entryUb.GetValue(i);
        bool isNewUnique = isFirstElement || (entry != prevEntry);
        isFirstElement = false;
        prevEntry = entry;
        if (isNewUnique) {
            inclusiveSum++;
            UniqueUb().SetValue(
                tileUniqueCnt,
                static_cast<int32_t>(static_cast<int64_t>(entry) - static_cast<int64_t>(rankId_) * numEntriesPerRank_));
            tileUniqueCnt++;
        }
        int32_t compactIdx = runningOffset + inclusiveSum - 1;
        CompactIdxUb().SetValue(i, compactIdx);
        DirectFlagUb().SetValue(i, isNewUnique ? 1 : 0);
        if (i > 0) {
            DirectFlagUb().SetValue(i - 1, (canDirectCopy && prevIsNewUnique && prevCompact != compactIdx) ? 1 : 0);
        }
        prevIsNewUnique = isNewUnique;
        prevCompact = compactIdx;
    }
    if (batchLen > 0) {
        DirectFlagUb().SetValue(batchLen - 1, 0);
    }
    return tileUniqueCnt;
}

__aicore__ inline void EngramFetchGradUnique::WriteBatchUnique(uint32_t tileUniqueCnt, int32_t &runningUniqueOffset,
                                                               GM_ADDR uniqueLocalEntryOutGM)
{
    AscendC::GlobalTensor<int32_t> uniqueEntryOutGM;
    uniqueEntryOutGM.SetGlobalBuffer((__gm__ int32_t *)uniqueLocalEntryOutGM);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::DataCopyParams ueParams{1U, static_cast<uint16_t>(tileUniqueCnt * sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(uniqueEntryOutGM[runningUniqueOffset], UniqueUb(), ueParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    runningUniqueOffset += static_cast<int32_t>(tileUniqueCnt);
}

__aicore__ inline void EngramFetchGradUnique::ProcessNonDirectRow(int32_t pos, uint32_t subStart,
                                                                  AscendC::LocalTensor<uint8_t> &gradRaw,
                                                                  AscendC::LocalTensor<float> &castRow,
                                                                  GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();
    uint32_t hiddenDim = static_cast<uint32_t>(hiddenDim_);

    int32_t compactIdx = CompactIdxUb().GetValue(pos);
    if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
        return;
    }
    if (compactIdx != accumCompactIdx_) {
        if (accumDirty_) {
            FlushAccum(gradUniqueOutGM);
            accumIdx_ ^= 1U;
            if (flushPending_[accumIdx_]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[accumIdx_]);
                flushPending_[accumIdx_] = false;
            }
        }
        accumCompactIdx_ = compactIdx;
        accumRowCnt_ = 0;
    }
    AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
    uint32_t rowOff = static_cast<uint32_t>(pos - static_cast<int32_t>(subStart)) * inRowStride_;
    if (accumRowCnt_ == 0U) {
        if (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) {
            CastToFP32(accum, gradRaw[rowOff], hiddenDim);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::LocalTensor<float> gradFp32 = gradRaw[rowOff].ReinterpretCast<float>();
            AscendC::Adds<float>(accum, gradFp32, 0.0f, hiddenDim);
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else if (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) {
        CastToFP32(castRow, gradRaw[rowOff], hiddenDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(accum, accum, castRow, hiddenDim);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        AscendC::LocalTensor<float> gradFp32 = gradRaw[rowOff].ReinterpretCast<float>();
        AscendC::Add<float>(accum, accum, gradFp32, hiddenDim);
        AscendC::PipeBarrier<PIPE_V>();
    }
    accumRowCnt_++;
    accumDirty_ = true;
}
__aicore__ inline void EngramFetchGradUnique::AccumulateDupListWalk(uint32_t subStart, uint32_t subLen,
                                                                    uint32_t &cursor,
                                                                    AscendC::LocalTensor<uint8_t> &gradRaw,
                                                                    AscendC::LocalTensor<float> &castRow,
                                                                    GM_ADDR gradUniqueOutGM)
{
    uint32_t subEnd = subStart + subLen;
    while (cursor < accumCnt_) {
        int32_t p = AccumListUb().GetValue(cursor);
        if (p < 0 || static_cast<uint32_t>(p) >= subEnd) {
            break;
        }
        if (static_cast<uint32_t>(p) >= subStart) {
            ProcessNonDirectRow(p, subStart, gradRaw, castRow, gradUniqueOutGM);
        }
        cursor++;
    }
}
__aicore__ inline void EngramFetchGradUnique::EmitRunsFromDupList(uint32_t subStart, uint32_t subLen, uint32_t &cursor,
                                                                  AscendC::LocalTensor<uint8_t> &gradRaw,
                                                                  GM_ADDR gradUniqueOutGM)
{
    uint32_t subEnd = subStart + subLen;
    while (cursor < runCnt_) {
        int32_t a = RunStartUb().GetValue(cursor);
        int32_t l = RunLenUb().GetValue(cursor);
        if (a < 0 || l <= 0) {
            cursor++;
            continue;
        }
        if (static_cast<uint32_t>(a) + static_cast<uint32_t>(l) > subStart) {
            break;
        }
        cursor++;
    }
    for (uint32_t r = cursor; r < runCnt_; r++) {
        int32_t a = RunStartUb().GetValue(r);
        int32_t l = RunLenUb().GetValue(r);
        if (a < 0 || l <= 0) {
            continue;
        }
        uint32_t b = static_cast<uint32_t>(a) + static_cast<uint32_t>(l);
        uint32_t lo = (static_cast<uint32_t>(a) > subStart) ? static_cast<uint32_t>(a) : subStart;
        uint32_t hi = (b < subEnd) ? b : subEnd;
        if (lo >= hi) {
            continue;
        }
        int32_t compactFirst = CompactIdxUb().GetValue(static_cast<int32_t>(lo));
        if (compactFirst < 0 || compactFirst >= numEntriesPerRank_) {
            continue;
        }
        FlushDirectRun(gradRaw, lo - subStart, hi - lo, compactFirst, gradUniqueOutGM);
    }
}
__aicore__ inline void EngramFetchGradUnique::ProcessDirectSubBatchLoopV2(uint32_t batchLen, uint32_t maxGradPerBatch,
                                                                          AscendC::LocalTensor<uint8_t> &gradBase,
                                                                          GM_ADDR recvGradGM, GM_ADDR gradUniqueOutGM)
{
    constexpr uint32_t kBufs = 4U;
    event_t evtMte2V[kBufs];
    event_t evtVMte2[kBufs];
    event_t evtMte2Mte3[kBufs];
    event_t evtMte3Mte2[kBufs];
    for (uint32_t b = 0; b < kBufs; b++) {
        evtMte2V[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        evtVMte2[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
        evtMte2Mte3[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
        evtMte3Mte2[b] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    }
    AscendC::LocalTensor<float> castRow = castBuf_->Get<float>();
    AscendC::DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};
    constexpr uint32_t kBufBytes = Mc2Kernel::GRAD_PING_BYTES / 2U;
    uint32_t bufRows = kBufBytes / inRowStride_;
    if (bufRows < 1U) {
        bufRows = 1U;
    }
    if (bufRows < maxGradPerBatch) {
        maxGradPerBatch = bufRows;
    }

    uint32_t accumCursor = 0;
    uint32_t runCursor = 0;
    uint32_t tileIdx = 0;
    for (uint32_t subStart = 0; subStart < batchLen; subStart += maxGradPerBatch) {
        uint32_t subLen = batchLen - subStart;
        if (subLen > maxGradPerBatch) {
            subLen = maxGradPerBatch;
        }
        uint32_t b = tileIdx % kBufs;
        AscendC::LocalTensor<uint8_t> gradRaw = gradBase[b * kBufBytes];

        if (tileIdx >= kBufs) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
        }

        for (uint32_t j = 0; j < subLen; j++) {
            int32_t recvIdx = CompUb().GetValue(subStart + j);
            if (recvIdx < 0 || static_cast<uint32_t>(recvIdx) >= numRecv_) {
                recvIdx = 0;
            }
            GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(recvIdx) * hiddenBytes_;
            AscendC::DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
            AscendC::GlobalTensor<uint8_t> gradSrcGM;
            gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
            AscendC::DataCopyPad(gradRaw[static_cast<uint64_t>(j) * inRowStride_], gradSrcGM, gradParams, gradPad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V[b]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[b]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V[b]);

        AccumulateDupListWalk(subStart, subLen, accumCursor, gradRaw, castRow, gradUniqueOutGM);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[b]);
        EmitRunsFromDupList(subStart, subLen, runCursor, gradRaw, gradUniqueOutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
        tileIdx++;
    }

    uint32_t drainFrom = (tileIdx > kBufs) ? (tileIdx - kBufs) : 0U;
    for (uint32_t t = drainFrom; t < tileIdx; t++) {
        uint32_t b = t % kBufs;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
    }
    for (uint32_t b = 0; b < kBufs; b++) {
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evtMte2V[b]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evtVMte2[b]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[b]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[b]);
    }
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessScatterBatch(
    uint32_t cur, uint32_t end, int32_t &runningOffset, int32_t &runningUniqueOffset, int32_t &prevEntry,
    bool &isFirstElement, GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
    GM_ADDR recvGradGM, GM_ADDR sortCompanionGM)
{
    uint32_t batchLen = end - cur;
    if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
        batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
    }

    int32_t inclusiveSum = 0;
    bool singleRowMode = inRowStride_ > Mc2Kernel::GRAD_PING_BYTES;
    bool canDirectCopy = (inputDtype_ == outputDtype_);
    bool packedRows = (inRowStride_ == static_cast<uint32_t>(hiddenBytes_));
    bool useListPath = canDirectCopy && !singleRowMode && packedRows && inRowStride_ <= Mc2Kernel::GRAD_PING_BYTES / 2U;
    uint32_t tileUniqueCnt = ProcessBatchUnique(cur, batchLen, runningOffset, prevEntry, isFirstElement, inclusiveSum,
                                                recvLocalEntryOutGM, sortCompanionGM, useListPath);

    uint32_t maxGradPerBatch = Mc2Kernel::GRAD_PING_BYTES / inRowStride_;
    if (maxGradPerBatch < 1U) {
        maxGradPerBatch = 1U;
    }
    // 行宽超过 32KB 半缓冲时禁用 ping/pong 拆分：整缓冲单行、跨 tile 用 evt_0 串行，
    // 否则 tileIdx=1 写 gradPing[32K] 处的单行会越过 gradBuf_ 污染池内相邻缓冲
    bool canDirectCopyEarly = canDirectCopy;
    if (!canDirectCopyEarly && maxGradPerBatch > gradSubBatch_) {
        maxGradPerBatch = gradSubBatch_;
    }
    uint32_t gradBufHalf = Mc2Kernel::GRAD_PING_BYTES;

    AscendC::LocalTensor<uint8_t> gradPing = gradBuf_->Get<uint8_t>();
    AscendC::LocalTensor<uint8_t> gradPong = gradPing[gradBufHalf];

    AscendC::DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};

    bool needCast = (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT);
    uint32_t castHalfFloats = 0;
    uint32_t fp32StrideFloats = fp32RowStride_ / sizeof(float);
    if (needCast) {
        uint32_t castHalfBytes =
            (fp32RowStride_ * maxGradPerBatch + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
        castHalfFloats = castHalfBytes / sizeof(float);
    }

    event_t evtMte2V_0 = static_cast<event_t>(0);
    event_t evtMte2V_1 = static_cast<event_t>(0);
    event_t evtVMte2_0 = static_cast<event_t>(0);
    event_t evtVMte2_1 = static_cast<event_t>(0);
    event_t evtMte2Mte3_0 = static_cast<event_t>(0);
    event_t evtMte2Mte3_1 = static_cast<event_t>(0);
    event_t evtMte3Mte2_0 = static_cast<event_t>(0);
    event_t evtMte3Mte2_1 = static_cast<event_t>(0);

    if (useListPath) {
        ProcessDirectSubBatchLoopV2(batchLen, maxGradPerBatch, gradPing, recvGradGM, gradUniqueOutGM);
    } else if (canDirectCopy && !singleRowMode && inRowStride_ <= Mc2Kernel::GRAD_PING_BYTES / 2U) {
        ProcessDirectSubBatchLoop(batchLen, maxGradPerBatch, gradPing, recvGradGM, gradUniqueOutGM);
    } else {
        evtMte2V_0 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_V));
        evtMte2V_1 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_V));
        evtVMte2_0 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE2));
        evtVMte2_1 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE2));
        if (canDirectCopy) {
            evtMte2Mte3_0 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            evtMte2Mte3_1 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            evtMte3Mte2_0 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            evtMte3Mte2_1 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        }

        uint32_t tileIdx = 0;
        for (uint32_t subStart = 0; subStart < batchLen; subStart += maxGradPerBatch) {
            uint32_t subLen = batchLen - subStart;
            if (subLen > maxGradPerBatch) {
                subLen = maxGradPerBatch;
            }
            uint32_t bufIdx = tileIdx % 2U;
            AscendC::LocalTensor<uint8_t> gradRaw = (bufIdx == 0U || singleRowMode) ? gradPing : gradPong;
            bool useEvt0 = singleRowMode || bufIdx == 0U;

            if (tileIdx >= (singleRowMode ? 1U : 2U)) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(useEvt0 ? evtVMte2_0 : evtVMte2_1);
                if (canDirectCopy) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(useEvt0 ? evtMte3Mte2_0 : evtMte3Mte2_1);
                }
            }

            for (uint32_t j = 0; j < subLen; j++) {
                int32_t recvIdx = CompUb().GetValue(subStart + j);
                if (recvIdx < 0 || static_cast<uint32_t>(recvIdx) >= numRecv_) {
                    recvIdx = 0;
                }
                GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(recvIdx) * hiddenBytes_;
                AscendC::DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
                AscendC::GlobalTensor<uint8_t> gradSrcGM;
                gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
                AscendC::DataCopyPad(gradRaw[static_cast<uint64_t>(j) * inRowStride_], gradSrcGM, gradParams, gradPad);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(useEvt0 ? evtMte2V_0 : evtMte2V_1);
            if (canDirectCopy) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(useEvt0 ? evtMte2Mte3_0 : evtMte2Mte3_1);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(useEvt0 ? evtMte2V_0 : evtMte2V_1);

            if (canDirectCopy && needCast) {
                AscendC::LocalTensor<float> castRow = castBuf_->Get<float>();
                AccumulateDirectSubBatch(gradRaw, subStart, subLen, castRow, gradUniqueOutGM);
            } else if (needCast) {
                AscendC::LocalTensor<float> castPingF = castBuf_->Get<float>();
                AscendC::LocalTensor<float> castPongF = castPingF[castHalfFloats];
                AscendC::LocalTensor<float> gradFp32 = (bufIdx == 0U) ? castPingF : castPongF;
                // gradRaw 行带 32B 对齐 stride，cast 必须逐行进行（行间存在 padding 字节）
                for (uint32_t j = 0; j < subLen; j++) {
                    CastToFP32(gradFp32[j * fp32StrideFloats], gradRaw[j * inRowStride_],
                               static_cast<uint32_t>(hiddenDim_));
                }
                AscendC::PipeBarrier<PIPE_V>();
                AccumulateSubBatch(gradFp32, fp32StrideFloats, subStart, subLen, gradUniqueOutGM);
            } else {
                AscendC::LocalTensor<float> gradFp32 = gradRaw.ReinterpretCast<float>();
                AccumulateSubBatch(gradFp32, inRowStride_ / sizeof(float), subStart, subLen, gradUniqueOutGM);
            }

            if (canDirectCopy) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(useEvt0 ? evtMte2Mte3_0 : evtMte2Mte3_1);
                for (uint32_t j = 0; j < subLen; j++) {
                    if (DirectFlagUb().GetValue(subStart + j) != 0) {
                        int32_t compactIdx = CompactIdxUb().GetValue(subStart + j);
                        if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
                            continue;
                        }
                        FlushDirect(gradRaw, j, compactIdx, gradUniqueOutGM);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(useEvt0 ? evtMte3Mte2_0 : evtMte3Mte2_1);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(useEvt0 ? evtVMte2_0 : evtVMte2_1);
            tileIdx++;
        }

        if (tileIdx >= 1U) {
            uint32_t lastEvt = singleRowMode ? 0U : ((tileIdx - 1U) % 2U);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(lastEvt == 0U ? evtVMte2_0 : evtVMte2_1);
            if (canDirectCopy) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(lastEvt == 0U ? evtMte3Mte2_0 : evtMte3Mte2_1);
            }
        }
        if (!singleRowMode && tileIdx >= 2U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tileIdx % 2U == 0U ? evtVMte2_0 : evtVMte2_1);
            if (canDirectCopy) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tileIdx % 2U == 0U ? evtMte3Mte2_0 : evtMte3Mte2_1);
            }
        }
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evtMte2V_0);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evtMte2V_1);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evtVMte2_0);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evtVMte2_1);
        if (canDirectCopy) {
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3_0);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3_1);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2_0);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2_1);
        }
    }

    if (tileUniqueCnt > 0) {
        WriteBatchUnique(tileUniqueCnt, runningUniqueOffset, uniqueLocalEntryOutGM);
    }
    runningOffset += inclusiveSum;
    return cur + batchLen;
}

__aicore__ inline void EngramFetchGradUnique::ScatterAccumulateParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                                        GM_ADDR uniqueLocalEntryOutGM,
                                                                        GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM,
                                                                        GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                                        GM_ADDR sortCompanionGM)
{
    uint32_t start;
    uint32_t end;
    int32_t preCoreOffset;
    LoadCoreRange(numRecv, coreStartGM, segCountGM, start, end, preCoreOffset);
    if (start >= end) {
        return;
    }
    accumCompactIdx_ = -1;
    accumDirty_ = false;
    if (chunkElems_ > 0U) {
        ChunkedScatterRange(start, end, preCoreOffset, recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM,
                            recvGradGM, sortCompanionGM);
        return;
    }
    int32_t runningOffset = preCoreOffset;
    int32_t runningUniqueOffset = preCoreOffset;
    int32_t prevEntry = 0;
    bool isFirstElement = true;
    if (start > 0U) {
        AscendC::GlobalTensor<int32_t> sortedEntryGM;
        sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM);
        AscendC::LocalTensor<int32_t> prevUb = tempBuf_->Get<int32_t>();
        AscendC::DataCopyPadExtParams<int32_t> prevPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams prevParams{1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(prevUb, sortedEntryGM[start - 1U], prevParams, prevPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
        prevEntry = prevUb.GetValue(0);
        isFirstElement = false;
    }

    uint32_t cur = start;
    while (cur < end) {
        cur = ProcessScatterBatch(cur, end, runningOffset, runningUniqueOffset, prevEntry, isFirstElement,
                                  recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM, recvGradGM,
                                  sortCompanionGM);
    }
}

__aicore__ inline void EngramFetchGradUnique::WriteNumUnique(GM_ADDR segCountGM, GM_ADDR numUniqueOutGM)
{
    // Sum per-core segCount on core 0 and plain-write (do NOT atomic-add: numUniqueOutGM is
    // at::empty/uninitialized, so SetAtomicAdd would accumulate onto garbage -> wrong numUnique).
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
    AscendC::LocalTensor<int32_t> segCountUb = tempBuf_->Get<int32_t>();
    AscendC::DataCopyPadExtParams<int32_t> scPad{false, 0, 0, 0};
    AscendC::DataCopyExtParams scParams{1U, static_cast<uint32_t>(totalBlocks_ * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(segCountUb, segCountGMT, scParams, scPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    int32_t numUnique = 0;
    for (uint32_t core = 0; core < totalBlocks_; core++) {
        numUnique += segCountUb.GetValue(core);
    }

    if (aivId_ == 0) {
        AscendC::LocalTensor<int32_t> tmp = statusBuf_->Get<int32_t>();
        tmp.SetValue(0, numUnique);
        SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
        AscendC::GlobalTensor<int32_t> numUniqueGM;
        numUniqueGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM);
        AscendC::DataCopyParams p{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
        AscendC::DataCopyPad(numUniqueGM, tmp, p);
        SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    }
    AscendC::SyncAll<true>();
}

} // namespace EngramFetchGradUnique

#endif
