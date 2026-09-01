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
#include "../engram_fetch_grad_utils.h"

namespace EngramFetchGradUnique {

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

    // entryBuf_ int32 slot layout: one ENTRY_BATCH_CAP-sized slot per array.
    __aicore__ inline AscendC::LocalTensor<int32_t> CompUb()
    {
        return entryBuf_->Get<int32_t>();
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> UniqueUb()
    {
        return entryBuf_->Get<int32_t>()[Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> RecvIdxUb()
    {
        return entryBuf_->Get<int32_t>()[2 * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> CompactIdxUb()
    {
        return entryBuf_->Get<int32_t>()[3 * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> DirectFlagUb()
    {
        return entryBuf_->Get<int32_t>()[4 * Mc2Kernel::ENTRY_BATCH_CAP];
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
    __aicore__ inline uint32_t ProcessBatchUnique(uint32_t cur, uint32_t batchLen, int32_t runningOffset,
                                                  int32_t &prevEntry, bool &isFirstElement, int32_t &inclusiveSum,
                                                  GM_ADDR recvLocalEntryOutGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void WriteBatchUnique(uint32_t tileUniqueCnt, int32_t &runningUniqueOffset,
                                            GM_ADDR uniqueLocalEntryOutGM);

    __aicore__ inline void FlushAccum(GM_ADDR gradUniqueOutGM);
    __aicore__ inline void FlushDirect(AscendC::LocalTensor<uint8_t> &gradRaw, uint32_t elemIdx, int32_t compactIdx,
                                       GM_ADDR gradUniqueOutGM);
    __aicore__ inline void AccumulateSubBatch(AscendC::LocalTensor<float> &gradFp32, uint32_t subStart, uint32_t subLen,
                                              GM_ADDR gradUniqueOutGM);
    __aicore__ inline void CastToFP32(AscendC::LocalTensor<float> &outT, AscendC::LocalTensor<uint8_t> &gradRaw,
                                      uint32_t count);

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    int32_t numEntriesPerRank_{0};
    int64_t hiddenDim_{0};
    int64_t hiddenBytes_{0};
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

    int32_t myPreCoreOffset_{0};
    int32_t mySegCount_{0};

    int32_t accumCompactIdx_{-1};
    bool accumDirty_{false};
    uint32_t accumFloats_{0};
    uint32_t accumIdx_{0};
    bool flushPending_[2]{false, false};
    int32_t flushEvtVMte3Arr_[2]{0, 0};
    int32_t flushEvtMte3VArr_[2]{0, 0};
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
        uint32_t probeLen = rawEnd - rawStart + 1U;
        if (probeLen > Mc2Kernel::ENTRY_BATCH_CAP) {
            probeLen = Mc2Kernel::ENTRY_BATCH_CAP;
        }
        AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(probeLen * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(probeUb, sortedEntryGM[rawStart - 1U], cpParams, cpPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

        boundaryEntry = probeUb.GetValue(0);
        uint32_t p = 1U;
        while (p < probeLen && probeUb.GetValue(p) == boundaryEntry) {
            p++;
        }
        start = rawStart + p - 1U;
    }

    uint32_t localUniqueCount = 0;
    if (start < rawEnd) {
        AscendC::LocalTensor<int32_t> entryUb = tempBuf_->Get<int32_t>();
        int32_t prevEntry = 0;
        bool isFirstElement = (start == 0);
        if (!isFirstElement) {
            prevEntry = boundaryEntry;
        }

        uint32_t cur = start;
        while (cur < rawEnd) {
            uint32_t batchLen = rawEnd - cur;
            if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
                batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
            }
            AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            AscendC::DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
            SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
            for (uint32_t i = 0; i < batchLen; i++) {
                int32_t entry = entryUb.GetValue(i);
                bool isNewUnique;
                if (isFirstElement) {
                    isNewUnique = true;
                    isFirstElement = false;
                } else {
                    isNewUnique = (entry != prevEntry);
                }
                if (isNewUnique) {
                    localUniqueCount++;
                }
                prevEntry = entry;
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

__aicore__ inline void EngramFetchGradUnique::LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                            uint32_t &start, uint32_t &end, int32_t &preCoreOffset)
{
    AscendC::GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);

    AscendC::LocalTensor<int32_t> ub = tempBuf_->Get<int32_t>();
    uint32_t totalBytes = totalBlocks_ * static_cast<uint32_t>(sizeof(int32_t));
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyExtParams params{1U, totalBytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(ub, coreStartGMT, params, pad);
    AscendC::LocalTensor<int32_t> segUb = ub[totalBlocks_];
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
        if (castTailBytes > Mc2Kernel::ENTRY_BUF_BYTES) {
            RUNTIME_ABORT("FlushAccum cast staging overflow: need %u bytes, entryBuf=%u bytes, hiddenDim=%u",
                          castTailBytes, Mc2Kernel::ENTRY_BUF_BYTES, static_cast<uint32_t>(hiddenDim_));
        }
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
    uint32_t srcOffset = elemIdx * static_cast<uint32_t>(hiddenBytes_);
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

__aicore__ inline void EngramFetchGradUnique::CastToFP32(AscendC::LocalTensor<float> &outT,
                                                         AscendC::LocalTensor<uint8_t> &gradRaw, uint32_t count)
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
                                                                 uint32_t subStart, uint32_t subLen,
                                                                 GM_ADDR gradUniqueOutGM)
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
        AscendC::Add<float>(accum, accum, gradFp32[j * static_cast<uint32_t>(hiddenDim_)],
                            static_cast<uint32_t>(hiddenDim_));
        AscendC::PipeBarrier<PIPE_V>();
        accumDirty_ = true;
    }
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessBatchUnique(uint32_t cur, uint32_t batchLen,
                                                                     int32_t runningOffset, int32_t &prevEntry,
                                                                     bool &isFirstElement, int32_t &inclusiveSum,
                                                                     GM_ADDR recvLocalEntryOutGM,
                                                                     GM_ADDR sortCompanionGM)
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
        RecvIdxUb().SetValue(i, CompUb().GetValue(i));
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
    uint32_t tileUniqueCnt = ProcessBatchUnique(cur, batchLen, runningOffset, prevEntry, isFirstElement, inclusiveSum,
                                                recvLocalEntryOutGM, sortCompanionGM);

    uint32_t maxGradPerBatch = Mc2Kernel::TILE_BYTES / static_cast<uint32_t>(hiddenBytes_);
    if (maxGradPerBatch < 1U) {
        maxGradPerBatch = 1U;
    }
    // 行宽超过 32KB 半缓冲时禁用 ping/pong 拆分：整缓冲单行、跨 tile 用 evt_0 串行，
    // 否则 tileIdx=1 写 gradPing[32K] 处的单行会越过 gradBuf_ 污染池内相邻缓冲
    bool singleRowMode = static_cast<uint32_t>(hiddenBytes_) > Mc2Kernel::GRAD_PING_BYTES;
    if (maxGradPerBatch > gradSubBatch_) {
        maxGradPerBatch = gradSubBatch_;
    }
    uint32_t gradBufHalf = Mc2Kernel::GRAD_PING_BYTES;

    AscendC::LocalTensor<uint8_t> gradPing = gradBuf_->Get<uint8_t>();
    AscendC::LocalTensor<uint8_t> gradPong = gradPing[gradBufHalf];

    AscendC::DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};

    bool needCast = (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT);
    bool canDirectCopy = (inputDtype_ == outputDtype_);
    uint32_t castHalfFloats = 0;
    if (needCast) {
        uint32_t castHalfBytes =
            (static_cast<uint32_t>(hiddenDim_) * sizeof(float) * maxGradPerBatch + Mc2Kernel::UB_ALIGN - 1U) /
            Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
        castHalfFloats = castHalfBytes / sizeof(float);
    }

    event_t evtMte2V_0 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtMte2V_1 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtVMte2_0 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE2));
    event_t evtVMte2_1 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE2));
    event_t evtMte2Mte3_0 = static_cast<event_t>(0);
    event_t evtMte2Mte3_1 = static_cast<event_t>(0);
    event_t evtMte3Mte2_0 = static_cast<event_t>(0);
    event_t evtMte3Mte2_1 = static_cast<event_t>(0);
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
            int32_t recvIdx = RecvIdxUb().GetValue(subStart + j);
            if (recvIdx < 0 || static_cast<uint32_t>(recvIdx) >= numRecv_) {
                recvIdx = 0;
            }
            GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(recvIdx) * hiddenBytes_;
            AscendC::DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
            AscendC::GlobalTensor<uint8_t> gradSrcGM;
            gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
            AscendC::DataCopyPad(gradRaw[static_cast<uint64_t>(j) * static_cast<uint32_t>(hiddenBytes_)], gradSrcGM,
                                 gradParams, gradPad);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(useEvt0 ? evtMte2V_0 : evtMte2V_1);
        if (canDirectCopy) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(useEvt0 ? evtMte2Mte3_0 : evtMte2Mte3_1);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(useEvt0 ? evtMte2V_0 : evtMte2V_1);

        if (needCast) {
            AscendC::LocalTensor<float> castPingF = castBuf_->Get<float>();
            AscendC::LocalTensor<float> castPongF = castPingF[castHalfFloats];
            AscendC::LocalTensor<float> gradFp32 = (bufIdx == 0U) ? castPingF : castPongF;
            uint32_t castCount = subLen * static_cast<uint32_t>(hiddenDim_);
            CastToFP32(gradFp32, gradRaw, castCount);
            AscendC::PipeBarrier<PIPE_V>();
            AccumulateSubBatch(gradFp32, subStart, subLen, gradUniqueOutGM);
        } else {
            AscendC::LocalTensor<float> gradFp32 = gradRaw.ReinterpretCast<float>();
            AccumulateSubBatch(gradFp32, subStart, subLen, gradUniqueOutGM);
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
    int32_t runningOffset = preCoreOffset;
    int32_t runningUniqueOffset = preCoreOffset;
    int32_t prevEntry = 0;
    bool isFirstElement = true;

    accumCompactIdx_ = -1;
    accumDirty_ = false;

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
