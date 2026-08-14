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

class EngramFetchGradUnique {
public:
    __aicore__ inline EngramFetchGradUnique() = default;

    __aicore__ inline void Init(
        uint32_t aivId, uint32_t totalBlocks, uint32_t rankId, uint32_t numRanks,
        int32_t numEntriesPerRank, int64_t hiddenDim, int64_t hiddenBytes,
        int32_t inputDtype, int32_t outputDtype, AscendC::TPipe *pipe,
        AscendC::TBuf<> &pingBuf, AscendC::TBuf<> &pongBuf, AscendC::TBuf<> &indicesBuf,
        AscendC::TBuf<> &tempBuf, AscendC::TBuf<> &statusBuf, AscendC::TBuf<> &gradSumBuf);

    __aicore__ inline void Run(uint32_t numRecv,
                               GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR numUniqueOutGM,
                               GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM,
                               GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM);

    __aicore__ inline void WriteNumUniqueZero(GM_ADDR numUniqueOutGM);

private:
    __aicore__ inline void ZeroGradUnique(uint32_t numRecv, GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM);
    __aicore__ inline void CountUniquesParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                GM_ADDR coreStartGM, GM_ADDR segCountGM);
    __aicore__ inline void FixCoreStartForSkippedCores(GM_ADDR coreStartGM, GM_ADDR segCountGM);
    __aicore__ inline void ScatterAddAtomicParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                    GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
                                                    GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM,
                                                    GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void CastGradUniqueToOutput(GM_ADDR segCountGM, GM_ADDR numUniqueOutGM,
                                                  GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM);
    __aicore__ inline void LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, uint32_t &start, uint32_t &end);
    __aicore__ inline int32_t ComputePreCoreOffset(GM_ADDR segCountGM);
    __aicore__ inline uint32_t ProcessScatterBatch(
        uint32_t cur, uint32_t end, int32_t &runningOffset, int32_t &runningUniqueOffset,
        int32_t &prevEntry, bool &isFirstElement,
        GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
        GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void ScatterGradSubBatch(uint32_t subStart, uint32_t subLen,
                                               GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM);
    __aicore__ inline void AtomicAddGrad(AscendC::GlobalTensor<float> &atomicDstGM, AscendC::LocalTensor<float> &srcT,
                                         uint32_t subStart, uint32_t subLen);
    __aicore__ inline uint32_t ProcessBatchUnique(uint32_t cur, uint32_t batchLen, int32_t runningOffset,
                                                  int32_t &prevEntry, bool &isFirstElement,
                                                  int32_t &inclusiveSum,
                                                  GM_ADDR recvLocalEntryOutGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void WriteBatchUnique(uint32_t tileUniqueCnt, int32_t &runningUniqueOffset,
                                            GM_ADDR uniqueLocalEntryOutGM);

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    int32_t numEntriesPerRank_{0};
    int64_t hiddenDim_{0};
    int64_t hiddenBytes_{0};
    int32_t inputDtype_{0};
    int32_t outputDtype_{0};
    AscendC::TPipe *pipe_{nullptr};
    AscendC::TBuf<> *pingBuf_{nullptr};
    AscendC::TBuf<> *pongBuf_{nullptr};
    AscendC::TBuf<> *indicesBuf_{nullptr};
    AscendC::TBuf<> *tempBuf_{nullptr};
    AscendC::TBuf<> *statusBuf_{nullptr};
    AscendC::TBuf<> *gradSumBuf_{nullptr};
};

__aicore__ inline void EngramFetchGradUnique::Init(
    uint32_t aivId, uint32_t totalBlocks, uint32_t rankId, uint32_t numRanks,
    int32_t numEntriesPerRank, int64_t hiddenDim, int64_t hiddenBytes,
    int32_t inputDtype, int32_t outputDtype, AscendC::TPipe *pipe,
    AscendC::TBuf<> &pingBuf, AscendC::TBuf<> &pongBuf, AscendC::TBuf<> &indicesBuf,
    AscendC::TBuf<> &tempBuf, AscendC::TBuf<> &statusBuf, AscendC::TBuf<> &gradSumBuf)
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
    pingBuf_ = &pingBuf;
    pongBuf_ = &pongBuf;
    indicesBuf_ = &indicesBuf;
    tempBuf_ = &tempBuf;
    statusBuf_ = &statusBuf;
    gradSumBuf_ = &gradSumBuf;
}

__aicore__ inline void EngramFetchGradUnique::Run(
    uint32_t numRecv,
    GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR numUniqueOutGM,
    GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM,
    GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM)
{
    ZeroGradUnique(numRecv, gradUniqueOutGM, gradUniqueFp32GM);
    CountUniquesParallel(numRecv, recvLocalEntryOutGM, coreStartGM, segCountGM);
    AscendC::SyncAll<true>();

    FixCoreStartForSkippedCores(coreStartGM, segCountGM);
    AscendC::SyncAll<true>();

    ScatterAddAtomicParallel(numRecv, recvLocalEntryOutGM, uniqueLocalEntryOutGM,
                             gradUniqueOutGM, gradUniqueFp32GM, recvGradGM,
                             coreStartGM, segCountGM, sortCompanionGM);
    AscendC::SyncAll<true>();

    CastGradUniqueToOutput(segCountGM, numUniqueOutGM, gradUniqueOutGM, gradUniqueFp32GM);
    AscendC::SyncAll<true>();
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

__aicore__ inline void EngramFetchGradUnique::ZeroGradUnique(uint32_t numRecv, GM_ADDR gradUniqueOutGM,
                                                             GM_ADDR gradUniqueFp32GM)
{
    uint32_t zeroCount = numRecv;
    uint64_t totalFloats = static_cast<uint64_t>(zeroCount) * static_cast<uint64_t>(hiddenDim_);
    uint64_t chunk = (totalFloats + totalBlocks_ - 1U) / totalBlocks_;
    uint64_t start = static_cast<uint64_t>(aivId_) * chunk;
    uint64_t end = start + chunk;
    if (end > totalFloats) {
        end = totalFloats;
    }

    if (start < end) {
        AscendC::LocalTensor<float> zeroBuf = pongBuf_->Get<float>();
        uint32_t tileFloats = Mc2Kernel::TILE_BYTES / sizeof(float);
        AscendC::Duplicate<float>(zeroBuf, 0.0f, tileFloats);
        SyncFunc<AscendC::HardEvent::V_MTE3>(*pipe_);

        GM_ADDR dstBase = (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT) ? gradUniqueOutGM : gradUniqueFp32GM;
        AscendC::GlobalTensor<float> dstGM;
        dstGM.SetGlobalBuffer((__gm__ float *)dstBase);
        for (uint64_t off = start; off < end; off += tileFloats) {
            uint64_t thisLen = end - off;
            if (thisLen > tileFloats) {
                thisLen = tileFloats;
            }
            AscendC::DataCopyParams params{1U, static_cast<uint16_t>(thisLen * sizeof(float)), 0U, 0U};
            AscendC::DataCopyPad(dstGM[off], zeroBuf, params);
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    }
}

__aicore__ inline void EngramFetchGradUnique::CountUniquesParallel(
    uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR coreStartGM, GM_ADDR segCountGM)
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

        int32_t boundaryEntry = probeUb.GetValue(0);
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
            AscendC::DataCopyPadExtParams<int32_t> onePad{false, 0, 0, 0};
            AscendC::DataCopyExtParams oneParams{1U, sizeof(int32_t), 0U, 0U, 0U};
            AscendC::DataCopyPad(entryUb, sortedEntryGM[start - 1U], oneParams, onePad);
            SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
            prevEntry = entryUb.GetValue(0);
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
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::DataCopyParams startParams{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(coreStartGMT[aivId_], cntUb, startParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);

    cntUb.SetValue(0, static_cast<int32_t>(localUniqueCount));
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
    AscendC::DataCopyParams p{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(segCountGMT[aivId_], cntUb, p);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
}

__aicore__ inline void EngramFetchGradUnique::FixCoreStartForSkippedCores(GM_ADDR coreStartGM, GM_ADDR segCountGM)
{
    if (aivId_ != 0) {
        return;
    }

    AscendC::GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);

    int32_t nextValidStart = -1;
    for (int32_t core = static_cast<int32_t>(totalBlocks_) - 1; core >= 0; core--) {
        int32_t segCnt = segCountGMT.GetValue(core);
        if (segCnt == 0) {
            if (nextValidStart >= 0) {
                AscendC::LocalTensor<int32_t> fixUb = statusBuf_->Get<int32_t>();
                fixUb.SetValue(0, nextValidStart);
                SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
                AscendC::DataCopyParams fixParams{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
                AscendC::DataCopyPad(coreStartGMT[core], fixUb, fixParams);
                SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
            }
        } else {
            nextValidStart = coreStartGMT.GetValue(core);
        }
    }
}

__aicore__ inline void EngramFetchGradUnique::LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM,
                                                            uint32_t &start, uint32_t &end)
{
    AscendC::GlobalTensor<int32_t> coreStartGM_;
    coreStartGM_.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::LocalTensor<int32_t> startUb = tempBuf_->Get<int32_t>();
    uint32_t readCores = aivId_ + 2U;
    if (readCores > totalBlocks_) {
        readCores = totalBlocks_;
    }
    if (aivId_ == totalBlocks_ - 1U) {
        readCores = totalBlocks_;
    }
    AscendC::DataCopyPadExtParams<int32_t> csPad{false, 0, 0, 0};
    AscendC::DataCopyExtParams csParams{1U, static_cast<uint32_t>(readCores * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(startUb, coreStartGM_, csParams, csPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    start = static_cast<uint32_t>(startUb.GetValue(aivId_));
    if (aivId_ == totalBlocks_ - 1U) {
        end = numRecv;
    } else {
        end = static_cast<uint32_t>(startUb.GetValue(aivId_ + 1U));
    }
}

__aicore__ inline int32_t EngramFetchGradUnique::ComputePreCoreOffset(GM_ADDR segCountGM)
{
    int32_t preCoreOffset = 0;
    if (aivId_ > 0) {
        AscendC::GlobalTensor<int32_t> segCountGMT;
        segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
        AscendC::LocalTensor<int32_t> segCountUb = tempBuf_->Get<int32_t>();
        AscendC::DataCopyPadExtParams<int32_t> scPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams scParams{1U, static_cast<uint32_t>(aivId_ * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(segCountUb, segCountGMT, scParams, scPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
        for (uint32_t core = 0; core < aivId_; core++) {
            preCoreOffset += segCountUb.GetValue(core);
        }
    }
    return preCoreOffset;
}

__aicore__ inline void EngramFetchGradUnique::AtomicAddGrad(
    AscendC::GlobalTensor<float> &atomicDstGM, AscendC::LocalTensor<float> &srcT, uint32_t subStart, uint32_t subLen)
{
    AscendC::LocalTensor<int32_t> pingInt32 = pingBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> compactIdxBuf = pingInt32[3 * Mc2Kernel::ENTRY_BATCH_CAP];
    constexpr uint32_t MAX_BLOCK_BYTES = 65535U;
    uint32_t totalBytes = static_cast<uint32_t>(hiddenDim_) * sizeof(float);
    uint32_t numBlocks = (totalBytes + MAX_BLOCK_BYTES - 1U) / MAX_BLOCK_BYTES;
    for (uint32_t j = 0; j < subLen; j++) {
        int32_t compactIdx = compactIdxBuf.GetValue(subStart + j);
        AscendC::SetAtomicAdd<float>();
        for (uint32_t b = 0; b < numBlocks; b++) {
            uint32_t byteOffset = b * MAX_BLOCK_BYTES;
            uint32_t elemOffset = byteOffset / sizeof(float);
            uint32_t remaining = totalBytes - byteOffset;
            uint16_t blkBytes = static_cast<uint16_t>(
                remaining > MAX_BLOCK_BYTES ? MAX_BLOCK_BYTES : remaining);
            AscendC::DataCopyParams atomicParams{1U, blkBytes, 0U, 0U};
            AscendC::DataCopyPad(atomicDstGM[static_cast<uint64_t>(compactIdx) * hiddenDim_ + elemOffset],
                                 srcT[static_cast<uint32_t>(j) * static_cast<uint32_t>(hiddenDim_) + elemOffset],
                                 atomicParams);
        }
        AscendC::SetAtomicNone();
    }
}

__aicore__ inline void EngramFetchGradUnique::ScatterGradSubBatch(
    uint32_t subStart, uint32_t subLen, GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM)
{
    AscendC::LocalTensor<int32_t> pingInt32 = pingBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> recvIdxBuf = pingInt32[2 * Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::LocalTensor<uint8_t> gradRaw = pongBuf_->Get<uint8_t>();
    AscendC::LocalTensor<float> gradFp32 = gradSumBuf_->Get<float>();
    GM_ADDR atomicDstBase = (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT) ? gradUniqueOutGM : gradUniqueFp32GM;
    AscendC::GlobalTensor<float> atomicDstGM;
    atomicDstGM.SetGlobalBuffer((__gm__ float *)atomicDstBase);

    AscendC::DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};
    for (uint32_t j = 0; j < subLen; j++) {
        int32_t recvIdx = recvIdxBuf.GetValue(subStart + j);
        GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(recvIdx) * hiddenBytes_;
        AscendC::DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
        AscendC::GlobalTensor<uint8_t> gradSrcGM;
        gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
        AscendC::DataCopyPad(gradRaw[static_cast<uint32_t>(j) * static_cast<uint32_t>(hiddenBytes_)],
                             gradSrcGM, gradParams, gradPad);
    }

    if (inputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT) {
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>(*pipe_);
        AscendC::LocalTensor<float> inT = gradRaw.ReinterpretCast<float>();
        AtomicAddGrad(atomicDstGM, inT, subStart, subLen);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>(*pipe_);
    } else {
        SyncFunc<AscendC::HardEvent::MTE2_V>(*pipe_);
        uint32_t castCount = subLen * static_cast<uint32_t>(hiddenDim_);
        if (inputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
            AscendC::LocalTensor<bfloat16_t> inT = gradRaw.ReinterpretCast<bfloat16_t>();
            AscendC::Cast(gradFp32, inT, AscendC::RoundMode::CAST_NONE, castCount);
        } else {
            AscendC::LocalTensor<half> inT = gradRaw.ReinterpretCast<half>();
            AscendC::Cast(gradFp32, inT, AscendC::RoundMode::CAST_NONE, castCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
        SyncFunc<AscendC::HardEvent::V_MTE3>(*pipe_);
        AtomicAddGrad(atomicDstGM, gradFp32, subStart, subLen);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>(*pipe_);
    }
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessBatchUnique(
    uint32_t cur, uint32_t batchLen, int32_t runningOffset,
    int32_t &prevEntry, bool &isFirstElement, int32_t &inclusiveSum,
    GM_ADDR recvLocalEntryOutGM, GM_ADDR sortCompanionGM)
{
    AscendC::LocalTensor<int32_t> entryUb = indicesBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> pingInt32 = pingBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> compUb = pingInt32;
    AscendC::LocalTensor<int32_t> uniqueUb = pingInt32[Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::LocalTensor<int32_t> recvIdxBuf = pingInt32[2 * Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::LocalTensor<int32_t> compactIdxBuf = pingInt32[3 * Mc2Kernel::ENTRY_BATCH_CAP];

    AscendC::GlobalTensor<int32_t> sortedEntryGM;
    AscendC::GlobalTensor<int32_t> compGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM);
    compGM.SetGlobalBuffer((__gm__ int32_t *)sortCompanionGM);
    AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    AscendC::DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
    AscendC::DataCopyPad(compUb, compGM[cur], cpParams, cpPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    inclusiveSum = 0;
    uint32_t tileUniqueCnt = 0;
    for (uint32_t i = 0; i < batchLen; i++) {
        int32_t entry = entryUb.GetValue(i);
        bool isNewUnique = isFirstElement || (entry != prevEntry);
        isFirstElement = false;
        prevEntry = entry;
        if (isNewUnique) {
            inclusiveSum++;
            uniqueUb.SetValue(tileUniqueCnt, entry - static_cast<int32_t>(rankId_) * numEntriesPerRank_);
            tileUniqueCnt++;
        }
        compactIdxBuf.SetValue(i, runningOffset + inclusiveSum - 1);
        recvIdxBuf.SetValue(i, compUb.GetValue(i));
    }
    return tileUniqueCnt;
}

__aicore__ inline void EngramFetchGradUnique::WriteBatchUnique(uint32_t tileUniqueCnt, int32_t &runningUniqueOffset,
                                                               GM_ADDR uniqueLocalEntryOutGM)
{
    AscendC::LocalTensor<int32_t> pingInt32 = pingBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> uniqueUb = pingInt32[Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::GlobalTensor<int32_t> uniqueEntryOutGM;
    uniqueEntryOutGM.SetGlobalBuffer((__gm__ int32_t *)uniqueLocalEntryOutGM);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::DataCopyParams ueParams{1U, static_cast<uint16_t>(tileUniqueCnt * sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(uniqueEntryOutGM[runningUniqueOffset], uniqueUb, ueParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    runningUniqueOffset += static_cast<int32_t>(tileUniqueCnt);
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessScatterBatch(
    uint32_t cur, uint32_t end, int32_t &runningOffset, int32_t &runningUniqueOffset,
    int32_t &prevEntry, bool &isFirstElement,
    GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
    GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM, GM_ADDR sortCompanionGM)
{
    uint32_t batchLen = end - cur;
    if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
        batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
    }

    int32_t inclusiveSum = 0;
    uint32_t tileUniqueCnt = ProcessBatchUnique(
        cur, batchLen, runningOffset, prevEntry, isFirstElement, inclusiveSum,
        recvLocalEntryOutGM, sortCompanionGM);

    uint32_t maxGradPerBatch = Mc2Kernel::TILE_BYTES / static_cast<uint32_t>(hiddenBytes_);
    if (maxGradPerBatch < 1U) {
        maxGradPerBatch = 1U;
    }
    if (maxGradPerBatch > Mc2Kernel::GRAD_SUB_BATCH) {
        maxGradPerBatch = Mc2Kernel::GRAD_SUB_BATCH;
    }
    for (uint32_t subStart = 0; subStart < batchLen; subStart += maxGradPerBatch) {
        uint32_t subLen = batchLen - subStart;
        if (subLen > maxGradPerBatch) {
            subLen = maxGradPerBatch;
        }
        ScatterGradSubBatch(subStart, subLen, gradUniqueOutGM, gradUniqueFp32GM, recvGradGM);
    }

    if (tileUniqueCnt > 0) {
        WriteBatchUnique(tileUniqueCnt, runningUniqueOffset, uniqueLocalEntryOutGM);
    }
    runningOffset += inclusiveSum;
    return cur + batchLen;
}

__aicore__ inline void EngramFetchGradUnique::ScatterAddAtomicParallel(
    uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM,
    GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM,
    GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM)
{
    uint32_t start;
    uint32_t end;
    LoadCoreRange(numRecv, coreStartGM, start, end);
    if (start >= end) {
        return;
    }

    int32_t preCoreOffset = ComputePreCoreOffset(segCountGM);
    int32_t runningOffset = preCoreOffset;
    int32_t runningUniqueOffset = preCoreOffset;
    int32_t prevEntry = 0;
    bool isFirstElement = true;

    uint32_t cur = start;
    while (cur < end) {
        cur = ProcessScatterBatch(cur, end, runningOffset, runningUniqueOffset, prevEntry, isFirstElement,
                                  recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM,
                                  gradUniqueFp32GM, recvGradGM, sortCompanionGM);
    }
}

__aicore__ inline void EngramFetchGradUnique::CastGradUniqueToOutput(
    GM_ADDR segCountGM, GM_ADDR numUniqueOutGM, GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM)
{
    AscendC::GlobalTensor<int32_t> segCountGM_;
    segCountGM_.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
    AscendC::LocalTensor<int32_t> segCountUb = tempBuf_->Get<int32_t>();
    AscendC::DataCopyPadExtParams<int32_t> scPad{false, 0, 0, 0};
    AscendC::DataCopyExtParams scParams{1U, static_cast<uint32_t>(totalBlocks_ * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(segCountUb, segCountGM_, scParams, scPad);
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

    if (numUnique == 0 || outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT) {
        AscendC::SyncAll<true>();
        return;
    }

    uint32_t totalElements = static_cast<uint32_t>(numUnique) * static_cast<uint32_t>(hiddenDim_);
    uint32_t chunk = (totalElements + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t start = aivId_ * chunk;
    uint32_t end = start + chunk;
    if (end > totalElements) {
        end = totalElements;
    }
    if (start >= end) {
        AscendC::SyncAll<true>();
        return;
    }

    AscendC::GlobalTensor<float> gradUniqueFp32GMT;
    gradUniqueFp32GMT.SetGlobalBuffer((__gm__ float *)gradUniqueFp32GM);
    AscendC::GlobalTensor<uint8_t> gradUniqueOutGMT;
    gradUniqueOutGMT.SetGlobalBuffer((__gm__ uint8_t *)gradUniqueOutGM);

    AscendC::LocalTensor<float> fp32Tile = pingBuf_->Get<float>();
    uint32_t tileLen = Mc2Kernel::TILE_BYTES / sizeof(float);

    uint32_t off = start;
    while (off < end) {
        uint32_t thisLen = end - off;
        if (thisLen > tileLen) {
            thisLen = tileLen;
        }

        AscendC::DataCopyExtParams inParams{1U, static_cast<uint32_t>(thisLen * sizeof(float)), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<float> inPad{false, 0, 0, 0};
        AscendC::DataCopyPad(fp32Tile, gradUniqueFp32GMT[off], inParams, inPad);
        SyncFunc<AscendC::HardEvent::MTE2_V>(*pipe_);

        if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
            AscendC::LocalTensor<bfloat16_t> outTile = pongBuf_->Get<bfloat16_t>();
            AscendC::Cast(outTile, fp32Tile, AscendC::RoundMode::CAST_RINT, thisLen);
            SyncFunc<AscendC::HardEvent::V_MTE3>(*pipe_);
            AscendC::DataCopyParams outParams{1U, static_cast<uint16_t>(thisLen * sizeof(bfloat16_t)), 0U, 0U};
            AscendC::DataCopyPad(gradUniqueOutGMT[static_cast<uint64_t>(off) * sizeof(bfloat16_t)],
                                 outTile.ReinterpretCast<uint8_t>(), outParams);
        } else {
            AscendC::LocalTensor<half> outTile = pongBuf_->Get<half>();
            AscendC::Cast(outTile, fp32Tile, AscendC::RoundMode::CAST_RINT, thisLen);
            SyncFunc<AscendC::HardEvent::V_MTE3>(*pipe_);
            AscendC::DataCopyParams outParams{1U, static_cast<uint16_t>(thisLen * sizeof(half)), 0U, 0U};
            AscendC::DataCopyPad(gradUniqueOutGMT[static_cast<uint64_t>(off) * sizeof(half)],
                                 outTile.ReinterpretCast<uint8_t>(), outParams);
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>(*pipe_);
        off += thisLen;
    }
    AscendC::SyncAll<true>();
}

} // namespace EngramFetchGradUnique

#endif
