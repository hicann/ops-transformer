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
 * \file moe_v3_counting_sort_unfull_load.h
 * \brief 计数排序非全载（CutOrigin）模板
 */
#ifndef MOE_V3_COUNTING_SORT_UNFULL_LOAD_H
#define MOE_V3_COUNTING_SORT_UNFULL_LOAD_H

#include "moe_v3_common.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t EXERPT_TOKENS_CUMSUM = 0;

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ScatterPairsStableSimt(
    int32_t batchSize, __ubuf__ int32_t *flatIdxLocalAddr, __ubuf__ int32_t *rankLocalAddr, __gm__ int32_t *dstGmAddr)
{
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < batchSize; i += static_cast<int32_t>(blockDim.x)) {
        dstGmAddr[rankLocalAddr[i]] = flatIdxLocalAddr[i];
    }
}

template <typename T>
class MoeV3CutOriginPhaseAB {
public:
    __aicore__ inline MoeV3CutOriginPhaseAB(){};
    __aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokens, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FilterAndCountChunked();
    __aicore__ inline void CountChunkWithFilter(int64_t chunkLength, int64_t gmFlatOffset, int64_t &pairCursor);
    // 过滤核心：接收 CountChunkWithFilter 载入的 expertIdxLocal，调用方不再重复 DataCopyPad
    __aicore__ inline void CountChunkFilteredWithLoaded(int64_t chunkLength, int64_t gmFlatOffset,
                                                        LocalTensor<int32_t> &expertIdxLocal, int64_t &pairCursor);
    // 过滤两段式：构造 [expertStart_, expertEnd_) 命中掩码 / 按掩码收拢命中并计数写回（细分以缩小
    // CountChunkWithFilter）
    __aicore__ inline void BuildChunkFilterMask(int64_t chunkLength, int64_t alignedLen,
                                                LocalTensor<int32_t> &expertIdxLocal,
                                                LocalTensor<float> &expertIdxFp32Local,
                                                LocalTensor<uint8_t> &compareMask0, LocalTensor<uint8_t> &compareMask1,
                                                LocalTensor<uint8_t> &gatherMaskLocal);
    __aicore__ inline void GatherAndCountFilteredHits(int64_t chunkLength, int64_t gmFlatOffset,
                                                      LocalTensor<int32_t> &expertIdxLocal,
                                                      LocalTensor<uint8_t> &gatherMaskLocal,
                                                      LocalTensor<int32_t> &gatheredExpertLocal,
                                                      LocalTensor<int32_t> &flatIdxBufferLocal,
                                                      LocalTensor<int32_t> &gatheredIdxLocal, int64_t &pairCursor);
    __aicore__ inline void WritePairBatchToWs(int64_t pairCursor, int64_t count, LocalTensor<int32_t> &idxSrc,
                                              LocalTensor<int32_t> &expertSrc);
    __aicore__ inline void WriteExpertCountToWorkspace();
    // Phase B
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void ScatterToSortedRowIdx();
    __aicore__ inline void WriteExpandedExpertIdx();
    __aicore__ inline void WriteExpertIdxValue();
    __aicore__ inline void WriteExpertTotalCount();
    __aicore__ inline void WriteExpertTokens();

private:
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;

    TPipe *pipe_;
    TBuf<TPosition::VECCALC> buf_;

    // GM tensors
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> pairsWorkspaceGm_;
    GlobalTensor<int32_t> expertCountWorkspaceGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<int32_t> expertIdxValueGm_;
    GlobalTensor<int64_t> expertTokensGm_;

    // Tiling params
    int64_t blockIdx_;
    int64_t n_;
    int64_t k_;
    int64_t expertStart_;
    int64_t expertEnd_;
    int64_t actualExpertNum_;
    int64_t expertNum_;
    int64_t rowIdxType_;
    int64_t filterNeedCoreNum_;
    int64_t filterChunkSize_;
    int64_t expertTokensNumFlag_;
    int64_t expertTokensNumType_;
    int64_t dropPadMode_;

    // Per-core distribution（Phase A）
    int64_t coreFlatStart_;
    int64_t coreEntries_;

    // Derived constants
    int64_t expertCountStride_;
    int64_t pairsPerCore_;
    int64_t maxChunks_;
    int64_t chunkAligned_;
    int64_t maskBytes_;

    int64_t dropPadNeedCoreNum_;
    int64_t dropPadPerCoreRows_;
    int64_t dropPadLastCoreRows_;

    // UB offsets
    int64_t expertCountLocalOffset_;
    int64_t totalCountLocalOffset_;
    int64_t prefixSumLocalOffset_;
    int64_t oneCoreExpertCountLocalOffset_; // batchBuf
    int64_t expertTokensLocalOffset_;
    int64_t expertCountAlign_;
    int64_t persistentSize_;
    int64_t totalBufSize_;
    int64_t batchBufSize_;
    int64_t pairsBatchElements_;
    int64_t scatterFlatIdxOffset_;
    int64_t scatterExpertIdxOffset_;
    int64_t scatterIdxBufOffset_;

    // Results
    int64_t coreTotalPairs_;
    int64_t expertTotalCount_;
};

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokens,
                                                      GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling,
                                                      TPipe *pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();

    n_ = tiling->n;
    k_ = tiling->k;
    expertStart_ = tiling->expertStart;
    expertEnd_ = tiling->expertEnd;
    actualExpertNum_ = tiling->actualExpertNum;
    expertNum_ = tiling->expertNum;
    rowIdxType_ = tiling->rowIdxType;
    filterNeedCoreNum_ = tiling->countingSortParamsOp.filterNeedCoreNum;
    expertTokensNumFlag_ = tiling->expertTokensNumFlag;
    expertTokensNumType_ = tiling->expertTokensNumType;
    dropPadMode_ = tiling->dropPadMode;
    filterChunkSize_ = COUNTING_SORT_FILTER_CHUNK_SIZE;

    int64_t filterPerCoreTokens = tiling->countingSortParamsOp.filterPerCoreTokens;
    int64_t coreTokenStart = blockIdx_ * filterPerCoreTokens;
    int64_t coreTokenEnd = Min(coreTokenStart + filterPerCoreTokens, n_);
    if (blockIdx_ == filterNeedCoreNum_ - 1) {
        coreTokenEnd = n_;
    }
    coreEntries_ = (coreTokenEnd - coreTokenStart) * k_;
    coreFlatStart_ = coreTokenStart * k_;

    expertCountStride_ = AlignElem(actualExpertNum_, static_cast<int64_t>(8));
    chunkAligned_ = AlignElem(filterChunkSize_, static_cast<int64_t>(8));
    maxChunks_ = Ceil(coreEntries_, filterChunkSize_);
    maskBytes_ = AlignElem(Ceil(chunkAligned_, static_cast<int64_t>(8)), static_cast<int64_t>(sizeof(int8_t)));

    // Workspace stride: consistent across all cores (use max possible per-core entries)
    int64_t lastCoreTokens = n_ - (filterNeedCoreNum_ - 1) * filterPerCoreTokens;
    int64_t maxCoreTokens = Max(filterPerCoreTokens, lastCoreTokens);
    int64_t maxCoreEntries = maxCoreTokens * k_;
    pairsPerCore_ = AlignElem(maxCoreEntries, static_cast<int64_t>(8)) * 2;

    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);

    if (rowIdxType_ == GATHER) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> expandedRowIdxGmTmp = expandedRowIdxGm_[filterPerCoreTokens * k_ * blockIdx_];
            if (blockIdx_ == filterNeedCoreNum_ - 1) {
                InitGlobalMemory(expandedRowIdxGmTmp, lastCoreTokens * k_, -1);
            } else {
                InitGlobalMemory(expandedRowIdxGmTmp, maxCoreEntries, -1);
            }
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
        SyncAll();
    }
    int64_t wsOffset = tiling->countingSortParamsOp.pairsWsOffset;
    pairsWorkspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + wsOffset);
    wsOffset += filterNeedCoreNum_ * pairsPerCore_;
    expertCountWorkspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + wsOffset);

    sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                    Align(n_ * k_, static_cast<int64_t>(sizeof(int32_t))));
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                        Align(n_ * k_, static_cast<int64_t>(sizeof(int32_t))) * 2 +
                                        Align(actualExpertNum_, static_cast<int64_t>(sizeof(int32_t))));
    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace);
    expertIdxValueGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                      Align(n_ * k_, static_cast<int64_t>(sizeof(int32_t))) * 2 +
                                      Align(actualExpertNum_, static_cast<int64_t>(sizeof(int32_t))));

    // SCATTER + useGatherCopy（逆表协议）：本模板只产"位置->源"正表(expandedRowIdx)，
    // 不产 stage3 RowIdxGather 消费/产出的 "源->位置" 逆表(workspace+Align(n*k))。
    // GatherOut 在 useGatherCopy 下按全源域 [0,n*k) 遍历逆表，仅 outIndex<0 视为空位；
    // 专家子区间(ep_==1)时正表只覆盖 [0,expertTotalCount)，大量无效源槽若残留旧值会被误当合法 dest，
    // 因此在阶段切换前把逆表槽整段预填 -1（与 GATHER 模式预填 expandedRowIdx=-1 同理）。
    // ep_==0 为全域正置换，RowIdxGather 会写满每个有效源，无需预填。
    if (rowIdxType_ == SCATTER && tiling->useGatherCopy == 1) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> sortedRowIdxGmTmp = sortedRowIdxGm_[filterPerCoreTokens * k_ * blockIdx_];
            if (blockIdx_ == filterNeedCoreNum_ - 1) {
                InitGlobalMemory(sortedRowIdxGmTmp, lastCoreTokens * k_, -1);
            } else {
                InitGlobalMemory(sortedRowIdxGmTmp, maxCoreEntries, -1);
            }
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
        SyncAll();
    }
    dropPadNeedCoreNum_ = tiling->srcToDstDropPadParamsOp.needCoreNum;
    dropPadPerCoreRows_ = tiling->srcToDstDropPadParamsOp.perCoreRows;
    dropPadLastCoreRows_ = tiling->srcToDstDropPadParamsOp.lastCoreRows;
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokens);
    }

    // UB 布局及各段大小由 tiling 下发（host 统一计算），kernel 仅按偏移使用
    expertCountLocalOffset_ = 0;
    expertCountAlign_ = tiling->countingSortParamsOp.coutSortExpertCountAlign;
    persistentSize_ = tiling->countingSortParamsOp.coutSortPersistentSize;
    batchBufSize_ = tiling->countingSortParamsOp.coutSortBatchBufSize;
    pairsBatchElements_ = tiling->countingSortParamsOp.coutSortPairsBatchElements;

    totalCountLocalOffset_ = persistentSize_;
    prefixSumLocalOffset_ = persistentSize_ + expertCountAlign_;
    oneCoreExpertCountLocalOffset_ = persistentSize_ + 2 * expertCountAlign_;
    // expertTokensLocal 复用 prefixSum + batchBuf 区（scalar loop 后 prefixSum 不再需要）
    expertTokensLocalOffset_ = prefixSumLocalOffset_;

    // 离散搬出临时区：pairs batch（flatIdx + expertIdx）+ idxBuf，放在 batchBuf 之后
    int64_t pairsBatchSlot = AlignElem(pairsBatchElements_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    int64_t scatterBase = persistentSize_ + 2 * expertCountAlign_ + batchBufSize_;
    scatterFlatIdxOffset_ = scatterBase;
    scatterExpertIdxOffset_ = scatterBase + pairsBatchSlot;
    scatterIdxBufOffset_ = scatterExpertIdxOffset_ + pairsBatchSlot;

    totalBufSize_ = tiling->countingSortParamsOp.coutSortTotalBufSize;
    pipe_->InitBuffer(buf_, totalBufSize_);
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::Process()
{
    if (blockIdx_ >= filterNeedCoreNum_) {
        SyncAll(); // 等 Phase A：各核计数写回 GM 完成
        SyncAll(); // 等 Phase B：离散搬出 sortedRowIdx 完成
        SyncAll(); // 等收尾：dropPad 桥接产物 / ExpertTokens 写 GM 完成
        return;
    }
    // Phase A: filter + count
    FilterAndCountChunked();
    WriteExpertCountToWorkspace();
    SyncAll();

    // Phase B: global offset + discrete scatter + bridge products
    ComputeGlobalOffset();
    ScatterToSortedRowIdx();
    SyncAll();
    if (dropPadMode_ == DROP_PAD_MODE) {
        // dropPad：离散搬出后补两个桥接产物（expandedExpertIdx + expertIdxValue），
        // 供 Stage3 RowIdxGatherDropPad 消费；expertTotalCount 在 dropPad 下不写（同一槽位是 expertIdxValue）。
        WriteExpandedExpertIdx();
        WriteExpertIdxValue();
    } else {
        WriteExpertTotalCount();
    }
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        WriteExpertTokens();
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::FilterAndCountChunked()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int64_t pairCursor = 0;

    for (int64_t chunkIdx = 0; chunkIdx < maxChunks_; chunkIdx++) {
        int64_t chunkStart = chunkIdx * filterChunkSize_;
        int64_t chunkLength = Min(filterChunkSize_, coreEntries_ - chunkStart);
        int64_t gmFlatOffset = coreFlatStart_ + chunkStart;

        CountChunkWithFilter(chunkLength, gmFlatOffset, pairCursor);
    }
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::CountChunkWithFilter(int64_t chunkLength, int64_t gmFlatOffset,
                                                                      int64_t &pairCursor)
{
    int64_t expertIdxOff = persistentSize_;
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxOff / sizeof(int32_t)];

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(chunkLength * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdxLocal, expertIdxGm_[gmFlatOffset], copyParams, padParams);

    // 载入同步：MTE2 载入完成 + 上一 chunk 的 MTE3 搬出完成
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    CountChunkFilteredWithLoaded(chunkLength, gmFlatOffset, expertIdxLocal, pairCursor);
}

// 过滤核心：expertIdxLocal 由 CountChunkWithFilter 载入后传入。
template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::CountChunkFilteredWithLoaded(int64_t chunkLength, int64_t gmFlatOffset,
                                                                              LocalTensor<int32_t> &expertIdxLocal,
                                                                              int64_t &pairCursor)
{
    int64_t off = persistentSize_ + AlignElem(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    int64_t expertIdxFp32Off = off;
    off += AlignElem(chunkAligned_ * static_cast<int64_t>(sizeof(float)), BLOCK_BYTES);
    int64_t compareMask0Off = off;
    off += maskBytes_;
    int64_t compareMask1Off = off;
    off += maskBytes_;
    int64_t gatherMaskOff = off;
    off += maskBytes_;
    int64_t flatIdxBufferOff = off;
    off += AlignElem(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    int64_t gatheredExpertOff = off;
    off += AlignElem(chunkAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    int64_t gatheredIdxOff = off;

    LocalTensor<float> expertIdxFp32Local = buf_.Get<float>()[expertIdxFp32Off / sizeof(float)];
    LocalTensor<uint8_t> compareMask0 = buf_.Get<uint8_t>()[compareMask0Off];
    LocalTensor<uint8_t> compareMask1 = buf_.Get<uint8_t>()[compareMask1Off];
    LocalTensor<uint8_t> gatherMaskLocal = buf_.Get<uint8_t>()[gatherMaskOff];
    LocalTensor<int32_t> flatIdxBufferLocal = buf_.Get<int32_t>()[flatIdxBufferOff / sizeof(int32_t)];
    LocalTensor<int32_t> gatheredExpertLocal = buf_.Get<int32_t>()[gatheredExpertOff / sizeof(int32_t)];
    LocalTensor<int32_t> gatheredIdxLocal = buf_.Get<int32_t>()[gatheredIdxOff / sizeof(int32_t)];

    int64_t alignedLen = Ceil(chunkLength, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;

    BuildChunkFilterMask(chunkLength, alignedLen, expertIdxLocal, expertIdxFp32Local, compareMask0, compareMask1,
                         gatherMaskLocal);
    GatherAndCountFilteredHits(chunkLength, gmFlatOffset, expertIdxLocal, gatherMaskLocal, gatheredExpertLocal,
                               flatIdxBufferLocal, gatheredIdxLocal, pairCursor);
}

// 段1：构造 [expertStart_, expertEnd_) 命中掩码
template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::BuildChunkFilterMask(int64_t chunkLength, int64_t alignedLen,
                                                                      LocalTensor<int32_t> &expertIdxLocal,
                                                                      LocalTensor<float> &expertIdxFp32Local,
                                                                      LocalTensor<uint8_t> &compareMask0,
                                                                      LocalTensor<uint8_t> &compareMask1,
                                                                      LocalTensor<uint8_t> &gatherMaskLocal)
{
    Cast(expertIdxFp32Local, expertIdxLocal, RoundMode::CAST_ROUND, alignedLen);
    PipeBarrier<PIPE_V>();

    if (chunkLength < alignedLen) {
        for (int32_t i = 0; i < alignedLen - chunkLength; i++) {
            expertIdxFp32Local.SetValue(chunkLength + i, static_cast<float>(-1));
        }
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    }

    CompareScalar(compareMask0, expertIdxFp32Local, static_cast<float>(expertStart_), CMPMODE::GE, alignedLen);
    PipeBarrier<PIPE_V>();
    CompareScalar(compareMask1, expertIdxFp32Local, static_cast<float>(expertEnd_), CMPMODE::LT, alignedLen);
    PipeBarrier<PIPE_V>();
    And(gatherMaskLocal.ReinterpretCast<uint16_t>(), compareMask0.ReinterpretCast<uint16_t>(),
        compareMask1.ReinterpretCast<uint16_t>(), Ceil(alignedLen, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE / 2);
    PipeBarrier<PIPE_V>();
}

// 段2：按掩码收拢命中并计数写回 pairs
template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::GatherAndCountFilteredHits(
    int64_t chunkLength, int64_t gmFlatOffset, LocalTensor<int32_t> &expertIdxLocal,
    LocalTensor<uint8_t> &gatherMaskLocal, LocalTensor<int32_t> &gatheredExpertLocal,
    LocalTensor<int32_t> &flatIdxBufferLocal, LocalTensor<int32_t> &gatheredIdxLocal, int64_t &pairCursor)
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    uint64_t rsvdCnt = 0;
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = DST_REP_STRIDE;
    gatherMaskParams.src1RepeatStride = DST_REP_STRIDE;
    GatherMask(gatheredExpertLocal, expertIdxLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(chunkLength), gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();

    int64_t filteredInChunk = static_cast<int64_t>(rsvdCnt);
    if (filteredInChunk > 0) {
        ArithProgression<int32_t>(flatIdxBufferLocal, static_cast<int32_t>(gmFlatOffset), 1,
                                  static_cast<int32_t>(chunkLength));
        PipeBarrier<PIPE_V>();

        uint64_t idxRsvdCnt = 0;
        GatherMask(gatheredIdxLocal, flatIdxBufferLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
                   static_cast<uint32_t>(chunkLength), gatherMaskParams, idxRsvdCnt);
        PipeBarrier<PIPE_V>();

        Adds(gatheredExpertLocal, gatheredExpertLocal, static_cast<int32_t>(-expertStart_),
             static_cast<int32_t>(filteredInChunk));

        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

        for (int64_t j = 0; j < filteredInChunk; j++) {
            int32_t expertOffset = gatheredExpertLocal.GetValue(j);
            int32_t curCount = expertCountLocal.GetValue(expertOffset);
            expertCountLocal.SetValue(expertOffset, curCount + 1);
        }

        WritePairBatchToWs(pairCursor, filteredInChunk, gatheredIdxLocal, gatheredExpertLocal);
        pairCursor += filteredInChunk;
    }
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WritePairBatchToWs(int64_t pairCursor, int64_t count,
                                                                    LocalTensor<int32_t> &idxSrc,
                                                                    LocalTensor<int32_t> &expertSrc)
{
    if (count <= 0) {
        return;
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyExtParams wpCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairCursor], idxSrc, wpCopyParams);
    DataCopyPad(pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairsPerCore_ / 2 + pairCursor], expertSrc, wpCopyParams);
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WriteExpertCountToWorkspace()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertCountStride_ * sizeof(int32_t)),
                                 0, 0, 0};
    DataCopyPad(expertCountWorkspaceGm_[blockIdx_ * expertCountStride_], expertCountLocal, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::ComputeGlobalOffset()
{
    int64_t batchCores = batchBufSize_ / expertCountAlign_;
    if (batchCores > filterNeedCoreNum_) {
        batchCores = filterNeedCoreNum_;
    }
    if (batchCores < 1) {
        batchCores = 1;
    }
    LocalTensor<int32_t> batchBufLocal = buf_.Get<int32_t>()[oneCoreExpertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> totalCountLocal = buf_.Get<int32_t>()[totalCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> prefixSumLocal = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    Duplicate(totalCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    Duplicate(prefixSumLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();

    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    coreTotalPairs_ = 0;
    for (int64_t batchStart = 0; batchStart < filterNeedCoreNum_; batchStart += batchCores) {
        int64_t batchEnd = Min(batchStart + batchCores, filterNeedCoreNum_);
        int64_t curBatchSize = batchEnd - batchStart;

        int64_t batchElements = curBatchSize * expertCountStride_;
        DataCopyExtParams loadParams{static_cast<uint16_t>(1), static_cast<uint32_t>(batchElements * sizeof(int32_t)),
                                     0, 0, 0};
        DataCopyPad(batchBufLocal, expertCountWorkspaceGm_[batchStart * expertCountStride_], loadParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        for (int64_t c = 0; c < curBatchSize; c++) {
            Add(totalCountLocal, totalCountLocal, batchBufLocal[c * expertCountStride_], expertCountStride_);
            PipeBarrier<PIPE_V>();
        }
        for (int64_t c = 0; c < curBatchSize; c++) {
            int64_t globalCoreIdx = batchStart + c;
            if (globalCoreIdx < blockIdx_) {
                Add(prefixSumLocal, prefixSumLocal, batchBufLocal[c * expertCountStride_], expertCountStride_);
                PipeBarrier<PIPE_V>();
            }
        }

        // 捕获本核 pair 总数（sum over experts of this core's expertCount）
        if (batchStart <= blockIdx_ && blockIdx_ < batchEnd) {
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            int64_t coreIdxInBatch = blockIdx_ - batchStart;
            for (int64_t e = 0; e < actualExpertNum_; e++) {
                coreTotalPairs_ += batchBufLocal.GetValue(coreIdxInBatch * expertCountStride_ + e);
            }
        }
        if (batchStart + batchCores < filterNeedCoreNum_) {
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        }
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int64_t cumulativeSum = 0;
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t totalForExpert = totalCountLocal.GetValue(e);
        int32_t prefixForExpert = prefixSumLocal.GetValue(e);
        expertCountLocal.SetValue(e, static_cast<int32_t>(cumulativeSum) + prefixForExpert);
        cumulativeSum += totalForExpert;
    }
    expertTotalCount_ = cumulativeSum;
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::ScatterToSortedRowIdx()
{
    if (coreTotalPairs_ <= 0) {
        return;
    }

    // Invalidate cache to see pairs written in Phase A
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(pairsWorkspaceGm_);
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> flatIdxLocal = buf_.Get<int32_t>()[scatterFlatIdxOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[scatterExpertIdxOffset_ / sizeof(int32_t)];

    // 目标 GM 基址：GATHER -> sortedRowIdx（workspace+Align(n*k)）；SCATTER -> expandedRowIdx（输出）
    __gm__ int32_t *dstGmAddr = (rowIdxType_ == GATHER) ? (__gm__ int32_t *)sortedRowIdxGm_.GetPhyAddr() :
                                                          (__gm__ int32_t *)expandedRowIdxGm_.GetPhyAddr();
    __ubuf__ int32_t *flatIdxAddr = (__ubuf__ int32_t *)flatIdxLocal.GetPhyAddr();
    __ubuf__ int32_t *rankAddr = (__ubuf__ int32_t *)expertIdxLocal.GetPhyAddr();

    DataCopyPadExtParams<int32_t> intPadParams{false, 0, 0, 0};

    int64_t pairCursor = 0;
    while (pairCursor < coreTotalPairs_) {
        int64_t batchSize = Min(pairsBatchElements_, coreTotalPairs_ - pairCursor);

        DataCopyExtParams pairLoadParams{static_cast<uint16_t>(1), static_cast<uint32_t>(batchSize * sizeof(int32_t)),
                                         0, 0, 0};
        DataCopyPad(flatIdxLocal, pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairCursor], pairLoadParams,
                    intPadParams);
        DataCopyPad(expertIdxLocal, pairsWorkspaceGm_[blockIdx_ * pairsPerCore_ + pairsPerCore_ / 2 + pairCursor],
                    pairLoadParams, intPadParams);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S); // MTE2 → scalar：UB 数据就绪
        for (int64_t i = 0; i < batchSize; i++) {
            int32_t e = expertIdxLocal.GetValue(i);
            int32_t pos = expertCountLocal.GetValue(e);
            expertIdxLocal.SetValue(i, pos);
            expertCountLocal.SetValue(e, pos + 1);
        }

        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        asc_vf_call<ScatterPairsStableSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, static_cast<int32_t>(batchSize), flatIdxAddr,
                                            rankAddr, dstGmAddr);
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);

        pairCursor += batchSize;
    }
}

// dropPad 专用
template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WriteExpandedExpertIdx()
{
    LocalTensor<int32_t> totalCountLocal = buf_.Get<int32_t>()[totalCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> prefixSumLocal = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> fillBuf = buf_.Get<int32_t>()[scatterFlatIdxOffset_ / sizeof(int32_t)];

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // 本核全局输出区间 [globalStart, globalEnd)：globalStart = sum_e prefixSumLocal[e]（之前所有核的 token 数）
    int64_t globalStart = 0;
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        globalStart += static_cast<int64_t>(prefixSumLocal.GetValue(e));
    }
    int64_t globalEnd = globalStart + coreTotalPairs_;

    // 每专家连续 run：expert e 占全局位置 [globalPrefix, globalPrefix + totalCountLocal[e])
    int64_t globalPrefix = 0;
    DataCopyExtParams runCp{1, 0, 0, 0, 0};
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t count = totalCountLocal.GetValue(e);
        int64_t runStart = Max(globalStart, globalPrefix);
        int64_t runEnd = Min(globalEnd, globalPrefix + static_cast<int64_t>(count));
        globalPrefix += count;
        if (runStart >= runEnd) {
            continue;
        }
        int32_t globalExpertId = static_cast<int32_t>(e + expertStart_);
        for (int64_t off = runStart; off < runEnd; off += pairsBatchElements_) {
            int64_t n = Min(pairsBatchElements_, runEnd - off);
            Duplicate(fillBuf, globalExpertId, static_cast<int32_t>(n));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            runCp.blockLen = static_cast<uint32_t>(n * sizeof(int32_t));
            DataCopyPad(expandedExpertIdxGm_[off], fillBuf, runCp);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }

    // 无效尾区 [expertTotalCount, n*k) 填 -1：blockIdx 0 负责（与所有核的有效写 [0, expertTotalCount) 无重叠）
    if (blockIdx_ == 0 && expertTotalCount_ < n_ * k_) {
        for (int64_t off = expertTotalCount_; off < n_ * k_; off += pairsBatchElements_) {
            int64_t n = Min(pairsBatchElements_, n_ * k_ - off);
            Duplicate(fillBuf, static_cast<int32_t>(-1), static_cast<int32_t>(n));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            runCp.blockLen = static_cast<uint32_t>(n * sizeof(int32_t));
            DataCopyPad(expandedExpertIdxGm_[off], fillBuf, runCp);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}

// dropPad 专用
template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WriteExpertIdxValue()
{
    if (blockIdx_ != 0) {
        return;
    }
    LocalTensor<int32_t> totalCountLocal = buf_.Get<int32_t>()[totalCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> idxBuf = buf_.Get<int32_t>()[scatterIdxBufOffset_ / sizeof(int32_t)];
    DataCopyExtParams cp{1, static_cast<uint32_t>(2 * sizeof(int32_t)), 0, 0, 0};

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    for (int64_t i = 0; i < dropPadNeedCoreNum_; i++) {
        int64_t segStart = i * dropPadPerCoreRows_;
        int64_t segLen = (i == dropPadNeedCoreNum_ - 1) ? dropPadLastCoreRows_ : dropPadPerCoreRows_;
        int64_t segEnd = segStart + segLen;

        int32_t lastExpertId = -1;
        int32_t lastTokenCount = 0;
        if (segStart < expertTotalCount_) {
            // 段内最后一个有效 token 的全局位置
            int64_t lastValidPos = Min(segEnd, expertTotalCount_) - 1;
            // 线性找 e：globalPrefix[e] <= lastValidPos < globalPrefix[e] + totalCountLocal[e]
            int64_t e = 0;
            int64_t prefix = 0;
            while (e < actualExpertNum_) {
                int32_t cnt = totalCountLocal.GetValue(e);
                if (lastValidPos >= prefix && lastValidPos < prefix + static_cast<int64_t>(cnt)) {
                    break;
                }
                prefix += cnt;
                e++;
            }
            // 边界专家在段内计数 = lastValidPos - max(segStart, prefix) + 1
            lastExpertId = static_cast<int32_t>(e + expertStart_);
            lastTokenCount = static_cast<int32_t>(lastValidPos - Max(segStart, prefix) + 1);
        }
        idxBuf.SetValue(0, lastExpertId);
        idxBuf.SetValue(1, lastTokenCount);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expertIdxValueGm_[i * 2], idxBuf, cp);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WriteExpertTotalCount()
{
    // 非 dropPad 才写：dropPad 场景同一 GM 槽位是 expertIdxValueGm_，且 Stage3 dropPad 不读此值
    if (blockIdx_ == 0 && dropPadMode_ != DROP_PAD_MODE) {
        LocalTensor<int32_t> totalCountBuf = buf_.Get<int32_t>()[scatterIdxBufOffset_ / sizeof(int32_t)];
        totalCountBuf.SetValue(0, static_cast<int32_t>(expertTotalCount_));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(expertTotalCountGm_[0], totalCountBuf, copyParams);
    }
}

template <typename T>
__aicore__ inline void MoeV3CutOriginPhaseAB<T>::WriteExpertTokens()
{
    if (blockIdx_ == 0 && expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        LocalTensor<int32_t> totalCountLocal = buf_.Get<int32_t>()[totalCountLocalOffset_ / sizeof(int32_t)];
        LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];

        if (expertTokensNumType_ == EXERPT_TOKENS_COUNT) {
            Cast(expertTokensLocal, totalCountLocal, RoundMode::CAST_NONE, static_cast<int32_t>(actualExpertNum_));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>(actualExpertNum_ * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);

        } else if (expertTokensNumType_ == EXERPT_TOKENS_NONE) {
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            int64_t cumsum = 0;
            for (int64_t e = 0; e < actualExpertNum_; e++) {
                cumsum += static_cast<int64_t>(totalCountLocal.GetValue(e));
                expertTokensLocal.SetValue(e, cumsum);
            }
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>(actualExpertNum_ * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);

        } else if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
            int64_t kvTotalElements =
                (actualExpertNum_ == expertNum_) ? actualExpertNum_ * 2 : (actualExpertNum_ + 1) * 2;
            Duplicate(expertTokensLocal, static_cast<int64_t>(0), static_cast<int32_t>(kvTotalElements));
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            int64_t kvOffset = 0;
            for (int64_t e = 0; e < actualExpertNum_; e++) {
                int32_t count = totalCountLocal.GetValue(e);
                if (count != 0) {
                    expertTokensLocal.SetValue(kvOffset * 2, static_cast<int64_t>(e + expertStart_));
                    expertTokensLocal.SetValue(kvOffset * 2 + 1, static_cast<int64_t>(count));
                    kvOffset++;
                }
            }
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>(kvTotalElements * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);
        }
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_UNFULL_LOAD_H
