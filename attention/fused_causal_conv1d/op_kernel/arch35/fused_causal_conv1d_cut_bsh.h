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
 * \file fused_causal_conv1d_cut_bsh.h
 * \brief BSH kernel — simplified rewrite with CopyIn/Compute separation,
 *        unified metadata (PrepareBatchMeta), unified cache writeback (WriteCacheToStates),
 *        batch-level sync, and distributed APC fill from xLocal (UB).
 */

#ifndef FUSED_CAUSAL_CONV1D_CUT_BSH_H
#define FUSED_CAUSAL_CONV1D_CUT_BSH_H

#include "kernel_operator.h"
#include "vf/fused_causal_conv1d_128dim.h"
#include "vf/fused_causal_conv1d_state_single_tail.h"
#include "vf/fused_causal_conv1d_no_state_single_tail.h"
#include "vf/fused_causal_conv1d_no_state_double_tail.h"
#include "fused_causal_conv1d_cut_bsh_struct.h"

namespace FusedCausalConv1dCutBSHNs {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t DIM_ALIGN = 64;

template <typename T>
class FusedCausalConv1dCutBSH {
public:
    __aicore__ inline FusedCausalConv1dCutBSH() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR numAcceptedToken, GM_ADDR numComputedTokens,
                                GM_ADDR blockIdxFirst, GM_ADDR blockIdxLast, GM_ADDR initialStateIdx, GM_ADDR y,
                                GM_ADDR workspace, const FusedCausalConv1dCutBSHTilingData *tiling);
    __aicore__ inline void Process();

private:
    // ---- Init helpers ----
    __aicore__ inline void ParseTiling(const FusedCausalConv1dCutBSHTilingData *t);
    __aicore__ inline void InitQueues();
    __aicore__ inline void LoadMetadata();

    // ---- Three-stage pipeline (BH template pattern) ----
    // CopyIn:  MTE2 loads x + weight into queues
    __aicore__ inline void CopyIn(uint32_t curBsStart, uint32_t curBS, uint32_t dimStart, uint32_t dimSize);
    // Compute: DeQue, load cache per-batch, conv all tokens, write cache, output y
    __aicore__ inline void Compute(uint32_t curBsStart, uint32_t curBS, uint32_t dimStart, uint32_t dimSize,
                                   uint32_t dimBlocks, uint32_t iStart, uint32_t batchIdx, uint32_t curSeqIdx,
                                   uint64_t coreGlobalEnd, uint32_t &endBatchIdx);

    __aicore__ inline void WriteCacheToStates(LocalTensor<T> &cacheLocal, LocalTensor<T> &xLocal, uint32_t chunkEnd,
                                              uint32_t curBatchLen, uint32_t cachedStateLen, bool hasCacheLocal,
                                              int64_t writeCIdx, uint32_t dimStart, uint32_t dimSize,
                                              uint32_t dimBlocks, uint32_t cacheSkipBlocks, uint64_t batchGmEnd);

    // Deferred write (post-SyncAll) for first batch on non-first BS cores.
    // Reads cache prefix from GM and x suffix from GM.
    __aicore__ inline void WriteDeferredCacheSimple();
    __aicore__ inline void ExecuteDeferredRunningCache();

    // ---- Distributed APC prefix fill ----
    __aicore__ inline void FillApcChunksInRange(uint32_t curBatch, uint64_t batchSt, uint32_t batchLen,
                                                uint32_t curBsStart, uint32_t curBS, LocalTensor<T> &xLocal,
                                                uint32_t dimStart, uint32_t dimSize, uint32_t dimBlocks,
                                                uint32_t cacheSkip);
    // cache-prefix + collision-deferred chunks: handled once pre-SyncAll.
    // O(1) early-exit when neither cache-prefix nor collision exists.
    __aicore__ inline void FillApcCachePrefixChunks();

    // ---- inplace finalize: copy y from workspace back to xGM_ ----
    //  仅 inplace_==true 时调用，在 FillApcCachePrefixChunks 之后执行。
    //  按本核 BS/Dim 切分遍历自己的 y 输出范围，借用 xQueue_ 做 UB 中转搬运。
    __aicore__ inline void CopyYToX();

    // ---- Batch metadata (single source of truth) ----
    struct BatchMeta {
        int64_t readCIdx;
        int64_t writeCIdx;
        uint32_t mtpOffset;
        uint32_t cachedStateLen;
        bool useZeroCache;
        uint32_t lastResetIdx; // conv_mode==1: number of leading tokens to zero out
    };
    __aicore__ inline void PrepareBatchMeta(uint32_t batchIdx, uint32_t curSeqIdx, BatchMeta &m);

    // ---- Helpers ----
    __aicore__ inline uint32_t FindBatchIdx(uint64_t globalSeqIdx);
    __aicore__ inline void ResolveCacheIndices(uint32_t batchIdx, int64_t &readCIdx, int64_t &writeCIdx);
    template <HardEvent E>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<E>(eventId);
        WaitFlag<E>(eventId);
    }
    static __aicore__ inline uint32_t AlignUp(uint32_t n, uint32_t align)
    {
        return (n + align - 1) / align * align;
    }

    // ---- Helper: 2D cache_indices read for APC ----
    __aicore__ inline int64_t GetApcCacheLine(uint32_t batchIdx, uint32_t blockIdx)
    {
        return static_cast<int64_t>(
            cacheIndices2DGM_.GetValue(static_cast<uint64_t>(batchIdx) * maxNumBlocks_ + blockIdx));
    }

    // Cache batch-invariant APC params (firstBlk/lastBlk/numComputed/lastFull/readCIdx).
    // Called at the start of FillApcChunksInRange; no-op when curBatch == cached.
    __aicore__ inline void RefreshApcState(uint32_t curBatch, uint32_t batchLen);

    __aicore__ inline uint32_t GetDimStart() const
    {
        if (dimIdx_ < dimRemainderCores_) {
            return dimIdx_ * dimBlockFactor_;
        }
        return dimRemainderCores_ * dimBlockFactor_ + (dimIdx_ - dimRemainderCores_) * dimBlockTailFactor_;
    }

    // GetDimTotal：本核负责的 dim 总长度（元素单位）。last core 额外承担 dimRemainderElems_。
    __aicore__ inline uint32_t GetDimTotal() const
    {
        uint32_t dimSize = isDimTailCore_ ? dimBlockTailFactor_ : dimBlockFactor_;
        if (isLastDimCore_)
            dimSize += dimRemainderElems_;
        return dimSize;
    }

    // GetMaxUbDim：本核的 UB 主块 dim 大小（用于 buffer 容量与 UB 循环切分）。
    __aicore__ inline uint32_t GetMaxUbDim() const
    {
        if (isLastDimCore_)
            return lastCoreubFactorDim_;
        if (isDimTailCore_)
            return tailBlockubFactorDim_;
        return ubFactorDim_;
    }

    // ======================== Members ========================

    // ---- Tiling params ----
    uint32_t dim_;
    uint32_t kernelWidth_;
    uint32_t batchSize_;
    uint32_t stateLen_;

    uint32_t loopNumBS_;
    uint32_t loopNumDim_;
    uint32_t ubFactorBS_;
    uint32_t ubTailFactorBS_;
    uint32_t ubFactorDim_;
    uint32_t ubTailFactorDim_;

    uint32_t tailBlockloopNumBS_;
    uint32_t tailBlockloopNumDim_;
    uint32_t tailBlockubFactorBS_;
    uint32_t tailBlockubTailFactorBS_;
    uint32_t tailBlockubFactorDim_;
    uint32_t tailBlockubTailFactorDim_;

    // dimRemainderElems_ == 0 时等价于普通尾核参数（向后兼容）
    uint32_t dimRemainderElems_;
    uint32_t lastCoreloopNumDim_;
    uint32_t lastCoreubFactorDim_;
    uint32_t lastCoreubTailFactorDim_;

    uint32_t dimCoreNum_;
    uint32_t dimRemainderCores_;
    uint32_t dimBlockFactor_;
    uint32_t dimBlockTailFactor_;

    uint32_t bsRemainderCores_;
    uint32_t bsBlockFactor_;
    uint32_t bsBlockTailFactor_;

    uint32_t realCoreNum_;

    uint32_t xStride_;
    uint32_t cacheStride0_;
    uint32_t cacheStride1_;
    uint32_t residualConnection_;

    int64_t padSlotId_;

    bool apcEnabled_;
    uint32_t blockSize_;
    uint32_t maxNumBlocks_;
    uint32_t convMode_;
    bool hasAcceptTokenNum_;
    bool hasNumComputedTokens_;
    bool hasCacheIndices_;
    bool inplace_;

    // Runtime core index
    uint32_t bsIdx_;
    uint32_t dimIdx_;

    bool isBsTailCore_;
    bool isDimTailCore_;
    bool isLastDimCore_;

    // ---- Pipeline & GM ----
    TPipe pipe_;

    GlobalTensor<T> xGM_;
    GlobalTensor<T> weightGM_;
    GlobalTensor<T> cacheStatesGM_;
    GlobalTensor<int32_t> cacheIndicesGM_;
    GlobalTensor<int32_t> cacheIndices2DGM_;
    GlobalTensor<int32_t> seqStartIndexGM_;
    GlobalTensor<int32_t> numComputedTokensGM_;
    GlobalTensor<int32_t> acceptTokenGM_;
    GlobalTensor<int32_t> blockIdxFirstGM_;
    GlobalTensor<int32_t> blockIdxLastGM_;
    GlobalTensor<int32_t> initialStateIdxGM_;
    GlobalTensor<T> yGM_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> cacheQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECIN, 1> weightQueue_;

    // ---- TBuf metadata ----
    TBuf<TPosition::VECCALC> seqStartBuf_;
    TBuf<TPosition::VECCALC> cacheIdxBuf_;
    TBuf<TPosition::VECCALC> numComputedBuf_;
    TBuf<TPosition::VECCALC> acceptTokenBuf_;
    TBuf<TPosition::VECCALC> blockIdxFirstBuf_;
    TBuf<TPosition::VECCALC> blockIdxLastBuf_;
    TBuf<TPosition::VECCALC> initialStateIdxBuf_;

    // Persistent LocalTensors
    LocalTensor<int32_t> seqStartLocal_;
    LocalTensor<int32_t> cacheIdxLocal_;
    LocalTensor<int32_t> numComputedLocal_;
    LocalTensor<int32_t> acceptTokenLocal_;
    LocalTensor<int32_t> blockIdxFirstLocal_;
    LocalTensor<int32_t> blockIdxLastLocal_;
    LocalTensor<int32_t> initialStateIdxLocal_;

    // ---- Per-batch APC state cache (avoids redundant TBuf reads + scalar math) ----
    // All values are batch-invariant.  Cache populated on first access for a batch,
    // reused until curBatch changes.
    struct BatchApcState {
        uint32_t batchIdx;
        int32_t firstBlk;
        int32_t lastBlk;
        int32_t nBlockToFill;
        int32_t lastFull;
        int64_t readCIdx;
    } apcState_;

    // ---- Cross-core state (simplified: 1 struct instead of 4 vars) ----
    struct DeferredWriteState {
        bool active;
        uint32_t batchIdx;
        int64_t writeCIdx;
    } deferred_;
    struct CacheDeferredState {
        bool active;
        uint32_t batchIdx;
        int64_t writeCIdx;
    } cacheDeferred_; // deferred running cache for readCIdx==writeCIdx prefix batches
    uint32_t lastBatchIdx_;
    uint32_t firstBatchIdx_;
};

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::ParseTiling(const FusedCausalConv1dCutBSHTilingData *tiling)
{
    dim_ = tiling->dim;
    kernelWidth_ = tiling->kernelWidth;
    batchSize_ = tiling->batch;

    loopNumBS_ = tiling->loopNumBS;
    loopNumDim_ = tiling->loopNumDim;
    ubFactorBS_ = tiling->ubFactorBS;
    ubTailFactorBS_ = tiling->ubTailFactorBS;
    ubFactorDim_ = tiling->ubFactorDim;
    ubTailFactorDim_ = tiling->ubTailFactorDim;

    tailBlockloopNumBS_ = tiling->tailBlockloopNumBS;
    tailBlockloopNumDim_ = tiling->tailBlockloopNumDim;
    tailBlockubFactorBS_ = tiling->tailBlockubFactorBS;
    tailBlockubTailFactorBS_ = tiling->tailBlockubTailFactorBS;
    tailBlockubFactorDim_ = tiling->tailBlockubFactorDim;
    tailBlockubTailFactorDim_ = tiling->tailBlockubTailFactorDim;

    dimRemainderElems_ = static_cast<uint32_t>(tiling->dimRemainderElems);
    lastCoreloopNumDim_ = static_cast<uint32_t>(tiling->lastCoreloopNumDim);
    lastCoreubFactorDim_ = static_cast<uint32_t>(tiling->lastCoreubFactorDim);
    lastCoreubTailFactorDim_ = static_cast<uint32_t>(tiling->lastCoreubTailFactorDim);

    dimCoreNum_ = tiling->dimCoreNum;
    dimRemainderCores_ = tiling->dimRemainderCores;
    dimBlockFactor_ = tiling->dimBlockFactor;
    dimBlockTailFactor_ = tiling->dimBlockTailFactor;

    bsRemainderCores_ = tiling->bsRemainderCores;
    bsBlockFactor_ = tiling->bsBlockFactor;
    bsBlockTailFactor_ = tiling->bsBlockTailFactor;

    realCoreNum_ = tiling->realCoreNum;

    xStride_ = tiling->xStride;
    cacheStride0_ = tiling->cacheStride0;
    cacheStride1_ = tiling->cacheStride1;
    residualConnection_ = tiling->residualConnection;

    padSlotId_ = tiling->padSlotId;

    apcEnabled_ = (tiling->apcEnabled != 0);
    blockSize_ = static_cast<uint32_t>(tiling->blockSize);
    maxNumBlocks_ = static_cast<uint32_t>(tiling->maxNumBlocks);
    convMode_ = static_cast<uint32_t>(tiling->convMode);
    hasAcceptTokenNum_ = (tiling->hasAcceptTokenNum != 0);
    hasNumComputedTokens_ = (tiling->hasNumComputedTokens != 0);
    hasCacheIndices_ = (tiling->hasCacheIndices != 0);
    inplace_ = (tiling->inplace != 0);

    stateLen_ = tiling->stateLen; // 来自逻辑 shape dim[1]，非连续 stride 除法不可靠
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates,
                                                        GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
                                                        GM_ADDR numAcceptedToken, GM_ADDR numComputedTokens,
                                                        GM_ADDR blockIdxFirst, GM_ADDR blockIdxLast,
                                                        GM_ADDR initialStateIdx, GM_ADDR y, GM_ADDR workspace,
                                                        const FusedCausalConv1dCutBSHTilingData *tiling)
{
    (void)y; // y may be unused when inplace_==true; suppress warning
    ParseTiling(tiling);

    uint32_t blockIdx = GetBlockIdx();
    uint32_t dimCN = (dimCoreNum_ == 0) ? 1 : dimCoreNum_;
    bsIdx_ = blockIdx / dimCN;
    dimIdx_ = blockIdx - bsIdx_ * dimCN;

    isBsTailCore_ = (bsIdx_ >= bsRemainderCores_);
    isDimTailCore_ = (dimIdx_ >= dimRemainderCores_);
    isLastDimCore_ = (dimIdx_ == dimCoreNum_ - 1) && (dimRemainderElems_ > 0);

    // ---- GM binding ----
    uint64_t cuSeqLen = tiling->cuSeqLen;

    xGM_.SetGlobalBuffer((__gm__ T *)x, cuSeqLen * static_cast<uint64_t>(xStride_));
    weightGM_.SetGlobalBuffer((__gm__ T *)weight, static_cast<uint64_t>(kernelWidth_) * dim_);
    cacheStatesGM_.SetGlobalBuffer((__gm__ T *)convStates);
    seqStartIndexGM_.SetGlobalBuffer((__gm__ int32_t *)queryStartLoc, batchSize_ + 1);

    // inplace=true: yGM_ 指向 workspace 上的 y backup 区（避免污染 xGM_）。
    uint64_t yStride = inplace_ ? static_cast<uint64_t>(xStride_) : static_cast<uint64_t>(dim_);
    if (inplace_) {
        yGM_.SetGlobalBuffer((__gm__ T *)workspace, cuSeqLen * yStride);
    } else {
        yGM_.SetGlobalBuffer((__gm__ T *)y, cuSeqLen * yStride);
    }

    if (apcEnabled_) {
        cacheIndices2DGM_.SetGlobalBuffer((__gm__ int32_t *)cacheIndices,
                                          static_cast<uint64_t>(batchSize_) * maxNumBlocks_);
        if (initialStateIdx != nullptr) {
            initialStateIdxGM_.SetGlobalBuffer((__gm__ int32_t *)initialStateIdx, batchSize_);
        }
        if (blockIdxFirst != nullptr) {
            blockIdxFirstGM_.SetGlobalBuffer((__gm__ int32_t *)blockIdxFirst, batchSize_);
        }
        if (blockIdxLast != nullptr) {
            blockIdxLastGM_.SetGlobalBuffer((__gm__ int32_t *)blockIdxLast, batchSize_);
        }
    } else {
        if (hasCacheIndices_) {
            cacheIndicesGM_.SetGlobalBuffer((__gm__ int32_t *)cacheIndices, batchSize_);
        }
    }

    if (hasNumComputedTokens_ && numComputedTokens != nullptr) {
        numComputedTokensGM_.SetGlobalBuffer((__gm__ int32_t *)numComputedTokens, batchSize_);
    }
    if (hasAcceptTokenNum_ && numAcceptedToken != nullptr) {
        acceptTokenGM_.SetGlobalBuffer((__gm__ int32_t *)numAcceptedToken, batchSize_);
    }

    InitQueues();

    // Init cross-core state
    deferred_.active = false;
    deferred_.batchIdx = 0;
    deferred_.writeCIdx = 0;
    cacheDeferred_.active = false;
    cacheDeferred_.batchIdx = 0;
    cacheDeferred_.writeCIdx = 0;
    lastBatchIdx_ = 0;
    firstBatchIdx_ = 0;
    apcState_.batchIdx = 0xFFFFFFFF; // sentinel: APC state not cached
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::InitQueues()
{
    uint32_t K = kernelWidth_;

    uint32_t maxUbBS = isBsTailCore_ ? tailBlockubFactorBS_ : ubFactorBS_;
    uint32_t maxUbDim = GetMaxUbDim();

    uint32_t weightBufBytes = AlignUp(K * maxUbDim * sizeof(T), ALIGN_BYTES);
    uint32_t cacheBufBytes = AlignUp(stateLen_ * maxUbDim * sizeof(T), ALIGN_BYTES);
    uint32_t startLocBytes = AlignUp((batchSize_ + 1) * sizeof(int32_t), ALIGN_BYTES);
    uint32_t xBufBytes = AlignUp(maxUbBS * maxUbDim * sizeof(T), ALIGN_BYTES);
    uint32_t yBufBytes = xBufBytes;

    pipe_.InitBuffer(xQueue_, BUFFER_NUM, xBufBytes);
    pipe_.InitBuffer(yQueue_, BUFFER_NUM, yBufBytes);
    pipe_.InitBuffer(cacheQueue_, BUFFER_NUM, cacheBufBytes);
    pipe_.InitBuffer(weightQueue_, BUFFER_NUM, weightBufBytes);

    // ---- TBuf ----
    pipe_.InitBuffer(seqStartBuf_, startLocBytes);
    uint32_t metaBytes = AlignUp(batchSize_ * sizeof(int32_t), ALIGN_BYTES);

    if (hasCacheIndices_ && !apcEnabled_) {
        pipe_.InitBuffer(cacheIdxBuf_, metaBytes);
    }
    if (hasNumComputedTokens_) {
        pipe_.InitBuffer(numComputedBuf_, metaBytes);
    }
    if (hasAcceptTokenNum_) {
        pipe_.InitBuffer(acceptTokenBuf_, metaBytes);
    }
    if (apcEnabled_) {
        pipe_.InitBuffer(blockIdxFirstBuf_, metaBytes);
        pipe_.InitBuffer(blockIdxLastBuf_, metaBytes);
        pipe_.InitBuffer(initialStateIdxBuf_, metaBytes);
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::LoadMetadata()
{
    seqStartLocal_ = seqStartBuf_.Get<int32_t>();
    {
        DataCopyExtParams cpParams{1, static_cast<uint16_t>((batchSize_ + 1) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(seqStartLocal_, seqStartIndexGM_[0], cpParams, padParams);
    }

    DataCopyExtParams metaCp{1, static_cast<uint16_t>(batchSize_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> metaPad{false, 0, 0, 0};

    if (hasCacheIndices_ && !apcEnabled_) {
        cacheIdxLocal_ = cacheIdxBuf_.Get<int32_t>();
        DataCopyPad(cacheIdxLocal_, cacheIndicesGM_[0], metaCp, metaPad);
    }

    if (apcEnabled_) {
        blockIdxFirstLocal_ = blockIdxFirstBuf_.Get<int32_t>();
        blockIdxLastLocal_ = blockIdxLastBuf_.Get<int32_t>();
        initialStateIdxLocal_ = initialStateIdxBuf_.Get<int32_t>();
        DataCopyPad(blockIdxFirstLocal_, blockIdxFirstGM_[0], metaCp, metaPad);
        DataCopyPad(blockIdxLastLocal_, blockIdxLastGM_[0], metaCp, metaPad);
        DataCopyPad(initialStateIdxLocal_, initialStateIdxGM_[0], metaCp, metaPad);
    }

    if (hasNumComputedTokens_) {
        numComputedLocal_ = numComputedBuf_.Get<int32_t>();
        DataCopyPad(numComputedLocal_, numComputedTokensGM_[0], metaCp, metaPad);
    }

    if (hasAcceptTokenNum_) {
        acceptTokenLocal_ = acceptTokenBuf_.Get<int32_t>();
        DataCopyPad(acceptTokenLocal_, acceptTokenGM_[0], metaCp, metaPad);
    }
}

template <typename T>
__aicore__ inline uint32_t FusedCausalConv1dCutBSH<T>::FindBatchIdx(uint64_t globalSeqIdx)
{
    uint32_t lo = 0;
    uint32_t hi = batchSize_ - 1;
    while (lo < hi) {
        uint32_t mid = (lo + hi + 1) / 2;
        if ((uint64_t)seqStartLocal_.GetValue(mid) <= globalSeqIdx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::ResolveCacheIndices(uint32_t batchIdx, int64_t &readCIdx,
                                                                       int64_t &writeCIdx)
{
    if (apcEnabled_) {
        int32_t initIdx = initialStateIdxLocal_.GetValue(batchIdx);
        int32_t lastIdx = blockIdxLastLocal_.GetValue(batchIdx);
        readCIdx = GetApcCacheLine(batchIdx, static_cast<uint32_t>(initIdx));
        writeCIdx = GetApcCacheLine(batchIdx, static_cast<uint32_t>(lastIdx));
    } else if (hasCacheIndices_) {
        readCIdx = static_cast<int64_t>(cacheIdxLocal_.GetValue(batchIdx));
        writeCIdx = readCIdx;
    } else {
        readCIdx = static_cast<int64_t>(batchIdx);
        writeCIdx = readCIdx;
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::PrepareBatchMeta(uint32_t batchIdx, uint32_t curSeqIdx, BatchMeta &m)
{
    uint32_t K = kernelWidth_;

    ResolveCacheIndices(batchIdx, m.readCIdx, m.writeCIdx);

    // ---- useZeroCache ----
    m.useZeroCache = false;
    if (hasNumComputedTokens_) {
        m.useZeroCache = (numComputedLocal_.GetValue(batchIdx) == 0);
    }

    // ---- mtpOffset ----
    m.mtpOffset = 0;
    if (m.useZeroCache) {
        m.mtpOffset = 0;
    } else if (hasAcceptTokenNum_) {
        int32_t accepted = acceptTokenLocal_.GetValue(batchIdx);
        if (accepted > 0) {
            m.mtpOffset = static_cast<uint32_t>(accepted - 1);
        }
    } else {
        m.mtpOffset = stateLen_ - (K - 1);
    }

    // ---- cachedStateLen ----
    if (m.useZeroCache) {
        m.cachedStateLen = K - 1;
    } else if (hasAcceptTokenNum_) {
        m.cachedStateLen = m.mtpOffset + (K - 1);
        if (m.cachedStateLen > stateLen_)
            m.cachedStateLen = stateLen_;
    } else {
        m.cachedStateLen = stateLen_;
    }

    // ---- lastResetIdx (conv_mode==1) ----
    m.lastResetIdx = 0;
    if (convMode_ == 1 && hasNumComputedTokens_) {
        int32_t nct = numComputedLocal_.GetValue(batchIdx);
        int32_t lri = static_cast<int32_t>(K - 1) - nct;
        if (lri > 0) {
            uint32_t globalReset = static_cast<uint32_t>(lri);
            // lastResetIdx is relative to this UB block's token range
            if (globalReset > curSeqIdx) {
                m.lastResetIdx = globalReset - curSeqIdx;
            }
        }
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::Process()
{
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= realCoreNum_) {
        return;
    }

    LoadMetadata();

    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    uint32_t K = kernelWidth_;

    // ---- Compute bsStart (K-1 overlap between BS cores) ----
    uint64_t bsStart;
    if (bsIdx_ < bsRemainderCores_) {
        uint64_t effectiveStep = bsBlockFactor_ - (K - 1);
        bsStart = (uint64_t)bsIdx_ * effectiveStep;
    } else {
        uint64_t effectiveStepMain = bsBlockFactor_ - (K - 1);
        uint64_t effectiveStepTail = bsBlockTailFactor_ - (K - 1);
        bsStart = (uint64_t)bsRemainderCores_ * effectiveStepMain +
                  (uint64_t)(bsIdx_ - bsRemainderCores_) * effectiveStepTail;
    }

    // ---- Compute dimStart ----
    uint64_t dimStart = GetDimStart();

    // ---- Tail core adjustments ----
    uint32_t loopBS = isBsTailCore_ ? tailBlockloopNumBS_ : loopNumBS_;
    uint32_t factBS = isBsTailCore_ ? tailBlockubFactorBS_ : ubFactorBS_;
    uint32_t tailBS = isBsTailCore_ ? tailBlockubTailFactorBS_ : ubTailFactorBS_;

    uint32_t loopDim, factDim, tailDim;
    if (isLastDimCore_) {
        loopDim = lastCoreloopNumDim_;
        factDim = lastCoreubFactorDim_;
        tailDim = lastCoreubTailFactorDim_;
    } else if (isDimTailCore_) {
        loopDim = tailBlockloopNumDim_;
        factDim = tailBlockubFactorDim_;
        tailDim = tailBlockubTailFactorDim_;
    } else {
        loopDim = loopNumDim_;
        factDim = ubFactorDim_;
        tailDim = ubTailFactorDim_;
    }

    // ---- BS-dim step for inner loop (K-1 overlap within a core) ----
    uint16_t step = (factBS > K - 1) ? (factBS - (K - 1)) : 1;

    bool isBsFirstCore = (bsIdx_ == 0);

    // iStart: first BS core starts from token 0; others from K-1 (overlap)
    uint32_t iStart = isBsFirstCore ? 0 : (K - 1);

    uint32_t dimCN = (dimCoreNum_ == 0) ? 1 : dimCoreNum_;
    uint32_t bsCoreNum = realCoreNum_ / dimCN;

    uint64_t coreGlobalEnd;
    if (bsIdx_ + 1 < bsCoreNum) {
        uint32_t nextBsIdx = bsIdx_ + 1;
        uint64_t nextBsStart;
        if (nextBsIdx < bsRemainderCores_) {
            nextBsStart = (uint64_t)nextBsIdx * (bsBlockFactor_ - (K - 1));
        } else {
            uint64_t effStepMain = bsBlockFactor_ - (K - 1);
            uint64_t effStepTail = bsBlockTailFactor_ - (K - 1);
            nextBsStart =
                (uint64_t)bsRemainderCores_ * effStepMain + (uint64_t)(nextBsIdx - bsRemainderCores_) * effStepTail;
        }
        coreGlobalEnd = nextBsStart + (K - 1);
    } else {
        coreGlobalEnd = (uint64_t)-1; // last BS core has no successor
    }

    uint64_t curBsOff = 0;
    uint32_t curBatchIdx = FindBatchIdx(bsStart);
    firstBatchIdx_ = curBatchIdx;

    for (uint32_t bsLoop = 0; bsLoop < loopBS; bsLoop++) {
        uint32_t curBS = (bsLoop == loopBS - 1) ? tailBS : factBS;
        uint64_t curBsStart = bsStart + curBsOff;
        uint64_t actualStart = curBsStart + iStart;

        // Advance batchIdx if we've passed batch boundaries
        while (curBatchIdx + 1 < batchSize_ && actualStart >= (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1)) {
            curBatchIdx++;
        }

        uint64_t batchStartIdx = (uint64_t)seqStartLocal_.GetValue(curBatchIdx);
        uint32_t curSeqIdx = (uint32_t)(actualStart - batchStartIdx);

        // Dynamic check: first batch on this core may need deferred write.
        if (bsLoop == 0) {
            uint64_t firstBatchEnd = (uint64_t)seqStartLocal_.GetValue(curBatchIdx + 1);
            bool endsInThisCore = (firstBatchEnd <= coreGlobalEnd);
            bool isFirstBatchDeferred = !(isBsFirstCore || curSeqIdx == 0) && endsInThisCore;
            if (isFirstBatchDeferred) {
                deferred_.active = true;
                deferred_.batchIdx = curBatchIdx;
                BatchMeta dm;
                PrepareBatchMeta(curBatchIdx, curSeqIdx, dm);
                deferred_.writeCIdx = dm.writeCIdx;
            }
        }

        // ---- Dim loop ----
        uint32_t dimOff = (uint32_t)dimStart;
        uint32_t batchIdxForDim = curBatchIdx;
        uint32_t seqIdxForDim = curSeqIdx;
        for (uint32_t dimLoop = 0; dimLoop < loopDim; dimLoop++) {
            uint32_t curDim = (dimLoop == loopDim - 1) ? tailDim : factDim;

            CopyIn((uint32_t)curBsStart, curBS, dimOff, curDim);

            uint32_t dimBlocks = curDim * sizeof(T) / ALIGN_BYTES;
            batchIdxForDim = curBatchIdx;
            seqIdxForDim = curSeqIdx;
            Compute((uint32_t)curBsStart, curBS, dimOff, curDim, dimBlocks, iStart, batchIdxForDim, seqIdxForDim,
                    coreGlobalEnd, batchIdxForDim);

            dimOff += curDim;
        }
        curBatchIdx = batchIdxForDim;
        lastBatchIdx_ = curBatchIdx;

        curBsOff += step;
        iStart = K - 1; // subsequent BS blocks always start with K-1 overlap
    }

    SyncAll();

    WriteDeferredCacheSimple();

    FillApcCachePrefixChunks();
    SyncAll(); // cross-core barrier: ensure FillApc reads complete before deferred writes
    if (cacheDeferred_.active) {
        ExecuteDeferredRunningCache();
        cacheDeferred_.active = false;
    }

    // inplace=true：把 y workspace 搬回 xGM_ 完成 inplace 语义。
    if (inplace_) {
        SyncAll();
        CopyYToX();
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::CopyIn(uint32_t curBsStart, uint32_t curBS, uint32_t dimStart,
                                                          uint32_t dimSize)
{
    uint32_t K = kernelWidth_;
    uint32_t dimBlocks = dimSize * sizeof(T) / ALIGN_BYTES;
    uint32_t weightSkip = (dim_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t xSkip = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;

    // ---- Load weight [K, dimSize] ----
    LocalTensor<T> weightLocal = weightQueue_.AllocTensor<T>();
    {
        DataCopyExtParams wcp{static_cast<uint16_t>(K), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              weightSkip * ALIGN_BYTES, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(weightLocal, weightGM_[dimStart], wcp, padParams);
    }
    weightQueue_.EnQue(weightLocal);

    // ---- Load x [curBS, dimSize] ----
    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
    {
        DataCopyExtParams xcp{static_cast<uint16_t>(curBS), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                              xSkip * ALIGN_BYTES, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGM_[(uint64_t)curBsStart * xStride_ + dimStart], xcp, padParams);
    }
    xQueue_.EnQue(xLocal);

    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::Compute(uint32_t curBsStart, uint32_t curBS, uint32_t dimStart,
                                                           uint32_t dimSize, uint32_t dimBlocks, uint32_t iStart,
                                                           uint32_t batchIdx, uint32_t curSeqIdx,
                                                           uint64_t coreGlobalEnd, uint32_t &endBatchIdx)
{
    uint32_t K = kernelWidth_;
    uint32_t N = curBS;

    uint32_t xSkip = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t cacheSkip = (cacheStride1_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t ySkip = (inplace_ ? (xStride_ - dimSize) : (dim_ - dimSize)) * sizeof(T) / ALIGN_BYTES;
    uint64_t yRowStride = inplace_ ? xStride_ : dim_;

    // ---- DeQue ----
    LocalTensor<T> weightLocal = weightQueue_.DeQue<T>();
    LocalTensor<T> xLocal = xQueue_.DeQue<T>();
    LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();

    // Zero-init yLocal so padSlot tokens produce zero output
    Duplicate(yLocal, (T)0, N * dimSize);
    PipeBarrier<PIPE_V>();

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    uint32_t curBatch = batchIdx;
    uint32_t seqInBatch = curSeqIdx; // batch position of token at xLocal[iStart]
    int64_t curWriteCIdx = 0;

    // Track first batch for deferred write
    bool seenFirst = false;
    BatchMeta firstMeta;

    uint32_t i = iStart;
    uint32_t yOutOff = 0;

    while (i < N) {
        // ---- Determine current batch boundaries ----
        uint64_t batchSt = (uint64_t)seqStartLocal_.GetValue(curBatch);
        uint64_t batchEnd = (uint64_t)seqStartLocal_.GetValue(curBatch + 1);
        uint32_t batchLen = (uint32_t)(batchEnd - batchSt);

        // ---- Prepare metadata (once per batch per UB block) ----
        BatchMeta meta;
        PrepareBatchMeta(curBatch, seqInBatch, meta);
        curWriteCIdx = meta.writeCIdx;

        // ---- Save first batch info for deferred write ----
        if (!seenFirst) {
            seenFirst = true;
            firstMeta = meta;
        }

        // ---- padSlot skip ----
        if (curWriteCIdx == padSlotId_) {
            uint32_t remain = batchLen - seqInBatch;
            uint32_t skip = (N - i < remain) ? (N - i) : remain;
            i += skip;
            seqInBatch += skip;
            yOutOff += skip; // advance y output position (yLocal already zero-initialized)
            if (skip == remain && curBatch + 1 < batchSize_) {
                curBatch++;
                seqInBatch = 0;
            }
            continue;
        }

        // ---- Tokens remaining in this batch within this UB block ----
        uint32_t remainInBatch = batchLen - seqInBatch;
        uint32_t chunkSize = (N - i < remainInBatch) ? (N - i) : remainInBatch;
        bool isBatchEnd = (chunkSize == remainInBatch);

        // ---- Compute effectiveState for MTP OOB handling ----
        uint32_t effectiveState = K - 1;
        if (!meta.useZeroCache && hasAcceptTokenNum_) {
            if (meta.mtpOffset >= stateLen_) {
                effectiveState = 0;
            } else if (meta.mtpOffset + (K - 1) > stateLen_) {
                effectiveState = stateLen_ - meta.mtpOffset;
            }
        }

        // ---- Load cache for need-cache tokens (positions 0..K-2 in batch) ----
        LocalTensor<T> cacheLocal;
        bool cacheLoaded = false;
        if (seqInBatch < K - 1) {
            cacheLocal = cacheQueue_.AllocTensor<T>();
            cacheLoaded = true;
            if (!meta.useZeroCache && meta.cachedStateLen > 0) {
                uint64_t cacheGmOff = (uint64_t)meta.readCIdx * cacheStride0_ + dimStart;
                DataCopyExtParams ccp{static_cast<uint16_t>(meta.cachedStateLen),
                                      static_cast<uint16_t>(dimBlocks * ALIGN_BYTES), cacheSkip * ALIGN_BYTES, 0, 0};
                DataCopyPad(cacheLocal, cacheStatesGM_[cacheGmOff], ccp, padParams);
            } else {
                Duplicate(cacheLocal, (T)0, meta.cachedStateLen * dimSize);
            }
            cacheQueue_.EnQue(cacheLocal);
            cacheLocal = cacheQueue_.DeQue<T>();
        }

        // ---- Ensure MTE2 (x, weight, cache) and MTE3 (prior cache writeback) complete before VEC ----
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

        // ---- Conv for each token in this chunk ----
        uint32_t needCacheTokens = 0;
        if (seqInBatch < K - 1) {
            needCacheTokens = (K - 1 - seqInBatch < chunkSize) ? (K - 1 - seqInBatch) : chunkSize;
        }

        // conv_mode==1 + useZeroCache: all-zero output for the leading tokens
        bool allZeroConv = (convMode_ == 1 && meta.useZeroCache);
        uint32_t dimRem = dimSize - (dimSize / (DIM_ALIGN * 2)) * (DIM_ALIGN * 2);
        if (needCacheTokens > 0) {
            // tokens that need cache state (positions 0..K-2)
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            if (allZeroConv) {
                for (uint32_t j = 0; j < needCacheTokens; j++) {
                    if (residualConnection_) {
                        Adds(yLocal[(yOutOff + j) * dimSize], xLocal[(i + j) * dimSize], (T)0, dimSize);
                    } else {
                        Duplicate(yLocal[(yOutOff + j) * dimSize], (T)0, dimSize);
                    }
                }
            } else {
                for (uint32_t j = 0; j < needCacheTokens; j++) {
                    uint32_t seqPos = seqInBatch + j;
                    uint32_t sLen = (seqPos < effectiveState) ? (effectiveState - seqPos) : 0;
                    uint32_t xLen = K - sLen;
                    uint32_t xOff = (seqPos >= effectiveState) ? (seqPos - effectiveState) : 0;
                    LocalTensor<T> xSlice = xLocal[(i - seqInBatch + xOff) * dimSize];
                    LocalTensor<T> sSlice = cacheLocal[((sLen > 0) ? (meta.mtpOffset + seqPos) : 0) * dimSize];
                    LocalTensor<T> ySlice = yLocal[(yOutOff + j) * dimSize];
                    if (dimRem == 0) {
                        Conv1dNeedState(xSlice, weightLocal, sSlice, ySlice, static_cast<uint8_t>(sLen), xLen, dimSize,
                                        residualConnection_);
                    } else {
                        Conv1dNeedStateSingleTail(xSlice, weightLocal, sSlice, ySlice, static_cast<uint8_t>(sLen), xLen,
                                                  dimSize, residualConnection_);
                    }
                }
            }
        }

        // tokens that don't need cache (positions K-1..end)
        uint32_t noCacheTokens = chunkSize - needCacheTokens;
        if (noCacheTokens > 0) {
            uint32_t ncStart = i + needCacheTokens;
            // xSlice starts K-1 positions back
            uint32_t xStartIdx = ncStart - (K - 1);
            LocalTensor<T> xSlice = xLocal[xStartIdx * dimSize];
            LocalTensor<T> ySlice = yLocal[(yOutOff + needCacheTokens) * dimSize];
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            if (dimRem == 0) {
                Conv1dNoNeedState(xSlice, weightLocal, ySlice, noCacheTokens, dimSize, residualConnection_);
            } else if (dimRem > DIM_ALIGN) {
                Conv1dNoStateDoubleTail(xSlice, weightLocal, ySlice, noCacheTokens, dimSize, residualConnection_);
            } else {
                Conv1dNoStateSingleTail(xSlice, weightLocal, ySlice, noCacheTokens, dimSize, residualConnection_);
            }
        }

        // ---- conv_mode==1 zero fill (partial, non-useZeroCache case) ----
        if (meta.lastResetIdx > 0) {
            uint32_t resetInChunk = meta.lastResetIdx;
            if (resetInChunk > chunkSize)
                resetInChunk = chunkSize;
            for (uint32_t j = 0; j < resetInChunk; j++) {
                if (residualConnection_) {
                    Adds(yLocal[(yOutOff + j) * dimSize], xLocal[(i + j) * dimSize], (T)0, dimSize);
                } else {
                    Duplicate(yLocal[(yOutOff + j) * dimSize], (T)0, dimSize);
                }
            }
            meta.lastResetIdx -= resetInChunk;
        }

        // ---- Cache writeback ----
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

        if (isBatchEnd) {
            bool crossCoreBE = (batchEnd > coreGlobalEnd);
            bool deferForPrefix = false;
            if (meta.readCIdx == meta.writeCIdx && apcEnabled_ && blockSize_ > 0) {
                RefreshApcState(curBatch, batchLen);
                if (apcState_.nBlockToFill > 0) {
                    int32_t B = static_cast<int32_t>(blockSize_);
                    int32_t firstBdy = apcState_.lastFull - (apcState_.nBlockToFill - 1) * B;
                    deferForPrefix = (firstBdy > 0 && firstBdy < static_cast<int32_t>(K - 1));
                }
            }
            if (deferForPrefix) {
                cacheDeferred_.active = true;
                cacheDeferred_.batchIdx = curBatch;
                cacheDeferred_.writeCIdx = meta.writeCIdx;
            } else if (crossCoreBE || (deferred_.active && curBatch == deferred_.batchIdx)) {
                // Deferred: skip in-pipeline write; handled post-SyncAll
            } else {
                // chunkEnd: one-past-last row of current chunk in xLocal
                uint32_t chunkEnd = i + chunkSize;
                WriteCacheToStates(cacheLocal, xLocal, chunkEnd, batchLen, meta.cachedStateLen, cacheLoaded,
                                   curWriteCIdx, dimStart, dimSize, dimBlocks, cacheSkip, batchEnd);
            }
        } else if (cacheLoaded && seqInBatch == 0 && batchLen >= K) {
            uint32_t cFromCachePA, cFromXPA;
            if (batchLen >= stateLen_) {
                cFromCachePA = 0;
                cFromXPA = stateLen_;
            } else if (batchLen + meta.cachedStateLen >= stateLen_) {
                cFromCachePA = stateLen_ - batchLen;
                cFromXPA = batchLen;
            } else {
                cFromCachePA = meta.cachedStateLen;
                cFromXPA = batchLen;
            }
            bool crossCore = (batchEnd > coreGlobalEnd);
            bool needDeferredPA = crossCore || (deferred_.active && curBatch == deferred_.batchIdx);
            if (needDeferredPA) {
                // Deferred: skip; handled post-SyncAll
            } else if (cFromCachePA > 0) {
                if (seqInBatch + chunkSize >= K - 1) {
                    uint32_t tailRowOffPA = stateLen_ - cFromCachePA - cFromXPA;
                    uint32_t cacheRowStartPA = meta.cachedStateLen - cFromCachePA;
                    uint64_t csPrefixOff =
                        (uint64_t)curWriteCIdx * cacheStride0_ + (uint64_t)tailRowOffPA * cacheStride1_ + dimStart;
                    DataCopyExtParams wcpPA{static_cast<uint16_t>(cFromCachePA),
                                            static_cast<uint16_t>(dimBlocks * ALIGN_BYTES), 0, cacheSkip * ALIGN_BYTES,
                                            0};
                    DataCopyPad(cacheStatesGM_[csPrefixOff], cacheLocal[cacheRowStartPA * dimSize], wcpPA);
                }
            }
        }

        // ---- Free cache tensor ----
        if (cacheLoaded) {
            cacheQueue_.FreeTensor(cacheLocal);
        }

        // ---- Advance ----
        i += chunkSize;
        yOutOff += chunkSize;
        seqInBatch += chunkSize;
        if (isBatchEnd && curBatch + 1 < batchSize_) {
            curBatch++;
            seqInBatch = 0;
        }
    }

    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    {
        // yGmOff: global position of the first PROCESSED token
        uint64_t yGmOff = (uint64_t)(curBsStart + iStart) * yRowStride + dimStart;
        DataCopyExtParams ycp{static_cast<uint16_t>(yOutOff), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES), 0,
                              ySkip * ALIGN_BYTES, 0};
        DataCopyPad(yGM_[yGmOff], yLocal, ycp);
    }

    if (apcEnabled_ && blockSize_ > 0) {
        for (uint32_t b = batchIdx; b <= curBatch && b < batchSize_; b++) {
            uint64_t bSt = (uint64_t)seqStartLocal_.GetValue(b);
            uint32_t bLen = (uint32_t)((uint64_t)seqStartLocal_.GetValue(b + 1) - bSt);
            FillApcChunksInRange(b, bSt, bLen, curBsStart, curBS, xLocal, dimStart, dimSize, dimBlocks, cacheSkip);
        }
    }

    weightQueue_.FreeTensor(weightLocal);
    xQueue_.FreeTensor(xLocal);
    yQueue_.FreeTensor(yLocal);

    endBatchIdx = curBatch;
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::WriteCacheToStates(
    LocalTensor<T> &cacheLocal, LocalTensor<T> &xLocal, uint32_t chunkEnd, uint32_t curBatchLen,
    uint32_t cachedStateLen, bool hasCacheLocal, int64_t writeCIdx, uint32_t dimStart, uint32_t dimSize,
    uint32_t dimBlocks, uint32_t cacheSkipBlocks, uint64_t batchGmEnd)
{
    // ---- Decompose: how many rows from cache vs from x ----
    uint32_t cFromCache, cFromX;
    if (curBatchLen >= stateLen_) {
        cFromCache = 0;
        cFromX = stateLen_;
    } else if (curBatchLen + cachedStateLen >= stateLen_) {
        cFromCache = stateLen_ - curBatchLen;
        cFromX = curBatchLen;
    } else {
        cFromCache = cachedStateLen;
        cFromX = curBatchLen;
    }

    if (!hasCacheLocal) {
        cFromCache = 0;
    }

    uint32_t totalRows = cFromCache + cFromX;
    if (totalRows == 0)
        return;

    uint32_t tailRowOff = stateLen_ - totalRows;
    uint64_t csBase = (uint64_t)writeCIdx * cacheStride0_ + (uint64_t)tailRowOff * cacheStride1_ + dimStart;

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // Write cache prefix (last cFromCache rows)
    if (cFromCache > 0) {
        uint32_t cacheRowStart = cachedStateLen - cFromCache;
        DataCopyExtParams wcp{static_cast<uint16_t>(cFromCache), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES), 0,
                              cacheSkipBlocks * ALIGN_BYTES, 0};
        DataCopyPad(cacheStatesGM_[csBase], cacheLocal[cacheRowStart * dimSize], wcp);
    }

    // Write x suffix (last cFromX rows of this batch)
    if (cFromX > 0) {
        if (chunkEnd >= cFromX) {
            uint32_t xRowStart = chunkEnd - cFromX;
            DataCopyExtParams wcp{static_cast<uint16_t>(cFromX), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES), 0,
                                  cacheSkipBlocks * ALIGN_BYTES, 0};
            DataCopyPad(cacheStatesGM_[csBase + (uint64_t)cFromCache * cacheStride1_], xLocal[xRowStart * dimSize],
                        wcp);
        } else {
            // Fallback: re-read batch tail from GM via xQueue_ as temp buffer.
            uint32_t xSkip = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;
            LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();
            uint64_t xSrcOff = (batchGmEnd - cFromX) * xStride_ + dimStart;
            DataCopyExtParams rcp{static_cast<uint16_t>(cFromX), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES),
                                  xSkip * ALIGN_BYTES, 0, 0};
            DataCopyPad(tmpBuf, xGM_[xSrcOff], rcp, padParams);
            xQueue_.EnQue(tmpBuf);
            tmpBuf = xQueue_.DeQue<T>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams wcp{static_cast<uint16_t>(cFromX), static_cast<uint16_t>(dimBlocks * ALIGN_BYTES), 0,
                                  cacheSkipBlocks * ALIGN_BYTES, 0};
            DataCopyPad(cacheStatesGM_[csBase + (uint64_t)cFromCache * cacheStride1_], tmpBuf, wcp);
            xQueue_.FreeTensor(tmpBuf);
        }
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::WriteDeferredCacheSimple()
{
    if (!deferred_.active)
        return;
    if (deferred_.writeCIdx == padSlotId_)
        return;
    if (cacheDeferred_.active && cacheDeferred_.batchIdx == deferred_.batchIdx)
        return; // running cache deferred to after prefix fill

    uint32_t batchIdx = deferred_.batchIdx;
    int64_t writeCIdx = deferred_.writeCIdx;

    uint32_t dimStart = GetDimStart();
    uint32_t dimSize = GetDimTotal();

    uint64_t bSt = (uint64_t)seqStartLocal_.GetValue(batchIdx);
    uint64_t bEnd = (uint64_t)seqStartLocal_.GetValue(batchIdx + 1);
    uint32_t curBatchLen = (uint32_t)(bEnd - bSt);

    BatchMeta m;
    PrepareBatchMeta(batchIdx, 0, m);

    uint32_t rowBlocks = dimSize * sizeof(T) / ALIGN_BYTES;
    uint32_t cacheSkip = (cacheStride1_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t xSkip = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;

    uint32_t cFromCache, cFromX;
    if (curBatchLen >= stateLen_) {
        cFromCache = 0;
        cFromX = stateLen_;
    } else if (curBatchLen + m.cachedStateLen >= stateLen_) {
        cFromCache = stateLen_ - curBatchLen;
        cFromX = curBatchLen;
    } else {
        cFromCache = m.cachedStateLen;
        cFromX = curBatchLen;
    }

    uint32_t totalRows = cFromCache + cFromX;
    if (totalRows == 0)
        return;

    // Use xQueue_ as temp buffer
    LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // Load cache prefix from GM
    if (cFromCache > 0) {
        if (!m.useZeroCache) {
            int64_t readCIdx = m.readCIdx;
            uint32_t srcRowOff = m.cachedStateLen - cFromCache;
            uint64_t cacheSrcOff = (uint64_t)readCIdx * cacheStride0_ + (uint64_t)srcRowOff * cacheStride1_ + dimStart;
            DataCopyExtParams rcp{static_cast<uint16_t>(cFromCache), static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                  cacheSkip * ALIGN_BYTES, 0, 0};
            DataCopyPad(tmpBuf, cacheStatesGM_[cacheSrcOff], rcp, padParams);
        } else {
            Duplicate(tmpBuf, (T)0, cFromCache * dimSize);
        }
    }

    // Load x suffix from GM
    if (cFromX > 0) {
        uint64_t xSrcOff = (bEnd - cFromX) * xStride_ + dimStart;
        DataCopyExtParams rcp{static_cast<uint16_t>(cFromX), static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                              xSkip * ALIGN_BYTES, 0, 0};
        DataCopyPad(tmpBuf[cFromCache * dimSize], xGM_[xSrcOff], rcp, padParams);
    }

    xQueue_.EnQue(tmpBuf);
    tmpBuf = xQueue_.DeQue<T>();

    uint32_t tailRowOff = stateLen_ - totalRows;
    uint64_t csDstOff = (uint64_t)writeCIdx * cacheStride0_ + (uint64_t)tailRowOff * cacheStride1_ + dimStart;
    DataCopyExtParams wcp{static_cast<uint16_t>(totalRows), static_cast<uint16_t>(rowBlocks * ALIGN_BYTES), 0,
                          cacheSkip * ALIGN_BYTES, 0};
    DataCopyPad(cacheStatesGM_[csDstOff], tmpBuf, wcp);

    xQueue_.FreeTensor(tmpBuf);
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::ExecuteDeferredRunningCache()
{
    if (!cacheDeferred_.active)
        return;
    int64_t writeCIdx = cacheDeferred_.writeCIdx;
    if (writeCIdx == padSlotId_)
        return;

    uint32_t batchIdx = cacheDeferred_.batchIdx;
    uint64_t bSt = (uint64_t)seqStartLocal_.GetValue(batchIdx);
    uint64_t bEnd = (uint64_t)seqStartLocal_.GetValue(batchIdx + 1);
    uint32_t curBatchLen = (uint32_t)(bEnd - bSt);

    BatchMeta m;
    PrepareBatchMeta(batchIdx, 0, m);

    uint32_t dimStart = GetDimStart();
    uint32_t dimSize = GetDimTotal();
    uint32_t rowBlocks = dimSize * sizeof(T) / ALIGN_BYTES;
    uint32_t xSkip = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t cacheSkip = (cacheStride1_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    uint32_t cFromCache, cFromX;
    if (curBatchLen >= stateLen_) {
        cFromCache = 0;
        cFromX = stateLen_;
    } else if (curBatchLen + m.cachedStateLen >= stateLen_) {
        cFromCache = stateLen_ - curBatchLen;
        cFromX = curBatchLen;
    } else {
        cFromCache = m.cachedStateLen;
        cFromX = curBatchLen;
    }

    uint32_t totalRows = cFromCache + cFromX;
    if (totalRows == 0)
        return;

    LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();

    // Read cache prefix from writeCIdx (original data: running cache was deferred)
    if (cFromCache > 0) {
        if (!m.useZeroCache) {
            uint32_t srcRowOff = m.cachedStateLen - cFromCache;
            uint64_t cacheSrcOff = (uint64_t)writeCIdx * cacheStride0_ + (uint64_t)srcRowOff * cacheStride1_ + dimStart;
            DataCopyExtParams rcp{static_cast<uint16_t>(cFromCache), static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                  cacheSkip * ALIGN_BYTES, 0, 0};
            DataCopyPad(tmpBuf, cacheStatesGM_[cacheSrcOff], rcp, padParams);
        } else {
            Duplicate(tmpBuf, (T)0, cFromCache * dimSize);
        }
    }

    // Read x suffix from xGM
    if (cFromX > 0) {
        uint64_t xSrcOff = (bEnd - cFromX) * xStride_ + dimStart;
        DataCopyExtParams rcp{static_cast<uint16_t>(cFromX), static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                              xSkip * ALIGN_BYTES, 0, 0};
        DataCopyPad(tmpBuf[cFromCache * dimSize], xGM_[xSrcOff], rcp, padParams);
    }

    xQueue_.EnQue(tmpBuf);
    tmpBuf = xQueue_.DeQue<T>();

    uint32_t tailRowOff = stateLen_ - totalRows;
    uint64_t csDstOff = (uint64_t)writeCIdx * cacheStride0_ + (uint64_t)tailRowOff * cacheStride1_ + dimStart;
    DataCopyExtParams wcp{static_cast<uint16_t>(totalRows), static_cast<uint16_t>(rowBlocks * ALIGN_BYTES), 0,
                          cacheSkip * ALIGN_BYTES, 0};
    DataCopyPad(cacheStatesGM_[csDstOff], tmpBuf, wcp);

    xQueue_.FreeTensor(tmpBuf);
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::RefreshApcState(uint32_t curBatch, uint32_t batchLen)
{
    if (apcState_.batchIdx == curBatch)
        return;

    apcState_.batchIdx = curBatch;
    apcState_.firstBlk = blockIdxFirstLocal_.GetValue(curBatch);
    apcState_.lastBlk = blockIdxLastLocal_.GetValue(curBatch);
    apcState_.nBlockToFill = apcState_.lastBlk - apcState_.firstBlk;

    int32_t numComputed = 0;
    if (hasNumComputedTokens_)
        numComputed = numComputedLocal_.GetValue(curBatch);

    int32_t B = static_cast<int32_t>(blockSize_);
    uint32_t seqCompletedOffsetToken =
        (numComputed >= 0) ? static_cast<uint32_t>(numComputed - (numComputed / B) * B) : 0;
    uint32_t seqCompletedOffset = static_cast<uint32_t>(B) - seqCompletedOffsetToken;

    int32_t seqEndOffsetRaw = static_cast<int32_t>(batchLen) - static_cast<int32_t>(seqCompletedOffset);
    uint32_t seqEndOffset = static_cast<uint32_t>(seqEndOffsetRaw - (seqEndOffsetRaw / B) * B);

    apcState_.lastFull = static_cast<int32_t>(batchLen) - static_cast<int32_t>(seqEndOffset);
    if (seqEndOffset == 0 && batchLen >= seqCompletedOffset)
        apcState_.lastFull -= B;

    int32_t initIdx = initialStateIdxLocal_.GetValue(curBatch);
    apcState_.readCIdx = GetApcCacheLine(curBatch, static_cast<uint32_t>(initIdx));
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::FillApcChunksInRange(uint32_t curBatch, uint64_t batchSt,
                                                                        uint32_t batchLen, uint32_t curBsStart,
                                                                        uint32_t curBS, LocalTensor<T> &xLocal,
                                                                        uint32_t dimStart, uint32_t dimSize,
                                                                        uint32_t dimBlocks, uint32_t cacheSkip)
{
    if (!apcEnabled_ || blockSize_ == 0)
        return;

    RefreshApcState(curBatch, batchLen);
    if (apcState_.nBlockToFill == 0)
        return;

    uint32_t K = kernelWidth_;
    int32_t BSize = static_cast<int32_t>(blockSize_);

    int32_t baseOff = static_cast<int32_t>(curBsStart) - static_cast<int32_t>(batchSt);
    int32_t minBdy = baseOff + static_cast<int32_t>(K - 1);
    if (minBdy < static_cast<int32_t>(K - 1))
        minBdy = static_cast<int32_t>(K - 1);
    if (minBdy < 1)
        minBdy = 1;
    int32_t xEnd = baseOff + static_cast<int32_t>(curBS) + 1;

    int32_t firstBdy = apcState_.lastFull - (apcState_.nBlockToFill - 1) * BSize;
    int32_t chunkSkip = 0;
    if (firstBdy < minBdy) {
        int32_t gap = minBdy - firstBdy;       // gap > 0 guaranteed here
        chunkSkip = (gap + BSize - 1) / BSize; // ceiling(gap / B)
        if (chunkSkip >= apcState_.nBlockToFill)
            return;
    }

    uint32_t rowBlocks = dimSize * sizeof(T) / ALIGN_BYTES;
    for (int32_t chunk = chunkSkip; chunk < apcState_.nBlockToFill; chunk++) {
        int32_t boundaryIdx = apcState_.lastFull - (apcState_.nBlockToFill - chunk - 1) * BSize;
        if (boundaryIdx >= xEnd)
            break; // past range
        if (boundaryIdx <= 0)
            continue;

        int32_t blkId = apcState_.firstBlk + chunk;
        int64_t cLine = GetApcCacheLine(curBatch, static_cast<uint32_t>(blkId));
        if (cLine == padSlotId_ || cLine == apcState_.readCIdx)
            continue;

        uint32_t chunkLen = K - 1;
        uint32_t maxAvail = (K - 1) + batchLen - static_cast<uint32_t>(boundaryIdx);
        if (chunkLen > maxAvail)
            chunkLen = maxAvail;
        if (chunkLen == 0)
            continue;

        // Write chunkLen rows from xLocal directly to APC cache line.
        uint32_t srcOffset = static_cast<uint32_t>(boundaryIdx) - (K - 1);
        uint32_t globalPos = static_cast<uint32_t>(batchSt) + srcOffset;
        uint32_t localPos = globalPos - curBsStart;
        uint32_t tailRowOff = (stateLen_ > chunkLen) ? (stateLen_ - chunkLen) : 0;
        uint64_t csDst = (uint64_t)cLine * cacheStride0_ + (uint64_t)tailRowOff * cacheStride1_ + dimStart;
        DataCopyExtParams wcp{static_cast<uint16_t>(chunkLen), static_cast<uint16_t>(rowBlocks * ALIGN_BYTES), 0,
                              cacheSkip * ALIGN_BYTES, 0};
        DataCopyPad(cacheStatesGM_[csDst], xLocal[localPos * dimSize], wcp);
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::FillApcCachePrefixChunks()
{
    if (!apcEnabled_)
        return;
    if (blockSize_ == 0)
        return;

    uint32_t K = kernelWidth_;

    uint32_t dimCN = (dimCoreNum_ == 0) ? 1 : dimCoreNum_;
    uint32_t bsCoreNum = realCoreNum_ / dimCN;

    uint64_t coreBsStart;
    if (bsIdx_ < bsRemainderCores_) {
        coreBsStart = (uint64_t)bsIdx_ * (bsBlockFactor_ - (K - 1));
    } else {
        uint64_t effStepMain = bsBlockFactor_ - (K - 1);
        uint64_t effStepTail = bsBlockTailFactor_ - (K - 1);
        coreBsStart = (uint64_t)bsRemainderCores_ * effStepMain + (uint64_t)(bsIdx_ - bsRemainderCores_) * effStepTail;
    }

    uint64_t coreBsEnd;
    if (bsIdx_ + 1 < bsCoreNum) {
        uint32_t nextBsIdx = bsIdx_ + 1;
        if (nextBsIdx < bsRemainderCores_) {
            coreBsEnd = (uint64_t)nextBsIdx * (bsBlockFactor_ - (K - 1));
        } else {
            uint64_t effStepMain = bsBlockFactor_ - (K - 1);
            uint64_t effStepTail = bsBlockTailFactor_ - (K - 1);
            coreBsEnd =
                (uint64_t)bsRemainderCores_ * effStepMain + (uint64_t)(nextBsIdx - bsRemainderCores_) * effStepTail;
        }
    } else {
        coreBsEnd = (uint64_t)seqStartLocal_.GetValue(batchSize_);
    }

    uint32_t dimStart = GetDimStart();
    uint32_t dimSize = GetDimTotal();

    uint32_t rowBlocks = dimSize * sizeof(T) / ALIGN_BYTES;
    uint32_t cacheSkip = (cacheStride1_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    uint32_t xSkip = (xStride_ - dimSize) * sizeof(T) / ALIGN_BYTES;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (uint32_t b = firstBatchIdx_; b <= lastBatchIdx_; b++) {
        if (b >= batchSize_)
            break;

        int32_t firstBlk = blockIdxFirstLocal_.GetValue(b);
        int32_t lastBlk = blockIdxLastLocal_.GetValue(b);
        if (lastBlk <= firstBlk)
            continue;

        uint64_t batchSt = (uint64_t)seqStartLocal_.GetValue(b);
        if (!(batchSt >= coreBsStart && batchSt < coreBsEnd))
            continue;

        uint32_t seqLen = static_cast<uint32_t>(seqStartLocal_.GetValue(b + 1) - batchSt);

        BatchMeta m;
        PrepareBatchMeta(b, 0, m);

        int32_t numComputed = 0;
        if (hasNumComputedTokens_) {
            numComputed = numComputedLocal_.GetValue(b);
        }

        int32_t B = static_cast<int32_t>(blockSize_);
        uint32_t seqCompletedOffsetToken =
            (numComputed >= 0) ? static_cast<uint32_t>(numComputed - (numComputed / B) * B) : 0;
        uint32_t seqCompletedOffset = static_cast<uint32_t>(B) - seqCompletedOffsetToken;
        int32_t seqEndOffsetRaw = static_cast<int32_t>(seqLen) - static_cast<int32_t>(seqCompletedOffset);
        uint32_t seqEndOffset = static_cast<uint32_t>(seqEndOffsetRaw - (seqEndOffsetRaw / B) * B);
        int32_t lastFull = static_cast<int32_t>(seqLen) - static_cast<int32_t>(seqEndOffset);
        if (seqEndOffset == 0 && seqLen >= seqCompletedOffset)
            lastFull -= B;
        int32_t nBlockToFill = lastBlk - firstBlk;

        int32_t firstBdy = lastFull - (nBlockToFill - 1) * B;
        bool hasCachePrefix = (firstBdy > 0 && firstBdy < static_cast<int32_t>(K - 1));

        int32_t initIdx = initialStateIdxLocal_.GetValue(b);
        bool hasCollision = (initIdx >= firstBlk && initIdx < lastBlk);
        int32_t collisionChunk = hasCollision ? (initIdx - firstBlk) : -1;

        if (!hasCachePrefix && !hasCollision)
            continue; // nothing to do for this batch

        if (hasCachePrefix) {
            for (int32_t chunk = 0; chunk < nBlockToFill; chunk++) {
                int32_t boundaryIdx = lastFull - (nBlockToFill - chunk - 1) * B;
                if (boundaryIdx >= static_cast<int32_t>(K - 1))
                    break; // past cache-prefix
                if (boundaryIdx <= 0)
                    continue;

                int32_t blkId = firstBlk + chunk;
                int64_t cLine = GetApcCacheLine(b, static_cast<uint32_t>(blkId));
                if (cLine == padSlotId_)
                    continue;

                uint32_t chunkLen = K - 1;
                uint32_t maxAvail = (K - 1) + seqLen - static_cast<uint32_t>(boundaryIdx);
                if (chunkLen > maxAvail)
                    chunkLen = maxAvail;
                if (chunkLen == 0)
                    continue;

                LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();
                uint32_t cacheRows = (K - 1) - static_cast<uint32_t>(boundaryIdx);
                if (cacheRows > chunkLen)
                    cacheRows = chunkLen;
                uint32_t xRows = chunkLen - cacheRows;

                if (m.useZeroCache) {
                    if (xRows == 0) {
                        Duplicate(tmpBuf, (T)0, chunkLen * dimSize);
                    } else {
                        Duplicate(tmpBuf, (T)0, cacheRows * dimSize);
                        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                        uint64_t xSrcOff = batchSt * xStride_ + dimStart;
                        DataCopyExtParams xrcp{static_cast<uint16_t>(xRows),
                                               static_cast<uint16_t>(rowBlocks * ALIGN_BYTES), xSkip * ALIGN_BYTES, 0,
                                               0};
                        DataCopyPad(tmpBuf[cacheRows * dimSize], xGM_[xSrcOff], xrcp, padParams);
                    }
                } else {
                    int64_t readCIdx = GetApcCacheLine(b, static_cast<uint32_t>(initIdx));
                    if (cacheRows > 0) {
                        uint64_t srcOff = m.mtpOffset + static_cast<uint32_t>(boundaryIdx);
                        uint64_t cacheSrcOff = (uint64_t)readCIdx * cacheStride0_ + srcOff * cacheStride1_ + dimStart;
                        DataCopyExtParams crcp{static_cast<uint16_t>(cacheRows),
                                               static_cast<uint16_t>(rowBlocks * ALIGN_BYTES), cacheSkip * ALIGN_BYTES,
                                               0, 0};
                        DataCopyPad(tmpBuf, cacheStatesGM_[cacheSrcOff], crcp, padParams);
                    }
                    if (xRows > 0) {
                        uint64_t xSrcOff = batchSt * xStride_ + dimStart;
                        DataCopyExtParams xrcp{static_cast<uint16_t>(xRows),
                                               static_cast<uint16_t>(rowBlocks * ALIGN_BYTES), xSkip * ALIGN_BYTES, 0,
                                               0};
                        DataCopyPad(tmpBuf[cacheRows * dimSize], xGM_[xSrcOff], xrcp, padParams);
                    }
                }

                xQueue_.EnQue(tmpBuf);
                tmpBuf = xQueue_.DeQue<T>();
                uint32_t tailRowOff = (stateLen_ > chunkLen) ? (stateLen_ - chunkLen) : 0;
                uint64_t csDst = (uint64_t)cLine * cacheStride0_ + (uint64_t)tailRowOff * cacheStride1_ + dimStart;
                DataCopyExtParams wcp{static_cast<uint16_t>(chunkLen), static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                      0, cacheSkip * ALIGN_BYTES, 0};
                DataCopyPad(cacheStatesGM_[csDst], tmpBuf, wcp);
                xQueue_.FreeTensor(tmpBuf);
            }
        }

        if (hasCollision) {
            int32_t chunk = collisionChunk;
            int32_t boundaryIdx = lastFull - (nBlockToFill - chunk - 1) * B;
            if (boundaryIdx < static_cast<int32_t>(K - 1))
                continue;

            // x-suffix chunk deferred due to readCIdx collision
            uint32_t chunkLen = K - 1;
            uint32_t maxAvail = (K - 1) + seqLen - static_cast<uint32_t>(boundaryIdx);
            if (chunkLen > maxAvail)
                chunkLen = maxAvail;
            if (chunkLen == 0)
                continue;

            int32_t blkId = firstBlk + chunk;
            int64_t cLine = GetApcCacheLine(b, static_cast<uint32_t>(blkId));
            if (cLine == padSlotId_)
                continue;

            LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();
            uint32_t srcOffset = static_cast<uint32_t>(boundaryIdx) - (K - 1);
            uint64_t xSrcOff = (batchSt + srcOffset) * xStride_ + dimStart;
            DataCopyExtParams rcp{static_cast<uint16_t>(chunkLen), static_cast<uint16_t>(rowBlocks * ALIGN_BYTES),
                                  xSkip * ALIGN_BYTES, 0, 0};
            DataCopyPad(tmpBuf, xGM_[xSrcOff], rcp, padParams);

            xQueue_.EnQue(tmpBuf);
            tmpBuf = xQueue_.DeQue<T>();
            uint32_t tailRowOff = (stateLen_ > chunkLen) ? (stateLen_ - chunkLen) : 0;
            uint64_t csDst = (uint64_t)cLine * cacheStride0_ + (uint64_t)tailRowOff * cacheStride1_ + dimStart;
            DataCopyExtParams wcp{static_cast<uint16_t>(chunkLen), static_cast<uint16_t>(rowBlocks * ALIGN_BYTES), 0,
                                  cacheSkip * ALIGN_BYTES, 0};
            DataCopyPad(cacheStatesGM_[csDst], tmpBuf, wcp);
            xQueue_.FreeTensor(tmpBuf);
        }
    }
}

template <typename T>
__aicore__ inline void FusedCausalConv1dCutBSH<T>::CopyYToX()
{
    uint32_t K = kernelWidth_;
    bool isBsFirstCore = (bsIdx_ == 0);

    // ---- 本核 BS 起点（核间 K-1 overlap）----
    uint64_t bsStart;
    if (!isBsTailCore_) {
        uint64_t effStep = bsBlockFactor_ - (K - 1);
        bsStart = (uint64_t)bsIdx_ * effStep;
    } else {
        uint64_t effStepMain = bsBlockFactor_ - (K - 1);
        uint64_t effStepTail = bsBlockTailFactor_ - (K - 1);
        bsStart = (uint64_t)bsRemainderCores_ * effStepMain + (uint64_t)(bsIdx_ - bsRemainderCores_) * effStepTail;
    }

    uint32_t bsFactor = isBsTailCore_ ? bsBlockTailFactor_ : bsBlockFactor_;
    uint32_t iStart = isBsFirstCore ? 0 : (K - 1);
    if (bsFactor <= iStart)
        return;
    uint32_t validRows = bsFactor - iStart;
    uint64_t bsBase = bsStart + iStart;

    // ---- 本核 Dim 起点与长度 ----
    uint32_t dimStart = GetDimStart();
    uint32_t dimTotal = GetDimTotal();
    if (dimTotal == 0)
        return;

    uint32_t maxUbBS = isBsTailCore_ ? tailBlockubFactorBS_ : ubFactorBS_;
    uint32_t maxUbDim = GetMaxUbDim();
    if (maxUbBS == 0 || maxUbDim == 0)
        return;

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    uint32_t bsDone = 0;
    while (bsDone < validRows) {
        uint32_t bsChunk = (validRows - bsDone > maxUbBS) ? maxUbBS : (validRows - bsDone);
        uint64_t rowBase = bsBase + bsDone;

        uint32_t dimDone = 0;
        while (dimDone < dimTotal) {
            uint32_t dimChunk = (dimTotal - dimDone > maxUbDim) ? maxUbDim : (dimTotal - dimDone);
            uint32_t skipBlocks = (xStride_ - dimChunk) * sizeof(T) / ALIGN_BYTES;
            uint32_t lineBytes = dimChunk * sizeof(T);
            uint64_t gmOff = rowBase * (uint64_t)xStride_ + dimStart + dimDone;

            LocalTensor<T> tmpBuf = xQueue_.AllocTensor<T>();
            DataCopyExtParams rcp{static_cast<uint16_t>(bsChunk), static_cast<uint16_t>(lineBytes),
                                  skipBlocks * ALIGN_BYTES, 0, 0};
            DataCopyPad(tmpBuf, yGM_[gmOff], rcp, padParams);
            xQueue_.EnQue(tmpBuf);
            tmpBuf = xQueue_.DeQue<T>();

            DataCopyExtParams wcp{static_cast<uint16_t>(bsChunk), static_cast<uint16_t>(lineBytes), 0,
                                  skipBlocks * ALIGN_BYTES, 0};
            DataCopyPad(xGM_[gmOff], tmpBuf, wcp);
            xQueue_.FreeTensor(tmpBuf);

            dimDone += dimChunk;
        }
        bsDone += bsChunk;
    }
}

} // namespace FusedCausalConv1dCutBSHNs

#endif // FUSED_CAUSAL_CONV1D_CUT_BSH_H
