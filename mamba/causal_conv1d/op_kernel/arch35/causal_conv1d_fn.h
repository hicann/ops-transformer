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
 * \file causal_conv1d_fn.h
 * \brief Prefill mode — token-tiled full-sequence causal conv1d.
 */

#ifndef CAUSAL_CONV1D_FN_H
#define CAUSAL_CONV1D_FN_H

#include "causal_conv1d.h"

namespace CausalConv1d {

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
class CausalConv1dFn : public CausalConv1d<T, inputModeKey, widthKey, hasBiasKey, activationKey> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR bias, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens,
                                GM_ADDR convStatesOut, GM_ADDR y, GM_ADDR workspace,
                                const CausalConv1dTilingData *tilingData)
    {
        (void)numAcceptedTokens;
        this->InitGlobalBuffers(x, weight, convStates, bias, queryStartLoc, cacheIndices, y, convStatesOut, tilingData);
        this->initialStateModeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(initialStateMode));
        if (tilingData->hasInitStateWorkspace) {
            this->initStateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
        }
        this->InitBuffers();
    }

    __aicore__ inline void Process()
    {
        if constexpr (IsVarlenInputModeKey(inputModeKey)) {
            this->template ProcessImpl<SEQ_PARTITION_MODE_VARLEN>();
        } else {
            this->template ProcessImpl<SEQ_PARTITION_MODE_BATCH>();
        }
    }

protected:
    GlobalTensor<T> initStateWorkspaceGm_;

    template <int32_t kWindowMode>
    __aicore__ inline void ProcessImpl()
    {
        int64_t dimStart;
        int32_t curBaseDim;
        int32_t batchIdx;
        int32_t batchCnt;
        int64_t cursor;
        int64_t cursorEnd;
        if (!this->template InitBlock<kWindowMode>(dimStart, curBaseDim, batchIdx, batchCnt, cursor, cursorEnd)) {
            return;
        }
        this->template ProcessBlock<kWindowMode>(dimStart, curBaseDim, batchIdx, batchCnt, cursor, cursorEnd);
    }

    template <int32_t kWindowMode>
    __aicore__ inline bool InitBlock(int64_t &dimStart, int32_t &curBaseDim, int32_t &batchIdx, int32_t &batchCnt,
                                     int64_t &cursor, int64_t &cursorEnd)
    {
        const int64_t dim = this->tilingData_->dim;
        const int64_t batch = this->tilingData_->batch;
        const int64_t seqLen = this->tilingData_->seqLen;
        const int64_t cuSeqlen = this->tilingData_->cuSeqlen;
        const int32_t baseDim = this->tilingData_->baseDim;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int32_t baseDimCnt = (dim + baseDim - 1) / baseDim;
        const int64_t tokensPerBlock = this->tilingData_->tokensPerBlock;
        const int64_t tokenBlockCnt = this->tilingData_->tokenBlockCnt;

        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        const int64_t coreCnt = tokenBlockCnt * baseDimCnt;
        if (static_cast<int64_t>(blockIdx) >= coreCnt) {
            return false;
        }

        const int32_t tokenTileId = blockIdx / baseDimCnt;
        const int32_t baseDimIdx = blockIdx % baseDimCnt;
        dimStart = baseDimIdx * baseDim;
        curBaseDim = (dimStart + baseDim <= dim) ? baseDim : (dim - dimStart);
        const int64_t tokenStart = static_cast<int64_t>(tokenTileId) * tokensPerBlock;
        const int64_t tokenEndRaw = tokenStart + tokensPerBlock;
        const int64_t tokenEnd = (tokenEndRaw <= cuSeqlen) ? tokenEndRaw : cuSeqlen;
        const bool valid = (tokenStart < cuSeqlen) && (curBaseDim > 0) && (tokenEnd > tokenStart);

        if (this->tilingData_->hasInitStateWorkspace) {
            if (valid && tokenTileId == 0) {
                this->PrefetchInitStatesToWorkspace(dimStart, curBaseDim);
            }
            SyncAll();
        }

        if (!valid) {
            return false;
        }

        this->InitCalcBuf();
        this->LoadWeightAndBias(dimStart, curBaseDim);

        if constexpr (kWindowMode == SEQ_PARTITION_MODE_VARLEN) {
            batchIdx = this->LocateBatchByToken(static_cast<int32_t>(tokenStart));
            batchCnt = static_cast<int32_t>(batch);
            cursor = tokenStart;
            cursorEnd = tokenEnd;
        } else {
            batchIdx = static_cast<int32_t>(tokenStart / (seqLen * coBatch));
            batchCnt = static_cast<int32_t>(batch / coBatch);
            // cursor in time-step space: one step covers coBatch flat positions
            cursor = tokenStart / coBatch;
            cursorEnd = tokenEnd / coBatch;
        }
        return true;
    }

    template <int32_t kWindowMode>
    __aicore__ inline void ProcessBlock(int64_t dimStart, int32_t curBaseDim, int32_t &batchIdx, int32_t batchCnt,
                                        int64_t &cursor, int64_t cursorEnd)
    {
        const int64_t dim = this->tilingData_->dim;
        const int64_t seqLen = this->tilingData_->seqLen;
        const int64_t stateLen = this->tilingData_->stateLen;
        const int64_t kernelWidth = this->tilingData_->kernelWidth;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const bool hasCacheIndices = this->tilingData_->hasCacheIndices;
        const bool hasInitialStateMode = this->tilingData_->hasInitialStateMode;

        while (cursor < cursorEnd && batchIdx < batchCnt) {
            int64_t curSeqStart;
            int64_t curSeqEnd;
            this->template GetSeqWindow<kWindowMode>(batchIdx, seqLen, coBatch, curSeqStart, curSeqEnd);

            if (cursor >= curSeqEnd) {
                ++batchIdx;
                continue;
            }

            const int64_t chunkEnd = (cursorEnd < curSeqEnd) ? cursorEnd : curSeqEnd;
            const int32_t runLen = static_cast<int32_t>(chunkEnd - cursor);
            if (runLen <= 0) {
                ++batchIdx;
                continue;
            }

            // curSeqFlatPos = curSeqStart + (cursor % seqLen).
            // For coBatch=1: cursor is flat position, curSeqStart = batchIdx * seqLen,
            //   cursor % seqLen = cursor - curSeqStart, so curSeqFlatPos = cursor.
            // For coBatch>1: cursor is time step, cursor % seqLen is batch-0 offset.
            // For varlen: cursor is flat position, curSeqFlatPos = cursor.
            int64_t curSeqFlatPos;
            if constexpr (kWindowMode == SEQ_PARTITION_MODE_BATCH) {
                curSeqFlatPos = curSeqStart + cursor % seqLen;
            } else {
                curSeqFlatPos = cursor;
            }

            int32_t cacheIdx;
            if (!this->ResolveBatchCacheIndex(batchIdx * coBatch, hasCacheIndices, cacheIdx)) {
                cursor = chunkEnd;
                if (cursor >= curSeqEnd) {
                    ++batchIdx;
                }
                continue;
            }

            const bool hasInit = this->ResolveBatchHasInit(batchIdx * coBatch, hasInitialStateMode);

            this->InitRing(cacheIdx, hasInit, curSeqStart, curSeqFlatPos, dimStart, curBaseDim, dim);
            {
                int64_t stStart;
                if constexpr (kWindowMode == SEQ_PARTITION_MODE_VARLEN) {
                    stStart = curSeqEnd - stateLen;
                } else {
                    stStart = curSeqStart + seqLen - stateLen;
                }
                this->RunSeq(curSeqFlatPos, runLen, dimStart, curBaseDim, cacheIdx, stStart, stateLen);
            }

            if (cursor + runLen >= curSeqEnd) {
                const int32_t aliveCount =
                    static_cast<int32_t>((kernelWidth - 1 < stateLen) ? (kernelWidth - 1) : stateLen);
                this->WriteBackState(cacheIdx, runLen, dimStart, curBaseDim, dim,
                                     static_cast<int32_t>(stateLen - aliveCount));
            }

            SetEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);

            cursor = chunkEnd;
            if (cursor >= curSeqEnd) {
                ++batchIdx;
            }
        }
    }

    __aicore__ inline void PrefetchInitStatesToWorkspace(int64_t dimStart, int32_t baseDimSize)
    {
        const int64_t dim = this->tilingData_->dim;
        const int64_t stateLen = this->tilingData_->stateLen;
        const int64_t batch = this->tilingData_->batch;
        const bool hasCacheIndices = this->tilingData_->hasCacheIndices;
        const bool hasInitialStateMode = this->tilingData_->hasInitialStateMode;
        const int32_t nullBlockId = static_cast<int32_t>(this->tilingData_->nullBlockId);
        const int32_t batchRows = static_cast<int32_t>(this->tilingData_->kernelWidth) + 1;
        const uint32_t blockBytes = static_cast<uint32_t>(baseDimSize) * sizeof(T);
        const uint32_t gapBytes = static_cast<uint32_t>(dim - baseDimSize) * sizeof(T);
        LocalTensor<T> ubBuf = this->inTensor_;

        for (int64_t b = 0; b < batch; ++b) {
            if (hasInitialStateMode && this->initialStateModeGm_.GetValue(b) == 0) {
                continue;
            }
            int64_t cacheIdx = b;
            if (hasCacheIndices) {
                cacheIdx = this->cacheIndicesGm_.GetValue(b);
                if (cacheIdx == nullBlockId) {
                    continue;
                }
            }
            const int64_t slotBase = cacheIdx * stateLen;
            const int64_t totalRows = slotBase + stateLen;
            for (int64_t row = slotBase; row < totalRows; row += batchRows) {
                const int64_t rowsThisBatch = (row + batchRows <= totalRows) ? batchRows : (totalRows - row);
                const int64_t rowOffset = row * dim + dimStart;

                DataCopyPad(ubBuf[0], this->convStatesGm_[rowOffset],
                            {static_cast<uint16_t>(rowsThisBatch), blockBytes, gapBytes, 0, 0}, {false, 0, 0, 0});
                SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

                DataCopyPad(this->initStateWorkspaceGm_[rowOffset], ubBuf[0],
                            {static_cast<uint16_t>(rowsThisBatch), blockBytes, 0, gapBytes, 0});
                SetEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            }
        }
    }

    __aicore__ inline int32_t LocateBatchByToken(int32_t tokenIdx) const
    {
        int32_t left = 0;
        int32_t right = static_cast<int32_t>(this->tilingData_->batch);
        while (left < right) {
            const int32_t mid = left + ((right - left) >> 1);
            const int32_t endVal = this->queryStartLocGm_.GetValue(mid + 1);
            if (tokenIdx < endVal) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int64_t seqStart, int64_t chunkStart,
                                    int64_t dimStart, int32_t baseDim, int64_t dim)
    {
        const int64_t width = this->tilingData_->kernelWidth;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        LocalTensor<T> ring = this->inTensor_;
        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);

        const int64_t histBegin = chunkStart - (width - 1);
        int32_t padLen;
        if (seqStart <= histBegin) {
            padLen = 0;
        } else if (seqStart == chunkStart) {
            padLen = static_cast<int32_t>(width - 1);
        } else {
            padLen = static_cast<int32_t>(seqStart - histBegin);
        }
        const int32_t loadLen = static_cast<int32_t>(width - 1 - padLen);

        bool hasGmHistoryCopy = this->LoadConvStates(ring, padLen, cacheIdx, histBegin, seqStart,
                                                     static_cast<int32_t>(width), dimStart, dim, baseDim, hasInit);
        hasGmHistoryCopy |=
            this->LoadInput(ring, loadLen, padLen, chunkStart, static_cast<int32_t>(width), dimStart, dim, baseDim);

        if (hasGmHistoryCopy) {
            SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        }

        const int32_t slot0 = CurrSlot(0, static_cast<int32_t>(width));
        const int64_t seqLen = this->tilingData_->seqLen;
        const uint32_t srcGapBatch = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);
        DataCopyPad(ring[slot0 * this->dimBufferSize_], this->xGm_[chunkStart * dim + dimStart],
                    {static_cast<uint16_t>(coBatch), blockBytes, srcGapBatch, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline bool LoadConvStates(LocalTensor<T> &ring, int32_t padLen, int32_t cacheIdx, int64_t histBegin,
                                          int64_t seqStart, int32_t width, int64_t dimStart, int64_t dim,
                                          int32_t baseDim, bool hasInit)
    {
        if (padLen <= 0) {
            return false;
        }
        if (!hasInit) {
            Duplicate(ring[0], static_cast<T>(0), padLen * this->dimBufferSize_);
            return false;
        }
        const int64_t stateLen = this->tilingData_->stateLen;
        const int64_t statePos = histBegin - seqStart + width - 1;
        const int64_t srcBase = static_cast<int64_t>(cacheIdx) * stateLen * dim + statePos * dim + dimStart;

        LoopModeParams loopParams;
        loopParams.loop1Size = static_cast<uint32_t>(this->tilingData_->coBatch);
        loopParams.loop2Size = 1;
        loopParams.loop1SrcStride = static_cast<uint64_t>(stateLen) * dim * sizeof(T);
        loopParams.loop1DstStride = static_cast<uint64_t>(baseDim) * sizeof(T);
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;

        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(padLen);
        copyParams.blockLen = static_cast<uint32_t>(baseDim) * sizeof(T);
        copyParams.srcStride = static_cast<uint32_t>(dim - baseDim) * sizeof(T);
        copyParams.dstStride = static_cast<uint32_t>(this->dimBufferSize_ - baseDim) * sizeof(T) / this->kUbBlockSize;
        copyParams.rsv = 0;

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        if (this->tilingData_->hasInitStateWorkspace) {
            DataCopyPad(ring[0], this->initStateWorkspaceGm_[srcBase], copyParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(ring[0], this->convStatesGm_[srcBase], copyParams, {false, 0, 0, 0});
        }
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        return true;
    }

    __aicore__ inline bool LoadInput(LocalTensor<T> &ring, int32_t loadLen, int32_t padLen, int64_t chunkStart,
                                     int32_t width, int64_t dimStart, int64_t dim, int32_t baseDim)
    {
        if (loadLen <= 0) {
            return false;
        }
        const int64_t seqLen = this->tilingData_->seqLen;
        const int64_t xGmRow = chunkStart - (width - 1) + padLen;
        const int64_t srcBase = xGmRow * dim + dimStart;

        LoopModeParams loopParams;
        loopParams.loop1Size = static_cast<uint32_t>(this->tilingData_->coBatch);
        loopParams.loop2Size = 1;
        loopParams.loop1SrcStride = static_cast<uint64_t>(seqLen) * dim * sizeof(T);
        loopParams.loop1DstStride = static_cast<uint64_t>(baseDim) * sizeof(T);
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;

        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(loadLen);
        copyParams.blockLen = static_cast<uint32_t>(baseDim) * sizeof(T);
        copyParams.srcStride = static_cast<uint32_t>(dim - baseDim) * sizeof(T);
        copyParams.dstStride = static_cast<uint32_t>(this->dimBufferSize_ - baseDim) * sizeof(T) / this->kUbBlockSize;
        copyParams.rsv = 0;

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(ring[padLen * this->dimBufferSize_], this->xGm_[srcBase], copyParams, {false, 0, 0, 0});
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        return true;
    }

    __aicore__ inline void RunSeq(int64_t start, int32_t len, int64_t dimStart, int32_t baseDim, int32_t cacheIdx,
                                  int64_t stateStart, int64_t stateLen)
    {
        const int64_t dim = this->tilingData_->dim;
        const int64_t kernelWidth = this->tilingData_->kernelWidth;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int64_t seqLen = this->tilingData_->seqLen;
        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        LocalTensor<T> ring = this->inTensor_;
        LocalTensor<T> outT = this->outTensor_;

        for (int32_t idx = 0; idx < len; ++idx) {
            WaitFlag<HardEvent::MTE2_V>((idx & 1) ? EVENT_ID1 : EVENT_ID0);

            if (idx + 1 < len) {
                const int32_t slotNext = NextSlot(idx, kernelWidth);
                if (idx > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>((idx & 1) ? EVENT_ID1 : EVENT_ID0);
                }
                const int64_t xGmBase = (start + idx + 1) * dim + dimStart;
                const uint32_t srcGapBatchX = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);
                DataCopyPad(ring[slotNext * this->dimBufferSize_], this->xGm_[xGmBase],
                            {static_cast<uint16_t>(coBatch), blockBytes, srcGapBatchX, 0, 0}, {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>((idx & 1) ? EVENT_ID0 : EVENT_ID1);
            }

            const int32_t outSlot = idx & 1;
            if (idx >= 2) {
                WaitFlag<HardEvent::MTE3_V>((idx & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            LocalTensor<T> outSlotT = outT[outSlot * this->dimBufferSize_];
            this->ComputeConv1d(ring, this->pool_.weight, this->pool_.bias, outSlotT, this->dimBufferSize_, idx);

            SetFlag<HardEvent::V_MTE3>((idx & 1) ? EVENT_ID0 : EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>((idx & 1) ? EVENT_ID0 : EVENT_ID1);

            const int64_t outGmBase = (start + idx) * dim + dimStart;
            const uint32_t dstGapBatchY = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);
            DataCopyPad(this->yGm_[outGmBase], outSlotT[0],
                        {static_cast<uint16_t>(coBatch), blockBytes, 0, dstGapBatchY, 0});

            // ---- dead slot state write: ring slot about to be overwritten → convStatesOutGm ----
            const int64_t deadPaddedIdx = start + idx - kernelWidth + 1;
            if (deadPaddedIdx >= stateStart) {
                const int32_t deadSlot = static_cast<int32_t>(idx % (kernelWidth + 1));
                const int64_t stateOffset = deadPaddedIdx - stateStart;
                const int64_t stateGmBase =
                    static_cast<int64_t>(cacheIdx) * stateLen * dim + stateOffset * dim + dimStart;
                DataCopyExtParams stateCp;
                stateCp.blockCount = static_cast<uint16_t>(coBatch);
                stateCp.blockLen = blockBytes;
                stateCp.srcStride = 0;
                stateCp.dstStride = static_cast<uint32_t>(stateLen * dim - baseDim) * sizeof(T);
                stateCp.rsv = 0;
                DataCopyPad(this->convStatesOutGm_[stateGmBase], ring[deadSlot * this->dimBufferSize_], stateCp);
            }

            if (idx + 2 < len) {
                SetFlag<HardEvent::MTE3_V>((idx & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            if (idx + 2 < len) {
                SetFlag<HardEvent::MTE3_MTE2>((idx & 1) ? EVENT_ID0 : EVENT_ID1);
            }
        }
    }
};

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
__aicore__ inline void RunCausalConv1dFn(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR bias,
                                         GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                         GM_ADDR numAcceptedTokens, GM_ADDR convStatesOut, GM_ADDR y, GM_ADDR workspace,
                                         const CausalConv1dTilingData *tilingData)
{
    CausalConv1dFn<T, inputModeKey, widthKey, hasBiasKey, activationKey> op;
    op.Init(x, weight, convStates, bias, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens,
            convStatesOut, y, workspace, tilingData);
    op.Process();
}

} // namespace CausalConv1d

#endif // CAUSAL_CONV1D_FN_H
