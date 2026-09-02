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
 * \file scatter_pa_kv_cache_nz_non_contiguous.h
 * \brief Non-contiguous with NZ format using strides
 */

#ifndef SCATTER_PA_KV_CACHE_NZ_NON_CONTIGUOUS_H_
#define SCATTER_PA_KV_CACHE_NZ_NON_CONTIGUOUS_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode, bool FIVE_DIM = false>
class ScatterPaKvCacheNzNonContiguous {
public:
    __aicore__ inline ScatterPaKvCacheNzNonContiguous(TPipe *pipe, const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR key_cache_out,
                                GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInKeyNz(int64_t tokenIdx, int64_t headIdx, int64_t lastDimIdx, int64_t headNum,
                                       int64_t lastDimNum);
    __aicore__ inline void CopyOutKeyNz(int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t lastDimIdx,
                                        int64_t headNum, int64_t lastDimNum);
    __aicore__ inline void CopyInValueNz(int64_t tokenIdx, int64_t headIdx, int64_t lastDimIdx, int64_t headNum,
                                         int64_t lastDimNum);
    __aicore__ inline void CopyOutValueNz(int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t lastDimIdx,
                                          int64_t headNum, int64_t lastDimNum);
    __aicore__ inline void InitLoopInfo();
    __aicore__ inline void ProcessKey(int64_t tokenIdx, int64_t blockIdx, int64_t blockOffset);
    __aicore__ inline void ProcessValue(int64_t tokenIdx, int64_t blockIdx, int64_t blockOffset);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> keyQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> valueQue_;
    const ScatterPaKvCacheTilingData *tilingData_;

    GlobalTensor<T1> keyGm_;
    GlobalTensor<T2> valueGm_;
    GlobalTensor<T1> keyCacheOutGm_;
    GlobalTensor<T2> valueCacheOutGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;

    int64_t blockIdx_{0};
    int64_t maxTokens_ = 0;
    int64_t lastDimK_{0};
    int64_t lastDimV_{0};
    int64_t kHeadLastDimLoop_{0};
    int64_t vHeadLastDimLoop_{0};
    int64_t maxHandleSize_{0};

    bool kFullyLoad_{false};
    bool vFullyLoad_{false};
    int64_t kLastDimIn_{0};
    int64_t kLastDimLoop_{0};
    int64_t kLastDimTail_{0};
    int64_t kNumHeadIn_{0};
    int64_t kNumHeadLoop_{0};
    int64_t kNumHeadTail_{0};

    int64_t vLastDimIn_{0};
    int64_t vLastDimLoop_{0};
    int64_t vLastDimTail_{0};
    int64_t vNumHeadIn_{0};
    int64_t vNumHeadLoop_{0};
    int64_t vNumHeadTail_{0};
};

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode, bool FIVE_DIM>
__aicore__ inline void ScatterPaKvCacheNzNonContiguous<T1, T2, IndexDtype, InOutMode, FIVE_DIM>::Init(
    GM_ADDR key, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR key_cache_out, GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    lastDimK_ = BLOCK_SIZE / sizeof(T1);
    lastDimV_ = BLOCK_SIZE / sizeof(T2);
    kHeadLastDimLoop_ = tilingData_->kHeadSize / lastDimK_;
    vHeadLastDimLoop_ = tilingData_->vHeadSize / lastDimV_;

    keyGm_.SetGlobalBuffer((__gm__ T1 *)(key));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slot_mapping));
    keyCacheOutGm_.SetGlobalBuffer((__gm__ T1 *)(key_cache_out));

    maxHandleSize_ = tilingData_->ubSize / DOUBLE_BUFFER;
    if constexpr (InOutMode == DUAL_IN_OUT) {
        maxHandleSize_ /= DUAL_IN_OUT;
        valueGm_.SetGlobalBuffer((__gm__ T2 *)(value));
        valueCacheOutGm_.SetGlobalBuffer((__gm__ T2 *)(value_cache_out));
        pipe_->InitBuffer(valueQue_, DOUBLE_BUFFER, maxHandleSize_);
    }
    pipe_->InitBuffer(keyQue_, DOUBLE_BUFFER, maxHandleSize_);
    maxTokens_ = tilingData_->numBlocks * tilingData_->blockSize;
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode, bool FIVE_DIM>
__aicore__ inline void ScatterPaKvCacheNzNonContiguous<T1, T2, IndexDtype, InOutMode, FIVE_DIM>::InitLoopInfo()
{
    int64_t ubHandleSize = maxHandleSize_ / sizeof(T1);
    int64_t kElementsPerHead = kHeadLastDimLoop_ * lastDimK_;

    kFullyLoad_ = (kElementsPerHead <= ubHandleSize);

    if (kFullyLoad_) {
        kLastDimIn_ = kHeadLastDimLoop_;
        kLastDimLoop_ = 1;
        kLastDimTail_ = 0;
        kNumHeadIn_ = ubHandleSize / kElementsPerHead;
        if (kNumHeadIn_ > tilingData_->numHead) {
            kNumHeadIn_ = tilingData_->numHead;
        }
        kNumHeadLoop_ = tilingData_->numHead / kNumHeadIn_;
        kNumHeadTail_ = tilingData_->numHead - kNumHeadIn_ * kNumHeadLoop_;
    } else {
        kLastDimIn_ = ubHandleSize / lastDimK_;
        kLastDimLoop_ = kHeadLastDimLoop_ / kLastDimIn_;
        kLastDimTail_ = kHeadLastDimLoop_ - kLastDimIn_ * kLastDimLoop_;
        kNumHeadIn_ = 1;
        kNumHeadLoop_ = tilingData_->numHead;
        kNumHeadTail_ = 0;
    }

    if constexpr (InOutMode == DUAL_IN_OUT) {
        int64_t ubHandleSizeV = maxHandleSize_ / sizeof(T2);
        int64_t vElementsPerHead = vHeadLastDimLoop_ * lastDimV_;

        vFullyLoad_ = (vElementsPerHead <= ubHandleSizeV);

        if (vFullyLoad_) {
            vLastDimIn_ = vHeadLastDimLoop_;
            vLastDimLoop_ = 1;
            vLastDimTail_ = 0;
            vNumHeadIn_ = ubHandleSizeV / vElementsPerHead;
            if (vNumHeadIn_ > tilingData_->numHead) {
                vNumHeadIn_ = tilingData_->numHead;
            }
            vNumHeadLoop_ = tilingData_->numHead / vNumHeadIn_;
            vNumHeadTail_ = tilingData_->numHead - vNumHeadIn_ * vNumHeadLoop_;
        } else {
            vLastDimIn_ = ubHandleSizeV / lastDimV_;
            vLastDimLoop_ = vHeadLastDimLoop_ / vLastDimIn_;
            vLastDimTail_ = vHeadLastDimLoop_ - vLastDimIn_ * vLastDimLoop_;
            vNumHeadIn_ = 1;
            vNumHeadLoop_ = tilingData_->numHead;
            vNumHeadTail_ = 0;
        }
    }
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode, bool FIVE_DIM>
__aicore__ inline void ScatterPaKvCacheNzNonContiguous<T1, T2, IndexDtype, InOutMode, FIVE_DIM>::CopyInKeyNz(
    int64_t tokenIdx, int64_t headIdx, int64_t lastDimIdx, int64_t headNum, int64_t lastDimNum)
{
    LocalTensor<T1> inputLocal = keyQue_.AllocTensor<T1>();

    int64_t keyOffset = tokenIdx * tilingData_->keyStride0 + headIdx * tilingData_->keyStride1 + lastDimIdx * lastDimK_;

    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(headNum), static_cast<uint32_t>(lastDimNum * lastDimK_ * sizeof(T1)),
        static_cast<uint32_t>((tilingData_->keyStride1 - lastDimNum * lastDimK_) * sizeof(T1)), 0, 0};
    DataCopyPadExtParams<T1> padParams = {false, 0, 0, static_cast<T1>(0)};
    DataCopyPad(inputLocal, keyGm_[keyOffset], copyParams, padParams);

    keyQue_.EnQue(inputLocal);
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode, bool FIVE_DIM>
__aicore__ inline void ScatterPaKvCacheNzNonContiguous<T1, T2, IndexDtype, InOutMode, FIVE_DIM>::CopyOutKeyNz(
    int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t lastDimIdx, int64_t headNum, int64_t lastDimNum)
{
    LocalTensor<T1> inputLocal = keyQue_.DeQue<T1>();

    int64_t dstStride = 0;
    if constexpr (FIVE_DIM) {
        dstStride = tilingData_->keyCacheStride2 * sizeof(T1) - BLOCK_SIZE;
    } else {
        dstStride = tilingData_->keyCacheStride1 * sizeof(T1) - BLOCK_SIZE;
    }
    int64_t keyIdxOffset = blockIdx * tilingData_->keyCacheStride0;
    for (int64_t h = 0; h < headNum; h++) {
        int64_t curHeadIdx = headIdx + h;
        int64_t keyCacheOffset = keyIdxOffset;
        if constexpr (FIVE_DIM) {
            keyCacheOffset += curHeadIdx * tilingData_->keyCacheStride1 + lastDimIdx * tilingData_->keyCacheStride2 +
                              blockOffset * tilingData_->keyCacheStride3;
        } else {
            keyCacheOffset += (curHeadIdx * kHeadLastDimLoop_ + lastDimIdx) * tilingData_->keyCacheStride1 +
                              blockOffset * tilingData_->keyCacheStride2;
        }

        int64_t ubOffset = h * lastDimNum * lastDimK_;
        DataCopyExtParams copyParams = {static_cast<uint16_t>(lastDimNum), BLOCK_SIZE, 0, dstStride, 0};

        DataCopyPad(keyCacheOutGm_[keyCacheOffset], inputLocal[ubOffset], copyParams);
    }

    keyQue_.FreeTensor(inputLocal);
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode, bool FIVE_DIM>
__aicore__ inline void ScatterPaKvCacheNzNonContiguous<T1, T2, IndexDtype, InOutMode, FIVE_DIM>::CopyInValueNz(
    int64_t tokenIdx, int64_t headIdx, int64_t lastDimIdx, int64_t headNum, int64_t lastDimNum)
{
    LocalTensor<T2> inputLocal = valueQue_.AllocTensor<T2>();

    int64_t valueOffset =
        tokenIdx * tilingData_->valueStride0 + headIdx * tilingData_->valueStride1 + lastDimIdx * lastDimV_;

    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(headNum), static_cast<uint32_t>(lastDimNum * lastDimV_ * sizeof(T2)),
        static_cast<uint32_t>((tilingData_->valueStride1 - lastDimNum * lastDimV_) * sizeof(T2)), 0, 0};
    DataCopyPadExtParams<T2> padParams = {false, 0, 0, static_cast<T2>(0)};
    DataCopyPad(inputLocal, valueGm_[valueOffset], copyParams, padParams);

    valueQue_.EnQue(inputLocal);
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode, bool FIVE_DIM>
__aicore__ inline void ScatterPaKvCacheNzNonContiguous<T1, T2, IndexDtype, InOutMode, FIVE_DIM>::CopyOutValueNz(
    int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t lastDimIdx, int64_t headNum, int64_t lastDimNum)
{
    LocalTensor<T2> inputLocal = valueQue_.DeQue<T2>();

    int64_t dstStride = 0;
    if constexpr (FIVE_DIM) {
        dstStride = tilingData_->valueCacheStride2 * sizeof(T2) - BLOCK_SIZE;
    } else {
        dstStride = tilingData_->valueCacheStride1 * sizeof(T2) - BLOCK_SIZE;
    }
    int64_t blockIdxOffset = blockIdx * tilingData_->valueCacheStride0;
    for (int64_t h = 0; h < headNum; h++) {
        int64_t curHeadIdx = headIdx + h;
        int64_t valueCacheOffset = blockIdxOffset;
        if constexpr (FIVE_DIM) {
            valueCacheOffset += curHeadIdx * tilingData_->valueCacheStride1 +
                                lastDimIdx * tilingData_->valueCacheStride2 +
                                blockOffset * tilingData_->valueCacheStride3;
        } else {
            valueCacheOffset += (curHeadIdx * vHeadLastDimLoop_ + lastDimIdx) * tilingData_->valueCacheStride1 +
                                blockOffset * tilingData_->valueCacheStride2;
        }

        int64_t ubOffset = h * lastDimNum * lastDimV_;

        DataCopyExtParams copyParams = {static_cast<uint16_t>(lastDimNum), BLOCK_SIZE, 0, dstStride, 0};

        DataCopyPad(valueCacheOutGm_[valueCacheOffset], inputLocal[ubOffset], copyParams);
    }

    valueQue_.FreeTensor(inputLocal);
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode, bool FIVE_DIM>
__aicore__ inline void ScatterPaKvCacheNzNonContiguous<T1, T2, IndexDtype, InOutMode, FIVE_DIM>::ProcessKey(
    int64_t tokenIdx, int64_t blockIdx, int64_t blockOffset)
{
    if (kFullyLoad_) {
        for (int64_t headLoop = 0; headLoop < kNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop * kNumHeadIn_;
            CopyInKeyNz(tokenIdx, headIdx, 0, kNumHeadIn_, kHeadLastDimLoop_);
            CopyOutKeyNz(blockIdx, blockOffset, headIdx, 0, kNumHeadIn_, kHeadLastDimLoop_);
        }
        if (kNumHeadTail_ > 0) {
            int64_t headIdx = kNumHeadLoop_ * kNumHeadIn_;
            CopyInKeyNz(tokenIdx, headIdx, 0, kNumHeadTail_, kHeadLastDimLoop_);
            CopyOutKeyNz(blockIdx, blockOffset, headIdx, 0, kNumHeadTail_, kHeadLastDimLoop_);
        }
    } else {
        for (int64_t headLoop = 0; headLoop < kNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop;
            for (int64_t lastDimLoop = 0; lastDimLoop < kLastDimLoop_; lastDimLoop++) {
                int64_t lastDimIdx = lastDimLoop * kLastDimIn_;
                CopyInKeyNz(tokenIdx, headIdx, lastDimIdx, 1, kLastDimIn_);
                CopyOutKeyNz(blockIdx, blockOffset, headIdx, lastDimIdx, 1, kLastDimIn_);
            }
            if (kLastDimTail_ > 0) {
                int64_t lastDimIdx = kLastDimLoop_ * kLastDimIn_;
                CopyInKeyNz(tokenIdx, headIdx, lastDimIdx, 1, kLastDimTail_);
                CopyOutKeyNz(blockIdx, blockOffset, headIdx, lastDimIdx, 1, kLastDimTail_);
            }
        }
    }
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode, bool FIVE_DIM>
__aicore__ inline void ScatterPaKvCacheNzNonContiguous<T1, T2, IndexDtype, InOutMode, FIVE_DIM>::ProcessValue(
    int64_t tokenIdx, int64_t blockIdx, int64_t blockOffset)
{
    if (vFullyLoad_) {
        for (int64_t headLoop = 0; headLoop < vNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop * vNumHeadIn_;
            CopyInValueNz(tokenIdx, headIdx, 0, vNumHeadIn_, vHeadLastDimLoop_);
            CopyOutValueNz(blockIdx, blockOffset, headIdx, 0, vNumHeadIn_, vHeadLastDimLoop_);
        }
        if (vNumHeadTail_ > 0) {
            int64_t headIdx = vNumHeadLoop_ * vNumHeadIn_;
            CopyInValueNz(tokenIdx, headIdx, 0, vNumHeadTail_, vHeadLastDimLoop_);
            CopyOutValueNz(blockIdx, blockOffset, headIdx, 0, vNumHeadTail_, vHeadLastDimLoop_);
        }
    } else {
        for (int64_t headLoop = 0; headLoop < vNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop;
            for (int64_t lastDimLoop = 0; lastDimLoop < vLastDimLoop_; lastDimLoop++) {
                int64_t lastDimIdx = lastDimLoop * vLastDimIn_;
                CopyInValueNz(tokenIdx, headIdx, lastDimIdx, 1, vLastDimIn_);
                CopyOutValueNz(blockIdx, blockOffset, headIdx, lastDimIdx, 1, vLastDimIn_);
            }
            if (vLastDimTail_ > 0) {
                int64_t lastDimIdx = vLastDimLoop_ * vLastDimIn_;
                CopyInValueNz(tokenIdx, headIdx, lastDimIdx, 1, vLastDimTail_);
                CopyOutValueNz(blockIdx, blockOffset, headIdx, lastDimIdx, 1, vLastDimTail_);
            }
        }
    }
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode, bool FIVE_DIM>
__aicore__ inline void ScatterPaKvCacheNzNonContiguous<T1, T2, IndexDtype, InOutMode, FIVE_DIM>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    int64_t startTaskId = blockIdx_ * tilingData_->blockFactor;
    int64_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;

    InitLoopInfo();

    for (int64_t loopToken = 0; loopToken < curBlockFactor; loopToken++) {
        IndexDtype slotValue = slotMappingGm_.GetValue(startTaskId + loopToken);
        if (slotValue < 0 || slotValue >= maxTokens_) {
            continue;
        }

        int64_t tokenIdx = startTaskId + loopToken;
        int64_t blockIdx = slotValue / tilingData_->blockSize;
        int64_t blockOffset = slotValue % tilingData_->blockSize;

        ProcessKey(tokenIdx, blockIdx, blockOffset);

        if constexpr (InOutMode == DUAL_IN_OUT) {
            ProcessValue(tokenIdx, blockIdx, blockOffset);
        }
    }
}

} // namespace ScatterPaKvCache
#endif // SCATTER_PA_KV_CACHE_NZ_NON_CONTIGUOUS_H_
