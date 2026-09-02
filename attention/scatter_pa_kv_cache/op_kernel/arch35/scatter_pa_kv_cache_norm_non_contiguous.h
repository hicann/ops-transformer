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
 * \file scatter_pa_kv_cache_norm_non_contiguous.h
 * \brief Non-contiguous with ND format using strides
 */

#ifndef SCATTER_PA_KV_CACHE_NORM_NON_CONTIGUOUS_H_
#define SCATTER_PA_KV_CACHE_NORM_NON_CONTIGUOUS_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T, typename IndexDtype, int64_t InOutMode>
class ScatterPaKvCacheNormNonContiguous {
public:
    __aicore__ inline ScatterPaKvCacheNormNonContiguous(TPipe *pipe,
                                                        const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR key_cache_out,
                                GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInKey(int64_t tokenIdx, int64_t headIdx, int64_t sizeOffset, int64_t headNum,
                                     int64_t headSize);
    __aicore__ inline void CopyOutKey(int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t sizeOffset,
                                      int64_t headNum, int64_t headSize);
    __aicore__ inline void CopyInValue(int64_t tokenIdx, int64_t headIdx, int64_t sizeOffset, int64_t headNum,
                                       int64_t headSize);
    __aicore__ inline void CopyOutValue(int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t sizeOffset,
                                        int64_t headNum, int64_t headSize);
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void InitLoopInfo();
    __aicore__ inline void ProcessKey(int64_t tokenIdx, int64_t blockIdx, int64_t blockOffset);
    __aicore__ inline void ProcessValue(int64_t tokenIdx, int64_t blockIdx, int64_t blockOffset);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> keyQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> valueQue_;
    const ScatterPaKvCacheTilingData *tilingData_;

    GlobalTensor<T> keyGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<T> keyCacheOutGm_;
    GlobalTensor<T> valueCacheOutGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;

    int64_t blockIdx_{0};
    int64_t maxTokens_ = 0;
    int64_t blockFactorOffset_ = 0;
    int64_t alignKHeadSize_ = 0;
    int64_t alignVHeadSize_ = 0;
    int64_t maxHandleSize_{0};

    bool kFullyLoad_{false};
    bool vFullyLoad_{false};
    int64_t kHeadSizeIn_{0};
    int64_t kHeadSizeloop_{0};
    int64_t kHeadSizeTail_{0};
    int64_t kNumHeadIn_{0};
    int64_t kNumHeadLoop_{0};
    int64_t kNumHeadTail_{0};

    int64_t vHeadSizeIn_{0};
    int64_t vHeadSizeloop_{0};
    int64_t vHeadSizeTail_{0};
    int64_t vNumHeadIn_{0};
    int64_t vNumHeadLoop_{0};
    int64_t vNumHeadTail_{0};
};

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormNonContiguous<T, IndexDtype, InOutMode>::Init(
    GM_ADDR key, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR key_cache_out, GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    blockFactorOffset_ = blockIdx_ * tilingData_->blockFactor;
    keyGm_.SetGlobalBuffer((__gm__ T *)(key));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slot_mapping) + blockIdx_ * tilingData_->blockFactor);
    keyCacheOutGm_.SetGlobalBuffer((__gm__ T *)(key_cache_out));

    maxHandleSize_ = tilingData_->ubSize / DOUBLE_BUFFER;
    if constexpr (InOutMode == DUAL_IN_OUT) {
        maxHandleSize_ /= DUAL_IN_OUT;
        valueGm_.SetGlobalBuffer((__gm__ T *)(value));
        valueCacheOutGm_.SetGlobalBuffer((__gm__ T *)(value_cache_out));
        pipe_->InitBuffer(valueQue_, DOUBLE_BUFFER, maxHandleSize_);
    }
    pipe_->InitBuffer(keyQue_, DOUBLE_BUFFER, maxHandleSize_);
    maxTokens_ = tilingData_->numBlocks * tilingData_->blockSize;
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline int64_t ScatterPaKvCacheNormNonContiguous<T, IndexDtype, InOutMode>::RoundUp(int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormNonContiguous<T, IndexDtype, InOutMode>::CopyInKey(
    int64_t tokenIdx, int64_t headIdx, int64_t sizeOffset, int64_t headNum, int64_t headSize)
{
    LocalTensor<T> inputKeyLocal = keyQue_.AllocTensor<T>();

    int64_t keyOffset = tokenIdx * tilingData_->keyStride0 + headIdx * tilingData_->keyStride1 + sizeOffset;

    DataCopyExtParams keyParams = {static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)),
                                   static_cast<uint32_t>((tilingData_->keyStride1 - headSize) * sizeof(T)),
                                   static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(inputKeyLocal, keyGm_[keyOffset], keyParams, padParams);

    keyQue_.EnQue(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormNonContiguous<T, IndexDtype, InOutMode>::CopyOutKey(
    int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t sizeOffset, int64_t headNum, int64_t headSize)
{
    LocalTensor<T> inputKeyLocal = keyQue_.DeQue<T>();

    int64_t keyCacheOffset = blockIdx * tilingData_->keyCacheStride0 + blockOffset * tilingData_->keyCacheStride1 +
                             headIdx * tilingData_->keyCacheStride2 + sizeOffset;

    DataCopyExtParams outKeyCacheParams = {
        static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>((tilingData_->keyCacheStride2 - headSize) * sizeof(T)), static_cast<uint32_t>(0)};

    DataCopyPad(keyCacheOutGm_[keyCacheOffset], inputKeyLocal, outKeyCacheParams);

    keyQue_.FreeTensor(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormNonContiguous<T, IndexDtype, InOutMode>::CopyInValue(
    int64_t tokenIdx, int64_t headIdx, int64_t sizeOffset, int64_t headNum, int64_t headSize)
{
    LocalTensor<T> inputValueLocal = valueQue_.AllocTensor<T>();

    int64_t valueOffset = tokenIdx * tilingData_->valueStride0 + headIdx * tilingData_->valueStride1 + sizeOffset;

    DataCopyExtParams valueParams = {static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)),
                                     static_cast<uint32_t>((tilingData_->valueStride1 - headSize) * sizeof(T)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(inputValueLocal, valueGm_[valueOffset], valueParams, padParams);

    valueQue_.EnQue(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormNonContiguous<T, IndexDtype, InOutMode>::CopyOutValue(
    int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t sizeOffset, int64_t headNum, int64_t headSize)
{
    LocalTensor<T> inputValueLocal = valueQue_.DeQue<T>();
    int64_t valueCacheOffset = blockIdx * tilingData_->valueCacheStride0 +
                               blockOffset * tilingData_->valueCacheStride1 + headIdx * tilingData_->valueCacheStride2 +
                               sizeOffset;

    DataCopyExtParams outValueCacheParams = {
        static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>((tilingData_->valueCacheStride2 - headSize) * sizeof(T)), static_cast<uint32_t>(0)};

    DataCopyPad(valueCacheOutGm_[valueCacheOffset], inputValueLocal, outValueCacheParams);

    valueQue_.FreeTensor(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormNonContiguous<T, IndexDtype, InOutMode>::InitLoopInfo()
{
    int64_t ubHandleSize = maxHandleSize_ / BLOCK_SIZE * BLOCK_SIZE;
    ubHandleSize = ubHandleSize / sizeof(T);
    alignKHeadSize_ = RoundUp(tilingData_->kHeadSize);

    kFullyLoad_ = (alignKHeadSize_ <= ubHandleSize);

    if (kFullyLoad_) {
        kHeadSizeIn_ = tilingData_->kHeadSize;
        kHeadSizeloop_ = 1;
        kHeadSizeTail_ = 0;
        kNumHeadIn_ = ubHandleSize / alignKHeadSize_;
        if (kNumHeadIn_ > tilingData_->numHead) {
            kNumHeadIn_ = tilingData_->numHead;
        }
        kNumHeadLoop_ = tilingData_->numHead / kNumHeadIn_;
        kNumHeadTail_ = tilingData_->numHead - kNumHeadIn_ * kNumHeadLoop_;
    } else {
        kHeadSizeIn_ = ubHandleSize;
        kHeadSizeloop_ = tilingData_->kHeadSize / kHeadSizeIn_;
        kHeadSizeTail_ = tilingData_->kHeadSize - kHeadSizeIn_ * kHeadSizeloop_;
        kNumHeadIn_ = 1;
        kNumHeadLoop_ = tilingData_->numHead;
        kNumHeadTail_ = 0;
    }

    if constexpr (InOutMode == DUAL_IN_OUT) {
        alignVHeadSize_ = RoundUp(tilingData_->vHeadSize);

        vFullyLoad_ = (alignVHeadSize_ <= ubHandleSize);

        if (vFullyLoad_) {
            vHeadSizeIn_ = tilingData_->vHeadSize;
            vHeadSizeloop_ = 1;
            vHeadSizeTail_ = 0;
            vNumHeadIn_ = ubHandleSize / alignVHeadSize_;
            if (vNumHeadIn_ > tilingData_->numHead) {
                vNumHeadIn_ = tilingData_->numHead;
            }
            vNumHeadLoop_ = tilingData_->numHead / vNumHeadIn_;
            vNumHeadTail_ = tilingData_->numHead - vNumHeadIn_ * vNumHeadLoop_;
        } else {
            vHeadSizeIn_ = ubHandleSize;
            vHeadSizeloop_ = tilingData_->vHeadSize / vHeadSizeIn_;
            vHeadSizeTail_ = tilingData_->vHeadSize - vHeadSizeIn_ * vHeadSizeloop_;
            vNumHeadIn_ = 1;
            vNumHeadLoop_ = tilingData_->numHead;
            vNumHeadTail_ = 0;
        }
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormNonContiguous<T, IndexDtype, InOutMode>::ProcessKey(int64_t tokenIdx,
                                                                                               int64_t blockIdx,
                                                                                               int64_t blockOffset)
{
    if (kFullyLoad_) {
        for (int64_t headLoop = 0; headLoop < kNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop * kNumHeadIn_;
            CopyInKey(tokenIdx, headIdx, 0, kNumHeadIn_, kHeadSizeIn_);
            CopyOutKey(blockIdx, blockOffset, headIdx, 0, kNumHeadIn_, kHeadSizeIn_);
        }
        if (kNumHeadTail_ > 0) {
            int64_t headIdx = kNumHeadLoop_ * kNumHeadIn_;
            CopyInKey(tokenIdx, headIdx, 0, kNumHeadTail_, kHeadSizeIn_);
            CopyOutKey(blockIdx, blockOffset, headIdx, 0, kNumHeadTail_, kHeadSizeIn_);
        }
    } else {
        for (int64_t headLoop = 0; headLoop < kNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop;
            for (int64_t sizeLoop = 0; sizeLoop < kHeadSizeloop_; sizeLoop++) {
                int64_t sizeOffset = sizeLoop * kHeadSizeIn_;
                CopyInKey(tokenIdx, headIdx, sizeOffset, 1, kHeadSizeIn_);
                CopyOutKey(blockIdx, blockOffset, headIdx, sizeOffset, 1, kHeadSizeIn_);
            }
            if (kHeadSizeTail_ > 0) {
                int64_t sizeOffset = kHeadSizeloop_ * kHeadSizeIn_;
                CopyInKey(tokenIdx, headIdx, sizeOffset, 1, kHeadSizeTail_);
                CopyOutKey(blockIdx, blockOffset, headIdx, sizeOffset, 1, kHeadSizeTail_);
            }
        }
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormNonContiguous<T, IndexDtype, InOutMode>::ProcessValue(int64_t tokenIdx,
                                                                                                 int64_t blockIdx,
                                                                                                 int64_t blockOffset)
{
    if (vFullyLoad_) {
        for (int64_t headLoop = 0; headLoop < vNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop * vNumHeadIn_;
            CopyInValue(tokenIdx, headIdx, 0, vNumHeadIn_, vHeadSizeIn_);
            CopyOutValue(blockIdx, blockOffset, headIdx, 0, vNumHeadIn_, vHeadSizeIn_);
        }
        if (vNumHeadTail_ > 0) {
            int64_t headIdx = vNumHeadLoop_ * vNumHeadIn_;
            CopyInValue(tokenIdx, headIdx, 0, vNumHeadTail_, vHeadSizeIn_);
            CopyOutValue(blockIdx, blockOffset, headIdx, 0, vNumHeadTail_, vHeadSizeIn_);
        }
    } else {
        for (int64_t headLoop = 0; headLoop < vNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop;
            for (int64_t sizeLoop = 0; sizeLoop < vHeadSizeloop_; sizeLoop++) {
                int64_t sizeOffset = sizeLoop * vHeadSizeIn_;
                CopyInValue(tokenIdx, headIdx, sizeOffset, 1, vHeadSizeIn_);
                CopyOutValue(blockIdx, blockOffset, headIdx, sizeOffset, 1, vHeadSizeIn_);
            }
            if (vHeadSizeTail_ > 0) {
                int64_t sizeOffset = vHeadSizeloop_ * vHeadSizeIn_;
                CopyInValue(tokenIdx, headIdx, sizeOffset, 1, vHeadSizeTail_);
                CopyOutValue(blockIdx, blockOffset, headIdx, sizeOffset, 1, vHeadSizeTail_);
            }
        }
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormNonContiguous<T, IndexDtype, InOutMode>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;

    InitLoopInfo();
    for (int64_t idx = 0; idx < curBlockFactor; idx++) {
        int64_t slotIdx = slotMappingGm_.GetValue(idx);
        if (slotIdx < 0 || slotIdx >= maxTokens_) {
            continue;
        }
        int64_t blockIdx = slotIdx / tilingData_->blockSize;
        int64_t blockOffset = slotIdx % tilingData_->blockSize;
        int64_t tokenIdx = blockFactorOffset_ + idx;

        ProcessKey(tokenIdx, blockIdx, blockOffset);

        if constexpr (InOutMode == DUAL_IN_OUT) {
            ProcessValue(tokenIdx, blockIdx, blockOffset);
        }
    }
}
} // namespace ScatterPaKvCache
#endif // SCATTER_PA_KV_CACHE_NORM_NON_CONTIGUOUS_H_
