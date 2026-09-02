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
 * \file scatter_pa_kv_cache_omni_not_fully_load.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_OMNI_NOT_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_OMNI_NOT_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T, typename IndexDtype, int64_t InOutMode>
class ScatterPaKvCacheOmniNotFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheOmniNotFullyLoad(TPipe *pipe, const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void CopyInKey(int64_t loopIdx, int64_t k, int64_t keyOffset, int64_t handleNum);
    __aicore__ inline void CopyInValue(int64_t loopIdx, int64_t k, int64_t valueOffset, int64_t handleNum);
    __aicore__ inline void CopyOutKey(int64_t loopIdx, int64_t startIdx, int64_t handleNum);
    __aicore__ inline void CopyOutValue(int64_t loopIdx, int64_t startIdx, int64_t handleNum);
    __aicore__ inline void HandleKeyCache(int64_t iter, int64_t offsetIndex);
    __aicore__ inline void HandleValueCache(int64_t iter, int64_t offsetIndex);
    __aicore__ inline void UpdateKeyCache(int64_t k, int64_t startIdx, int64_t keyOffset);
    __aicore__ inline void UpdateValueCache(int64_t k, int64_t startIdx, int64_t valueOffset);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputKeyQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputValueQueue_;
    int64_t keyBlockOffset_{0};
    int64_t blockIdx_{0};
    int64_t valueBlockOffset_{0};
    int64_t kvBlockOffset_{0};
    int64_t count_{0};

    GlobalTensor<T> inputKeyGm_;
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T> outputKeyCacheGm_;
    GlobalTensor<T> outputValueCacheGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;
    GlobalTensor<IndexDtype> compressSeqOffsetGm_;
    GlobalTensor<IndexDtype> compressLensGm_;
    GlobalTensor<IndexDtype> seqLensGm_;
    const ScatterPaKvCacheTilingData *tilingData_;
};

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::Init(
    GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in,
    GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, GM_ADDR key_cache_out,
    GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    kvBlockOffset_ = GetBlockIdx() * tilingData_->blockFactor;
    inputKeyGm_.SetGlobalBuffer((__gm__ T *)(key));

    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slot_mapping) + GetBlockIdx() * tilingData_->blockFactor);
    compressSeqOffsetGm_.SetGlobalBuffer((__gm__ IndexDtype *)(compress_seq_offset) +
                                         GetBlockIdx() * tilingData_->blockFactor);
    compressLensGm_.SetGlobalBuffer((__gm__ IndexDtype *)(compress_lens) + GetBlockIdx() * tilingData_->blockFactor);
    seqLensGm_.SetGlobalBuffer((__gm__ IndexDtype *)(seq_lens));
    outputKeyCacheGm_.SetGlobalBuffer((__gm__ T *)(key_cache_out));

    int64_t maxKHandleNum = tilingData_->kHandleNumPerLoop > tilingData_->kTailHandleNum ?
                                tilingData_->kHandleNumPerLoop :
                                tilingData_->kTailHandleNum;
    int64_t maxVHandleNum = tilingData_->vHandleNumPerLoop > tilingData_->vTailHandleNum ?
                                tilingData_->vHandleNumPerLoop :
                                tilingData_->vTailHandleNum;
    pipe_->InitBuffer(inputKeyQueue_, 1, RoundUp(maxKHandleNum) * sizeof(T));

    if constexpr (InOutMode == DUAL_IN_OUT) {
        inputValueGm_.SetGlobalBuffer((__gm__ T *)(value));
        outputValueCacheGm_.SetGlobalBuffer((__gm__ T *)(value_cache_out));
        pipe_->InitBuffer(inputValueQueue_, 1, RoundUp(maxVHandleNum) * sizeof(T));
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline int64_t ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::RoundUp(int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::HandleKeyCache(int64_t iter,
                                                                                                  int64_t offsetIndex)
{
    int64_t startIdx = slotMappingGm_.GetValue(iter);
    int64_t keyOffset = (kvBlockOffset_ + iter) / tilingData_->numHead * tilingData_->keyStride0 +
                        (kvBlockOffset_ + iter) % tilingData_->numHead * tilingData_->keyStride2;
    for (int64_t k = 0; k < offsetIndex; k++) {
        if (k >= tilingData_->seqLen) {
            break;
        }
        int64_t kStartIdx = startIdx + k + count_;
        if (kStartIdx < 0 || kStartIdx >= tilingData_->numBlocks * tilingData_->blockSize) {
            continue;
        }
        UpdateKeyCache(k, kStartIdx * tilingData_->kHeadSize, keyOffset);
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::CopyInKey(int64_t loopIdx, int64_t k,
                                                                                             int64_t keyOffset,
                                                                                             int64_t handleNum)
{
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.AllocTensor<T>();
    DataCopyExtParams inKeyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(handleNum * sizeof(T)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = 0;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    DataCopyPad(inputKeyLocal,
                inputKeyGm_[keyOffset + k * tilingData_->keyStride1 + loopIdx * tilingData_->kHandleNumPerLoop],
                inKeyParams, padParams);
    inputKeyQueue_.EnQue(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::CopyOutKey(int64_t loopIdx,
                                                                                              int64_t startIdx,
                                                                                              int64_t handleNum)
{
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.DeQue<T>();

    DataCopyExtParams outKeyCacheParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(handleNum * sizeof(T)),
                                           static_cast<uint32_t>(0), static_cast<uint32_t>(0),
                                           static_cast<uint32_t>(0)};

    DataCopyPad(outputKeyCacheGm_[startIdx + loopIdx * tilingData_->kHandleNumPerLoop], inputKeyLocal,
                outKeyCacheParams);
    inputKeyQueue_.FreeTensor(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::CopyInValue(int64_t loopIdx,
                                                                                               int64_t k,
                                                                                               int64_t valueOffset,
                                                                                               int64_t handleNum)
{
    LocalTensor<T> inputValueLocal = inputValueQueue_.AllocTensor<T>();
    DataCopyExtParams inValueParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(handleNum * sizeof(T)),
                                       static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = 0;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    DataCopyPad(inputValueLocal,
                inputValueGm_[valueOffset + k * tilingData_->valueStride1 + loopIdx * tilingData_->vHandleNumPerLoop],
                inValueParams, padParams);
    inputValueQueue_.EnQue(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::CopyOutValue(int64_t loopIdx,
                                                                                                int64_t startIdx,
                                                                                                int64_t handleNum)
{
    LocalTensor<T> inputValueLocal = inputValueQueue_.DeQue<T>();

    DataCopyExtParams outValueCacheParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(handleNum * sizeof(T)),
                                             static_cast<uint32_t>(0), static_cast<uint32_t>(0),
                                             static_cast<uint32_t>(0)};

    DataCopyPad(outputValueCacheGm_[startIdx + loopIdx * tilingData_->vHandleNumPerLoop], inputValueLocal,
                outValueCacheParams);
    inputValueQueue_.FreeTensor(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::UpdateKeyCache(int64_t k,
                                                                                                  int64_t startIdx,
                                                                                                  int64_t keyOffset)
{
    for (int64_t i = 0; i < tilingData_->kLoopNum; i++) {
        CopyInKey(i, k, keyOffset, tilingData_->kHandleNumPerLoop);
        CopyOutKey(i, startIdx, tilingData_->kHandleNumPerLoop);
    }

    // handle tail
    CopyInKey(tilingData_->kLoopNum, k, keyOffset, tilingData_->kTailHandleNum);
    CopyOutKey(tilingData_->kLoopNum, startIdx, tilingData_->kTailHandleNum);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::HandleValueCache(int64_t iter,
                                                                                                    int64_t offsetIndex)
{
    int64_t startIdx = slotMappingGm_.GetValue(iter);
    int64_t valueOffset = (kvBlockOffset_ + iter) / tilingData_->numHead * tilingData_->valueStride0 +
                          (kvBlockOffset_ + iter) % tilingData_->numHead * tilingData_->valueStride2;
    for (int64_t k = 0; k < offsetIndex; k++) {
        if (k >= tilingData_->seqLen) {
            break;
        }
        int64_t vStartIdx = startIdx + k + count_;
        if (vStartIdx < 0 || vStartIdx >= tilingData_->numBlocks * tilingData_->blockSize) {
            continue;
        }
        UpdateValueCache(k, vStartIdx * tilingData_->vHeadSize, valueOffset);
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::UpdateValueCache(int64_t k,
                                                                                                    int64_t startIdx,
                                                                                                    int64_t valueOffset)
{
    for (int64_t i = 0; i < tilingData_->vLoopNum; i++) {
        CopyInValue(i, k, valueOffset, tilingData_->vHandleNumPerLoop);
        CopyOutValue(i, startIdx, tilingData_->vHandleNumPerLoop);
    }

    // handle tail
    CopyInValue(tilingData_->vLoopNum, k, valueOffset, tilingData_->vTailHandleNum);
    CopyOutValue(tilingData_->vLoopNum, startIdx, tilingData_->vTailHandleNum);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniNotFullyLoad<T, IndexDtype, InOutMode>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;
    int64_t offsetIndex = 0;
    int64_t wins = 0;
    for (int64_t i = 0; i < curBlockFactor; i++) {
        // step1: update KvCache
        count_ = 0;
        offsetIndex = compressSeqOffsetGm_.GetValue(i);
        HandleKeyCache(i, offsetIndex);
        if constexpr (InOutMode == DUAL_IN_OUT) {
            HandleValueCache(i, offsetIndex);
        }
        count_ += offsetIndex;
        wins = compressLensGm_.GetValue(i);
        // step2: update KvCache
        offsetIndex =
            seqLensGm_.GetValue((kvBlockOffset_ + i) / tilingData_->numHead) - wins - compressSeqOffsetGm_.GetValue(i);
        if (offsetIndex > 0) {
            HandleKeyCache(i, offsetIndex);
            if constexpr (InOutMode == DUAL_IN_OUT) {
                HandleValueCache(i, offsetIndex);
            }
        }
    }
}

} // namespace ScatterPaKvCache

#endif // SCATTER_PA_KV_CACHE_OMNI_NOT_FULLY_LOAD_H_
