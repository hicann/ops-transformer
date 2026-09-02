/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file scatter_pa_kv_cache_normal_not_fully_load.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_NORMAL_NOT_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_NORMAL_NOT_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT = false>
class ScatterPaKvCacheNormalNotFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheNormalNotFullyLoad(TPipe *pipe,
                                                         const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInKey(int64_t iter, int64_t blockOffset, int64_t handleNumPerLoop);
    __aicore__ inline void CopyOutKey(int64_t iter, int64_t kStartIdx, int64_t handleNumPerLoop);
    __aicore__ inline void CopyInValue(int64_t iter, int64_t blockOffset, int64_t handleNumPerLoop);
    __aicore__ inline void CopyOutValue(int64_t iter, int64_t vStartIdx, int64_t handleNumPerLoop);
    __aicore__ inline int64_t RoundUp(int64_t x);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> keyQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> valueQueue_;
    const ScatterPaKvCacheTilingData *tilingData_;

    GlobalTensor<T> keyGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<T> keyCacheOutGm_;
    GlobalTensor<T> valueCacheOutGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;

    int64_t blockIdx_{0};
    int64_t maxTokens_ = 0;
    int64_t blockFactorOffset_ = 0;
};

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalNotFullyLoad<T, IndexDtype, InOutMode, NCT>::Init(
    GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in,
    GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, GM_ADDR key_cache_out,
    GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    blockFactorOffset_ = blockIdx_ * tilingData_->blockFactor;
    keyGm_.SetGlobalBuffer((__gm__ T *)(key));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slot_mapping) + GetBlockIdx() * tilingData_->blockFactor);
    keyCacheOutGm_.SetGlobalBuffer((__gm__ T *)(key_cache_out));
    int64_t kMaxHandleNumPerLoop = tilingData_->kHandleNumPerLoop > tilingData_->kTailHandleNum ?
                                       tilingData_->kHandleNumPerLoop :
                                       tilingData_->kTailHandleNum;
    pipe_->InitBuffer(keyQueue_, 1, RoundUp(kMaxHandleNumPerLoop) * sizeof(T));

    if constexpr (InOutMode == DUAL_IN_OUT) {
        valueGm_.SetGlobalBuffer((__gm__ T *)(value));
        valueCacheOutGm_.SetGlobalBuffer((__gm__ T *)(value_cache_out));
        int64_t vMaxHandleNumPerLoop = tilingData_->vHandleNumPerLoop > tilingData_->vTailHandleNum ?
                                           tilingData_->vHandleNumPerLoop :
                                           tilingData_->vTailHandleNum;
        pipe_->InitBuffer(valueQueue_, 1, RoundUp(vMaxHandleNumPerLoop) * sizeof(T));
    }
    maxTokens_ = tilingData_->numBlocks * tilingData_->blockSize;
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline int64_t ScatterPaKvCacheNormalNotFullyLoad<T, IndexDtype, InOutMode, NCT>::RoundUp(int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalNotFullyLoad<T, IndexDtype, InOutMode, NCT>::CopyInKey(
    int64_t iter, int64_t blockOffset, int64_t handleNumPerLoop)
{
    LocalTensor<T> inputKeyLocal = keyQueue_.AllocTensor<T>();

    DataCopyExtParams keyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(handleNumPerLoop * sizeof(T)),
                                   static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(inputKeyLocal, keyGm_[blockOffset + iter * tilingData_->kHandleNumPerLoop], keyParams, padParams);

    keyQueue_.EnQue(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalNotFullyLoad<T, IndexDtype, InOutMode, NCT>::CopyOutKey(
    int64_t iter, int64_t kStartIdx, int64_t handleNumPerLoop)
{
    LocalTensor<T> inputKeyLocal = keyQueue_.DeQue<T>();

    DataCopyExtParams outKeyCacheParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(handleNumPerLoop * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    event_t eventIdV2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdV2ToS);
    WaitFlag<HardEvent::V_S>(eventIdV2ToS);

    DataCopyPad(keyCacheOutGm_[kStartIdx + iter * tilingData_->kHandleNumPerLoop], inputKeyLocal, outKeyCacheParams);

    keyQueue_.FreeTensor(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalNotFullyLoad<T, IndexDtype, InOutMode, NCT>::CopyInValue(
    int64_t iter, int64_t blockOffset, int64_t handleNumPerLoop)
{
    LocalTensor<T> inputValueLocal = valueQueue_.AllocTensor<T>();
    DataCopyExtParams keyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(handleNumPerLoop * sizeof(T)),
                                   static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(inputValueLocal, valueGm_[blockOffset + iter * tilingData_->vHandleNumPerLoop], keyParams, padParams);

    valueQueue_.EnQue(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalNotFullyLoad<T, IndexDtype, InOutMode, NCT>::CopyOutValue(
    int64_t iter, int64_t vStartIdx, int64_t handleNumPerLoop)
{
    LocalTensor<T> inputValueLocal = valueQueue_.DeQue<T>();

    DataCopyExtParams outValueCacheParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(handleNumPerLoop * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    event_t eventIdV2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdV2ToS);
    WaitFlag<HardEvent::V_S>(eventIdV2ToS);

    DataCopyPad(valueCacheOutGm_[vStartIdx + iter * tilingData_->vHandleNumPerLoop], inputValueLocal,
                outValueCacheParams);

    valueQueue_.FreeTensor(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalNotFullyLoad<T, IndexDtype, InOutMode, NCT>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;

    for (int64_t idx = 0; idx < curBlockFactor; idx++) {
        int64_t kOffsetIn = (blockFactorOffset_ + idx) * tilingData_->kHandleNumPerCore;
        if constexpr (NCT) {
            kOffsetIn = (blockFactorOffset_ + idx) * tilingData_->kStride + tilingData_->kOffset;
            if (kOffsetIn < 0) {
                continue;
            }
        }
        int64_t startIdx = slotMappingGm_.GetValue(idx);
        if (startIdx < 0 || startIdx >= maxTokens_) {
            continue;
        }
        int64_t blockIdx = startIdx / tilingData_->blockSize;
        int64_t blockOffset = startIdx % tilingData_->blockSize;
        int64_t kStartIdx = blockIdx * tilingData_->keyCacheStride0 + blockOffset * tilingData_->keyCacheStride1;
        // key main loop
        for (int64_t i = 0; i < tilingData_->kLoopNum; i++) {
            CopyInKey(i, kOffsetIn, tilingData_->kHandleNumPerLoop);
            CopyOutKey(i, kStartIdx, tilingData_->kHandleNumPerLoop);
        }
        // tail
        CopyInKey(tilingData_->kLoopNum, kOffsetIn, tilingData_->kTailHandleNum);
        CopyOutKey(tilingData_->kLoopNum, kStartIdx, tilingData_->kTailHandleNum);

        if constexpr (InOutMode == DUAL_IN_OUT) {
            int64_t vOffsetIn = (blockFactorOffset_ + idx) * tilingData_->vHandleNumPerCore;
            if constexpr (NCT) {
                vOffsetIn = (blockFactorOffset_ + idx) * tilingData_->vStride + tilingData_->vOffset;
                if (vOffsetIn < 0) {
                    continue;
                }
            }
            int64_t vStartIdx =
                blockIdx * tilingData_->valueCacheStride0 + blockOffset * tilingData_->valueCacheStride1;
            // value main loop
            for (int64_t i = 0; i < tilingData_->vLoopNum; i++) {
                CopyInValue(i, vOffsetIn, tilingData_->vHandleNumPerLoop);
                CopyOutValue(i, vStartIdx, tilingData_->vHandleNumPerLoop);
            }
            // tail
            CopyInValue(tilingData_->vLoopNum, vOffsetIn, tilingData_->vTailHandleNum);
            CopyOutValue(tilingData_->vLoopNum, vStartIdx, tilingData_->vTailHandleNum);
        }
    }
}
} // namespace ScatterPaKvCache
#endif // SCATTER_PA_KV_CACHE_NORMAL_NOT_FULLY_LOAD_H_
