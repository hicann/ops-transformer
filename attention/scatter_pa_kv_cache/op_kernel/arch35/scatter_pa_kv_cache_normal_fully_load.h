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
 * \file scatter_pa_kv_cache_normal_fully_load.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_NORMAL_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_NORMAL_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT = false>
class ScatterPaKvCacheNormalFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheNormalFullyLoad(TPipe *pipe, const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t curBlockFactor);
    __aicore__ inline void CopyOut(int64_t curBlockFactor);
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void CalcStartIdx(LocalTensor<IndexDtype> slotMappingLocal, int64_t curBlockFactor,
                                        int64_t handleNumPerCore);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputKeyQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputValueQueue_;

    GlobalTensor<T> inputKeyGm_;
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T> outputKeyCacheGm_;
    GlobalTensor<T> outputValueCacheGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;

    const ScatterPaKvCacheTilingData *tilingData_;

    int64_t blockIdx_{0};

    TBuf<TPosition::VECCALC> kSlotMappingBuf_;
    TBuf<TPosition::VECCALC> vSlotMappingBuf_;
    int64_t maxTokens_ = 0;
    int64_t blockFactorOffset_ = 0;
};

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, InOutMode, NCT>::Init(
    GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in,
    GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, GM_ADDR key_cache_out,
    GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    blockFactorOffset_ = blockIdx_ * tilingData_->blockFactor;
    inputKeyGm_.SetGlobalBuffer((__gm__ T *)(key));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slot_mapping) + GetBlockIdx() * tilingData_->blockFactor);
    outputKeyCacheGm_.SetGlobalBuffer((__gm__ T *)(key_cache_out));
    int64_t maxBlockFactor = tilingData_->blockFactor > tilingData_->tailBlockFactor ? tilingData_->blockFactor :
                                                                                       tilingData_->tailBlockFactor;
    pipe_->InitBuffer(kSlotMappingBuf_, RoundUp(maxBlockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(inputKeyQueue_, 1, maxBlockFactor * RoundUp(tilingData_->kHandleNumPerCore) * sizeof(T));
    if constexpr (InOutMode == DUAL_IN_OUT) {
        inputValueGm_.SetGlobalBuffer((__gm__ T *)(value));
        outputValueCacheGm_.SetGlobalBuffer((__gm__ T *)(value_cache_out));
        pipe_->InitBuffer(vSlotMappingBuf_, RoundUp(maxBlockFactor) * sizeof(IndexDtype));
        pipe_->InitBuffer(inputValueQueue_, 1, maxBlockFactor * RoundUp(tilingData_->vHandleNumPerCore) * sizeof(T));
    }
    maxTokens_ = tilingData_->numBlocks * tilingData_->blockSize;
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline int64_t ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, InOutMode, NCT>::RoundUp(int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, InOutMode, NCT>::CalcStartIdx(
    LocalTensor<IndexDtype> slotMappingLocal, int64_t curBlockFactor, int64_t handleNumPerCore)
{
    DataCopyExtParams slotMappingParams{1, static_cast<uint32_t>(curBlockFactor * sizeof(IndexDtype)), 0, 0, 0};
    DataCopyPadExtParams<IndexDtype> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                    static_cast<IndexDtype>(0)};
    DataCopyPad(slotMappingLocal, slotMappingGm_, slotMappingParams, padParamIdx);
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, InOutMode, NCT>::CopyIn(int64_t curBlockFactor)
{
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.AllocTensor<T>();

    DataCopyExtParams kParams = {static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(tilingData_->kHandleNumPerCore * sizeof(T)),
                                 static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyExtParams vParams = {static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(tilingData_->vHandleNumPerCore * sizeof(T)),
                                 static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = 0;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    for (int64_t i = 0; i < curBlockFactor; i++) {
        int64_t kOffsetIn = (blockFactorOffset_ + i) * tilingData_->kHandleNumPerCore;
        if constexpr (NCT) {
            kOffsetIn = (blockFactorOffset_ + i) * tilingData_->kStride + tilingData_->kOffset;
            if (kOffsetIn < 0) {
                continue;
            }
        }
        DataCopyPad(inputKeyLocal[i * RoundUp(tilingData_->kHandleNumPerCore)], inputKeyGm_[kOffsetIn], kParams,
                    padParams);
    }
    inputKeyQueue_.EnQue(inputKeyLocal);
    if constexpr (InOutMode == DUAL_IN_OUT) {
        LocalTensor<T> inputValueLocal = inputValueQueue_.AllocTensor<T>();
        for (int64_t i = 0; i < curBlockFactor; i++) {
            int64_t vOffsetIn = (blockFactorOffset_ + i) * tilingData_->vHandleNumPerCore;
            if constexpr (NCT) {
                vOffsetIn = (blockFactorOffset_ + i) * tilingData_->vStride + tilingData_->vOffset;
                if (vOffsetIn < 0) {
                    continue;
                }
            }
            DataCopyPad(inputValueLocal[i * RoundUp(tilingData_->vHandleNumPerCore)], inputValueGm_[vOffsetIn], vParams,
                        padParams);
        }
        inputValueQueue_.EnQue(inputValueLocal);
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, InOutMode, NCT>::CopyOut(int64_t curBlockFactor)
{
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.DeQue<T>();
    LocalTensor<IndexDtype> kSlotMappingLocal = kSlotMappingBuf_.Get<IndexDtype>();

    DataCopyExtParams outKeyCacheParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_->kHandleNumPerCore * sizeof(T)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    for (int64_t i = 0; i < curBlockFactor; i++) {
        int64_t kStartIdx = kSlotMappingLocal.GetValue(i);
        if (kStartIdx < 0 || kStartIdx >= maxTokens_) {
            continue;
        }
        int64_t kBlockIdx = kStartIdx / tilingData_->blockSize;
        int64_t kBlockOffset = kStartIdx % tilingData_->blockSize;
        kStartIdx = kBlockIdx * tilingData_->keyCacheStride0 + kBlockOffset * tilingData_->keyCacheStride1;
        DataCopyPad(outputKeyCacheGm_[kStartIdx], inputKeyLocal[i * RoundUp(tilingData_->kHandleNumPerCore)],
                    outKeyCacheParams);
    }

    inputKeyQueue_.FreeTensor(inputKeyLocal);

    if constexpr (InOutMode == DUAL_IN_OUT) {
        LocalTensor<T> inputValueLocal = inputValueQueue_.DeQue<T>();
        LocalTensor<IndexDtype> vSlotMappingLocal = vSlotMappingBuf_.Get<IndexDtype>();
        DataCopyExtParams outValueCacheParams = {
            static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_->vHandleNumPerCore * sizeof(T)),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        for (int64_t i = 0; i < curBlockFactor; i++) {
            int64_t vStartIdx = vSlotMappingLocal.GetValue(i);
            if (vStartIdx < 0 || vStartIdx >= maxTokens_) {
                continue;
            }
            int64_t vBlockIdx = vStartIdx / tilingData_->blockSize;
            int64_t vBlockOffset = vStartIdx % tilingData_->blockSize;
            vStartIdx = vBlockIdx * tilingData_->valueCacheStride0 + vBlockOffset * tilingData_->valueCacheStride1;
            DataCopyPad(outputValueCacheGm_[vStartIdx], inputValueLocal[i * RoundUp(tilingData_->vHandleNumPerCore)],
                        outValueCacheParams);
        }
        inputValueQueue_.FreeTensor(inputValueLocal);
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, InOutMode, NCT>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;

    LocalTensor<IndexDtype> kSlotMappingLocal = kSlotMappingBuf_.Get<IndexDtype>();
    CalcStartIdx(kSlotMappingLocal, curBlockFactor, tilingData_->kHandleNumPerCore);
    if constexpr (InOutMode == DUAL_IN_OUT) {
        LocalTensor<IndexDtype> vSlotMappingLocal = vSlotMappingBuf_.Get<IndexDtype>();
        CalcStartIdx(vSlotMappingLocal, curBlockFactor, tilingData_->vHandleNumPerCore);
    }
    event_t eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    CopyIn(curBlockFactor);
    CopyOut(curBlockFactor);
}

} // namespace ScatterPaKvCache

#endif // SCATTER_PA_KV_CACHE_NORMAL_FULLY_LOAD_H_
