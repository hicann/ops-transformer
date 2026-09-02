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
 * \file scatter_pa_kv_cache_omni_fully_load.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_OMNI_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_OMNI_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T, typename IndexDtype, int64_t InOutMode>
class ScatterPaKvCacheOmniFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheOmniFullyLoad(TPipe *pipe, const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void CopyInKey(int64_t iter, int64_t offsetIndex, int64_t startIdx, int64_t curBlockFactor);
    __aicore__ inline void CopyInValue(int64_t iter, int64_t offsetIndex, int64_t startIdx, int64_t curBlockFactor);
    __aicore__ inline void CopyOutKey(int64_t iter, int64_t offsetIndex, int64_t startOffset, int64_t curBlockFactor);
    __aicore__ inline void CopyOutValue(int64_t iter, int64_t offsetIndex, int64_t startOffset, int64_t curBlockFactor);
    __aicore__ inline void CopyToUb(LocalTensor<IndexDtype> ubLocal, GlobalTensor<IndexDtype> gmGlobal,
                                    int64_t curBlockFactor);
    __aicore__ inline void CalcStartIdx(LocalTensor<IndexDtype> slotMappingLocal, int64_t curBlockFactor);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputKeyQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputValueQueue_;
    TBuf<TPosition::VECCALC> slotMappingBuf_;
    TBuf<TPosition::VECCALC> compressSeqOffsetBuf_;
    TBuf<TPosition::VECCALC> seqLensBuf_;
    TBuf<TPosition::VECCALC> compressLensBuf_;

    const ScatterPaKvCacheTilingData *tilingData_;

    GlobalTensor<T> inputKeyGm_;
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T> inputValueCacheGm_;
    GlobalTensor<T> outputKeyCacheGm_;
    GlobalTensor<T> outputValueCacheGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;
    GlobalTensor<IndexDtype> compressSeqOffsetGm_;
    GlobalTensor<IndexDtype> compressLensGm_;
    GlobalTensor<IndexDtype> seqLensGm_;
    int64_t blockIdx_{0};
    int64_t keyBlockOffset_{0};
    int64_t valueBlockOffset_{0};
    int64_t kvBlockOffset_{0};
    int64_t count_{0};
};

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniFullyLoad<T, IndexDtype, InOutMode>::Init(
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

    pipe_->InitBuffer(slotMappingBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(compressSeqOffsetBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(seqLensBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(compressLensBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(inputKeyQueue_, 1, (tilingData_->seqLen * RoundUp(tilingData_->kHeadSize)) * sizeof(T));

    if constexpr (InOutMode == DUAL_IN_OUT) {
        inputValueGm_.SetGlobalBuffer((__gm__ T *)(value));
        outputValueCacheGm_.SetGlobalBuffer((__gm__ T *)(value_cache_out));
        pipe_->InitBuffer(inputValueQueue_, 1, (tilingData_->seqLen * RoundUp(tilingData_->vHeadSize)) * sizeof(T));
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline int64_t ScatterPaKvCacheOmniFullyLoad<T, IndexDtype, InOutMode>::RoundUp(int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniFullyLoad<T, IndexDtype, InOutMode>::CalcStartIdx(
    LocalTensor<IndexDtype> slotMappingLocal, int64_t curBlockFactor)
{
    DataCopyExtParams slotMappingParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(curBlockFactor * sizeof(IndexDtype)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<IndexDtype> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                    static_cast<IndexDtype>(0)};
    DataCopyPad(slotMappingLocal, slotMappingGm_, slotMappingParams, padParamIdx);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniFullyLoad<T, IndexDtype, InOutMode>::CopyToUb(
    LocalTensor<IndexDtype> ubLocal, GlobalTensor<IndexDtype> gmGlobal, int64_t curBlockFactor)
{
    DataCopyExtParams params = {static_cast<uint16_t>(1), static_cast<uint32_t>(curBlockFactor * sizeof(IndexDtype)),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<IndexDtype> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                    static_cast<IndexDtype>(0)};
    DataCopyPad(ubLocal, gmGlobal, params, padParamIdx);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniFullyLoad<T, IndexDtype, InOutMode>::CopyInKey(int64_t iter,
                                                                                          int64_t offsetIndex,
                                                                                          int64_t startIdx,
                                                                                          int64_t curBlockFactor)
{
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.AllocTensor<T>();
    DataCopyExtParams inKeyParams = {static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(tilingData_->kHeadSize * sizeof(T)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = 0;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    int64_t offset = (kvBlockOffset_ + iter) / tilingData_->numHead * tilingData_->keyStride0 +
                     (kvBlockOffset_ + iter) % tilingData_->numHead * tilingData_->keyStride2;
    for (int64_t k = startIdx; k < offsetIndex; ++k) {
        if (k >= tilingData_->seqLen) {
            break;
        }
        DataCopyPad(inputKeyLocal[(k - startIdx) * RoundUp(tilingData_->kHeadSize)],
                    inputKeyGm_[offset + k * tilingData_->keyStride1], inKeyParams, padParams);
    }
    inputKeyQueue_.EnQue(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniFullyLoad<T, IndexDtype, InOutMode>::CopyInValue(int64_t iter,
                                                                                            int64_t offsetIndex,
                                                                                            int64_t startIdx,
                                                                                            int64_t curBlockFactor)
{
    LocalTensor<T> inputValueLocal = inputValueQueue_.AllocTensor<T>();

    DataCopyExtParams inValueParams = {static_cast<uint16_t>(1),
                                       static_cast<uint32_t>(tilingData_->vHeadSize * sizeof(T)),
                                       static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = 0;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    int64_t offset = (kvBlockOffset_ + iter) / tilingData_->numHead * tilingData_->valueStride0 +
                     (kvBlockOffset_ + iter) % tilingData_->numHead * tilingData_->valueStride2;
    for (int64_t k = startIdx; k < offsetIndex; ++k) {
        if (k >= tilingData_->seqLen) {
            break;
        }
        DataCopyPad(inputValueLocal[(k - startIdx) * RoundUp(tilingData_->vHeadSize)],
                    inputValueGm_[offset + k * tilingData_->valueStride1], inValueParams, padParams);
    }
    inputValueQueue_.EnQue(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniFullyLoad<T, IndexDtype, InOutMode>::CopyOutKey(int64_t iter,
                                                                                           int64_t offsetIndex,
                                                                                           int64_t startOffset,
                                                                                           int64_t curBlockFactor)
{
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.DeQue<T>();
    LocalTensor<IndexDtype> slotMappingLocal = slotMappingBuf_.Get<IndexDtype>();

    DataCopyExtParams outKeyCacheParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_->kHeadSize * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    int64_t kSlot = slotMappingLocal.GetValue(iter) + startOffset;
    for (int64_t k = 0; k < offsetIndex; k++) {
        if (k >= tilingData_->seqLen) {
            break;
        }
        int64_t kStartIdx = kSlot + k;
        if (kStartIdx < 0 || kStartIdx >= tilingData_->numBlocks * tilingData_->blockSize) {
            continue;
        }
        DataCopyPad(outputKeyCacheGm_[kStartIdx * tilingData_->kHeadSize],
                    inputKeyLocal[k * RoundUp(tilingData_->kHeadSize)], outKeyCacheParams);
    }
    inputKeyQueue_.FreeTensor(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniFullyLoad<T, IndexDtype, InOutMode>::CopyOutValue(int64_t iter,
                                                                                             int64_t offsetIndex,
                                                                                             int64_t startOffset,
                                                                                             int64_t curBlockFactor)
{
    LocalTensor<T> inputValueLocal = inputValueQueue_.DeQue<T>();
    LocalTensor<IndexDtype> slotMappingLocal = slotMappingBuf_.Get<IndexDtype>();

    DataCopyExtParams outValueCacheParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_->vHeadSize * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    int64_t vSlot = slotMappingLocal.GetValue(iter) + startOffset;
    for (int64_t k = 0; k < offsetIndex; k++) {
        if (k >= tilingData_->seqLen) {
            break;
        }
        int64_t vStartIdx = vSlot + k;
        if (vStartIdx < 0 || vStartIdx >= tilingData_->numBlocks * tilingData_->blockSize) {
            continue;
        }
        DataCopyPad(outputValueCacheGm_[vStartIdx * tilingData_->vHeadSize],
                    inputValueLocal[k * RoundUp(tilingData_->vHeadSize)], outValueCacheParams);
    }
    inputValueQueue_.FreeTensor(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheOmniFullyLoad<T, IndexDtype, InOutMode>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;
    LocalTensor<IndexDtype> slotMappingLocal = slotMappingBuf_.Get<IndexDtype>();
    LocalTensor<IndexDtype> compressSeqOffsetLocal = compressSeqOffsetBuf_.Get<IndexDtype>();
    LocalTensor<IndexDtype> compressLensLocal = compressLensBuf_.Get<IndexDtype>();
    CopyToUb(compressSeqOffsetLocal, compressSeqOffsetGm_, curBlockFactor);
    CopyToUb(compressLensLocal, compressLensGm_, curBlockFactor);
    CopyToUb(slotMappingLocal, slotMappingGm_, curBlockFactor);
    int64_t offsetIndex = 0;
    int64_t wins = 0;

    event_t eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

    for (int64_t i = 0; i < curBlockFactor; i++) {
        // step1: update KvCache
        count_ = 0;
        offsetIndex = compressSeqOffsetLocal.GetValue(i);
        CopyInKey(i, offsetIndex, 0, curBlockFactor);
        CopyOutKey(i, offsetIndex, 0, curBlockFactor);
        if constexpr (InOutMode == DUAL_IN_OUT) {
            CopyInValue(i, offsetIndex, 0, curBlockFactor);
            CopyOutValue(i, offsetIndex, 0, curBlockFactor);
        }
        count_ += offsetIndex;
        wins = compressLensLocal.GetValue(i);

        // step2: update KvCache
        offsetIndex = seqLensGm_.GetValue((kvBlockOffset_ + i) / tilingData_->numHead) - wins -
                      compressSeqOffsetLocal.GetValue(i);
        if (offsetIndex > 0) {
            CopyInKey(i, offsetIndex, 0, curBlockFactor);
            CopyOutKey(i, offsetIndex, count_, curBlockFactor);
            if constexpr (InOutMode == DUAL_IN_OUT) {
                CopyInValue(i, offsetIndex, 0, curBlockFactor);
                CopyOutValue(i, offsetIndex, count_, curBlockFactor);
            }
        }
    }
}
} // namespace ScatterPaKvCache

#endif // SCATTER_PA_KV_CACHE_OMNI_FULLY_LOAD_H_
