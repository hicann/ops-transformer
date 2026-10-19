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
 * \file scatter_pa_kv_cache_alibi_fully_load.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_ALIBI_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_ALIBI_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T, typename IndexDtype, int64_t InOutMode>
class ScatterPaKvCacheAlibiFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheAlibiFullyLoad(TPipe* pipe, const ScatterPaKvCacheTilingData* __restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void CopyInKey(int64_t iter, int64_t offsetIndex, int64_t seqLenOffset);
    __aicore__ inline void CopyInValue(int64_t iter, int64_t offsetIndex, int64_t seqLenOffset);
    __aicore__ inline void CopyOutKey(int64_t iter, int64_t offsetIndex, int64_t seqLenOffset);
    __aicore__ inline void CopyOutValue(int64_t iter, int64_t offsetIndex, int64_t seqLenOffset);
    __aicore__ inline void CopyToUb(LocalTensor<IndexDtype> ubLocal, GlobalTensor<IndexDtype> gmGlobal,
                                    int64_t curBlockFactor);

private:
    TPipe* pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputKeyQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputValueQueue_;
    TBuf<TPosition::VECCALC> slotMappingBuf_;
    TBuf<TPosition::VECCALC> compressLensBuf_;

    const ScatterPaKvCacheTilingData* tilingData_;

    GlobalTensor<T> inputKeyGm_;
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T> inputValueCacheGm_;
    GlobalTensor<T> outputKeyCacheGm_;
    GlobalTensor<T> outputValueCacheGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;
    GlobalTensor<IndexDtype> compressLensGm_;
    GlobalTensor<IndexDtype> seqLensGm_;
    int64_t keyBlockOffset_{0};
    int64_t valueBlockOffset_{0};
    int64_t kvBlockOffset_{0};
};

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiFullyLoad<T, IndexDtype, InOutMode>::Init(
    GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in,
    GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, GM_ADDR key_cache_out,
    GM_ADDR value_cache_out)
{
    kvBlockOffset_ = GetBlockIdx() * tilingData_->blockFactor;
    inputKeyGm_.SetGlobalBuffer((__gm__ T*)(key));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype*)(slot_mapping) + kvBlockOffset_);
    compressLensGm_.SetGlobalBuffer((__gm__ IndexDtype*)(compress_lens) + kvBlockOffset_);
    seqLensGm_.SetGlobalBuffer((__gm__ IndexDtype*)(seq_lens));
    outputKeyCacheGm_.SetGlobalBuffer((__gm__ T*)(key_cache_out));

    pipe_->InitBuffer(slotMappingBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(compressLensBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(inputKeyQueue_, 1, (tilingData_->seqLen * RoundUp(tilingData_->kHeadSize)) * sizeof(T));

    if constexpr (InOutMode == DUAL_IN_OUT) {
        inputValueGm_.SetGlobalBuffer((__gm__ T*)(value));
        outputValueCacheGm_.SetGlobalBuffer((__gm__ T*)(value_cache_out));
        pipe_->InitBuffer(inputValueQueue_, 1, (tilingData_->seqLen * RoundUp(tilingData_->vHeadSize)) * sizeof(T));
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline int64_t ScatterPaKvCacheAlibiFullyLoad<T, IndexDtype, InOutMode>::RoundUp(int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiFullyLoad<T, IndexDtype, InOutMode>::CopyToUb(
    LocalTensor<IndexDtype> ubLocal, GlobalTensor<IndexDtype> gmGlobal, int64_t curBlockFactor)
{
    DataCopyExtParams params = {static_cast<uint16_t>(1), static_cast<uint32_t>(curBlockFactor * sizeof(IndexDtype)),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<IndexDtype> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                    static_cast<IndexDtype>(0)};
    DataCopyPad(ubLocal, gmGlobal, params, padParamIdx);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiFullyLoad<T, IndexDtype, InOutMode>::CopyInKey(int64_t iter,
                                                                                           int64_t offsetIndex,
                                                                                           int64_t seqLenOffset)
{
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.AllocTensor<T>();
    int64_t kHeadSize = tilingData_->kHeadSize;
    int64_t roundUpKHeadSize = RoundUp(kHeadSize);
    int64_t seqLen = tilingData_->seqLen;
    int64_t keyStride1 = tilingData_->keyStride1;

    // 原 for 循环约束: k < offsetIndex, k < seqLen, k >= offsetIndex - seqLenOffset
    int64_t validStart = offsetIndex - seqLenOffset > 0 ? offsetIndex - seqLenOffset : 0;
    int64_t validEnd = offsetIndex < seqLen ? offsetIndex : seqLen;
    int64_t validCount = validEnd > validStart ? (validEnd - validStart) : 0;

    if (validCount > 0) {
        DataCopyExtParams params = {static_cast<uint16_t>(validCount), static_cast<uint32_t>(kHeadSize * sizeof(T)),
                                    static_cast<uint32_t>((keyStride1 - kHeadSize) * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<T>(0)};
        int64_t offset = (kvBlockOffset_ + iter) / tilingData_->numHead * tilingData_->keyStride0 +
                         (kvBlockOffset_ + iter) % tilingData_->numHead * tilingData_->keyStride2;
        int64_t gmSrcOffset = offset + (seqLenOffset - offsetIndex + validStart) * keyStride1;
        DataCopyPad(inputKeyLocal[validStart * roundUpKHeadSize], inputKeyGm_[gmSrcOffset], params, padParams);
    }
    inputKeyQueue_.EnQue(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiFullyLoad<T, IndexDtype, InOutMode>::CopyInValue(int64_t iter,
                                                                                             int64_t offsetIndex,
                                                                                             int64_t seqLenOffset)
{
    LocalTensor<T> inputValueLocal = inputValueQueue_.AllocTensor<T>();
    int64_t vHeadSize = tilingData_->vHeadSize;
    int64_t roundUpVHeadSize = RoundUp(vHeadSize);
    int64_t seqLen = tilingData_->seqLen;
    int64_t valueStride1 = tilingData_->valueStride1;

    // 原 for 循环约束: k < offsetIndex, k < seqLen, k >= offsetIndex - seqLenOffset
    int64_t validStart = offsetIndex - seqLenOffset > 0 ? offsetIndex - seqLenOffset : 0;
    int64_t validEnd = offsetIndex < seqLen ? offsetIndex : seqLen;
    int64_t validCount = validEnd > validStart ? (validEnd - validStart) : 0;

    if (validCount > 0) {
        DataCopyExtParams params = {static_cast<uint16_t>(validCount), static_cast<uint32_t>(vHeadSize * sizeof(T)),
                                    static_cast<uint32_t>((valueStride1 - vHeadSize) * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<T>(0)};
        int64_t offset = (kvBlockOffset_ + iter) / tilingData_->numHead * tilingData_->valueStride0 +
                         (kvBlockOffset_ + iter) % tilingData_->numHead * tilingData_->valueStride2;
        int64_t gmSrcOffset = offset + (seqLenOffset - offsetIndex + validStart) * valueStride1;
        DataCopyPad(inputValueLocal[validStart * roundUpVHeadSize], inputValueGm_[gmSrcOffset], params, padParams);
    }
    inputValueQueue_.EnQue(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiFullyLoad<T, IndexDtype, InOutMode>::CopyOutKey(int64_t iter,
                                                                                            int64_t offsetIndex,
                                                                                            int64_t seqLenOffset)
{
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.DeQue<T>();
    LocalTensor<IndexDtype> slotMappingLocal = slotMappingBuf_.Get<IndexDtype>();
    int64_t kHeadSize = tilingData_->kHeadSize;
    int64_t roundUpKHeadSize = RoundUp(kHeadSize);
    int64_t seqLen = tilingData_->seqLen;
    int64_t maxToken = tilingData_->numBlocks * tilingData_->blockSize;
    int64_t kSlot = slotMappingLocal.GetValue(iter);

    // 原 for 循环约束: k < offsetIndex, k < seqLen, k >= -kSlot, k >= offsetIndex-seqLenOffset, k < maxToken-kSlot
    int64_t validStart = -kSlot > (offsetIndex - seqLenOffset) ? -kSlot : (offsetIndex - seqLenOffset);
    validStart = validStart > 0 ? validStart : 0;
    int64_t validEnd = offsetIndex < seqLen ? offsetIndex : seqLen;
    validEnd = validEnd < (maxToken - kSlot) ? validEnd : (maxToken - kSlot);
    int64_t validCount = validEnd > validStart ? (validEnd - validStart) : 0;

    if (validCount > 0) {
        DataCopyExtParams params = {static_cast<uint16_t>(validCount), static_cast<uint32_t>(kHeadSize * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPad(outputKeyCacheGm_[(kSlot + validStart) * kHeadSize], inputKeyLocal[validStart * roundUpKHeadSize],
                    params);
    }
    inputKeyQueue_.FreeTensor(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiFullyLoad<T, IndexDtype, InOutMode>::CopyOutValue(int64_t iter,
                                                                                              int64_t offsetIndex,
                                                                                              int64_t seqLenOffset)
{
    LocalTensor<T> inputValueLocal = inputValueQueue_.DeQue<T>();
    LocalTensor<IndexDtype> slotMappingLocal = slotMappingBuf_.Get<IndexDtype>();
    int64_t vHeadSize = tilingData_->vHeadSize;
    int64_t roundUpVHeadSize = RoundUp(vHeadSize);
    int64_t seqLen = tilingData_->seqLen;
    int64_t maxToken = tilingData_->numBlocks * tilingData_->blockSize;
    int64_t vSlot = slotMappingLocal.GetValue(iter);

    // 原 for 循环约束: k < offsetIndex, k < seqLen, k >= -vSlot, k >= offsetIndex-seqLenOffset, k < maxToken-vSlot
    int64_t validStart = -vSlot > (offsetIndex - seqLenOffset) ? -vSlot : (offsetIndex - seqLenOffset);
    validStart = validStart > 0 ? validStart : 0;
    int64_t validEnd = offsetIndex < seqLen ? offsetIndex : seqLen;
    validEnd = validEnd < (maxToken - vSlot) ? validEnd : (maxToken - vSlot);
    int64_t validCount = validEnd > validStart ? (validEnd - validStart) : 0;

    if (validCount > 0) {
        DataCopyExtParams params = {static_cast<uint16_t>(validCount), static_cast<uint32_t>(vHeadSize * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPad(outputValueCacheGm_[(vSlot + validStart) * vHeadSize],
                    inputValueLocal[validStart * roundUpVHeadSize], params);
    }
    inputValueQueue_.FreeTensor(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiFullyLoad<T, IndexDtype, InOutMode>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t curBlockFactor =
        (GetBlockIdx() == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;
    LocalTensor<IndexDtype> slotMappingLocal = slotMappingBuf_.Get<IndexDtype>();
    LocalTensor<IndexDtype> compressLensLocal = compressLensBuf_.Get<IndexDtype>();
    CopyToUb(compressLensLocal, compressLensGm_, curBlockFactor);
    CopyToUb(slotMappingLocal, slotMappingGm_, curBlockFactor);

    event_t eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    int64_t curBatch = 0;
    int64_t seqLenOffset = 0;
    int64_t offsetIndex = 0;
    for (int64_t i = 0; i < curBlockFactor; i++) {
        offsetIndex = compressLensLocal.GetValue(i);
        curBatch = (kvBlockOffset_ + i) / tilingData_->numHead;
        seqLenOffset = seqLensGm_.GetValue(curBatch);
        CopyInKey(i, offsetIndex, seqLenOffset);
        CopyOutKey(i, offsetIndex, seqLenOffset);
        if constexpr (InOutMode == DUAL_IN_OUT) {
            CopyInValue(i, offsetIndex, seqLenOffset);
            CopyOutValue(i, offsetIndex, seqLenOffset);
        }
    }
}
} // namespace ScatterPaKvCache
#endif // SCATTER_PA_KV_CACHE_ALIBI_FULLY_LOAD_H_
