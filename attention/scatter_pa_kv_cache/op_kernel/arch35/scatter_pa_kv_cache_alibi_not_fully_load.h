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
 * \file scatter_pa_kv_cache_alibi_not_fully_load.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_ALIBI_NOT_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_ALIBI_NOT_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T, typename IndexDtype, int64_t InOutMode>
class ScatterPaKvCacheAlibiNotFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheAlibiNotFullyLoad(TPipe *pipe,
                                                        const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void CopyInKey(int64_t iter, int64_t loopIdx, int64_t curOffset, int64_t handleNumPerLoop);
    __aicore__ inline void CopyInValue(int64_t iter, int64_t loopIdx, int64_t curOffset, int64_t handleNumPerLoop);
    __aicore__ inline void CopyOutKey(int64_t loopIdx, int64_t startIdx, int64_t handleNumPerLoop);
    __aicore__ inline void CopyOutValue(int64_t loopIdx, int64_t startIdx, int64_t handleNumPerLoop);
    __aicore__ inline void CopyToUb(LocalTensor<IndexDtype> ubLocal, GlobalTensor<IndexDtype> gmGlobal,
                                    int64_t handleNumPerLoop);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputKeyQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputValueQueue_;
    TBuf<TPosition::VECCALC> slotMappingBuf_;
    TBuf<TPosition::VECCALC> compressLensBuf_;

    const ScatterPaKvCacheTilingData *tilingData_;

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
__aicore__ inline void ScatterPaKvCacheAlibiNotFullyLoad<T, IndexDtype, InOutMode>::Init(
    GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in,
    GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, GM_ADDR key_cache_out,
    GM_ADDR value_cache_out)
{
    kvBlockOffset_ = GetBlockIdx() * tilingData_->blockFactor;
    inputKeyGm_.SetGlobalBuffer((__gm__ T *)(key));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slot_mapping) + kvBlockOffset_);
    compressLensGm_.SetGlobalBuffer((__gm__ IndexDtype *)(compress_lens) + kvBlockOffset_);
    seqLensGm_.SetGlobalBuffer((__gm__ IndexDtype *)(seq_lens));
    outputKeyCacheGm_.SetGlobalBuffer((__gm__ T *)(key_cache_out));

    pipe_->InitBuffer(slotMappingBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(compressLensBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    int64_t kMaxHandleNumPerLoop = tilingData_->kHandleNumPerLoop > tilingData_->kTailHandleNum ?
                                       tilingData_->kHandleNumPerLoop :
                                       tilingData_->kTailHandleNum;
    int64_t vMaxHandleNumPerLoop = tilingData_->vHandleNumPerLoop > tilingData_->vTailHandleNum ?
                                       tilingData_->vHandleNumPerLoop :
                                       tilingData_->vTailHandleNum;
    pipe_->InitBuffer(inputKeyQueue_, 1, RoundUp(kMaxHandleNumPerLoop) * sizeof(T));

    if constexpr (InOutMode == DUAL_IN_OUT) {
        inputValueGm_.SetGlobalBuffer((__gm__ T *)(value));
        outputValueCacheGm_.SetGlobalBuffer((__gm__ T *)(value_cache_out));
        pipe_->InitBuffer(inputValueQueue_, 1, RoundUp(vMaxHandleNumPerLoop) * sizeof(T));
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline int64_t ScatterPaKvCacheAlibiNotFullyLoad<T, IndexDtype, InOutMode>::RoundUp(int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiNotFullyLoad<T, IndexDtype, InOutMode>::CopyToUb(
    LocalTensor<IndexDtype> ubLocal, GlobalTensor<IndexDtype> gmGlobal, int64_t curBlockFactor)
{
    DataCopyExtParams params = {static_cast<uint16_t>(1), static_cast<uint32_t>(curBlockFactor * sizeof(IndexDtype)),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<IndexDtype> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                    static_cast<IndexDtype>(0)};
    DataCopyPad(ubLocal, gmGlobal, params, padParamIdx);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiNotFullyLoad<T, IndexDtype, InOutMode>::CopyInKey(int64_t iter,
                                                                                              int64_t loopIdx,
                                                                                              int64_t curOffset,
                                                                                              int64_t handleNumPerLoop)
{
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.AllocTensor<T>();
    int64_t offset = (kvBlockOffset_ + iter) / tilingData_->numHead * tilingData_->keyStride0 +
                     (kvBlockOffset_ + iter) % tilingData_->numHead * tilingData_->keyStride2;

    DataCopyExtParams keyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(handleNumPerLoop * sizeof(T)),
                                   static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(inputKeyLocal,
                inputKeyGm_[offset + curOffset * tilingData_->keyStride1 + loopIdx * tilingData_->kHandleNumPerLoop],
                keyParams, padParams);
    inputKeyQueue_.EnQue(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiNotFullyLoad<T, IndexDtype, InOutMode>::CopyInValue(
    int64_t iter, int64_t loopIdx, int64_t curOffset, int64_t handleNumPerLoop)
{
    LocalTensor<T> inputValueLocal = inputValueQueue_.AllocTensor<T>();
    int64_t offset = (kvBlockOffset_ + iter) / tilingData_->numHead * tilingData_->valueStride0 +
                     (kvBlockOffset_ + iter) % tilingData_->numHead * tilingData_->valueStride2;

    DataCopyExtParams valueParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(handleNumPerLoop * sizeof(T)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(
        inputValueLocal,
        inputValueGm_[offset + curOffset * tilingData_->valueStride1 + loopIdx * tilingData_->vHandleNumPerLoop],
        valueParams, padParams);
    inputValueQueue_.EnQue(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiNotFullyLoad<T, IndexDtype, InOutMode>::CopyOutKey(int64_t loopIdx,
                                                                                               int64_t startIdx,
                                                                                               int64_t handleNumPerLoop)
{
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.DeQue<T>();

    DataCopyExtParams outKeyCacheParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(handleNumPerLoop * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outputKeyCacheGm_[startIdx * tilingData_->kHeadSize + loopIdx * tilingData_->kHandleNumPerLoop],
                inputKeyLocal, outKeyCacheParams);
    inputKeyQueue_.FreeTensor(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiNotFullyLoad<T, IndexDtype, InOutMode>::CopyOutValue(
    int64_t loopIdx, int64_t startIdx, int64_t handleNumPerLoop)
{
    LocalTensor<T> inputValueLocal = inputValueQueue_.DeQue<T>();

    DataCopyExtParams outValueCacheParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(handleNumPerLoop * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outputValueCacheGm_[startIdx * tilingData_->vHeadSize + loopIdx * tilingData_->vHandleNumPerLoop],
                inputValueLocal, outValueCacheParams);
    inputValueQueue_.FreeTensor(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheAlibiNotFullyLoad<T, IndexDtype, InOutMode>::Process()
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
    int64_t curOffset = 0;
    int64_t startIdx = 0;
    for (int64_t i = 0; i < curBlockFactor; i++) {
        offsetIndex = compressLensLocal.GetValue(i);
        curBatch = (kvBlockOffset_ + i) / tilingData_->numHead;
        seqLenOffset = seqLensGm_.GetValue(curBatch);
        LocalTensor<IndexDtype> slotMappingLocal = slotMappingBuf_.Get<IndexDtype>();
        for (int64_t k = 0; k < offsetIndex; k++) {
            if (k >= tilingData_->seqLen) {
                break;
            }
            curOffset = seqLenOffset - offsetIndex + k;
            startIdx = slotMappingLocal.GetValue(i) + k;
            if (startIdx < 0 || startIdx >= tilingData_->numBlocks * tilingData_->blockSize || curOffset < 0) {
                continue;
            }

            for (int64_t idx = 0; idx < tilingData_->kLoopNum; idx++) {
                CopyInKey(i, idx, curOffset, tilingData_->kHandleNumPerLoop);
                CopyOutKey(idx, startIdx, tilingData_->kHandleNumPerLoop);
            }
            CopyInKey(i, tilingData_->kLoopNum, curOffset, tilingData_->kTailHandleNum);
            CopyOutKey(tilingData_->kLoopNum, startIdx, tilingData_->kTailHandleNum);
            if constexpr (InOutMode == DUAL_IN_OUT) {
                for (int64_t idx = 0; idx < tilingData_->vLoopNum; idx++) {
                    CopyInValue(i, idx, curOffset, tilingData_->vHandleNumPerLoop);
                    CopyOutValue(idx, startIdx, tilingData_->vHandleNumPerLoop);
                }
                CopyInValue(i, tilingData_->vLoopNum, curOffset, tilingData_->vTailHandleNum);
                CopyOutValue(tilingData_->vLoopNum, startIdx, tilingData_->vTailHandleNum);
            }
        }
    }
}
} // namespace ScatterPaKvCache
#endif // SCATTER_PA_KV_CACHE_ALIBI_NOT_FULLY_LOAD_H_
