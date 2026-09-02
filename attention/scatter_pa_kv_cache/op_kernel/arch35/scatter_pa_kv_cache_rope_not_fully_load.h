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
 * \file scatter_pa_kv_cache_rope_fully_load.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_ROPE_NOT_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_ROPE_NOT_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T, typename U>
__simd_vf__ inline void CastToOriginNotFullyVf(__ubuf__ U *srcAddr, __ubuf__ T *dstAddr, uint32_t dataLen,
                                               uint16_t loopTimes)
{
    MicroAPI::RegTensor<U> srcValue;
    MicroAPI::MaskReg preg;
    uint32_t sregMask = dataLen;
    for (uint16_t j = 0; j < loopTimes; j++) {
        preg = MicroAPI::UpdateMask<uint32_t>(sregMask);
        MicroAPI::LoadAlign<U, MicroAPI::LoadDist::DIST_NORM>(srcValue, srcAddr + VL_B32 * j);
        if constexpr (IsSameType<T, int16_t>::value || IsSameType<T, uint16_t>::value) {
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr + VL_B32 * j,
                                                                        (MicroAPI::RegTensor<T> &)srcValue, preg);
        } else {
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK4_B32>(dstAddr + VL_B32 * j,
                                                                         (MicroAPI::RegTensor<T> &)srcValue, preg);
        }
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
class ScatterPaKvCacheRopeNotFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheRopeNotFullyLoad(TPipe *pipe, const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();
    __aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y);

private:
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void KVReduceMean(int64_t iter, int64_t startIdx, int64_t endIdx, int64_t curBlockFactor);
    __aicore__ inline void CopyInKey(int64_t loopIdx, int64_t k, int64_t startIdx, int64_t keyOffset,
                                     int64_t handleNum);
    __aicore__ inline void CopyInValue(int64_t loopIdx, int64_t k, int64_t startIdx, int64_t valueOffset,
                                       int64_t handleNum);
    __aicore__ inline void CopyOutKey(int64_t loopIdx, int64_t startIdx, int64_t handleNum);
    __aicore__ inline void CopyOutValue(int64_t loopIdx, int64_t startIdx, int64_t handleNum);
    __aicore__ inline void ReduceMeanKey(int64_t kStartIdx, int64_t keyOffset, int64_t startIdx, int64_t endIdx);
    __aicore__ inline void ReduceMeanValue(int64_t kStartIdx, int64_t valueOffset, int64_t startIdx, int64_t endIdx);
    __aicore__ inline void HandleKeyCache(int64_t iter, int64_t offsetIndex);
    __aicore__ inline void HandleValueCache(int64_t iter, int64_t offsetIndex);
    __aicore__ inline void UpdateKeyCache(int64_t k, int64_t startIdx, int64_t keyOffset);
    template <typename U>
    __aicore__ inline void CastToOrigin(LocalTensor<T> &dstLocal, LocalTensor<U> &srcLocal, uint32_t dataLen);
    __aicore__ inline void UpdateValueCache(int64_t k, int64_t startIdx, int64_t valueOffset);
    template <typename U>
    __aicore__ inline void ReduceMeanKeyPart(int64_t loopIdx, int64_t kStartIdx, int64_t startIdx, int64_t endIdx,
                                             int64_t keyOffset, int64_t handleNum);
    template <typename U>
    __aicore__ inline void ReduceMeanValuePart(int64_t loopIdx, int64_t kStartIdx, int64_t startIdx, int64_t endIdx,
                                               int64_t valueOffset, int64_t handleNum);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputKeyQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputValueQueue_;
    TBuf<TPosition::VECCALC> divideBuf_;
    TBuf<TPosition::VECCALC> castBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    int64_t keyBlockOffset_{0};
    int64_t headSize_{0};
    int64_t numHead_{0};
    int64_t blockIdx_{0};
    int64_t valueBlockOffset_{0};
    int64_t kvBlockOffset_{0};
    int64_t count_{0};
    int64_t seqLen_{0};

    GlobalTensor<T> inputKeyGm_;
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T> outputKeyCacheGm_;
    GlobalTensor<T> outputValueCacheGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;
    GlobalTensor<IndexDtype> compressSeqOffsetGm_;
    GlobalTensor<IndexDtype> compressLensGm_;
    GlobalTensor<IndexDtype> seqLensGm_;
    const ScatterPaKvCacheTilingData *tilingData_;
    static constexpr bool isNeedCast_ =
        (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value || IsSameType<T, hifloat8_t>::value ||
         IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value);
    static constexpr bool isInteger8or16_ = (IsSameType<T, uint8_t>::value || IsSameType<T, int8_t>::value ||
                                             IsSameType<T, uint16_t>::value || IsSameType<T, int16_t>::value);
};

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::Init(
    GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in,
    GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, GM_ADDR key_cache_out,
    GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    seqLen_ = tilingData_->seqLen;
    numHead_ = tilingData_->numHead;
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
    headSize_ = maxKHandleNum > maxVHandleNum ? maxKHandleNum : maxVHandleNum;
    pipe_->InitBuffer(tmpBuf_, RoundUp(headSize_) * sizeof(float));
    pipe_->InitBuffer(divideBuf_, RoundUp(headSize_) * sizeof(float));
    pipe_->InitBuffer(castBuf_, RoundUp(headSize_) * sizeof(float));
    pipe_->InitBuffer(inputKeyQueue_, 1, RoundUp(maxKHandleNum) * sizeof(T));

    if constexpr (InOutMode == DUAL_IN_OUT) {
        inputValueGm_.SetGlobalBuffer((__gm__ T *)(value));
        outputValueCacheGm_.SetGlobalBuffer((__gm__ T *)(value_cache_out));
        pipe_->InitBuffer(inputValueQueue_, 1, RoundUp(maxVHandleNum) * sizeof(T));
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline int64_t ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::RoundUp(int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline uint32_t ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::CeilDiv(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : ((x + y - 1) / y);
}
template <typename T, typename IndexDtype, int64_t InOutMode>
template <typename U>
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::CastToOrigin(
    LocalTensor<T> &dstLocal, LocalTensor<U> &srcLocal, uint32_t dataLen)
{
    __ubuf__ U *srcAddr = (__ubuf__ U *)srcLocal.GetPhyAddr();
    __ubuf__ T *dstAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();

    uint16_t loopTimes = CeilDiv(dataLen, VL_B32);

    CastToOriginNotFullyVf<T, U>((__ubuf__ U *)srcAddr, (__ubuf__ T *)dstAddr, dataLen, loopTimes);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
template <typename U>
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::ReduceMeanKeyPart(
    int64_t loopIdx, int64_t kStartIdx, int64_t startIdx, int64_t endIdx, int64_t keyOffset, int64_t handleNum)
{
    LocalTensor<U> kTmpLocal = tmpBuf_.Get<U>();
    LocalTensor<U> kDivideLocal = divideBuf_.Get<U>();
    LocalTensor<U> kCastLocal = castBuf_.Get<U>();
    Duplicate(kTmpLocal, static_cast<U>(0), handleNum);
    Duplicate(kDivideLocal, static_cast<U>(endIdx - startIdx), handleNum);
    DataCopyExtParams keyCacheOutParams{1, static_cast<uint32_t>(handleNum * sizeof(T)), 0, 0, 0};
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    for (int64_t k = startIdx; k < endIdx; k++) {
        if (k >= seqLen_) {
            break;
        }
        event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        if constexpr (isNeedCast_ || isInteger8or16_) {
            CopyInKey(loopIdx, k, startIdx, keyOffset, handleNum);
            LocalTensor<T> inputKeyLocal = inputKeyQueue_.DeQue<T>();
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            Cast(kCastLocal, inputKeyLocal, RoundMode::CAST_NONE, handleNum);
            Add(kTmpLocal, kTmpLocal, kCastLocal, handleNum);
            inputKeyQueue_.FreeTensor(inputKeyLocal);
        } else {
            CopyInKey(loopIdx, k, startIdx, keyOffset, handleNum);
            LocalTensor<T> inputKeyLocal = inputKeyQueue_.DeQue<T>();
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            Add(kTmpLocal, kTmpLocal, inputKeyLocal, handleNum);
            inputKeyQueue_.FreeTensor(inputKeyLocal);
        }
    }
    Div(kTmpLocal, kTmpLocal, kDivideLocal, handleNum);
    if constexpr (isNeedCast_) {
        LocalTensor<T> inputKeyLocal = inputKeyQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, hifloat8_t>::value) {
            Cast(inputKeyLocal, kTmpLocal, RoundMode::CAST_ROUND, handleNum);
        } else {
            Cast(inputKeyLocal, kTmpLocal, RoundMode::CAST_RINT, handleNum);
        }
        event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);
        DataCopyPad(outputKeyCacheGm_[kStartIdx + loopIdx * tilingData_->kHandleNumPerLoop], inputKeyLocal,
                    keyCacheOutParams);
        inputKeyQueue_.FreeTensor(inputKeyLocal);
    } else if constexpr (isInteger8or16_) {
        LocalTensor<T> inputKeyLocal = inputKeyQueue_.AllocTensor<T>();
        CastToOrigin<U>(inputKeyLocal, kTmpLocal, handleNum);
        event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);
        DataCopyPad(outputKeyCacheGm_[kStartIdx + loopIdx * tilingData_->kHandleNumPerLoop], inputKeyLocal,
                    keyCacheOutParams);
        inputKeyQueue_.FreeTensor(inputKeyLocal);

    } else {
        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopyPad(outputKeyCacheGm_[kStartIdx + loopIdx * tilingData_->kHandleNumPerLoop], kTmpLocal,
                    keyCacheOutParams);
    }
    event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
template <typename U>
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::ReduceMeanValuePart(
    int64_t loopIdx, int64_t kStartIdx, int64_t startIdx, int64_t endIdx, int64_t valueOffset, int64_t handleNum)
{
    LocalTensor<U> vTmpLocal = tmpBuf_.Get<U>();
    LocalTensor<U> vDivideLocal = divideBuf_.Get<U>();
    LocalTensor<U> vCastLocal = castBuf_.Get<U>();
    Duplicate(vTmpLocal, static_cast<U>(0), handleNum);
    Duplicate(vDivideLocal, static_cast<U>(endIdx - startIdx), handleNum);
    DataCopyExtParams outValueCacheParams{1, static_cast<uint32_t>(handleNum * sizeof(T)), 0, 0, 0};
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    for (int64_t k = startIdx; k < endIdx; k++) {
        if (k >= seqLen_) {
            break;
        }
        event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        if constexpr (isNeedCast_ || isInteger8or16_) {
            CopyInValue(loopIdx, k, startIdx, valueOffset, handleNum);
            LocalTensor<T> inputValueLocal = inputValueQueue_.DeQue<T>();
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            Cast(vCastLocal, inputValueLocal, RoundMode::CAST_NONE, handleNum);
            Add(vTmpLocal, vTmpLocal, vCastLocal, handleNum);
            inputValueQueue_.FreeTensor(inputValueLocal);
        } else {
            CopyInValue(loopIdx, k, startIdx, valueOffset, handleNum);
            LocalTensor<T> inputValueLocal = inputValueQueue_.DeQue<T>();
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            Add(vTmpLocal, vTmpLocal, inputValueLocal, handleNum);
            inputValueQueue_.FreeTensor(inputValueLocal);
        }
    }
    Div(vTmpLocal, vTmpLocal, vDivideLocal, handleNum);
    if constexpr (isNeedCast_) {
        LocalTensor<T> inputValueLocal = inputValueQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, hifloat8_t>::value) {
            Cast(inputValueLocal, vTmpLocal, RoundMode::CAST_ROUND, handleNum);
        } else {
            Cast(inputValueLocal, vTmpLocal, RoundMode::CAST_RINT, handleNum);
        }
        event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);
        DataCopyPad(outputValueCacheGm_[kStartIdx + loopIdx * tilingData_->vHandleNumPerLoop], inputValueLocal,
                    outValueCacheParams);
        inputValueQueue_.FreeTensor(inputValueLocal);
    } else if constexpr (isInteger8or16_) {
        LocalTensor<T> inputValueLocal = inputValueQueue_.AllocTensor<T>();
        CastToOrigin<U>(inputValueLocal, vTmpLocal, handleNum);
        event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);
        DataCopyPad(outputValueCacheGm_[kStartIdx + loopIdx * tilingData_->kHandleNumPerLoop], inputValueLocal,
                    outValueCacheParams);
        inputValueQueue_.FreeTensor(inputValueLocal);
    } else {
        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopyPad(outputValueCacheGm_[kStartIdx + loopIdx * tilingData_->vHandleNumPerLoop], vTmpLocal,
                    outValueCacheParams);
    }
    event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::ReduceMeanKey(int64_t kStartIdx,
                                                                                                 int64_t keyOffset,
                                                                                                 int64_t startIdx,
                                                                                                 int64_t endIdx)
{
    if constexpr (isNeedCast_) {
        for (int64_t i = 0; i < tilingData_->kLoopNum; ++i) {
            ReduceMeanKeyPart<float>(i, kStartIdx, startIdx, endIdx, keyOffset, tilingData_->kHandleNumPerLoop);
        }
        ReduceMeanKeyPart<float>(tilingData_->kLoopNum, kStartIdx, startIdx, endIdx, keyOffset,
                                 tilingData_->kTailHandleNum);
    } else if constexpr (IsSameType<T, int8_t>::value || IsSameType<T, int16_t>::value) {
        for (int64_t i = 0; i < tilingData_->kLoopNum; ++i) {
            ReduceMeanKeyPart<int32_t>(i, kStartIdx, startIdx, endIdx, keyOffset, tilingData_->kHandleNumPerLoop);
        }
        ReduceMeanKeyPart<int32_t>(tilingData_->kLoopNum, kStartIdx, startIdx, endIdx, keyOffset,
                                   tilingData_->kTailHandleNum);
    } else if constexpr (IsSameType<T, uint8_t>::value || IsSameType<T, uint16_t>::value) {
        for (int64_t i = 0; i < tilingData_->kLoopNum; ++i) {
            ReduceMeanKeyPart<uint32_t>(i, kStartIdx, startIdx, endIdx, keyOffset, tilingData_->kHandleNumPerLoop);
        }
        ReduceMeanKeyPart<uint32_t>(tilingData_->kLoopNum, kStartIdx, startIdx, endIdx, keyOffset,
                                    tilingData_->kTailHandleNum);
    } else {
        for (int64_t i = 0; i < tilingData_->kLoopNum; ++i) {
            ReduceMeanKeyPart<T>(i, kStartIdx, startIdx, endIdx, keyOffset, tilingData_->kHandleNumPerLoop);
        }
        ReduceMeanKeyPart<T>(tilingData_->kLoopNum, kStartIdx, startIdx, endIdx, keyOffset,
                             tilingData_->kTailHandleNum);
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::ReduceMeanValue(int64_t kStartIdx,
                                                                                                   int64_t valueOffset,
                                                                                                   int64_t startIdx,
                                                                                                   int64_t endIdx)
{
    if constexpr (isNeedCast_) {
        for (int64_t i = 0; i < tilingData_->vLoopNum; ++i) {
            ReduceMeanValuePart<float>(i, kStartIdx, startIdx, endIdx, valueOffset, tilingData_->vHandleNumPerLoop);
        }
        ReduceMeanValuePart<float>(tilingData_->vLoopNum, kStartIdx, startIdx, endIdx, valueOffset,
                                   tilingData_->vTailHandleNum);
    } else if constexpr (IsSameType<T, int8_t>::value || IsSameType<T, int16_t>::value) {
        for (int64_t i = 0; i < tilingData_->vLoopNum; ++i) {
            ReduceMeanValuePart<int32_t>(i, kStartIdx, startIdx, endIdx, valueOffset, tilingData_->vHandleNumPerLoop);
        }
        ReduceMeanValuePart<int32_t>(tilingData_->vLoopNum, kStartIdx, startIdx, endIdx, valueOffset,
                                     tilingData_->vTailHandleNum);
    } else if constexpr (IsSameType<T, uint8_t>::value || IsSameType<T, uint16_t>::value) {
        for (int64_t i = 0; i < tilingData_->vLoopNum; ++i) {
            ReduceMeanValuePart<uint32_t>(i, kStartIdx, startIdx, endIdx, valueOffset, tilingData_->vHandleNumPerLoop);
        }
        ReduceMeanValuePart<uint32_t>(tilingData_->vLoopNum, kStartIdx, startIdx, endIdx, valueOffset,
                                      tilingData_->vTailHandleNum);
    } else {
        for (int64_t i = 0; i < tilingData_->vLoopNum; ++i) {
            ReduceMeanValuePart<T>(i, kStartIdx, startIdx, endIdx, valueOffset, tilingData_->vHandleNumPerLoop);
        }
        ReduceMeanValuePart<T>(tilingData_->vLoopNum, kStartIdx, startIdx, endIdx, valueOffset,
                               tilingData_->vTailHandleNum);
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::KVReduceMean(int64_t iter,
                                                                                                int64_t startIdx,
                                                                                                int64_t endIdx,
                                                                                                int64_t curBlockFactor)
{
    if (endIdx - startIdx <= 0) {
        return;
    }
    int64_t keyOffset = (kvBlockOffset_ + iter) / numHead_ * tilingData_->keyStride0 +
                        (kvBlockOffset_ + iter) % numHead_ * tilingData_->keyStride2;
    int64_t kStartIdx = slotMappingGm_.GetValue(iter) + count_;
    if (kStartIdx < 0 || kStartIdx >= tilingData_->numBlocks * tilingData_->blockSize) {
        return;
    }
    ReduceMeanKey(kStartIdx * tilingData_->kHeadSize, keyOffset, startIdx, endIdx);
    if constexpr (InOutMode == DUAL_IN_OUT) {
        int64_t valueOffset = (kvBlockOffset_ + iter) / numHead_ * tilingData_->valueStride0 +
                              (kvBlockOffset_ + iter) % numHead_ * tilingData_->valueStride2;
        ReduceMeanValue(kStartIdx * tilingData_->vHeadSize, valueOffset, startIdx, endIdx);
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::HandleKeyCache(int64_t iter,
                                                                                                  int64_t offsetIndex)
{
    int64_t startIdx = slotMappingGm_.GetValue(iter);
    int64_t keyOffset = (kvBlockOffset_ + iter) / numHead_ * tilingData_->keyStride0 +
                        (kvBlockOffset_ + iter) % numHead_ * tilingData_->keyStride2;
    for (int64_t k = 0; k < offsetIndex; k++) {
        if (k >= seqLen_) {
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
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::CopyInKey(int64_t loopIdx, int64_t k,
                                                                                             int64_t startIdx,
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
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::CopyOutKey(int64_t loopIdx,
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
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::CopyInValue(
    int64_t loopIdx, int64_t k, int64_t startIdx, int64_t valueOffset, int64_t handleNum)
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
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::CopyOutValue(int64_t loopIdx,
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
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::UpdateKeyCache(int64_t k,
                                                                                                  int64_t startIdx,
                                                                                                  int64_t keyOffset)
{
    for (int64_t i = 0; i < tilingData_->kLoopNum; i++) {
        CopyInKey(i, k, startIdx, keyOffset, tilingData_->kHandleNumPerLoop);
        CopyOutKey(i, startIdx, tilingData_->kHandleNumPerLoop);
    }

    // handle tail
    CopyInKey(tilingData_->kLoopNum, k, startIdx, keyOffset, tilingData_->kTailHandleNum);
    CopyOutKey(tilingData_->kLoopNum, startIdx, tilingData_->kTailHandleNum);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::HandleValueCache(int64_t iter,
                                                                                                    int64_t offsetIndex)
{
    int64_t startIdx = slotMappingGm_.GetValue(iter);
    int64_t valueOffset = (kvBlockOffset_ + iter) / numHead_ * tilingData_->valueStride0 +
                          (kvBlockOffset_ + iter) % numHead_ * tilingData_->valueStride2;
    for (int64_t k = 0; k < offsetIndex; k++) {
        if (k >= seqLen_) {
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
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::UpdateValueCache(int64_t k,
                                                                                                    int64_t startIdx,
                                                                                                    int64_t valueOffset)
{
    for (int64_t i = 0; i < tilingData_->vLoopNum; i++) {
        CopyInValue(i, k, startIdx, valueOffset, tilingData_->vHandleNumPerLoop);
        CopyOutValue(i, startIdx, tilingData_->vHandleNumPerLoop);
    }

    // handle tail
    CopyInValue(tilingData_->vLoopNum, k, startIdx, valueOffset, tilingData_->vTailHandleNum);
    CopyOutValue(tilingData_->vLoopNum, startIdx, tilingData_->vTailHandleNum);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeNotFullyLoad<T, IndexDtype, InOutMode>::Process()
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

        // step2: reduce mean
        wins = compressLensGm_.GetValue(i);
        int64_t endIdx = wins + offsetIndex >= seqLen_ ? seqLen_ : wins + offsetIndex;
        KVReduceMean(i, offsetIndex, endIdx, curBlockFactor);

        // step3: update KvCache
        count_++;
        offsetIndex = seqLensGm_.GetValue((kvBlockOffset_ + i) / numHead_) - wins - compressSeqOffsetGm_.GetValue(i);
        if (offsetIndex > 0) {
            HandleKeyCache(i, offsetIndex);
            if constexpr (InOutMode == DUAL_IN_OUT) {
                HandleValueCache(i, offsetIndex);
            }
        }
    }
}

} // namespace ScatterPaKvCache

#endif // SCATTER_PA_KV_CACHE_ROPE_NOT_FULLY_LOAD_H_
