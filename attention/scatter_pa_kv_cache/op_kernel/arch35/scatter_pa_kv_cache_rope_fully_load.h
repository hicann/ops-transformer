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

#ifndef SCATTER_PA_KV_CACHE_ROPE_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_ROPE_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
constexpr uint32_t VL_B32 = VECTOR_LENGTH / sizeof(uint32_t);

template <typename T, typename U>
__simd_vf__ inline void CastToOriginFullyVf(__ubuf__ U *srcAddr, __ubuf__ T *dstAddr, uint32_t dataLen,
                                            uint16_t loopNum)
{
    MicroAPI::RegTensor<U> valueSrc;
    MicroAPI::MaskReg curpreg;
    uint32_t sregMask = dataLen;
    for (uint16_t j = 0; j < loopNum; j++) {
        curpreg = MicroAPI::UpdateMask<uint32_t>(sregMask);
        MicroAPI::LoadAlign<U, MicroAPI::LoadDist::DIST_NORM>(valueSrc, srcAddr + VL_B32 * j);
        if constexpr (IsSameType<T, int16_t>::value || IsSameType<T, uint16_t>::value) {
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr + VL_B32 * j,
                                                                        (MicroAPI::RegTensor<T> &)valueSrc, curpreg);
        } else {
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK4_B32>(dstAddr + VL_B32 * j,
                                                                         (MicroAPI::RegTensor<T> &)valueSrc, curpreg);
        }
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
class ScatterPaKvCacheRopeFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheRopeFullyLoad(TPipe *pipe, const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();
    __aicore__ inline uint32_t CeilDivFully(uint32_t x, uint32_t y);

private:
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void KVReduceMean(int64_t iter, int64_t startIdx, int64_t endIdx, int64_t curBlockFactor);
    __aicore__ inline void CopyInKey(int64_t iter, int64_t offsetIndex, int64_t startIdx, int64_t curBlockFactor);
    __aicore__ inline void CopyInValue(int64_t iter, int64_t offsetIndex, int64_t startIdx, int64_t curBlockFactor);
    __aicore__ inline void CopyOutKey(int64_t iter, int64_t offsetIndex, int64_t startOffset, int64_t curBlockFactor);
    __aicore__ inline void CopyOutValue(int64_t iter, int64_t offsetIndex, int64_t startOffset, int64_t curBlockFactor);
    __aicore__ inline void CopyToUb(LocalTensor<IndexDtype> ubLocal, GlobalTensor<IndexDtype> gmGlobal,
                                    int64_t curBlockFactor);
    template <typename U>
    __aicore__ inline void ReduceMeanKey(int64_t iter, int64_t startIdx, int64_t endIdx, int64_t curBlockFactor);
    template <typename U>
    __aicore__ inline void ReduceMeanValue(int64_t iter, int64_t startIdx, int64_t endIdx, int64_t curBlockFactor);
    __aicore__ inline void CalcStartIdx(LocalTensor<IndexDtype> slotMappingLocal, int64_t curBlockFactor);
    template <typename U>
    __aicore__ inline void CastToOrigin(LocalTensor<T> &dstLocal, LocalTensor<U> &srcLocal, uint32_t dataLen);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputKeyQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputValueQueue_;
    TBuf<TPosition::VECCALC> slotMappingBuf_;
    TBuf<TPosition::VECCALC> compressSeqOffsetBuf_;
    TBuf<TPosition::VECCALC> seqLensBuf_;
    TBuf<TPosition::VECCALC> compressLensBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> divideBuf_;
    TBuf<TPosition::VECCALC> castBuf_;

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
    int64_t numHead_{0};
    int64_t count_{0};
    int64_t headSize_{0};
    int64_t seqLen_{0};
    static constexpr bool isNeedCast_ =
        (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value || IsSameType<T, hifloat8_t>::value ||
         IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value);
    static constexpr bool isIntger8or16_ = (IsSameType<T, int8_t>::value || IsSameType<T, int16_t>::value ||
                                            IsSameType<T, uint8_t>::value || IsSameType<T, uint16_t>::value);
};

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::Init(
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

    pipe_->InitBuffer(slotMappingBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(compressSeqOffsetBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(seqLensBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(compressLensBuf_, RoundUp(tilingData_->blockFactor) * sizeof(IndexDtype));
    headSize_ = tilingData_->kHeadSize > tilingData_->vHeadSize ? tilingData_->kHeadSize : tilingData_->vHeadSize;
    pipe_->InitBuffer(tmpBuf_, RoundUp(headSize_) * sizeof(float));
    pipe_->InitBuffer(divideBuf_, RoundUp(headSize_) * sizeof(float));
    pipe_->InitBuffer(castBuf_, RoundUp(headSize_) * sizeof(float));
    pipe_->InitBuffer(inputKeyQueue_, 1, (seqLen_ * RoundUp(tilingData_->kHeadSize)) * sizeof(T));

    if constexpr (InOutMode == DUAL_IN_OUT) {
        inputValueGm_.SetGlobalBuffer((__gm__ T *)(value));
        outputValueCacheGm_.SetGlobalBuffer((__gm__ T *)(value_cache_out));
        pipe_->InitBuffer(inputValueQueue_, 1, (seqLen_ * RoundUp(tilingData_->vHeadSize)) * sizeof(T));
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline int64_t ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::RoundUp(int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::CalcStartIdx(
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
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::CopyToUb(
    LocalTensor<IndexDtype> ubLocal, GlobalTensor<IndexDtype> gmGlobal, int64_t curBlockFactor)
{
    DataCopyExtParams params = {static_cast<uint16_t>(1), static_cast<uint32_t>(curBlockFactor * sizeof(IndexDtype)),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<IndexDtype> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                    static_cast<IndexDtype>(0)};
    DataCopyPad(ubLocal, gmGlobal, params, padParamIdx);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
template <typename U>
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::ReduceMeanKey(int64_t iter,
                                                                                              int64_t startIdx,
                                                                                              int64_t endIdx,
                                                                                              int64_t curBlockFactor)
{
    LocalTensor<U> tmpLocal = tmpBuf_.Get<U>();
    LocalTensor<U> divideLocal = divideBuf_.Get<U>();
    LocalTensor<U> castLocal = castBuf_.Get<U>();
    Duplicate(tmpLocal, static_cast<U>(0), headSize_);
    Duplicate(divideLocal, static_cast<U>(endIdx - startIdx), headSize_);
    CopyInKey(iter, endIdx, startIdx, curBlockFactor);
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    LocalTensor<T> inputKeyLocal = inputKeyQueue_.DeQue<T>();
    for (int64_t i = startIdx; i < endIdx; ++i) {
        if constexpr (isNeedCast_ || isIntger8or16_) {
            Cast(castLocal, inputKeyLocal[(i - startIdx) * RoundUp(tilingData_->kHeadSize)], RoundMode::CAST_NONE,
                 tilingData_->kHeadSize);
            Add(tmpLocal, tmpLocal, castLocal, tilingData_->kHeadSize);
        } else {
            Add(tmpLocal, tmpLocal, inputKeyLocal[(i - startIdx) * RoundUp(tilingData_->kHeadSize)],
                tilingData_->kHeadSize);
        }
    }
    Div(tmpLocal, tmpLocal, divideLocal, tilingData_->kHeadSize);
    DataCopyExtParams outKeyCacheParams{1, static_cast<uint32_t>(tilingData_->kHeadSize * sizeof(T)), 0, 0, 0};
    LocalTensor<IndexDtype> slotMappingLocal = slotMappingBuf_.Get<IndexDtype>();
    int64_t kStartIdx = slotMappingLocal.GetValue(iter) + count_;
    if (kStartIdx >= 0 && kStartIdx < tilingData_->numBlocks * tilingData_->blockSize) {
        if constexpr (isNeedCast_) {
            if constexpr (IsSameType<T, hifloat8_t>::value) {
                Cast(inputKeyLocal, tmpLocal, RoundMode::CAST_ROUND, tilingData_->kHeadSize);
            } else {
                Cast(inputKeyLocal, tmpLocal, RoundMode::CAST_RINT, tilingData_->kHeadSize);
            }
            event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);
            DataCopyPad(outputKeyCacheGm_[kStartIdx * tilingData_->kHeadSize], inputKeyLocal, outKeyCacheParams);
        } else if constexpr (isIntger8or16_) {
            CastToOrigin<U>(inputKeyLocal, tmpLocal, tilingData_->kHeadSize);
            event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);
            DataCopyPad(outputKeyCacheGm_[kStartIdx * tilingData_->kHeadSize], inputKeyLocal, outKeyCacheParams);
        } else {
            event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
            DataCopyPad(outputKeyCacheGm_[kStartIdx * tilingData_->kHeadSize], tmpLocal, outKeyCacheParams);
        }
    }
    event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
    inputKeyQueue_.FreeTensor(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline uint32_t ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::CeilDivFully(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : ((x + y - 1) / y);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
template <typename U>
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::CastToOrigin(LocalTensor<T> &dstLocal,
                                                                                             LocalTensor<U> &srcLocal,
                                                                                             uint32_t dataLen)
{
    __ubuf__ U *srcPhyAddr = (__ubuf__ U *)srcLocal.GetPhyAddr();
    __ubuf__ T *dstPhyAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();

    uint16_t loopNum = CeilDivFully(dataLen, VL_B32);

    CastToOriginFullyVf<T, U>((__ubuf__ U *)srcPhyAddr, (__ubuf__ T *)dstPhyAddr, dataLen, loopNum);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
template <typename U>
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::ReduceMeanValue(int64_t iter,
                                                                                                int64_t startIdx,
                                                                                                int64_t endIdx,
                                                                                                int64_t curBlockFactor)
{
    LocalTensor<U> tmpLocal = tmpBuf_.Get<U>();
    LocalTensor<U> divideLocal = divideBuf_.Get<U>();
    LocalTensor<U> castLocal = castBuf_.Get<U>();
    Duplicate(tmpLocal, static_cast<U>(0), headSize_);
    Duplicate(divideLocal, static_cast<U>(endIdx - startIdx), headSize_);
    CopyInValue(iter, endIdx, startIdx, curBlockFactor);
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    LocalTensor<T> inputValueLocal = inputValueQueue_.DeQue<T>();
    for (int64_t i = startIdx; i < endIdx; ++i) {
        if constexpr (isNeedCast_ || isIntger8or16_) {
            Cast(castLocal, inputValueLocal[(i - startIdx) * RoundUp(tilingData_->vHeadSize)], RoundMode::CAST_NONE,
                 tilingData_->vHeadSize);
            Add(tmpLocal, tmpLocal, castLocal, tilingData_->vHeadSize);
        } else {
            Add(tmpLocal, tmpLocal, inputValueLocal[(i - startIdx) * RoundUp(tilingData_->vHeadSize)],
                tilingData_->vHeadSize);
        }
    }
    Div(tmpLocal, tmpLocal, divideLocal, tilingData_->vHeadSize);
    DataCopyExtParams outValueCacheParams{1, static_cast<uint32_t>(tilingData_->vHeadSize * sizeof(T)), 0, 0, 0};
    LocalTensor<IndexDtype> slotMappingLocal = slotMappingBuf_.Get<IndexDtype>();
    int64_t vStartIdx = slotMappingLocal.GetValue(iter) + count_;
    if (vStartIdx >= 0 && vStartIdx < tilingData_->numBlocks * tilingData_->blockSize) {
        if constexpr (isNeedCast_) {
            if constexpr (IsSameType<T, hifloat8_t>::value) {
                Cast(inputValueLocal, tmpLocal, RoundMode::CAST_ROUND, tilingData_->vHeadSize);
            } else {
                Cast(inputValueLocal, tmpLocal, RoundMode::CAST_RINT, tilingData_->vHeadSize);
            }
            event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);
            DataCopyPad(outputValueCacheGm_[vStartIdx * tilingData_->vHeadSize], inputValueLocal, outValueCacheParams);
        } else if constexpr (isIntger8or16_) {
            CastToOrigin<U>(inputValueLocal, tmpLocal, tilingData_->vHeadSize);
            event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);
            DataCopyPad(outputValueCacheGm_[vStartIdx * tilingData_->vHeadSize], inputValueLocal, outValueCacheParams);
        } else {
            event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
            DataCopyPad(outputValueCacheGm_[vStartIdx * tilingData_->vHeadSize], tmpLocal, outValueCacheParams);
        }
    }
    event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
    inputValueQueue_.FreeTensor(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::KVReduceMean(int64_t iter,
                                                                                             int64_t startIdx,
                                                                                             int64_t endIdx,
                                                                                             int64_t curBlockFactor)
{
    if (endIdx - startIdx <= 0) {
        return;
    }

    if constexpr (isNeedCast_) {
        ReduceMeanKey<float>(iter, startIdx, endIdx, curBlockFactor);
        if constexpr (InOutMode == DUAL_IN_OUT) {
            ReduceMeanValue<float>(iter, startIdx, endIdx, curBlockFactor);
        }
    } else if constexpr (IsSameType<T, int8_t>::value || IsSameType<T, int16_t>::value) {
        ReduceMeanKey<int32_t>(iter, startIdx, endIdx, curBlockFactor);
        if constexpr (InOutMode == DUAL_IN_OUT) {
            ReduceMeanValue<int32_t>(iter, startIdx, endIdx, curBlockFactor);
        }
    } else if constexpr (IsSameType<T, uint8_t>::value || IsSameType<T, uint16_t>::value) {
        ReduceMeanKey<uint32_t>(iter, startIdx, endIdx, curBlockFactor);
        if constexpr (InOutMode == DUAL_IN_OUT) {
            ReduceMeanValue<uint32_t>(iter, startIdx, endIdx, curBlockFactor);
        }
    } else {
        ReduceMeanKey<T>(iter, startIdx, endIdx, curBlockFactor);
        if constexpr (InOutMode == DUAL_IN_OUT) {
            ReduceMeanValue<T>(iter, startIdx, endIdx, curBlockFactor);
        }
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::CopyInKey(int64_t iter,
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

    int64_t offset = (kvBlockOffset_ + iter) / numHead_ * tilingData_->keyStride0 +
                     (kvBlockOffset_ + iter) % numHead_ * tilingData_->keyStride2;
    for (int64_t k = startIdx; k < offsetIndex; ++k) {
        if (k >= seqLen_) {
            break;
        }
        DataCopyPad(inputKeyLocal[(k - startIdx) * RoundUp(tilingData_->kHeadSize)],
                    inputKeyGm_[offset + k * tilingData_->keyStride1], inKeyParams, padParams);
    }
    inputKeyQueue_.EnQue(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::CopyInValue(int64_t iter,
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
    int64_t offset = (kvBlockOffset_ + iter) / numHead_ * tilingData_->valueStride0 +
                     (kvBlockOffset_ + iter) % numHead_ * tilingData_->valueStride2;
    for (int64_t k = startIdx; k < offsetIndex; ++k) {
        if (k >= seqLen_) {
            break;
        }
        DataCopyPad(inputValueLocal[(k - startIdx) * RoundUp(tilingData_->vHeadSize)],
                    inputValueGm_[offset + k * tilingData_->valueStride1], inValueParams, padParams);
    }
    inputValueQueue_.EnQue(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::CopyOutKey(int64_t iter,
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
        if (k >= seqLen_) {
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
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::CopyOutValue(int64_t iter,
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
        if (k >= seqLen_) {
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
__aicore__ inline void ScatterPaKvCacheRopeFullyLoad<T, IndexDtype, InOutMode>::Process()
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
    CalcStartIdx(slotMappingLocal, curBlockFactor);
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
        // step2: reduce mean
        wins = compressLensLocal.GetValue(i);
        int64_t endIdx = wins + offsetIndex >= seqLen_ ? seqLen_ : wins + offsetIndex;
        KVReduceMean(i, offsetIndex, endIdx, curBlockFactor);

        // step3: update KvCache
        count_++;
        offsetIndex = seqLensGm_.GetValue((kvBlockOffset_ + i) / numHead_) - wins - compressSeqOffsetLocal.GetValue(i);
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
#endif // SCATTER_PA_KV_CACHE_ROPE_FULLY_LOAD_H_
