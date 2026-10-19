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

constexpr Reg::CastTrait castTraitB322B64 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};

template <typename indexDtype, typename OffsetDtype>
__simd_vf__ inline void ComputeOffsetVF(__ubuf__ OffsetDtype* offsetAddr, __ubuf__ indexDtype* slotLocalPtr,
                                        uint32_t dataLen, uint16_t loopCnt, uint32_t vfLen, OffsetDtype blockSize,
                                        OffsetDtype maxTokenValue, OffsetDtype cacheStride0, OffsetDtype cacheStride1)
{
    using OffsetRegType =
        typename std::conditional<IsSameType<OffsetDtype, int64_t>::value,
                                  typename AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo>,
                                  typename AscendC::Reg::RegTensor<int32_t>>::type;
    OffsetRegType inReg;
    OffsetRegType v1;
    OffsetRegType outReg;
    OffsetRegType divReg;
    OffsetRegType mulReg;
    OffsetRegType subReg;
    OffsetRegType tempReg;
    OffsetRegType invalidOffsetReg;
    Reg::MaskReg cmpMask;
    Reg::RegTensor<int32_t> castReg;
    Reg::MaskReg pregLoop;
    uint32_t regLen = dataLen;
    for (uint16_t i = 0; i < static_cast<uint16_t>(loopCnt); i++) {
        if constexpr (IsSameType<OffsetDtype, int64_t>::value) {
            pregLoop = Reg::UpdateMask<int64_t, Reg::RegTraitNumTwo>(regLen);
        } else {
            pregLoop = Reg::UpdateMask<int32_t>(regLen);
        }

        Reg::Duplicate(v1, blockSize, pregLoop);
        Reg::Duplicate(invalidOffsetReg, static_cast<OffsetDtype>(-1), pregLoop);
        if constexpr (IsSameType<indexDtype, int32_t>::value && IsSameType<OffsetDtype, int64_t>::value) {
            Reg::LoadAlign(castReg, slotLocalPtr + i * vfLen);
            Reg::Cast<int64_t, int32_t, castTraitB322B64>(inReg, castReg, pregLoop);
        } else {
            Reg::LoadAlign(inReg, slotLocalPtr + i * vfLen);
        }
        Reg::Compares<OffsetDtype, CMPMODE::LT>(cmpMask, inReg, maxTokenValue, pregLoop);
        Reg::Select(inReg, inReg, invalidOffsetReg, cmpMask);
        Reg::Div(divReg, inReg, v1, pregLoop);
        Reg::Mul(mulReg, divReg, v1, pregLoop);
        Reg::Sub(subReg, inReg, mulReg, pregLoop);
        Reg::Muls(outReg, divReg, cacheStride0, pregLoop);
        Reg::Muls(tempReg, subReg, cacheStride1, pregLoop);
        Reg::Add(outReg, outReg, tempReg, pregLoop);
        Reg::StoreAlign(offsetAddr + i * vfLen, outReg, pregLoop);
    }
}

template <typename T, typename IndexDtype, typename OffsetDtype, int64_t InOutMode, bool NCT = false>
class ScatterPaKvCacheNormalFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheNormalFullyLoad(TPipe* pipe, const ScatterPaKvCacheTilingData* __restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset,
                                GM_ADDR seq_lens, GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(LocalTensor<T>& inputLocal, GlobalTensor<T>& inputGm, int64_t blockOffset,
                                  int64_t handleNum, int64_t curFactor, int64_t nctStride, int64_t nctOffset);
    __aicore__ inline void CopyOut(GlobalTensor<T>& outputGm, LocalTensor<T>& inputLocal,
                                   LocalTensor<OffsetDtype>& offsetLocal, LocalTensor<IndexDtype>& slotMapLocal,
                                   int64_t stride0, int64_t stride1, int64_t handleNum, int64_t curFactor);
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void CopyInSlotMap(LocalTensor<IndexDtype> slotMappingLocal, int64_t curBlockFactor);
    __aicore__ inline void ComputeOutOffset(LocalTensor<OffsetDtype> offsetLocal,
                                            LocalTensor<IndexDtype> slotMappingLocal, int64_t curFactor,
                                            int64_t stride0, int64_t stride1);

private:
    TPipe* pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputKeyQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputValueQueue_;

    GlobalTensor<T> inputKeyGm_;
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T> outputKeyCacheGm_;
    GlobalTensor<T> outputValueCacheGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;

    const ScatterPaKvCacheTilingData* tilingData_;

    int64_t blockIdx_{0};

    TQue<TPosition::VECIN, 1> kSlotMappingQue_;
    TQue<TPosition::VECIN, 1> vSlotMappingQue_;
    TQue<TPosition::VECOUT, 1> kOffsetQue_;
    TQue<TPosition::VECOUT, 1> vOffsetQue_;
    int64_t maxTokens_ = 0;
    int64_t blockFactorOffset_ = 0;
};

template <typename T, typename IndexDtype, typename OffsetDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, OffsetDtype, InOutMode, NCT>::Init(
    GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in,
    GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, GM_ADDR key_cache_out,
    GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    blockFactorOffset_ = blockIdx_ * tilingData_->blockFactor;
    inputKeyGm_.SetGlobalBuffer((__gm__ T*)(key));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype*)(slot_mapping));
    outputKeyCacheGm_.SetGlobalBuffer((__gm__ T*)(key_cache_out));
    pipe_->InitBuffer(kSlotMappingQue_, 1, RoundUp(tilingData_->ubBlockFactor) * sizeof(IndexDtype));
    pipe_->InitBuffer(inputKeyQueue_, 1,
                      tilingData_->ubBlockFactor * RoundUp(tilingData_->kHandleNumPerCore) * sizeof(T));
    pipe_->InitBuffer(kOffsetQue_, 1, RoundUp(tilingData_->ubBlockFactor) * sizeof(OffsetDtype));
    if constexpr (InOutMode == DUAL_IN_OUT) {
        inputValueGm_.SetGlobalBuffer((__gm__ T*)(value));
        outputValueCacheGm_.SetGlobalBuffer((__gm__ T*)(value_cache_out));
        pipe_->InitBuffer(vSlotMappingQue_, 1, RoundUp(tilingData_->ubBlockFactor) * sizeof(IndexDtype));
        pipe_->InitBuffer(inputValueQueue_, 1,
                          tilingData_->ubBlockFactor * RoundUp(tilingData_->vHandleNumPerCore) * sizeof(T));
        pipe_->InitBuffer(vOffsetQue_, 1, RoundUp(tilingData_->ubBlockFactor) * sizeof(OffsetDtype));
    }
    maxTokens_ = tilingData_->numBlocks * tilingData_->blockSize;
}

template <typename T, typename IndexDtype, typename OffsetDtype, int64_t InOutMode, bool NCT>
__aicore__ inline int64_t ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, OffsetDtype, InOutMode, NCT>::RoundUp(
    int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, typename OffsetDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, OffsetDtype, InOutMode, NCT>::CopyInSlotMap(
    LocalTensor<IndexDtype> slotMappingLocal, int64_t curBlockFactor)
{
    DataCopyExtParams slotMappingParams{1, static_cast<uint32_t>(curBlockFactor * sizeof(IndexDtype)), 0, 0, 0};
    DataCopyPadExtParams<IndexDtype> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                    static_cast<IndexDtype>(0)};
    DataCopyPad(slotMappingLocal, slotMappingGm_[blockFactorOffset_], slotMappingParams, padParamIdx);
}

template <typename T, typename IndexDtype, typename OffsetDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, OffsetDtype, InOutMode, NCT>::CopyIn(
    LocalTensor<T>& inputLocal, GlobalTensor<T>& inputGm, int64_t blockOffset, int64_t handleNum, int64_t curFactor,
    int64_t nctStride, int64_t nctOffset)
{
    DataCopyExtParams params;
    params.blockCount = curFactor;
    params.blockLen = handleNum * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = 0;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    if constexpr (NCT) {
        int64_t offsetIn = blockOffset * nctStride + nctOffset;
        params.blockCount = 1;
        for (int64_t i = 0; i < curFactor; i++) {
            if (offsetIn < 0) {
                continue;
            }
            DataCopyPad(inputLocal[i * RoundUp(handleNum)], inputGm[offsetIn], params, padParams);
            offsetIn += nctStride;
        }
    } else {
        int64_t offsetIn = blockOffset * handleNum;
        DataCopyPad(inputLocal, inputGm[offsetIn], params, padParams);
    }
}

template <typename T, typename IndexDtype, typename OffsetDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, OffsetDtype, InOutMode, NCT>::ComputeOutOffset(
    LocalTensor<OffsetDtype> offsetLocal, LocalTensor<IndexDtype> slotMappingLocal, int64_t curFactor, int64_t stride0,
    int64_t stride1)
{
    __local_mem__ IndexDtype* slotLocalPtr = (__local_mem__ IndexDtype*)slotMappingLocal.GetPhyAddr();
    __local_mem__ OffsetDtype* offsetAddr = (__local_mem__ OffsetDtype*)offsetLocal.GetPhyAddr();
    OffsetDtype cacheStride0 = static_cast<OffsetDtype>(stride0 * sizeof(T));
    OffsetDtype cacheStride1 = static_cast<OffsetDtype>(stride1 * sizeof(T));
    OffsetDtype blockSize = static_cast<OffsetDtype>(tilingData_->blockSize);
    OffsetDtype maxTokenValue = static_cast<OffsetDtype>(maxTokens_);

    uint32_t dataLen = curFactor;
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint32_t loopCnt = Ops::Base::CeilDiv(dataLen, vfLen);
    ComputeOffsetVF<IndexDtype, OffsetDtype>((__ubuf__ OffsetDtype*)offsetAddr, (__ubuf__ IndexDtype*)slotLocalPtr,
                                             dataLen, loopCnt, vfLen, blockSize, maxTokenValue, cacheStride0,
                                             cacheStride1);
}

template <typename T, typename IndexDtype, typename OffsetDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, OffsetDtype, InOutMode, NCT>::CopyOut(
    GlobalTensor<T>& outputGm, LocalTensor<T>& inputLocal, LocalTensor<OffsetDtype>& offsetLocal,
    LocalTensor<IndexDtype>& slotMapLocal, int64_t stride0, int64_t stride1, int64_t handleNum, int64_t curFactor)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
    ScatterParams scatterDataCopyParams;
    scatterDataCopyParams.offsetNum = curFactor;
    scatterDataCopyParams.burstLen = handleNum * sizeof(T);
    scatterDataCopyParams.srcStride = RoundUp(handleNum) * sizeof(T);
    ScatterDataCopy(outputGm, inputLocal, offsetLocal, scatterDataCopyParams);
#else
    DataCopyExtParams outKeyCacheParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(handleNum * sizeof(T)),
                                           static_cast<uint32_t>(0), static_cast<uint32_t>(0),
                                           static_cast<uint32_t>(0)};
    for (int64_t i = 0; i < curFactor; i++) {
        int64_t startIdx = slotMapLocal.GetValue(i);
        if (startIdx < 0 || startIdx >= maxTokens_) {
            continue;
        }
        int64_t blockIdx = startIdx / tilingData_->blockSize;
        int64_t blockOffset = startIdx % tilingData_->blockSize;
        startIdx = blockIdx * stride0 + blockOffset * stride1;
        DataCopyPad(outputGm[startIdx], inputLocal[i * RoundUp(handleNum)], outKeyCacheParams);
    }
#endif
}

template <typename T, typename IndexDtype, typename OffsetDtype, int64_t InOutMode, bool NCT>
__aicore__ inline void ScatterPaKvCacheNormalFullyLoad<T, IndexDtype, OffsetDtype, InOutMode, NCT>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;
    int64_t blockLoop = Ops::Base::CeilDiv(curBlockFactor, tilingData_->ubBlockFactor);
    int64_t tailUbBlockFactor = curBlockFactor - (blockLoop - 1) * tilingData_->ubBlockFactor;

    for (int32_t i = 0; i < blockLoop; i++) {
        int64_t curUbFactor = i == (blockLoop - 1) ? tailUbBlockFactor : tilingData_->ubBlockFactor;
        LocalTensor<IndexDtype> kSlotMappingLocal = kSlotMappingQue_.AllocTensor<IndexDtype>();
        CopyInSlotMap(kSlotMappingLocal, curUbFactor);
        LocalTensor<T> inputKeyLocal = inputKeyQueue_.AllocTensor<T>();
        CopyIn(inputKeyLocal, inputKeyGm_, blockFactorOffset_, tilingData_->kHandleNumPerCore, curUbFactor,
               tilingData_->kStride, tilingData_->kOffset);
        kSlotMappingQue_.EnQue(kSlotMappingLocal);
        kSlotMappingLocal = kSlotMappingQue_.DeQue<IndexDtype>();
        LocalTensor<OffsetDtype> kOffsetLocal;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
        kOffsetLocal = kOffsetQue_.AllocTensor<OffsetDtype>();
        ComputeOutOffset(kOffsetLocal, kSlotMappingLocal, curUbFactor, tilingData_->keyCacheStride0,
                         tilingData_->keyCacheStride1);
        kOffsetQue_.EnQue(kOffsetLocal);
        kOffsetLocal = kOffsetQue_.DeQue<OffsetDtype>();
#endif
        inputKeyQueue_.EnQue(inputKeyLocal);
        inputKeyLocal = inputKeyQueue_.DeQue<T>();
        CopyOut(outputKeyCacheGm_, inputKeyLocal, kOffsetLocal, kSlotMappingLocal, tilingData_->keyCacheStride0,
                tilingData_->keyCacheStride1, tilingData_->kHandleNumPerCore, curUbFactor);
        inputKeyQueue_.FreeTensor(inputKeyLocal);
        kSlotMappingQue_.FreeTensor(kSlotMappingLocal);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
        kOffsetQue_.FreeTensor(kOffsetLocal);
#endif
        if constexpr (InOutMode == DUAL_IN_OUT) {
            LocalTensor<IndexDtype> vSlotMappingLocal = vSlotMappingQue_.AllocTensor<IndexDtype>();
            CopyInSlotMap(vSlotMappingLocal, curUbFactor);
            LocalTensor<T> inputValueLocal = inputValueQueue_.AllocTensor<T>();
            CopyIn(inputValueLocal, inputValueGm_, blockFactorOffset_, tilingData_->vHandleNumPerCore, curUbFactor,
                   tilingData_->vStride, tilingData_->vOffset);
            vSlotMappingQue_.EnQue(vSlotMappingLocal);
            vSlotMappingLocal = vSlotMappingQue_.DeQue<IndexDtype>();
            LocalTensor<OffsetDtype> vOffsetLocal;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
            vOffsetLocal = vOffsetQue_.AllocTensor<OffsetDtype>();
            ComputeOutOffset(vOffsetLocal, vSlotMappingLocal, curUbFactor, tilingData_->valueCacheStride0,
                             tilingData_->valueCacheStride1);
            vOffsetQue_.EnQue(vOffsetLocal);
            vOffsetLocal = vOffsetQue_.DeQue<OffsetDtype>();
#endif
            inputValueQueue_.EnQue(inputValueLocal);
            inputValueLocal = inputValueQueue_.DeQue<T>();
            CopyOut(outputValueCacheGm_, inputValueLocal, vOffsetLocal, vSlotMappingLocal,
                    tilingData_->valueCacheStride0, tilingData_->valueCacheStride1, tilingData_->vHandleNumPerCore,
                    curUbFactor);
            inputValueQueue_.FreeTensor(inputValueLocal);
            vSlotMappingQue_.FreeTensor(vSlotMappingLocal);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
            vOffsetQue_.FreeTensor(vOffsetLocal);
#endif
        }
        blockFactorOffset_ += curUbFactor;
    }
}

} // namespace ScatterPaKvCache

#endif // SCATTER_PA_KV_CACHE_NORMAL_FULLY_LOAD_H_
