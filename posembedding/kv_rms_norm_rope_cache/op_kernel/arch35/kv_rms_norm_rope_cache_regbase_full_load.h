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
 * \file kv_rms_norm_rope_cache_regbase_full_load.h
 * \brief
 */

#ifndef KV_RMS_NORM_ROPE_CACHE_REGBASE_FULL_LOAD_H_
#define KV_RMS_NORM_ROPE_CACHE_REGBASE_FULL_LOAD_H_
#include "kv_rms_norm_rope_cache_regbase_base.h"

namespace KvRmsNormRopeCache {

using namespace AscendC;
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
class KvRmsNormRopeCacheRegbaseFullLoad : public KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE> {
public:
    __aicore__ inline KvRmsNormRopeCacheRegbaseFullLoad(TPipe *pipe,
                                                        const KvRmsNormRopeCacheRegbaseFullLoadTilingData *tiling)
    {
        tilingData_ = tiling;
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache,
                                GM_ADDR v_cache, GM_ADDR k_scale, GM_ADDR v_scale, GM_ADDR k_offset, GM_ADDR v_offset,
                                GM_ADDR k_out, GM_ADDR v_out)
    {
        // 获取分块操作中的单个块的大小。判断是否是最后一块，是最后一块，则等于剩余元素的数量，否则等于固定的单核处理的行数
        int64_t currentBlockFactor = tilingData_->blockFactor;
        if (this->blockIdx == (this->useCoreNum - 1)) {
            currentBlockFactor =
                tilingData_->batchSize * tilingData_->seqLength - (this->useCoreNum - 1) * tilingData_->blockFactor;
        }
        this->reciprocal = tilingData_->reciprocal;
        this->epsilon = tilingData_->epsilon;
        this->dv = tilingData_->dv;
        this->dvAlign = tilingData_->dvAlign;
        this->dvB8Align = tilingData_->dvB8Align;
        this->dkAlign = tilingData_->dkAlign;
        this->dk = tilingData_->dk;
        this->ubFactor = tilingData_->ubFactor;
        // NZ参数
        dk0 = BLOCK_SIZE / sizeof(T_K_CACHE);
        dk1 = this->dk / dk0;
        dv0 = BLOCK_SIZE / sizeof(T_V_CACHE);
        dv1 = this->dv / dv0;

        copyOutKParams.blockCount = 1;
        copyOutKParams.blockLen = tilingData_->dk * sizeof(T_K_CACHE);
        copyOutKParams.srcStride = 0;
        copyOutKParams.dstStride = 0;

        copyOutKParamsNz.blockCount = dk1;
        copyOutKParamsNz.blockLen = dk0 * sizeof(T_K_CACHE);
        copyOutKParamsNz.srcStride = 0;
        copyOutKParamsNz.dstStride = (tilingData_->blockSize - 1) * dk0 * sizeof(T_K_CACHE);

        copyOutVParams.blockCount = 1;
        copyOutVParams.blockLen = tilingData_->dv * sizeof(T_V_CACHE);
        copyOutVParams.srcStride = 0;
        copyOutVParams.dstStride = 0;

        copyOutVParamsNz.blockCount = dv1;
        copyOutVParamsNz.blockLen = dv0 * sizeof(T_V_CACHE);
        copyOutVParamsNz.srcStride = 0;
        copyOutVParamsNz.dstStride = (tilingData_->blockSize - 1) * dv0 * sizeof(T_V_CACHE);

        // ubLoop: 表示需要多少个子块来覆盖singleA大小的数据
        this->ubLoop = (currentBlockFactor + this->ubFactor - 1) / this->ubFactor;
        // ubLoop：最后一块
        this->ubTail = currentBlockFactor - (this->ubLoop - 1) * this->ubFactor;
        dKV = (tilingData_->dk + tilingData_->dv);
        kScaleCopyLen = tilingData_->kScaleType == 1 ? 1 : tilingData_->halfDk;
        kScaleCopyOffset = tilingData_->kScaleType == 1 ? 0 : tilingData_->halfDk;
        kOffsetCopyLen = tilingData_->kOffsetType == 1 ? 1 : tilingData_->halfDk;
        kOffsetCopyOffset = tilingData_->kOffsetType == 1 ? 0 : tilingData_->halfDk;
        vScaleCopyLen = tilingData_->vScaleType == 1 ? 1 : tilingData_->dv;
        vOffsetCopyLen = tilingData_->vOffsetType == 1 ? 1 : tilingData_->dv;

        kVFLoop = ops::FloorDiv(tilingData_->halfDk, static_cast<int64_t>(VL_FP32));
        kVFTail = tilingData_->halfDk % VL_FP32;

        this->dvLoopCount = ops::CeilDiv(tilingData_->dv, static_cast<int64_t>(VL_FP32));

        // init global memory
        this->kvGm.SetGlobalBuffer((__gm__ T_KV *)kv + this->blockIdx * tilingData_->blockFactor * dKV);
        this->gammaGm.SetGlobalBuffer((__gm__ T_KV *)gamma);
        this->cosGm.SetGlobalBuffer((__gm__ T_KV *)cos);
        this->sinGm.SetGlobalBuffer((__gm__ T_KV *)sin);
        this->indexGm.SetGlobalBuffer((__gm__ int64_t *)index);
        this->kCacheGm.SetGlobalBuffer((__gm__ T_K_CACHE *)k_cache);
        this->vCacheGm.SetGlobalBuffer((__gm__ T_V_CACHE *)v_cache);
        this->kScaleGm.SetGlobalBuffer((__gm__ float *)k_scale);
        this->vScaleGm.SetGlobalBuffer((__gm__ float *)v_scale);
        this->kOffsetGm.SetGlobalBuffer((__gm__ float *)k_offset);
        this->vOffsetGm.SetGlobalBuffer((__gm__ float *)v_offset);
        this->kOutGm.SetGlobalBuffer((__gm__ T_KV *)k_out + this->blockIdx * tilingData_->blockFactor * this->dk);
        this->vOutGm.SetGlobalBuffer((__gm__ T_KV *)v_out + this->blockIdx * tilingData_->blockFactor * this->dv);

        // init pipe
        pipe_->InitBuffer(inQueueGamma, BUFFER_COUNT_SINGLE, tilingData_->dvAlign * sizeof(T_KV));
        pipe_->InitBuffer(inQueueCosSin, BUFFER_COUNT_DOUBLE,
                          this->ubFactor * COS_SIN_CHANNEL_COUNT * tilingData_->halfDkAlign * sizeof(T_KV));
        pipe_->InitBuffer(inQueueX, BUFFER_COUNT_DOUBLE, this->ubFactor * tilingData_->inUbSize);
        pipe_->InitBuffer(outQueue, BUFFER_COUNT_DOUBLE, this->ubFactor * tilingData_->outUbSize);
        pipe_->InitBuffer(wsBuffer0, WORKSPACE_FLOAT_VECTOR_COUNT * VL_FP32 * sizeof(float));
        pipe_->InitBuffer(wsBuffer1, this->ubFactor * tilingData_->rmsNormWspSize);
        if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                      IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
            pipe_->InitBuffer(kScaleOffsetQueue, BUFFER_COUNT_SINGLE,
                              K_SCALE_OFFSET_CHANNEL_COUNT * tilingData_->halfDkAlign * sizeof(float));
        }
        if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                      IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
            pipe_->InitBuffer(vScaleOffsetQueue, BUFFER_COUNT_SINGLE,
                              V_SCALE_OFFSET_CHANNEL_COUNT * tilingData_->dvAlign * sizeof(float));
        }
    }

    __aicore__ inline void ScatterUpdateK(int64_t rows, int64_t startIdx)
    {
        DataCopyExtParams copyOutParams =
            tilingData_->cacheMode == PA_NZ_CACHE_MODE ? copyOutKParamsNz : copyOutKParams;
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t seqIndex = 0;
        int64_t batchId = 0;
        int64_t gmOffset = 0;
        for (int64_t i = 0; i < rows; i++) {
            seqIndex = this->indexGm(startIdx + i);
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            if (seqIndex >= 0 && seqIndex < tilingData_->cacheRowLimit) {
                if (tilingData_->cacheMode == NORM_CACHE_MODE) {
                    batchId = (startIdx + i) / tilingData_->seqLength;
                    gmOffset = (batchId * tilingData_->cacheLength + seqIndex) * tilingData_->dk;
                } else if (tilingData_->cacheMode == PA_CACHE_MODE) {
                    gmOffset = seqIndex * tilingData_->dk;
                } else if (tilingData_->cacheMode == PA_NZ_CACHE_MODE) {
                    int64_t pageId = seqIndex / tilingData_->blockSize;
                    int64_t tokenOffsetInCurrentPage = seqIndex % tilingData_->blockSize;
                    gmOffset = (pageId * dk1 * tilingData_->blockSize + tokenOffsetInCurrentPage) * dk0;
                } else {
                    return;
                }
                if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                              IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                    DataCopyPad(this->kCacheGm[gmOffset], kQuantLocal[i * tilingData_->dkB8Align], copyOutParams);
                } else {
                    DataCopyPad(this->kCacheGm[gmOffset], kOutLocal[i * tilingData_->dkAlign], copyOutParams);
                }
            }
        }
    }

    __aicore__ inline void ScatterBlkUpdateK(int64_t rows, int64_t startIdx)
    {
        DataCopyExtParams copyOutParams =
            tilingData_->cacheMode == PA_BLK_NZ_CACHE_MODE ? copyOutKParamsNz : copyOutKParams;
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t indexPageLength = (tilingData_->seqLength + tilingData_->blockSize - 1) / tilingData_->blockSize;
        int64_t gmOffset = 0;
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t batchId = tokenId / tilingData_->seqLength;
            int64_t tokenIdInCurrentBatch = tokenId % tilingData_->seqLength;
            int64_t indexPageId = tokenIdInCurrentBatch / tilingData_->blockSize;
            int64_t indexOffset = batchId * indexPageLength + indexPageId;
            int64_t pageOffset = this->indexGm(indexOffset);
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            int64_t tokenOffsetInCurrentPage = tokenIdInCurrentBatch % tilingData_->blockSize;
            int64_t pageId = pageOffset / tilingData_->blockSize;
            if (pageOffset >= 0 && pageOffset < tilingData_->cacheRowLimit) {
                if (tilingData_->cacheMode == PA_BLK_NZ_CACHE_MODE) {
                    gmOffset = (pageId * dk1 * tilingData_->blockSize + tokenOffsetInCurrentPage) * dk0;
                } else {
                    gmOffset = (pageId * tilingData_->blockSize + tokenOffsetInCurrentPage) * tilingData_->dk;
                }
                if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                              IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                    DataCopyPad(this->kCacheGm[gmOffset], kQuantLocal[i * tilingData_->dkB8Align], copyOutParams);
                } else {
                    DataCopyPad(this->kCacheGm[gmOffset], kOutLocal[i * tilingData_->dkAlign], copyOutParams);
                }
            }
        }
    }

    __aicore__ inline void ScatterUpdateV(int64_t rows, int64_t startIdx)
    {
        DataCopyExtParams copyOutParams =
            tilingData_->cacheMode == PA_NZ_CACHE_MODE ? copyOutVParamsNz : copyOutVParams;
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t seqIndex = 0;
        int64_t batchId = 0;
        int64_t gmOffset = 0;
        for (int64_t i = 0; i < rows; i++) {
            seqIndex = this->indexGm(startIdx + i);
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            if (seqIndex >= 0 && seqIndex < tilingData_->cacheRowLimit) {
                if (tilingData_->cacheMode == NORM_CACHE_MODE) {
                    batchId = (startIdx + i) / tilingData_->seqLength;
                    gmOffset = (batchId * tilingData_->cacheLength + seqIndex) * tilingData_->dv;
                } else if (tilingData_->cacheMode == PA_CACHE_MODE) {
                    gmOffset = seqIndex * tilingData_->dv;
                } else if (tilingData_->cacheMode == PA_NZ_CACHE_MODE) {
                    int64_t pageId = seqIndex / tilingData_->blockSize;
                    int64_t tokenOffsetInCurrentPage = seqIndex % tilingData_->blockSize;
                    gmOffset = (pageId * dv1 * tilingData_->blockSize + tokenOffsetInCurrentPage) * dv0;
                } else {
                    return;
                }
                if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                              IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                    DataCopyPad(this->vCacheGm[gmOffset], vQuantLocal[i * tilingData_->dvB8Align], copyOutParams);
                } else {
                    DataCopyPad(this->vCacheGm[gmOffset], vOutLocal[i * tilingData_->dvAlign], copyOutParams);
                }
            }
        }
    }

    __aicore__ inline void ScatterBlkUpdateV(int64_t rows, int64_t startIdx)
    {
        DataCopyExtParams copyOutParams =
            tilingData_->cacheMode == PA_BLK_NZ_CACHE_MODE ? copyOutVParamsNz : copyOutVParams;
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t indexPageLength = (tilingData_->seqLength + tilingData_->blockSize - 1) / tilingData_->blockSize;
        int64_t gmOffset = 0;
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t batchId = tokenId / tilingData_->seqLength;
            int64_t tokenIdInCurrentBatch = tokenId % tilingData_->seqLength;
            int64_t indexPageId = tokenIdInCurrentBatch / tilingData_->blockSize;
            int64_t indexOffset = batchId * indexPageLength + indexPageId;
            int64_t pageOffset = this->indexGm(indexOffset);
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            int64_t tokenOffsetInCurrentPage = tokenIdInCurrentBatch % tilingData_->blockSize;
            int64_t pageId = pageOffset / tilingData_->blockSize;
            if (pageOffset >= 0 && pageOffset < tilingData_->cacheRowLimit) {
                if (tilingData_->cacheMode == PA_BLK_NZ_CACHE_MODE) {
                    gmOffset = (pageId * dv1 * tilingData_->blockSize + tokenOffsetInCurrentPage) * dv0;
                } else {
                    gmOffset = (pageId * tilingData_->blockSize + tokenOffsetInCurrentPage) * tilingData_->dv;
                }
                if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                              IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                    DataCopyPad(this->vCacheGm[gmOffset], vQuantLocal[i * tilingData_->dvB8Align], copyOutParams);
                } else {
                    DataCopyPad(this->vCacheGm[gmOffset], vOutLocal[i * tilingData_->dvAlign], copyOutParams);
                }
            }
        }
    }

    __aicore__ inline void RopeAsymQuantWithKvVF(
        __ubuf__ T_KV *outFront, __ubuf__ T_KV *outBack, __ubuf__ T_K_CACHE *quantFront, __ubuf__ T_K_CACHE *quantBack,
        __ubuf__ T_KV *x, __ubuf__ T_KV *realCos, __ubuf__ T_KV *imgCos, __ubuf__ T_KV *realSin, __ubuf__ T_KV *imgSin,
        __ubuf__ float *realKScale, __ubuf__ float *imgKScale, __ubuf__ float *realKOffset, __ubuf__ float *imgKOffset,
        __ubuf__ float *ws, uint16_t row, uint16_t colLoopCount, uint32_t colCosSin, uint32_t xOffset,
        uint32_t cosSinOffset, uint32_t quantOutOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1, offset_0,
                offset_1;
            AscendC::MicroAPI::UnalignRegForStore UReg0, UReg1, UReg2, UReg3;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                __ubuf__ T_KV *out0 = outFront + rowId * xOffset;
                __ubuf__ T_KV *out1 = outBack + rowId * xOffset;
                __ubuf__ T_K_CACHE *out2 = quantFront + rowId * quantOutOffset;
                __ubuf__ T_K_CACHE *out3 = quantBack + rowId * quantOutOffset;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                       xOffset, cosSinOffset, rowId, i, mask, maskAll);
                    StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, VL_FP32);
                    LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                    AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                    LoadTensorForDtypeT(realKOffset, offset_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKOffset, offset_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Add(offset_0, offset_0, scale_0, mask);
                    AscendC::MicroAPI::Add(offset_1, offset_1, scale_1, mask);
                    StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, VL_FP32);
                }
                mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                RopeBasicComputeVF(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                   xOffset, cosSinOffset, rowId, colLoopCount, mask, maskAll);
                StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, kVFTail);
                StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, kVFTail);
                LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                LoadTensorForDtypeT(realKOffset, offset_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKOffset, offset_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Add(offset_0, offset_0, scale_0, mask);
                AscendC::MicroAPI::Add(offset_1, offset_1, scale_1, mask);
                StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, kVFTail);
                StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, kVFTail);
                AscendC::MicroAPI::StoreUnAlignPost(out0, UReg0, 0);
                AscendC::MicroAPI::StoreUnAlignPost(out1, UReg1, 0);
                AscendC::MicroAPI::StoreUnAlignPost(out2, UReg2, 0);
                AscendC::MicroAPI::StoreUnAlignPost(out3, UReg3, 0);
            }
        }
    }

    __aicore__ inline void RopeSymQuantWithKvVF(
        __ubuf__ T_KV *outFront, __ubuf__ T_KV *outBack, __ubuf__ T_K_CACHE *quantFront, __ubuf__ T_K_CACHE *quantBack,
        __ubuf__ T_KV *x, __ubuf__ T_KV *realCos, __ubuf__ T_KV *imgCos, __ubuf__ T_KV *realSin, __ubuf__ T_KV *imgSin,
        __ubuf__ float *realKScale, __ubuf__ float *imgKScale, __ubuf__ float *ws, uint16_t row, uint16_t colLoopCount,
        uint32_t colCosSin, uint32_t xOffset, uint32_t cosSinOffset, uint32_t quantOutOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1;
            AscendC::MicroAPI::UnalignRegForStore UReg0, UReg1, UReg2, UReg3;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                __ubuf__ T_KV *out0 = outFront + rowId * xOffset;
                __ubuf__ T_KV *out1 = outBack + rowId * xOffset;
                __ubuf__ T_K_CACHE *out2 = quantFront + rowId * quantOutOffset;
                __ubuf__ T_K_CACHE *out3 = quantBack + rowId * quantOutOffset;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                       xOffset, cosSinOffset, rowId, i, mask, maskAll);
                    StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, VL_FP32);
                    LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                    AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                    StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, VL_FP32);
                }
                mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                RopeBasicComputeVF(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                   xOffset, cosSinOffset, rowId, colLoopCount, mask, maskAll);
                StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, kVFTail);
                StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, kVFTail);
                LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, kVFTail);
                StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, kVFTail);
                AscendC::MicroAPI::StoreUnAlignPost(out0, UReg0, 0);
                AscendC::MicroAPI::StoreUnAlignPost(out1, UReg1, 0);
                AscendC::MicroAPI::StoreUnAlignPost(out2, UReg2, 0);
                AscendC::MicroAPI::StoreUnAlignPost(out3, UReg3, 0);
            }
        }
    }

    __aicore__ inline void RopeAsymQuantVF(__ubuf__ T_K_CACHE *quantFront, __ubuf__ T_K_CACHE *quantBack,
                                           __ubuf__ T_KV *x, __ubuf__ T_KV *realCos, __ubuf__ T_KV *imgCos,
                                           __ubuf__ T_KV *realSin, __ubuf__ T_KV *imgSin, __ubuf__ float *realKScale,
                                           __ubuf__ float *imgKScale, __ubuf__ float *realKOffset,
                                           __ubuf__ float *imgKOffset, __ubuf__ float *ws, uint16_t row,
                                           uint16_t colLoopCount, uint32_t colCosSin, uint32_t xOffset,
                                           uint32_t cosSinOffset, uint32_t quantOutOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1, offset_0,
                offset_1;
            AscendC::MicroAPI::UnalignRegForStore UReg2, UReg3;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                __ubuf__ T_K_CACHE *out2 = quantFront + rowId * quantOutOffset;
                __ubuf__ T_K_CACHE *out3 = quantBack + rowId * quantOutOffset;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                       xOffset, cosSinOffset, rowId, i, mask, maskAll);
                    LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                    AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                    LoadTensorForDtypeT(realKOffset, offset_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKOffset, offset_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Add(offset_0, offset_0, scale_0, mask);
                    AscendC::MicroAPI::Add(offset_1, offset_1, scale_1, mask);
                    StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, VL_FP32);
                }
                mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                RopeBasicComputeVF(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                   xOffset, cosSinOffset, rowId, colLoopCount, mask, maskAll);
                LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                LoadTensorForDtypeT(realKOffset, offset_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKOffset, offset_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Add(offset_0, offset_0, scale_0, mask);
                AscendC::MicroAPI::Add(offset_1, offset_1, scale_1, mask);
                StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, kVFTail);
                StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, kVFTail);
                AscendC::MicroAPI::StoreUnAlignPost(out2, UReg2, 0);
                AscendC::MicroAPI::StoreUnAlignPost(out3, UReg3, 0);
            }
        }
    }

    __aicore__ inline void RopeSymQuantVF(__ubuf__ T_K_CACHE *quantFront, __ubuf__ T_K_CACHE *quantBack,
                                          __ubuf__ T_KV *x, __ubuf__ T_KV *realCos, __ubuf__ T_KV *imgCos,
                                          __ubuf__ T_KV *realSin, __ubuf__ T_KV *imgSin, __ubuf__ float *realKScale,
                                          __ubuf__ float *imgKScale, __ubuf__ float *ws, uint16_t row,
                                          uint16_t colLoopCount, uint32_t colCosSin, uint32_t xOffset,
                                          uint32_t cosSinOffset, uint32_t quantOutOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1;
            AscendC::MicroAPI::UnalignRegForStore UReg2, UReg3;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                __ubuf__ T_K_CACHE *out2 = quantFront + rowId * quantOutOffset;
                __ubuf__ T_K_CACHE *out3 = quantBack + rowId * quantOutOffset;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                       xOffset, cosSinOffset, rowId, i, mask, maskAll);
                    LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                    AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                    StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, VL_FP32);
                }
                mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                RopeBasicComputeVF(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                   xOffset, cosSinOffset, rowId, colLoopCount, mask, maskAll);
                LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, kVFTail);
                StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, kVFTail);
                AscendC::MicroAPI::StoreUnAlignPost(out2, UReg2, 0);
                AscendC::MicroAPI::StoreUnAlignPost(out3, UReg3, 0);
            }
        }
    }

    __aicore__ inline void RopeBasicComputeVF(
        __ubuf__ T_KV *x, AscendC::MicroAPI::RegTensor<float> &x_0, AscendC::MicroAPI::RegTensor<float> &x_1,
        AscendC::MicroAPI::RegTensor<float> &cos_0, AscendC::MicroAPI::RegTensor<float> &cos_1,
        AscendC::MicroAPI::RegTensor<float> &sin_0, AscendC::MicroAPI::RegTensor<float> &sin_1, __ubuf__ T_KV *realCos,
        __ubuf__ T_KV *imgCos, __ubuf__ T_KV *realSin, __ubuf__ T_KV *imgSin, __ubuf__ float *ws, uint32_t xOffset,
        uint32_t cosSinOffset, uint16_t rowId, uint16_t colId, AscendC::MicroAPI::MaskReg mask,
        AscendC::MicroAPI::MaskReg maskAll)
    {
        // cast to float32
        LoadTensorForDtypeT<T_KV>(x, x_0, maskAll, colId * CONST_TWO * VL_FP32 + rowId * xOffset);
        LoadTensorForDtypeT<T_KV>(x, x_1, maskAll, (colId * CONST_TWO + CONST_ONE) * VL_FP32 + rowId * xOffset);
        StoreTensorForDtypeTOut<float>(ws, x_0, maskAll, 0);
        StoreTensorForDtypeTOut<float>(ws, x_1, maskAll, VL_FP32);
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        // get x
        AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(x_0, x_1,
                                                                                          ((__ubuf__ float *)(ws)));
        LoadTensorForDtypeT(realCos, cos_0, mask, colId * VL_FP32 + rowId * cosSinOffset);
        LoadTensorForDtypeT(imgCos, cos_1, mask, colId * VL_FP32 + rowId * cosSinOffset);
        LoadTensorForDtypeT(realSin, sin_0, mask, colId * VL_FP32 + rowId * cosSinOffset);
        LoadTensorForDtypeT(imgSin, sin_1, mask, colId * VL_FP32 + rowId * cosSinOffset);
        AscendC::MicroAPI::Mul(cos_0, x_0, cos_0, mask);
        AscendC::MicroAPI::Mul(cos_1, x_1, cos_1, mask);
        AscendC::MicroAPI::Muls(x_1, x_1, -1, mask);
        AscendC::MicroAPI::Mul(sin_0, x_1, sin_0, mask);
        AscendC::MicroAPI::Mul(sin_1, x_0, sin_1, mask);
        AscendC::MicroAPI::Add(sin_0, cos_0, sin_0, mask);
        AscendC::MicroAPI::Add(sin_1, cos_1, sin_1, mask);
    }

    __aicore__ inline void RopeVF(__ubuf__ T_KV *outFront, __ubuf__ T_KV *outBack, __ubuf__ T_KV *x,
                                  __ubuf__ T_KV *realCos, __ubuf__ T_KV *imgCos, __ubuf__ T_KV *realSin,
                                  __ubuf__ T_KV *imgSin, __ubuf__ float *ws, uint16_t row, uint16_t colLoopCount,
                                  uint32_t colCosSin, uint32_t xOffset, uint32_t cosSinOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1;
            AscendC::MicroAPI::UnalignRegForStore UReg0;
            AscendC::MicroAPI::UnalignRegForStore UReg1;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                // 跨行地址重刷新
                __ubuf__ T_KV *out0 = outFront + rowId * xOffset;
                __ubuf__ T_KV *out1 = outBack + rowId * xOffset;
                // 整块
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                       xOffset, cosSinOffset, rowId, i, mask, maskAll);
                    StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, VL_FP32);
                }
                // 尾块
                mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                RopeBasicComputeVF(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                   xOffset, cosSinOffset, rowId, colLoopCount, mask, maskAll);
                StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, kVFTail);
                StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, kVFTail);
                AscendC::MicroAPI::StoreUnAlignPost(out0, UReg0, 0);
                AscendC::MicroAPI::StoreUnAlignPost(out1, UReg1, 0);
            }
        }
    }

    __aicore__ inline void Rope(const LocalTensor<T_KV> &xLocal, const LocalTensor<T_KV> &realCosLocal,
                                const LocalTensor<T_KV> &imgCosLocal, const LocalTensor<T_KV> &realSinLocal,
                                const LocalTensor<T_KV> &imgSinLocal, const LocalTensor<float> &realKScaleLocal,
                                const LocalTensor<float> &imgKScaleLocal, const LocalTensor<float> &realKOffsetLocal,
                                const LocalTensor<float> &imgKOffsetLocal, const LocalTensor<float> &wsLocal,
                                uint16_t rowNum)
    {
        uint16_t row = rowNum;
        uint32_t colCosSin = tilingData_->halfDk;
        uint32_t xOffset = tilingData_->dkAlign;
        uint32_t cosSinOffset = tilingData_->halfDkAlign;
        uint32_t quantOutOffset = tilingData_->dkB8Align;

        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xLocal.GetPhyAddr();
        __ubuf__ T_KV *realCos = (__ubuf__ T_KV *)realCosLocal.GetPhyAddr();
        __ubuf__ T_KV *imgCos = (__ubuf__ T_KV *)imgCosLocal.GetPhyAddr();
        __ubuf__ T_KV *realSin = (__ubuf__ T_KV *)realSinLocal.GetPhyAddr();
        __ubuf__ T_KV *imgSin = (__ubuf__ T_KV *)imgSinLocal.GetPhyAddr();
        __ubuf__ float *ws = (__ubuf__ float *)wsLocal.GetPhyAddr();

        if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                      IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
            // scale/offset 仅在量化路径下由 Process 分配，非量化实例中这四个 LocalTensor 是默认构造的，
            // 取址必须留在本分支内，否则会读到未初始化的 TBuffAddr
            __ubuf__ float *realKScale = (__ubuf__ float *)realKScaleLocal.GetPhyAddr();
            __ubuf__ float *imgKScale = (__ubuf__ float *)imgKScaleLocal.GetPhyAddr();
            __ubuf__ float *realKOffset = (__ubuf__ float *)realKOffsetLocal.GetPhyAddr();
            __ubuf__ float *imgKOffset = (__ubuf__ float *)imgKOffsetLocal.GetPhyAddr();
            if (tilingData_->isOutputKv > 0) {
                kOutLocal = outQueue.AllocTensor<T_KV>();
                kQuantLocal =
                    kOutLocal
                        .template ReinterpretCast<T_K_CACHE>()[this->ubFactor * tilingData_->dkB8Align * sizeof(T_KV)];
                __ubuf__ T_KV *outFront = (__ubuf__ T_KV *)kOutLocal.GetPhyAddr();
                __ubuf__ T_KV *outBack = (__ubuf__ T_KV *)kOutLocal.GetPhyAddr() + tilingData_->halfDk;
                __ubuf__ T_K_CACHE *quantFront = (__ubuf__ T_K_CACHE *)kQuantLocal.GetPhyAddr();
                __ubuf__ T_K_CACHE *quantBack = (__ubuf__ T_K_CACHE *)kQuantLocal.GetPhyAddr() + tilingData_->halfDk;
                if (tilingData_->kOffsetType > 0) {
                    RopeAsymQuantWithKvVF(outFront, outBack, quantFront, quantBack, x, realCos, imgCos, realSin, imgSin,
                                          realKScale, imgKScale, realKOffset, imgKOffset, ws, row, kVFLoop, colCosSin,
                                          xOffset, cosSinOffset, quantOutOffset);
                } else {
                    RopeSymQuantWithKvVF(outFront, outBack, quantFront, quantBack, x, realCos, imgCos, realSin, imgSin,
                                         realKScale, imgKScale, ws, row, kVFLoop, colCosSin, xOffset, cosSinOffset,
                                         quantOutOffset);
                }
                outQueue.EnQue(kOutLocal);
                kOutLocal = outQueue.DeQue<T_KV>();
            } else {
                kQuantLocal = outQueue.AllocTensor<T_K_CACHE>();
                __ubuf__ T_K_CACHE *quantFront = (__ubuf__ T_K_CACHE *)kQuantLocal.GetPhyAddr();
                __ubuf__ T_K_CACHE *quantBack = (__ubuf__ T_K_CACHE *)kQuantLocal.GetPhyAddr() + tilingData_->halfDk;
                if (tilingData_->kOffsetType > 0) {
                    RopeAsymQuantVF(quantFront, quantBack, x, realCos, imgCos, realSin, imgSin, realKScale, imgKScale,
                                    realKOffset, imgKOffset, ws, row, kVFLoop, colCosSin, xOffset, cosSinOffset,
                                    quantOutOffset);
                } else {
                    RopeSymQuantVF(quantFront, quantBack, x, realCos, imgCos, realSin, imgSin, realKScale, imgKScale,
                                   ws, row, kVFLoop, colCosSin, xOffset, cosSinOffset, quantOutOffset);
                }
                outQueue.EnQue(kQuantLocal);
                kQuantLocal = outQueue.DeQue<T_K_CACHE>();
            }
        } else {
            kOutLocal = outQueue.AllocTensor<T_KV>();
            __ubuf__ T_KV *outFront = (__ubuf__ T_KV *)kOutLocal.GetPhyAddr();
            __ubuf__ T_KV *outBack = (__ubuf__ T_KV *)kOutLocal.GetPhyAddr() + tilingData_->halfDk;
            RopeVF(outFront, outBack, x, realCos, imgCos, realSin, imgSin, ws, row, kVFLoop, colCosSin, xOffset,
                   cosSinOffset);
            outQueue.EnQue(kOutLocal);
            kOutLocal = outQueue.DeQue<T_K_CACHE>();
        }
    }

    __aicore__ inline void RmsNorm(const LocalTensor<T_KV> &outVLocal, const LocalTensor<T_KV> &xLocal,
                                   const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &wsLocal,
                                   const LocalTensor<float> &vScaleTensor, const LocalTensor<float> &vOffsetTensor,
                                   int64_t calcRow)
    {
        // 需要量化
        if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                      IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
            // 需要输出中间结果
            if (tilingData_->isOutputKv > 0) {
                vQuantLocal =
                    outVLocal
                        .template ReinterpretCast<T_V_CACHE>()[this->ubFactor * tilingData_->dvB8Align * sizeof(T_KV)];
                // 需要量化+偏移
                if (tilingData_->vOffsetType > 0) {
                    this->RmsNormAsymQuantWithKvVF(outVLocal, vQuantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor,
                                                   vOffsetTensor, calcRow);
                } else { // 只需要量化
                    this->RmsNormSymQuantWithKvVF(outVLocal, vQuantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor,
                                                  calcRow);
                }
            } else {
                vQuantLocal = outVLocal.template ReinterpretCast<T_V_CACHE>();
                if (tilingData_->vOffsetType > 0) {
                    this->RmsNormAsymQuantVF(vQuantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor, vOffsetTensor,
                                             calcRow);
                } else {
                    this->RmsNormSymQuantVF(vQuantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor, calcRow);
                }
            }
        } else {
            // 普通情况， outLocal的类型需要指定
            this->RmsNormVF(outVLocal, xLocal, gammaLocal, wsLocal, calcRow);
        }
    }

    __aicore__ inline void Process()
    {
        // CopyIn gamma
        LocalTensor<T_KV> gammaLocal = inQueueGamma.AllocTensor<T_KV>();
        CopyInLineAlign<T_KV>(gammaLocal, this->gammaGm, 1, tilingData_->dv, tilingData_->dv);
        inQueueGamma.EnQue(gammaLocal);
        gammaLocal = inQueueGamma.DeQue<T_KV>();
        LocalTensor<float> workspaceBuffer0 = wsBuffer0.Get<float>();
        LocalTensor<float> workspaceBuffer1 = wsBuffer1.Get<float>();
        // 行，有几行相当于ubA
        int64_t calcRow = this->ubFactor;
        // 量化场景
        if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                      IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
            realKScaleLocal = kScaleOffsetQueue.AllocTensor<float>();
            imgKScaleLocal = realKScaleLocal[tilingData_->halfDkAlign];
            realKOffsetLocal = imgKScaleLocal[tilingData_->halfDkAlign];
            imgKOffsetLocal = realKOffsetLocal[tilingData_->halfDkAlign];
            CopyInLineAlign<float>(realKScaleLocal, this->kScaleGm, 1, kScaleCopyLen, kScaleCopyLen);
            CopyInLineAlign<float>(imgKScaleLocal, this->kScaleGm[kScaleCopyOffset], 1, kScaleCopyLen, kScaleCopyLen);
            if (tilingData_->kOffsetType > 0) {
                CopyInLineAlign<float>(realKOffsetLocal, this->kOffsetGm, 1, kOffsetCopyLen, kOffsetCopyLen);
                CopyInLineAlign<float>(imgKOffsetLocal, this->kOffsetGm[kOffsetCopyOffset], 1, kOffsetCopyLen,
                                       kOffsetCopyLen);
            }
            kScaleOffsetQueue.EnQue(realKScaleLocal);
            realKScaleLocal = kScaleOffsetQueue.DeQue<float>();
            // 将scale和offset Brc方便VF处理
            if (tilingData_->kScaleType == 1) {
                Duplicate<float>(realKScaleLocal, realKScaleLocal, tilingData_->halfDkAlign);
                Duplicate<float>(imgKScaleLocal, imgKScaleLocal, tilingData_->halfDkAlign);
            }
            if (tilingData_->kOffsetType == 1) {
                Duplicate<float>(realKOffsetLocal, realKOffsetLocal, tilingData_->halfDkAlign);
                Duplicate<float>(imgKOffsetLocal, imgKOffsetLocal, tilingData_->halfDkAlign);
            }
        }
        if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                      IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
            vScaleLocal = vScaleOffsetQueue.AllocTensor<float>();
            vOffsetLocal = vScaleLocal[tilingData_->dvAlign];
            CopyInLineAlign<float>(vScaleLocal, this->vScaleGm, 1, vScaleCopyLen, vScaleCopyLen);
            if (tilingData_->vOffsetType > 0) {
                CopyInLineAlign<float>(vOffsetLocal, this->vOffsetGm, 1, vOffsetCopyLen, vOffsetCopyLen);
            }
            vScaleOffsetQueue.EnQue(vScaleLocal);
            vScaleLocal = vScaleOffsetQueue.DeQue<float>();
            // 将scale和offset Brc方便VF处理
            if (tilingData_->vScaleType == 1) {
                Duplicate<float>(vScaleLocal, vScaleLocal, tilingData_->dvAlign);
            }
            if (tilingData_->vOffsetType == 1) {
                Duplicate<float>(vOffsetLocal, vOffsetLocal, tilingData_->dvAlign);
            }
        }
        for (int64_t loopIdx = 0; loopIdx < this->ubLoop; ++loopIdx) {
            if (loopIdx == this->ubLoop - 1) {
                calcRow = this->ubTail;
            }
            kvGlobalMemoryOffset = loopIdx * this->ubFactor * dKV;
            kOutGmOffset = loopIdx * this->ubFactor * this->dk;
            vOutGmOffset = loopIdx * this->ubFactor * this->dv;
            // 当前子块的偏移量
            int64_t startIdx = this->blockIdx * tilingData_->blockFactor + loopIdx * this->ubFactor;

            LocalTensor<T_KV> ropeLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> realCosLocal = inQueueCosSin.AllocTensor<T_KV>();
            LocalTensor<T_KV> imgCosLocal = realCosLocal[this->ubFactor * tilingData_->halfDkAlign];
            LocalTensor<T_KV> realSinLocal = imgCosLocal[this->ubFactor * tilingData_->halfDkAlign];
            LocalTensor<T_KV> imgSinLocal = realSinLocal[this->ubFactor * tilingData_->halfDkAlign];
            CopyInLineAlign<T_KV>(ropeLocal, this->kvGm[kvGlobalMemoryOffset + tilingData_->dv], calcRow, dKV,
                                  tilingData_->dk);
            if (tilingData_->cosSinNeedBrc == 0) {
                freqGlobalMemoryOffset =
                    this->blockIdx * tilingData_->blockFactor * this->dk + loopIdx * this->ubFactor * tilingData_->dk;
                CopyInLineAlign<T_KV>(realCosLocal, this->cosGm[freqGlobalMemoryOffset], calcRow, tilingData_->dk,
                                      tilingData_->halfDk);
                CopyInLineAlign<T_KV>(imgCosLocal, this->cosGm[freqGlobalMemoryOffset + tilingData_->halfDk], calcRow,
                                      tilingData_->dk, tilingData_->halfDk);
                CopyInLineAlign<T_KV>(realSinLocal, this->sinGm[freqGlobalMemoryOffset], calcRow, tilingData_->dk,
                                      tilingData_->halfDk);
                CopyInLineAlign<T_KV>(imgSinLocal, this->sinGm[freqGlobalMemoryOffset + tilingData_->halfDk], calcRow,
                                      tilingData_->dk, tilingData_->halfDk);
            } else {
                int64_t batchId = 0;
                int64_t ubOffset = 0;
                for (int64_t i = 0; i < calcRow; i++) {
                    batchId = (startIdx + i) / tilingData_->seqLength;
                    freqGlobalMemoryOffset = batchId * tilingData_->dk;
                    ubOffset = i * tilingData_->halfDkAlign;
                    CopyInLineAlign<T_KV>(realCosLocal[ubOffset], this->cosGm[freqGlobalMemoryOffset], 1,
                                          tilingData_->dk, tilingData_->halfDk);
                    CopyInLineAlign<T_KV>(imgCosLocal[ubOffset],
                                          this->cosGm[freqGlobalMemoryOffset + tilingData_->halfDk], 1, tilingData_->dk,
                                          tilingData_->halfDk);
                    CopyInLineAlign<T_KV>(realSinLocal[ubOffset], this->sinGm[freqGlobalMemoryOffset], 1,
                                          tilingData_->dk, tilingData_->halfDk);
                    CopyInLineAlign<T_KV>(imgSinLocal[ubOffset],
                                          this->sinGm[freqGlobalMemoryOffset + tilingData_->halfDk], 1, tilingData_->dk,
                                          tilingData_->halfDk);
                }
            }
            inQueueX.EnQue(ropeLocal);
            inQueueCosSin.EnQue(realCosLocal);
            ropeLocal = inQueueX.DeQue<T_KV>();
            realCosLocal = inQueueCosSin.DeQue<T_KV>();
            // Calc: RoPE
            Rope(ropeLocal, realCosLocal, imgCosLocal, realSinLocal, imgSinLocal, realKScaleLocal, imgKScaleLocal,
                 realKOffsetLocal, imgKOffsetLocal, workspaceBuffer0, calcRow);
            inQueueX.FreeTensor(ropeLocal);
            inQueueCosSin.FreeTensor(realCosLocal);
            // 拷出到kCache
            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateK(calcRow, startIdx);
            } else {
                ScatterBlkUpdateK(calcRow, startIdx);
            }

            // 拷出到kOut
            if (tilingData_->isOutputKv > 0) {
                CopyLineAlignOut<T_KV>(this->kOutGm[kOutGmOffset], kOutLocal, calcRow, this->dk, this->dk);
            }
            // FreeTensor 区分场景
            if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                          IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                if (tilingData_->isOutputKv > 0) {
                    outQueue.FreeTensor(kOutLocal);
                } else {
                    outQueue.FreeTensor(kQuantLocal);
                }
            } else {
                outQueue.FreeTensor(kOutLocal);
            }

            // CopyIn x: [ubFactor, RmsLength]
            LocalTensor<T_KV> xLocal = inQueueX.AllocTensor<T_KV>();
            CopyInLineAlign<T_KV>(xLocal, this->kvGm[kvGlobalMemoryOffset], calcRow, dKV, tilingData_->dv);
            inQueueX.EnQue(xLocal);
            xLocal = inQueueX.DeQue<T_KV>();

            // Calc: RmsNorm
            vOutLocal = outQueue.AllocTensor<T_KV>();
            this->RmsNorm(vOutLocal, xLocal, gammaLocal, workspaceBuffer1, vScaleLocal, vOffsetLocal, calcRow);
            inQueueX.FreeTensor(xLocal);
            outQueue.EnQue(vOutLocal);
            vOutLocal = outQueue.DeQue<T_KV>();
            // 拷出到vCache
            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateV(calcRow, startIdx);
            } else {
                ScatterBlkUpdateV(calcRow, startIdx);
            }
            // 拷出到vOut
            if (tilingData_->isOutputKv > 0) {
                CopyLineAlignOut<T_KV>(this->vOutGm[vOutGmOffset], vOutLocal, calcRow, this->dv, this->dv);
            }
            // FreeTensor 不论什么场景，都是申请的vOutLocal
            outQueue.FreeTensor(vOutLocal);
        }
        inQueueGamma.FreeTensor(gammaLocal);
        if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                      IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
            kScaleOffsetQueue.FreeTensor(realKScaleLocal);
        }
        if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                      IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
            vScaleOffsetQueue.FreeTensor(vScaleLocal);
        }
    }

private:
    const KvRmsNormRopeCacheRegbaseFullLoadTilingData *tilingData_;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueGamma, inQueueCosSin;
    TQue<QuePosition::VECIN, 1> kScaleOffsetQueue, vScaleOffsetQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> wsBuffer0, wsBuffer1;
    TPipe *pipe_ = nullptr;

    LocalTensor<T_KV> kOutLocal;
    LocalTensor<T_K_CACHE> kQuantLocal;
    LocalTensor<T_KV> vOutLocal;
    LocalTensor<T_V_CACHE> vQuantLocal;

    LocalTensor<float> realKScaleLocal;
    LocalTensor<float> imgKScaleLocal;
    LocalTensor<float> realKOffsetLocal;
    LocalTensor<float> imgKOffsetLocal;
    LocalTensor<float> vScaleLocal;
    LocalTensor<float> vOffsetLocal;

    event_t eventIDSToMTE3;

    int64_t dKV = 0;
    int64_t kScaleCopyLen = 0;
    int64_t kScaleCopyOffset = 0;
    int64_t kOffsetCopyLen = 0;
    int64_t kOffsetCopyOffset = 0;
    int64_t vScaleCopyLen = 0;
    int64_t vOffsetCopyLen = 0;
    int64_t dk0 = 0;
    int64_t dk1 = 0;
    int64_t dv0 = 0;
    int64_t dv1 = 0;

    uint16_t kVFLoop = 0;
    uint32_t kVFTail = 0;

    DataCopyExtParams copyOutKParams;
    DataCopyExtParams copyOutKParamsNz;
    DataCopyExtParams copyOutVParams;
    DataCopyExtParams copyOutVParamsNz;

    int64_t kvGlobalMemoryOffset = 0;
    int64_t freqGlobalMemoryOffset = 0;
    int64_t kOutGmOffset = 0;
    int64_t vOutGmOffset = 0;
};
} // namespace KvRmsNormRopeCache

#endif // KV_RMS_NORM_ROPE_CACHE_REGBASE_FULL_LOAD_H_
