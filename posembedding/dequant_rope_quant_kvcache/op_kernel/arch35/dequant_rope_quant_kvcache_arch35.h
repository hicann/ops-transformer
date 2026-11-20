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
 * \file dequant_rope_quant_kvcache.h
 * \brief
 */
#ifndef DEQUANT_ROPE_QUANT_KVCACHE_DEQUANT_ROPE_QUANT_KVCACHE_H
#define DEQUANT_ROPE_QUANT_KVCACHE_DEQUANT_ROPE_QUANT_KVCACHE_H

namespace DequantRopeQuantKvcache {

using namespace AscendC;

constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t DOUBLE_BUFFER_NUM = 1;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MAX_UINT8 = 255;
constexpr int64_t ROPE_LAST_DIM_SPLIT = 2;
constexpr int64_t FP16_ONE_BLOCK_NUM = 16;
constexpr int64_t FP32_ONE_BLOCK_NUM = 8;
constexpr int64_t ONE_REPEAT_SIZE = 256;
constexpr int64_t FP16_ONE_REPEAT_NUM = 128;
constexpr int64_t FP32_ONE_REPEAT_NUM = 64;
constexpr int64_t INDICES_MAX__NUM = 512;
constexpr int64_t OFFSET_OVERRIDE = 2;

template <typename T>
__aicore__ inline T GetDiv(const T& value1, const T& value2)
{
    if (value2 == 0) {
        return value2;
    }
    return (value1) / value2;
}

template <typename T>
__aicore__ inline T GetCeilInt(const T& value1, const T& value2)
{
    if (value2 == 0) {
        return value2;
    }
    return (value1 + value2 - 1) / value2;
}

template <typename T>
__aicore__ inline T GetRem(const T& value1, const T& value2)
{
    if (value2 == 0) {
        return value2;
    }
    return value1 % value2;
}
template <class T>
struct queType {
    using type = float;
};
template <>
struct queType<half> {
    using type = half;
};

template <typename XTYPE, typename BIASTYPE, typename COSTYPE>
class RopeQuantKvcacheV2 {
public:
    __aicore__ inline RopeQuantKvcacheV2(const DequantRopeQuantKvcacheTilingData* tilingData)
    {
        this->cacheSeqlen = tilingData->cacheSeqlen;
        this->seqlen = tilingData->seqlen;
        this->qHeadNum = tilingData->qHeadNum;
        this->kvHeadNum = tilingData->kvHeadNum;
        this->hiddenSize = tilingData->hiddenSize;
        this->qHiddenSize = tilingData->qHiddenSize;
        this->kHiddenSize = tilingData->kHiddenSize;
        this->vHiddenSize = tilingData->vHiddenSize;
        this->realCoreNum = tilingData->realCoreNum;
        this->frontCoreNum = tilingData->frontCoreNum;
        this->blockFactor = tilingData->blockFactor;
        this->tailCoreBlockFactor = tilingData->tailCoreBlockFactor;
        this->hasOffsetK = tilingData->hasOffsetK;
        this->hasOffsetV = tilingData->hasOffsetV;
        this->hiddenSizeFp32Align = tilingData->hiddenSizeFp32Align;
        this->hiddenSizeFp16Align = tilingData->hiddenSizeFp16Align;
        this->hiddenSizeInt8Align = tilingData->hiddenSizeInt8Align;
        this->OnceUBMaxS = tilingData->OnceUBMaxS;
        this->inputHiddenSize = this->qHiddenSize + this->kHiddenSize + this->vHiddenSize;
        this->isPA = tilingData->isPA;
        this->ifKVout = tilingData->ifKVout;
        this->hasBias = tilingData->hasBias;
        this->hasAS = tilingData->hasAS;
        this->batch = tilingData->batch;
        this->cacheBatchNum = tilingData->cacheBatchNum;
    }

    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR indice,
                                GM_ADDR weight_scale, GM_ADDR activation_scale, GM_ADDR bias, GM_ADDR quant_scale_k,
                                GM_ADDR quant_scale_v, GM_ADDR quant_offset_k, GM_ADDR quant_offset_v, GM_ADDR q_out,
                                GM_ADDR k_out, GM_ADDR v_out, GM_ADDR k_cache_out, GM_ADDR v_cache_out)
    {
        auto blockIdx = GetBlockIdx();
        if (blockIdx > this->frontCoreNum - 1) {
            sCoreOffset =
                this->blockFactor * this->frontCoreNum + (blockIdx - this->frontCoreNum) * this->tailCoreBlockFactor;
            coreCalcSNum = this->tailCoreBlockFactor;
        } else {
            sCoreOffset = this->blockFactor * blockIdx;
            coreCalcSNum = this->blockFactor;
        }
        bOffset = GetDiv(sCoreOffset, this->seqlen);
        coreCalcSLoop = GetCeilInt(coreCalcSNum, OnceUBMaxS);
        coreCalcSLastNum = coreCalcSNum - (coreCalcSLoop - 1) * OnceUBMaxS;
        uint64_t onceB = GetCeilInt(OnceUBMaxS, seqlen);
        uint64_t kvDataNum = OnceUBMaxS * this->kvHeadNum * this->hiddenSize; // 8*128
        this->cacheRowElems = this->kvHeadNum * this->hiddenSize;
        this->cacheRowAlign = (this->cacheRowElems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        uint64_t cacheBufNum = OnceUBMaxS * this->cacheRowAlign;
        uint64_t qkvBlockOffset = sCoreOffset * (this->qHiddenSize + this->kHiddenSize + this->vHiddenSize);
        uint64_t cossinBlockOffset = sCoreOffset * this->hiddenSize;
        uint64_t kvCacheBlockOffset = bOffset * this->cacheSeqlen * this->kvHeadNum * this->hiddenSize;
        uint64_t indiceOffset = bOffset;
        this->indiceUbNum = ((onceB + 1 + FP32_ONE_BLOCK_NUM - 1) / FP32_ONE_BLOCK_NUM) * FP32_ONE_BLOCK_NUM;
        uint64_t qDataNum = OnceUBMaxS * this->qHeadNum * this->hiddenSize;
        uint64_t qBlockOffset = blockIdx * qDataNum;
        inputGm.SetGlobalBuffer((__gm__ XTYPE*)qkv + qkvBlockOffset);
        cosGm.SetGlobalBuffer((__gm__ COSTYPE*)cos + cossinBlockOffset, this->blockFactor * this->hiddenSize);
        cosGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        sinGm.SetGlobalBuffer((__gm__ COSTYPE*)sin + cossinBlockOffset, this->blockFactor * this->hiddenSize);
        sinGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        quantScaleKGm.SetGlobalBuffer((__gm__ float*)quant_scale_k, this->kHiddenSize);
        quantScaleKGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        quantScaleVGm.SetGlobalBuffer((__gm__ float*)quant_scale_v, this->kHiddenSize);
        quantScaleVGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        if (this->hasOffsetK == true) {
            quantOffsetKGm.SetGlobalBuffer((__gm__ float*)quant_offset_k, this->kHiddenSize);
            quantOffsetKGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            pipe.InitBuffer(offsetKQueue, this->kHiddenSize * sizeof(float));
        }

        if (this->hasOffsetV == true) {
            quantOffsetVGm.SetGlobalBuffer((__gm__ float*)quant_offset_v, this->kHiddenSize);
            quantOffsetVGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            pipe.InitBuffer(offsetVQueue, this->kHiddenSize * sizeof(float));
        }

        indiceGm.SetGlobalBuffer((__gm__ int32_t*)indice);
        indiceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        outputGm.SetGlobalBuffer((__gm__ COSTYPE*)q_out + sCoreOffset * this->qHiddenSize);
        outputKGm.SetGlobalBuffer((__gm__ COSTYPE*)k_out + sCoreOffset * this->kHiddenSize);
        outputVGm.SetGlobalBuffer((__gm__ COSTYPE*)v_out + sCoreOffset * this->vHiddenSize);
        vCacheGm.SetGlobalBuffer((__gm__ int8_t*)v_cache_out);
        kCacheGm.SetGlobalBuffer((__gm__ int8_t*)k_cache_out);

        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            if (this->hasBias == true) {
                biasGM.SetGlobalBuffer((__gm__ BIASTYPE*)bias);
            }
            if (this->hasAS == true) {
                asGM.SetGlobalBuffer((__gm__ float*)activation_scale + sCoreOffset);
            }
            wsGM.SetGlobalBuffer((__gm__ float*)weight_scale);
        }

        if constexpr (IsSameType<COSTYPE, half>::value && !IsSameType<XTYPE, int32_t>::value) {
            pipe.InitBuffer(qQueue, BUFFER_NUM, qDataNum * sizeof(half));
            pipe.InitBuffer(kQueue, BUFFER_NUM, kvDataNum * sizeof(half));
            pipe.InitBuffer(vQueue, BUFFER_NUM, kvDataNum * sizeof(half));
        } else {
            pipe.InitBuffer(qQueue, BUFFER_NUM, qDataNum * sizeof(float));
            pipe.InitBuffer(kQueue, BUFFER_NUM, kvDataNum * sizeof(float));
            pipe.InitBuffer(vQueue, BUFFER_NUM, kvDataNum * sizeof(float));
        }
        if constexpr (IsSameType<COSTYPE, bfloat16_t>::value) {
            cosUbOffset = OnceUBMaxS * this->hiddenSize;
        } else {
            cosUbOffset = 0;
        }
        if constexpr (IsSameType<XTYPE, bfloat16_t>::value) {
            qUbOffset = qDataNum;
            kUbOffset = kvDataNum;
        } else {
            qUbOffset = 0;
            kUbOffset = 0;
        }

        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            if (this->hasBias == true) {
                pipe.InitBuffer(biasQueue, this->inputHiddenSize * sizeof(int32_t));
            }

            if (this->hasAS == true) {
                // as需要32字节对齐，ws和bias已对齐
                int64_t asUbNum = (OnceUBMaxS + FP32_ONE_BLOCK_NUM - 1) / FP32_ONE_BLOCK_NUM * FP32_ONE_BLOCK_NUM;
                pipe.InitBuffer(asQueue, BUFFER_NUM, asUbNum * sizeof(float));
                this->asOffset = 0;
            }
            pipe.InitBuffer(wsQueue, this->inputHiddenSize * sizeof(float));
        }

        pipe.InitBuffer(sinQueue, BUFFER_NUM, OnceUBMaxS * this->hiddenSize * sizeof(typename queType<COSTYPE>::type));
        pipe.InitBuffer(cosQueue, BUFFER_NUM, OnceUBMaxS * this->hiddenSize * sizeof(typename queType<COSTYPE>::type));
        pipe.InitBuffer(qOutQueue, BUFFER_NUM, qDataNum * sizeof(typename queType<COSTYPE>::type));

        pipe.InitBuffer(scaleKQueue, this->kHiddenSize * sizeof(float));
        pipe.InitBuffer(scaleVQueue, this->kHiddenSize * sizeof(float));

        pipe.InitBuffer(indiceQueue, BUFFER_NUM, this->indiceUbNum * sizeof(int32_t));

        pipe.InitBuffer(tmpBuf0, kvDataNum * sizeof(float));

        pipe.InitBuffer(kCacheOutQueue, BUFFER_NUM, cacheBufNum * sizeof(int8_t));
        pipe.InitBuffer(vCacheOutQueue, BUFFER_NUM, cacheBufNum * sizeof(int8_t));

        eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
        eventIdMTE2ToAS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
        eventIdVToMTE3V = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMTE3K = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        eventIdMTE3ToMTE2K = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        eventIdMTE2ToMTE3V = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        eventIdMTE3ToMTE2V = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        halfSize = hiddenSize / ROPE_LAST_DIM_SPLIT;
        oneRepeatNum = ONE_REPEAT_SIZE / sizeof(typename queType<COSTYPE>::type);
        repeatStride = hiddenSize * sizeof(typename queType<COSTYPE>::type) / BLOCK_SIZE;
        hRepeatTime = this->kvHeadNum * hiddenSize / FP32_ONE_REPEAT_NUM;
        hRepeatAligned = (this->kvHeadNum * hiddenSize % FP32_ONE_REPEAT_NUM == 0) ? 1 : 0;
        repeatStrideF = this->kvHeadNum * hiddenSize * sizeof(float) / BLOCK_SIZE;
    }

    __aicore__ inline void Process()
    {
        quantScaleKUb = scaleKQueue.Get<float>();
        quantScaleVUb = scaleVQueue.Get<float>();

        DataCopy(quantScaleKUb, quantScaleKGm, this->kHiddenSize);
        DataCopy(quantScaleVUb, quantScaleVGm, this->kHiddenSize);

        if (this->hasOffsetK == true) {
            offsetKUb = offsetKQueue.Get<float>();
            DataCopy(offsetKUb, quantOffsetKGm, this->kHiddenSize);
        }
        if (this->hasOffsetV == true) {
            offsetVUb = offsetVQueue.Get<float>();
            DataCopy(offsetVUb, quantOffsetVGm, this->kHiddenSize);
        }

        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            wsUb = wsQueue.Get<float>();
            DataCopy(wsUb, wsGM, inputHiddenSize);
            qWsUb = wsUb;
            kWsUb = wsUb[qHiddenSize];
            vWsUb = wsUb[qHiddenSize + kHiddenSize];
            if (this->hasBias == true) {
                biasUb = biasQueue.Get<BIASTYPE>();
                if constexpr (IsSameType<BIASTYPE, bfloat16_t>::value || IsSameType<BIASTYPE, half>::value) {
                    biasUbOffset = inputHiddenSize;
                } else {
                    biasUbOffset = 0;
                }

                DataCopy(biasUb[biasUbOffset], biasGM, inputHiddenSize);
            }
        }

        PipeBarrier<PIPE_ALL>();

        if (IsSameType<XTYPE, int32_t>::value && IsSameType<BIASTYPE, float>::value == false &&
            IsSameType<BIASTYPE, int32_t>::value == false && this->hasBias == true) {
            LocalTensor<float> biasInUb = biasUb.template ReinterpretCast<float>();
            Cast(biasInUb, biasUb[biasUbOffset], RoundMode::CAST_NONE, inputHiddenSize);
        }

        if (IsSameType<XTYPE, int32_t>::value && this->hasBias == true) {
            qBiasUb = biasUb;
            if constexpr (IsSameType<BIASTYPE, bfloat16_t>::value || IsSameType<BIASTYPE, half>::value) {
                kBiasUb = biasUb[qHiddenSize * OFFSET_OVERRIDE];
                vBiasUb = biasUb[(qHiddenSize + kHiddenSize) * OFFSET_OVERRIDE];
            } else {
                kBiasUb = biasUb[qHiddenSize];
                vBiasUb = biasUb[qHiddenSize + kHiddenSize];
            }
        }

        if (this->ifKVout == true) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2K);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2V);
        }

        dataCopyParamsK_ = {(uint16_t)OnceUBMaxS, static_cast<uint32_t>(this->kHiddenSize * sizeof(XTYPE)),
                            static_cast<uint32_t>((this->vHiddenSize + this->qHiddenSize) * sizeof(XTYPE)), 0, 0};
        dataCopyParamsQ_ = {(uint16_t)OnceUBMaxS, static_cast<uint32_t>(this->qHiddenSize * sizeof(XTYPE)),
                            static_cast<uint32_t>((this->kHiddenSize + this->vHiddenSize) * sizeof(XTYPE)), 0, 0};
        dataCopyParamsV_ = {(uint16_t)OnceUBMaxS, static_cast<uint32_t>(this->vHiddenSize * sizeof(XTYPE)),
                            static_cast<uint32_t>((this->qHiddenSize + this->kHiddenSize) * sizeof(XTYPE)), 0, 0};

        for (uint32_t ubLoopIndex = 0; ubLoopIndex < coreCalcSLoop - 1; ubLoopIndex++) {
            computeQKV(OnceUBMaxS);
        }

        dataCopyParamsK_ = {(uint16_t)coreCalcSLastNum, static_cast<uint32_t>(this->kHiddenSize * sizeof(XTYPE)),
                            static_cast<uint32_t>((this->vHiddenSize + this->qHiddenSize) * sizeof(XTYPE)), 0, 0};
        dataCopyParamsQ_ = {(uint16_t)coreCalcSLastNum, static_cast<uint32_t>(this->qHiddenSize * sizeof(XTYPE)),
                            static_cast<uint32_t>((this->kHiddenSize + this->vHiddenSize) * sizeof(XTYPE)), 0, 0};
        dataCopyParamsV_ = {(uint16_t)coreCalcSLastNum, static_cast<uint32_t>(this->vHiddenSize * sizeof(XTYPE)),
                            static_cast<uint32_t>((this->qHiddenSize + this->kHiddenSize) * sizeof(XTYPE)), 0, 0};
        computeQKV(coreCalcSLastNum);
        if (this->ifKVout == true) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2K);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2V);
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventIdMTE2ToS);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventIdMTE2ToAS);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3V);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3K);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2K);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3V);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2V);
    }

private:
    __aicore__ inline void copyCacheRows(GlobalTensor<int8_t>& cacheGm, const LocalTensor<int8_t>& src,
                                         int64_t dstOffset, int64_t srcIndex, int64_t sNum)
    {
        if (this->cacheRowAlign == this->cacheRowElems) {
            DataCopyExtParams cacheCopyParams = {1, static_cast<uint32_t>(sNum * this->cacheRowElems), 0, 0, 0};
            DataCopyPad(cacheGm[dstOffset], src[srcIndex * this->cacheRowAlign], cacheCopyParams);
        } else {
            for (int64_t i = 0; i < sNum; i++) {
                DataCopyExtParams cacheCopyParams = {1, static_cast<uint32_t>(this->cacheRowElems), 0, 0, 0};
                DataCopyPad(cacheGm[dstOffset + i * this->cacheRowElems], src[(srcIndex + i) * this->cacheRowAlign],
                            cacheCopyParams);
            }
        }
    }

    __aicore__ inline void copyOutcache(int64_t bIndex, int64_t sNum, int64_t sIndex, int64_t sIndexUb,
                                        LocalTensor<int8_t> vOut, LocalTensor<int8_t> kOut)
    {
        int64_t index = indiceUb.GetValue(bIndex);
        // cache写回长度不保证32字节对齐，使用DataCopyPad避免对齐异常；
        // 当行宽不是32B对齐时，UB源地址也要按行补齐，否则部分芯片会报VEC_ERROR
        if (this->isPA == true) {
            // page模式下indices为绝对页号，文档值域为[0, C_1*C_2)，越界时不写回
            if (index >= 0 && index < this->cacheBatchNum * this->cacheSeqlen) {
                bool hasLaterSameBlock = false;
                int64_t curToken = this->sCoreOffset + sIndexUb;
                for (int64_t j = curToken + sNum; j < this->batch; j++) {
                    if (indiceGm.GetValue(j) == index) {
                        hasLaterSameBlock = true;
                        break;
                    }
                }
                if (!hasLaterSameBlock) {
                    copyCacheRows(vCacheGm, vOut, index * this->cacheRowElems, sIndexUb, sNum);
                    copyCacheRows(kCacheGm, kOut, index * this->cacheRowElems, sIndexUb, sNum);
                }
            }
        } else {
            // contiguous模式下indices为行内起始偏移，文档值域为[0, C_2-S]，越界时不写回
            if (index >= 0 && index + this->seqlen <= this->cacheSeqlen) {
                int64_t dstOffset = ((bOffset + bIndex) * this->cacheSeqlen + index + sIndex) * this->cacheRowElems;
                copyCacheRows(vCacheGm, vOut, dstOffset, sIndexUb, sNum);
                copyCacheRows(kCacheGm, kOut, dstOffset, sIndexUb, sNum);
            }
        }
    }

    __aicore__ inline void computeQKV(int64_t onceS)
    {
        using RopeT = typename queType<COSTYPE>::type;
        // rope 的 Lo/Hi 两半按 halfSize 偏移做向量运算，要求该偏移按 32B 对齐；
        // D%16!=0 已被 tiling 拦截，但仍可能是 16B 对齐（D%32==16），此时走标量回退。
        const bool ropeUnaligned =
            (halfSize * static_cast<int64_t>(sizeof(RopeT)) % static_cast<int64_t>(BLOCK_SIZE)) != 0;
        int64_t cosOffset = sOffset * this->hiddenSize;

        LocalTensor<COSTYPE> cosInUb = cosQueue.AllocTensor<COSTYPE>();
        LocalTensor<COSTYPE> sinInUb = sinQueue.AllocTensor<COSTYPE>();

        DataCopy(cosInUb[cosUbOffset], cosGm[cosOffset], onceS * this->hiddenSize);
        DataCopy(sinInUb[cosUbOffset], sinGm[cosOffset], onceS * this->hiddenSize);

        sinQueue.EnQue(sinInUb);
        cosQueue.EnQue(cosInUb);
        sinInUb = sinQueue.DeQue<COSTYPE>();
        cosInUb = cosQueue.DeQue<COSTYPE>();

        indiceUb = indiceQueue.AllocTensor<int32_t>();
        bOffset = GetDiv(sCoreOffset, seqlen);
        // 拷贝长度取本批最多跨越的batch数上界，且以剩余batch数与UB容量为界，避免GM越界读与UB越界写
        int64_t onceB = GetCeilInt(onceS, seqlen);
        int64_t batchRemain = this->batch - this->bOffset;
        int64_t indiceCopyNum = AscendC::Std::min(onceB + 1, batchRemain);
        indiceCopyNum = indiceCopyNum < 0 ? 0 : indiceCopyNum;
        dataCopyParamsIndice_ = {1, static_cast<uint32_t>(indiceCopyNum * static_cast<int64_t>(sizeof(int32_t))), 0, 0,
                                 0};

        DataCopyPad(indiceUb, indiceGm[bOffset], dataCopyParamsIndice_, padParamsIndice_);
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

        if (IsSameType<XTYPE, int32_t>::value && this->hasAS == true) {
            DataCopyExtParams dataCopyParamsAs;
            DataCopyPadExtParams<float> padParamsAs;
            dataCopyParamsAs.blockCount = 1;
            dataCopyParamsAs.blockLen = onceS * sizeof(float);
            dataCopyParamsAs.srcStride = 0;
            dataCopyParamsAs.dstStride = 0;
            padParamsAs.isPad = false;
            padParamsAs.paddingValue = static_cast<float>(0);
            padParamsAs.leftPadding = 0;
            padParamsAs.rightPadding = 0;
            asUb = asQueue.AllocTensor<float>();
            DataCopyPad(asUb, asGM[asOffset], dataCopyParamsAs, padParamsAs);
            SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToAS);
        }

        LocalTensor<XTYPE> qCopyInUb = qQueue.AllocTensor<XTYPE>();
        DataCopyPad(qCopyInUb[qUbOffset], inputGm[sOffset * inputHiddenSize], dataCopyParamsQ_, padParamsQ_);

        if (this->ifKVout == true) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2K);
        }

        LocalTensor<XTYPE> kCopyInUb = kQueue.AllocTensor<XTYPE>();
        DataCopyPad(kCopyInUb[kUbOffset], inputGm[sOffset * inputHiddenSize + qHiddenSize], dataCopyParamsK_,
                    padParamsK_);

        kQueue.EnQue(kCopyInUb);
        qQueue.EnQue(qCopyInUb);

        LocalTensor<typename queType<COSTYPE>::type> cosUb =
            cosInUb.template ReinterpretCast<typename queType<COSTYPE>::type>();
        LocalTensor<typename queType<COSTYPE>::type> sinUb =
            sinInUb.template ReinterpretCast<typename queType<COSTYPE>::type>();
        if constexpr (IsSameType<COSTYPE, bfloat16_t>::value) {
            Cast(sinUb, sinInUb[cosUbOffset], RoundMode::CAST_NONE, onceS * this->hiddenSize);
            Cast(cosUb, cosInUb[cosUbOffset], RoundMode::CAST_NONE, onceS * this->hiddenSize);
            PipeBarrier<PIPE_V>();
        }

        if (this->hiddenSize > ROPE_LAST_DIM_SPLIT * oneRepeatNum || repeatStride > MAX_UINT8 || onceS > MAX_UINT8) {
            for (uint64_t r = 0; r < onceS; r++) {
                Muls(sinUb[r * hiddenSize], sinUb[r * hiddenSize], (typename queType<COSTYPE>::type) - 1.0, halfSize);
            }
        } else {
            Muls(sinUb, sinUb, (typename queType<COSTYPE>::type) - 1.0, halfSize, onceS,
                 {1, 1, repeatStride, repeatStride});
        }
        int64_t firstS = GetRem(sCoreOffset, seqlen);
        int64_t firstStepSeq = seqlen - firstS < onceS ? seqlen - firstS : onceS;
        int64_t seqLoop = GetDiv((onceS - firstStepSeq), seqlen);
        int64_t lastStepSeq = GetRem((onceS - firstStepSeq), seqlen);
        int64_t sIndexUb = 0;
        PipeBarrier<PIPE_V>();

        kCopyInUb = kQueue.DeQue<XTYPE>();
        qCopyInUb = qQueue.DeQue<XTYPE>();

        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            if (this->hasAS == true) {
                // dequantUb中标量读取asUb前需等待MTE2拷贝完成
                WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToAS);
            }
            dequantUb(kCopyInUb[kUbOffset], kBiasUb, kWsUb, onceS, this->kHiddenSize);
            dequantUb(qCopyInUb[qUbOffset], qBiasUb, qWsUb, onceS, this->qHiddenSize);

            if constexpr (IsSameType<COSTYPE, half>::value) {
                LocalTensor<COSTYPE> kInUb = kCopyInUb.template ReinterpretCast<COSTYPE>();
                LocalTensor<COSTYPE> qInUb = qCopyInUb.template ReinterpretCast<COSTYPE>();
                LocalTensor<float> kInF32Ub = kCopyInUb.template ReinterpretCast<float>();
                LocalTensor<float> qInF32Ub = qCopyInUb.template ReinterpretCast<float>();
                PipeBarrier<PIPE_V>();
                Cast(kInUb, kInF32Ub, RoundMode::CAST_RINT, onceS * this->kHiddenSize);
                Cast(qInUb, qInF32Ub, RoundMode::CAST_RINT, onceS * this->qHiddenSize);
                PipeBarrier<PIPE_V>();
            }
        }
        LocalTensor<typename queType<COSTYPE>::type> kInUb =
            kCopyInUb.template ReinterpretCast<typename queType<COSTYPE>::type>();
        LocalTensor<typename queType<COSTYPE>::type> qInUb =
            qCopyInUb.template ReinterpretCast<typename queType<COSTYPE>::type>();
        if constexpr (IsSameType<XTYPE, bfloat16_t>::value) {
            Cast(kInUb, kCopyInUb[kUbOffset], RoundMode::CAST_NONE, onceS * this->kHiddenSize);
            Cast(qInUb, qCopyInUb[qUbOffset], RoundMode::CAST_NONE, onceS * this->qHiddenSize);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> tmpBufFP32 = tmpBuf0.Get<float>();
        LocalTensor<typename queType<COSTYPE>::type> qOutUb = qOutQueue.AllocTensor<typename queType<COSTYPE>::type>();

        // k rope
        LocalTensor<typename queType<COSTYPE>::type> kOutUbF16 =
            tmpBufFP32.template ReinterpretCast<typename queType<COSTYPE>::type>();

        if (!ropeUnaligned && halfSize <= oneRepeatNum && kvHeadNum == 1 && repeatStride < MAX_UINT8 &&
            onceS < MAX_UINT8) {
            AscendC::SetMaskNorm();
            SetVectorMask<typename queType<COSTYPE>::type>(halfSize);

            Mul<typename queType<COSTYPE>::type, false>(kOutUbF16, kInUb[halfSize], sinUb, halfSize, onceS,
                                                        {1, 1, 1, repeatStride, repeatStride, repeatStride});
            Mul<typename queType<COSTYPE>::type, false>(kOutUbF16[halfSize], kInUb, sinUb[halfSize], halfSize, onceS,
                                                        {1, 1, 1, repeatStride, repeatStride, repeatStride});
            PipeBarrier<PIPE_V>();
            Mul<typename queType<COSTYPE>::type, false>(kInUb, kInUb, cosUb, halfSize, onceS,
                                                        {1, 1, 1, repeatStride, repeatStride, repeatStride});
            Mul<typename queType<COSTYPE>::type, false>(kInUb[halfSize], kInUb[halfSize], cosUb[halfSize], halfSize,
                                                        onceS, {1, 1, 1, repeatStride, repeatStride, repeatStride});
        } else if (!ropeUnaligned) {
            for (uint64_t loopS = 0; loopS < onceS; loopS++) {
                for (uint64_t loopH = 0; loopH < this->kvHeadNum; loopH++) {
                    Mul(kOutUbF16[(loopS * this->kvHeadNum + loopH) * this->hiddenSize],
                        kInUb[(loopS * this->kvHeadNum + loopH) * this->hiddenSize + halfSize],
                        sinUb[loopS * this->hiddenSize], halfSize);
                    Mul(kOutUbF16[(loopS * this->kvHeadNum + loopH) * this->hiddenSize + halfSize],
                        kInUb[(loopS * this->kvHeadNum + loopH) * this->hiddenSize],
                        sinUb[loopS * this->hiddenSize + halfSize], halfSize);
                }
            }
            PipeBarrier<PIPE_V>();
            for (uint64_t loopS = 0; loopS < onceS; loopS++) {
                for (uint64_t loopH = 0; loopH < this->kvHeadNum; loopH++) {
                    Mul(kInUb[(loopS * this->kvHeadNum + loopH) * this->hiddenSize],
                        kInUb[(loopS * this->kvHeadNum + loopH) * this->hiddenSize], cosUb[loopS * this->hiddenSize],
                        hiddenSize);
                }
            }
        } else {
            // 非 32B 对齐：逐元素标量计算，避免 half 类型 [halfSize] 偏移的向量访问
            // 标量 GetValue/SetValue 走 S 流水，必须与前置 MTE2/Vector 和后置 Vector 显式同步
            PipeBarrier<PIPE_ALL>();
            for (uint64_t loopS = 0; loopS < onceS; loopS++) {
                for (uint64_t loopH = 0; loopH < this->kvHeadNum; loopH++) {
                    int64_t base = static_cast<int64_t>((loopS * this->kvHeadNum + loopH) * this->hiddenSize);
                    int64_t cosBase = static_cast<int64_t>(loopS * this->hiddenSize);
                    for (int64_t i = 0; i < halfSize; ++i) {
                        float lo = static_cast<float>(kInUb.GetValue(base + i));
                        float hi = static_cast<float>(kInUb.GetValue(base + halfSize + i));
                        float sinLo = static_cast<float>(sinUb.GetValue(cosBase + i));
                        float sinHi = static_cast<float>(sinUb.GetValue(cosBase + halfSize + i));
                        float cosLo = static_cast<float>(cosUb.GetValue(cosBase + i));
                        float cosHi = static_cast<float>(cosUb.GetValue(cosBase + halfSize + i));
                        kOutUbF16.SetValue(base + i, static_cast<RopeT>(hi * sinLo));
                        kOutUbF16.SetValue(base + halfSize + i, static_cast<RopeT>(lo * sinHi));
                        kInUb.SetValue(base + i, static_cast<RopeT>(lo * cosLo));
                        kInUb.SetValue(base + halfSize + i, static_cast<RopeT>(hi * cosHi));
                    }
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<typename queType<COSTYPE>::type, float>::value) {
            Add(tmpBufFP32, kOutUbF16, kInUb, onceS * this->kHiddenSize);
        } else {
            Add(kInUb, kOutUbF16, kInUb, onceS * this->kHiddenSize);
            PipeBarrier<PIPE_V>();
            Cast(tmpBufFP32, kInUb, RoundMode::CAST_NONE, onceS * this->kHiddenSize);
        }

        if (this->ifKVout == true) {
            auto kRealOut = kInUb.template ReinterpretCast<COSTYPE>();
            if constexpr (IsSameType<COSTYPE, bfloat16_t>::value) {
                PipeBarrier<PIPE_V>();
                Cast(kRealOut, tmpBufFP32, RoundMode::CAST_RINT, onceS * this->kHiddenSize);
            }
            SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3K);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3K);
            DataCopy(outputKGm[sOffset * this->kHiddenSize], kRealOut, onceS * this->kHiddenSize);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2K);
        }
        // k quant

        kQueue.FreeTensor(kInUb);

        PipeBarrier<PIPE_V>();

        if (onceS * this->kvHeadNum < MAX_UINT8 && repeatStrideF < MAX_UINT8 && this->hRepeatAligned == 1) {
            AscendC::SetMaskNorm();
            SetVectorMask<float>(FP32_ONE_REPEAT_NUM);
            for (uint32_t r = 0; r < hRepeatTime; r++) {
                Div<float, false>(
                    tmpBufFP32[r * FP32_ONE_REPEAT_NUM], tmpBufFP32[r * FP32_ONE_REPEAT_NUM],
                    quantScaleKUb[r * FP32_ONE_REPEAT_NUM], FP32_ONE_REPEAT_NUM, onceS,
                    {1, 1, 1, static_cast<uint8_t>(repeatStrideF), static_cast<uint8_t>(repeatStrideF), 0});
            }
            PipeBarrier<PIPE_V>();
            if (this->hasOffsetK == true) {
                for (uint32_t r = 0; r < hRepeatTime; r++) {
                    Add<float, false>(
                        tmpBufFP32[r * FP32_ONE_REPEAT_NUM], tmpBufFP32[r * FP32_ONE_REPEAT_NUM],
                        offsetKUb[r * FP32_ONE_REPEAT_NUM], FP32_ONE_REPEAT_NUM, onceS,
                        {1, 1, 1, static_cast<uint8_t>(repeatStrideF), static_cast<uint8_t>(repeatStrideF), 0});
                }
            }
        } else {
            for (uint32_t r = 0; r < onceS; r++) {
                Div(tmpBufFP32[r * this->hiddenSize * this->kvHeadNum],
                    tmpBufFP32[r * this->hiddenSize * this->kvHeadNum], quantScaleKUb,
                    this->hiddenSize * this->kvHeadNum);
            }
            PipeBarrier<PIPE_V>();
            if (this->hasOffsetK == true) {
                for (uint32_t r = 0; r < onceS; r++) {
                    Add(tmpBufFP32[r * this->hiddenSize * this->kvHeadNum],
                        tmpBufFP32[r * this->hiddenSize * this->kvHeadNum], offsetKUb,
                        this->hiddenSize * this->kvHeadNum);
                }
            }
        }

        PipeBarrier<PIPE_V>();
        LocalTensor<half> kOutF16 = tmpBufFP32.template ReinterpretCast<half>();
        LocalTensor<int8_t> kOutUbS8 = kCacheOutQueue.AllocTensor<int8_t>();

        LocalTensor<int16_t> kOutUbS16 = tmpBufFP32.template ReinterpretCast<int16_t>();
        Cast(kOutUbS16, tmpBufFP32, RoundMode::CAST_RINT, onceS * this->kHiddenSize);
        PipeBarrier<PIPE_V>();
        Cast(kOutF16, kOutUbS16, RoundMode::CAST_NONE, onceS * this->kHiddenSize);
        PipeBarrier<PIPE_V>();
        if (this->cacheRowAlign == this->cacheRowElems) {
            Cast(kOutUbS8, kOutF16, RoundMode::CAST_NONE, onceS * this->kHiddenSize);
        } else {
            for (uint32_t r = 0; r < onceS; r++) {
                Cast(kOutUbS8[r * this->cacheRowAlign], kOutF16[r * this->kHiddenSize], RoundMode::CAST_NONE,
                     this->kHiddenSize);
            }
        }
        PipeBarrier<PIPE_V>();
        kCacheOutQueue.EnQue(kOutUbS8);
        LocalTensor<int8_t> kOut = kCacheOutQueue.DeQue<int8_t>();

        if (this->ifKVout == true) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2V);
        }
        LocalTensor<XTYPE> vCopyInUb = vQueue.AllocTensor<XTYPE>();
        DataCopyPad(vCopyInUb, inputGm[sOffset * inputHiddenSize + this->qHiddenSize + this->kHiddenSize],
                    dataCopyParamsV_, padParamsV_);
        if (this->ifKVout == true) {
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3V);
        }
        vQueue.EnQue(vCopyInUb);

        // q rope

        if (!ropeUnaligned && halfSize <= oneRepeatNum && repeatStride < MAX_UINT8 && this->qHeadNum < MAX_UINT8) {
            AscendC::SetMaskNorm();
            SetVectorMask<typename queType<COSTYPE>::type>(halfSize);
            for (uint64_t r = 0; r < onceS; r++) {
                Mul<typename queType<COSTYPE>::type, false>(
                    qOutUb[r * this->qHiddenSize], qInUb[r * this->qHiddenSize + halfSize], sinUb[r * this->hiddenSize],
                    halfSize, this->qHeadNum, {1, 1, 1, repeatStride, repeatStride, 0});
                Mul<typename queType<COSTYPE>::type, false>(qOutUb[r * this->qHiddenSize + halfSize],
                                                            qInUb[r * this->qHiddenSize],
                                                            sinUb[r * this->hiddenSize + halfSize], halfSize,
                                                            this->qHeadNum, {1, 1, 1, repeatStride, repeatStride, 0});
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < onceS; r++) {
                Mul<typename queType<COSTYPE>::type, false>(qInUb[r * this->qHiddenSize], qInUb[r * this->qHiddenSize],
                                                            cosUb[r * this->hiddenSize], halfSize, this->qHeadNum,
                                                            {1, 1, 1, repeatStride, repeatStride, 0});
                Mul<typename queType<COSTYPE>::type, false>(qInUb[r * this->qHiddenSize + halfSize],
                                                            qInUb[r * this->qHiddenSize + halfSize],
                                                            cosUb[r * this->hiddenSize + halfSize], halfSize,
                                                            this->qHeadNum, {1, 1, 1, repeatStride, repeatStride, 0});
            }
        } else if (!ropeUnaligned) {
            for (uint64_t loopS = 0; loopS < onceS; loopS++) {
                for (uint64_t loopH = 0; loopH < this->qHeadNum; loopH++) {
                    Mul(qOutUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize],
                        qInUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize + halfSize],
                        sinUb[loopS * this->hiddenSize], halfSize);
                    Mul(qOutUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize + halfSize],
                        qInUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize],
                        sinUb[loopS * this->hiddenSize + halfSize], halfSize);
                }
            }
            PipeBarrier<PIPE_V>();
            for (uint64_t loopS = 0; loopS < onceS; loopS++) {
                for (uint64_t loopH = 0; loopH < this->qHeadNum; loopH++) {
                    Mul(qInUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize],
                        qInUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize], cosUb[loopS * this->hiddenSize],
                        hiddenSize);
                }
            }
        } else {
            // 非 32B 对齐：逐元素标量计算，避免 half 类型 [halfSize] 偏移的向量访问
            // 标量 GetValue/SetValue 走 S 流水，必须与前置 MTE2/Vector 和后置 Vector 显式同步
            PipeBarrier<PIPE_ALL>();
            for (uint64_t loopS = 0; loopS < onceS; loopS++) {
                for (uint64_t loopH = 0; loopH < this->qHeadNum; loopH++) {
                    int64_t base = static_cast<int64_t>((loopS * this->qHeadNum + loopH) * this->hiddenSize);
                    int64_t cosBase = static_cast<int64_t>(loopS * this->hiddenSize);
                    for (int64_t i = 0; i < halfSize; ++i) {
                        float lo = static_cast<float>(qInUb.GetValue(base + i));
                        float hi = static_cast<float>(qInUb.GetValue(base + halfSize + i));
                        float sinLo = static_cast<float>(sinUb.GetValue(cosBase + i));
                        float sinHi = static_cast<float>(sinUb.GetValue(cosBase + halfSize + i));
                        float cosLo = static_cast<float>(cosUb.GetValue(cosBase + i));
                        float cosHi = static_cast<float>(cosUb.GetValue(cosBase + halfSize + i));
                        qOutUb.SetValue(base + i, static_cast<RopeT>(hi * sinLo));
                        qOutUb.SetValue(base + halfSize + i, static_cast<RopeT>(lo * sinHi));
                        qInUb.SetValue(base + i, static_cast<RopeT>(lo * cosLo));
                        qInUb.SetValue(base + halfSize + i, static_cast<RopeT>(hi * cosHi));
                    }
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
        sinQueue.FreeTensor(sinUb);
        cosQueue.FreeTensor(cosUb);
        PipeBarrier<PIPE_V>();
        Add(qOutUb, qOutUb, qInUb, onceS * this->qHiddenSize);
        LocalTensor<COSTYPE> qRealOut = qOutUb.template ReinterpretCast<COSTYPE>();
        if constexpr (IsSameType<COSTYPE, bfloat16_t>::value) {
            PipeBarrier<PIPE_V>();
            Cast(qRealOut, qOutUb, RoundMode::CAST_RINT, onceS * this->qHiddenSize);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
        qQueue.FreeTensor(qInUb);
        qOutQueue.EnQue(qOutUb);
        auto qOut = qOutQueue.DeQue<typename queType<COSTYPE>::type>();
        qRealOut = qOut.template ReinterpretCast<COSTYPE>();

        DataCopy(outputGm[sOffset * this->qHiddenSize], qRealOut, onceS * this->qHiddenSize);
        qOutQueue.FreeTensor(qOut);

        // v quant
        LocalTensor<int8_t> vOutUbS8 = vCacheOutQueue.AllocTensor<int8_t>();
        vCopyInUb = vQueue.DeQue<XTYPE>();
        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            dequantUb(vCopyInUb, vBiasUb, vWsUb, onceS, this->vHiddenSize);
        }
        if (IsSameType<XTYPE, int32_t>::value && this->hasAS == true) {
            asQueue.FreeTensor(asUb);
            this->asOffset += onceS;
        }
        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            PipeBarrier<PIPE_V>();
            LocalTensor<float> vFloatUb = vCopyInUb.template ReinterpretCast<float>();
            DataCopy(tmpBufFP32, vFloatUb, onceS * this->vHiddenSize);
            PipeBarrier<PIPE_V>();
        }
        if (this->ifKVout == true) {
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3V);
            LocalTensor<COSTYPE> vCopyOutUb = vCopyInUb.template ReinterpretCast<COSTYPE>();

            if constexpr (IsSameType<XTYPE, int32_t>::value) {
                LocalTensor<float> vFloatUb = vCopyInUb.template ReinterpretCast<float>();
                Cast(vCopyOutUb, vFloatUb, RoundMode::CAST_RINT, onceS * this->vHiddenSize);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3V);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3V);
            }
            DataCopy(outputVGm[sOffset * this->vHiddenSize], vCopyOutUb, onceS * this->vHiddenSize);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2V);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<COSTYPE> vCopyOutUb = vCopyInUb.template ReinterpretCast<COSTYPE>();
        if constexpr (!IsSameType<XTYPE, int32_t>::value) {
            Cast(tmpBufFP32, vCopyOutUb, RoundMode::CAST_NONE, onceS * this->vHiddenSize);
        }

        vQueue.FreeTensor(vCopyInUb);
        PipeBarrier<PIPE_V>();

        if (onceS * this->kvHeadNum < MAX_UINT8 && repeatStrideF < MAX_UINT8 && this->hRepeatAligned == 1) {
            AscendC::SetMaskNorm();
            SetVectorMask<float>(FP32_ONE_REPEAT_NUM);
            for (uint32_t r = 0; r < hRepeatTime; r++) {
                Div<float, false>(
                    tmpBufFP32[r * FP32_ONE_REPEAT_NUM], tmpBufFP32[r * FP32_ONE_REPEAT_NUM],
                    quantScaleVUb[r * FP32_ONE_REPEAT_NUM], FP32_ONE_REPEAT_NUM, onceS,
                    {1, 1, 1, static_cast<uint8_t>(repeatStrideF), static_cast<uint8_t>(repeatStrideF), 0});
            }
            PipeBarrier<PIPE_V>();
            if (this->hasOffsetV == true) {
                for (uint32_t r = 0; r < hRepeatTime; r++) {
                    Add<float, false>(
                        tmpBufFP32[r * FP32_ONE_REPEAT_NUM], tmpBufFP32[r * FP32_ONE_REPEAT_NUM],
                        offsetVUb[r * FP32_ONE_REPEAT_NUM], FP32_ONE_REPEAT_NUM, onceS,
                        {1, 1, 1, static_cast<uint8_t>(repeatStrideF), static_cast<uint8_t>(repeatStrideF), 0});
                }
            }
        } else {
            for (uint32_t r = 0; r < onceS; r++) {
                Div(tmpBufFP32[r * this->hiddenSize * this->kvHeadNum],
                    tmpBufFP32[r * this->hiddenSize * this->kvHeadNum], quantScaleVUb,
                    this->hiddenSize * this->kvHeadNum);
            }
            PipeBarrier<PIPE_V>();
            if (this->hasOffsetV == true) {
                for (uint32_t r = 0; r < onceS; r++) {
                    Add(tmpBufFP32[r * this->hiddenSize * this->kvHeadNum],
                        tmpBufFP32[r * this->hiddenSize * this->kvHeadNum], offsetVUb,
                        this->hiddenSize * this->kvHeadNum);
                }
            }
        }

        PipeBarrier<PIPE_V>();

        LocalTensor<int16_t> vOutUbS16 = tmpBufFP32.template ReinterpretCast<int16_t>();
        Cast(vOutUbS16, tmpBufFP32, RoundMode::CAST_RINT, onceS * this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> vOutUbF16 = tmpBufFP32.template ReinterpretCast<half>();
        Cast(vOutUbF16, vOutUbS16, RoundMode::CAST_NONE, onceS * this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        if (this->cacheRowAlign == this->cacheRowElems) {
            Cast(vOutUbS8, vOutUbF16, RoundMode::CAST_NONE, onceS * this->kvHeadNum * this->hiddenSize);
        } else {
            for (uint32_t r = 0; r < onceS; r++) {
                Cast(vOutUbS8[r * this->cacheRowAlign], vOutUbF16[r * this->kvHeadNum * this->hiddenSize],
                     RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);
            }
        }
        PipeBarrier<PIPE_V>();
        vCacheOutQueue.EnQue(vOutUbS8);
        LocalTensor<int8_t> vOut = vCacheOutQueue.DeQue<int8_t>();

        PipeBarrier<PIPE_V>();

        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

        copyOutcache(0, firstStepSeq, firstS, sIndexUb, vOut, kOut);
        sIndexUb += firstStepSeq;

        for (uint32_t seqLoopIndex = 1; seqLoopIndex < seqLoop + 1; seqLoopIndex++) {
            copyOutcache(seqLoopIndex, seqlen, 0, sIndexUb, vOut, kOut);
            sIndexUb += seqlen;
        }
        if (lastStepSeq != 0) {
            copyOutcache(seqLoop + 1, lastStepSeq, 0, sIndexUb, vOut, kOut);
        }

        indiceQueue.FreeTensor(indiceUb);
        kCacheOutQueue.FreeTensor(kOut);
        vCacheOutQueue.FreeTensor(vOut);

        sCoreOffset += onceS;
        sOffset += onceS;
    }

    __aicore__ inline void dequantUb(LocalTensor<XTYPE> copyInUb, LocalTensor<BIASTYPE> curBiasUb,
                                     LocalTensor<float> curWsUb, int64_t onceS, int64_t calcSize)
    {
        int64_t offset = 0;
        PipeBarrier<PIPE_ALL>();
        if (this->hasBias == true) {
            if constexpr (IsSameType<BIASTYPE, int32_t>::value) {
                for (uint32_t r = 0; r < onceS; r++) {
                    offset = calcSize * r;
                    Add(copyInUb[offset], copyInUb[offset], curBiasUb, calcSize);
                }

                PipeBarrier<PIPE_V>();

                LocalTensor<float> copyInFloatUb = copyInUb.template ReinterpretCast<float>();
                Cast(copyInFloatUb, copyInUb, RoundMode::CAST_RINT, onceS * calcSize);
                PipeBarrier<PIPE_V>();

                for (uint32_t r = 0; r < onceS; r++) {
                    offset = calcSize * r;
                    Mul(copyInFloatUb[offset], copyInFloatUb[offset], curWsUb, calcSize);
                }
                PipeBarrier<PIPE_V>();

                if (this->hasAS == true) {
                    for (uint32_t r = 0; r < onceS; r++) {
                        offset = calcSize * r;
                        Muls(copyInFloatUb[offset], copyInFloatUb[offset], asUb.GetValue(r), calcSize);
                    }
                }
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<float> copyInFloatUb = copyInUb.template ReinterpretCast<float>();
                Cast(copyInFloatUb, copyInUb, RoundMode::CAST_RINT, onceS * calcSize);
                PipeBarrier<PIPE_V>();

                for (uint32_t r = 0; r < onceS; r++) {
                    offset = calcSize * r;
                    Mul(copyInFloatUb[offset], copyInFloatUb[offset], curWsUb, calcSize);
                }
                PipeBarrier<PIPE_V>();

                if (this->hasAS == true) {
                    for (uint32_t r = 0; r < onceS; r++) {
                        offset = calcSize * r;
                        Muls(copyInFloatUb[offset], copyInFloatUb[offset], asUb.GetValue(r), calcSize);
                    }
                }
                PipeBarrier<PIPE_V>();
                LocalTensor<float> curBiasFloatUb = curBiasUb.template ReinterpretCast<float>();
                if (this->hasBias == true) {
                    for (uint32_t r = 0; r < onceS; r++) {
                        offset = calcSize * r;
                        Add(copyInFloatUb[offset], copyInFloatUb[offset], curBiasFloatUb, calcSize);
                    }
                }
                PipeBarrier<PIPE_V>();
            }
        } else {
            LocalTensor<float> copyInFloatUb = copyInUb.template ReinterpretCast<float>();

            Cast(copyInFloatUb, copyInUb, RoundMode::CAST_RINT, onceS * calcSize);

            PipeBarrier<PIPE_V>();

            for (uint32_t r = 0; r < onceS; r++) {
                offset = calcSize * r;
                Mul(copyInFloatUb[offset], copyInFloatUb[offset], curWsUb, calcSize);
            }
            PipeBarrier<PIPE_V>();

            if (this->hasAS == true) {
                for (uint32_t r = 0; r < onceS; r++) {
                    offset = calcSize * r;
                    Muls(copyInFloatUb[offset], copyInFloatUb[offset], asUb.GetValue(r), calcSize);
                }
            }
            PipeBarrier<PIPE_V>();
        }
    }

    /* global memory address */
    GlobalTensor<XTYPE> inputGm;

    GlobalTensor<COSTYPE> cosGm;
    GlobalTensor<COSTYPE> sinGm;

    GlobalTensor<int32_t> indiceGm;

    GlobalTensor<BIASTYPE> biasGM;
    GlobalTensor<float> asGM;
    GlobalTensor<float> wsGM;

    GlobalTensor<float> quantScaleKGm;
    GlobalTensor<float> quantScaleVGm;
    GlobalTensor<float> quantOffsetKGm;
    GlobalTensor<float> quantOffsetVGm;

    GlobalTensor<COSTYPE> outputGm;
    GlobalTensor<COSTYPE> outputKGm;
    GlobalTensor<COSTYPE> outputVGm;

    GlobalTensor<int8_t> kCacheGm;
    GlobalTensor<int8_t> vCacheGm;
    LocalTensor<float> quantScaleKUb;
    LocalTensor<float> quantScaleVUb;

    LocalTensor<float> offsetKUb;
    LocalTensor<float> offsetVUb;
    LocalTensor<int32_t> indiceUb;
    LocalTensor<BIASTYPE> biasUb;
    LocalTensor<BIASTYPE> qBiasUb;
    LocalTensor<BIASTYPE> kBiasUb;
    LocalTensor<BIASTYPE> vBiasUb;
    LocalTensor<float> asUb;
    LocalTensor<float> wsUb;
    LocalTensor<float> qWsUb;
    LocalTensor<float> kWsUb;
    LocalTensor<float> vWsUb;

    /* variable */
    uint64_t halfSize;
    uint8_t repeatStride;
    int64_t cacheSeqlen;
    int64_t seqlen;
    int64_t frontCoreNum;
    int64_t qHeadNum;
    int64_t kvHeadNum;
    int64_t inputHiddenSize;
    int64_t hiddenSize;
    int64_t qHiddenSize;
    int64_t kHiddenSize;
    int64_t vHiddenSize;
    int64_t realCoreNum;
    int64_t blockFactor;
    int64_t tailCoreBlockFactor;
    int64_t hasOffsetK;
    int64_t hasOffsetV;
    int64_t indiceUbNum;
    int64_t hiddenSizeFp32Align;
    int64_t hiddenSizeFp16Align;
    int64_t hiddenSizeInt8Align;
    int64_t OnceUBMaxS;
    int64_t sOffset = 0;
    int64_t bOffset;
    int64_t coreCalcSNum;
    int64_t coreCalcSLastNum;
    int64_t coreCalcSLoop;
    int64_t LastLoopS;
    int64_t sCoreOffset;
    int64_t oneRepeatNum;
    int64_t qUbOffset;
    int64_t kUbOffset;
    int64_t cosUbOffset;
    int64_t asOffset;
    int64_t biasUbOffset;
    int64_t hRepeatTime;
    int64_t hRepeatAligned;
    int64_t repeatStrideF;
    int64_t ifKVout;
    int64_t isPA;
    int64_t hasBias;
    int64_t hasAS;
    int64_t batch;
    int64_t cacheBatchNum;
    int64_t cacheRowElems;
    int64_t cacheRowAlign;

    int64_t indicesArray[INDICES_MAX__NUM];

    event_t eventIdMTE2ToS;
    event_t eventIdMTE2ToAS;
    event_t eventIdVToMTE3K;
    event_t eventIdMTE3ToMTE2K;
    event_t eventIdMTE2ToMTE3V;
    event_t eventIdMTE3ToMTE2V;
    event_t eventIdVToMTE3V;
    event_t eventIdMTE3ToVV;
    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> qQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> kQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> vQueue;

    TQue<QuePosition::VECIN, 1> sinQueue;
    TQue<QuePosition::VECIN, 1> cosQueue;
    TBuf<TPosition::VECCALC> scaleKQueue;
    TBuf<TPosition::VECCALC> scaleVQueue;
    TBuf<TPosition::VECCALC> offsetKQueue;
    TBuf<TPosition::VECCALC> offsetVQueue;
    TBuf<TPosition::VECCALC> biasQueue;
    TBuf<TPosition::VECCALC> wsQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> asQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> indiceQueue;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> qOutQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> kCacheOutQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> vCacheOutQueue;

    TBuf<TPosition::VECCALC> tmpBuf0;

    DataCopyExtParams dataCopyParamsIndice_;
    DataCopyPadExtParams<int32_t> padParamsIndice_{false, static_cast<int32_t>(0), 0, 0};

    DataCopyExtParams dataCopyParamsQ_;
    DataCopyPadExtParams<XTYPE> padParamsQ_;

    DataCopyExtParams dataCopyParamsK_;
    DataCopyPadExtParams<XTYPE> padParamsK_;

    DataCopyExtParams dataCopyParamsV_;
    DataCopyPadExtParams<XTYPE> padParamsV_;
};
} // namespace DequantRopeQuantKvcache
#endif // DEQUANT_ROPE_QUANT_KVCACHE_DEQUANT_ROPE_QUANT_KVCACHE_H
