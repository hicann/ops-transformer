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
 * \file rope_quant_kvcache.h
 * \brief
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace RopeQuantKvcacheND {
using namespace AscendC;
constexpr static int64_t TASK_NUM = 2;
constexpr static int64_t ROPE_TENSOR_NUM = 2;
constexpr static int64_t QUANTIZE_TENSOR_NUM = 2;
constexpr static int64_t ROPE_LAST_DIM_SPLIT = 2;
constexpr static int64_t FP16_ONE_BLOCK_NUM = 16;
constexpr static int64_t FP16_ONE_REPEAT_NUM = 128;
class RopeQuantKvcache {
public:
    __aicore__ inline RopeQuantKvcache(const RopeQuantKvcacheTilingData *tilingData)
    {
        this->cacheSeqlen = tilingData->cacheSeqlen;
        this->qHeadNum = tilingData->qHeadNum;
        this->kvHeadNum = tilingData->kvHeadNum;
        this->hiddenSize = tilingData->hiddenSize;
        this->qHiddenSize = tilingData->qHiddenSize;
        this->kHiddenSize = tilingData->kHiddenSize;
        this->vHiddenSize = tilingData->vHiddenSize;
    }

    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR quant_scale, GM_ADDR quant_offset,
                                GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR indice, GM_ADDR q_out, GM_ADDR k_cache_out,
                                GM_ADDR v_cache_out)
    {
        auto blockIdx = GetBlockIdx();
        auto batchId = (blockIdx / TASK_NUM);
        uint64_t kvDataNum = this->kvHeadNum * this->hiddenSize;
        uint64_t qkvBlockOffset = batchId * (this->qHiddenSize + this->kHiddenSize + this->vHiddenSize);
        uint64_t cossinBlockOffset = batchId * this->hiddenSize;
        uint64_t kvCacheBlockOffset = batchId * this->cacheSeqlen * kvDataNum;

        taskId = blockIdx % TASK_NUM;
        if (taskId == 1) {
            uint64_t qDataNum = this->qHeadNum * this->hiddenSize;
            uint64_t qBlockOffset = batchId * qDataNum;
            inputGm.SetGlobalBuffer((__gm__ half *)qkv + qkvBlockOffset);

            cosGm.SetGlobalBuffer((__gm__ half *)cos + cossinBlockOffset);
            cosGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            sinGm.SetGlobalBuffer((__gm__ half *)sin + cossinBlockOffset);
            sinGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            outputGm.SetGlobalBuffer((__gm__ half *)q_out + qBlockOffset);

            quantScaleGm.SetGlobalBuffer((__gm__ float *)quant_scale);
            quantScaleGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            quantOffsetGm.SetGlobalBuffer((__gm__ int32_t *)quant_offset);
            quantOffsetGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            indiceGm.SetGlobalBuffer((__gm__ int32_t *)indice + batchId);
            indiceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            int32_t idx = indiceGm.GetValue(0);
            vCacheGm.SetGlobalBuffer((__gm__ int8_t *)v_cache_out + kvCacheBlockOffset + idx * kvDataNum);

            pipe.InitBuffer(inQueue, 1,
                            qDataNum * sizeof(half) + kvDataNum * sizeof(half) +
                                this->hiddenSize * sizeof(half) * ROPE_TENSOR_NUM +
                                this->hiddenSize * sizeof(float) * QUANTIZE_TENSOR_NUM);
            pipe.InitBuffer(outQueue, 1, qDataNum * sizeof(half) + kvDataNum * sizeof(float));
        } else if (taskId == 0) {
            inputGm.SetGlobalBuffer((__gm__ half *)qkv + qkvBlockOffset + this->qHiddenSize);

            cosGm.SetGlobalBuffer((__gm__ half *)cos + cossinBlockOffset);
            cosGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            sinGm.SetGlobalBuffer((__gm__ half *)sin + cossinBlockOffset);
            sinGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            quantScaleGm.SetGlobalBuffer((__gm__ float *)quant_scale);
            quantScaleGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            quantOffsetGm.SetGlobalBuffer((__gm__ int32_t *)quant_offset);
            quantOffsetGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            indiceGm.SetGlobalBuffer((__gm__ int32_t *)indice + batchId);
            indiceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            int32_t idx = indiceGm.GetValue(0);
            kCacheGm.SetGlobalBuffer((__gm__ int8_t *)k_cache_out + kvCacheBlockOffset + idx * kvDataNum);

            pipe.InitBuffer(inQueue, 1,
                            kvDataNum * sizeof(half) + this->hiddenSize * sizeof(half) * ROPE_TENSOR_NUM +
                                this->hiddenSize * sizeof(float) * QUANTIZE_TENSOR_NUM);
            pipe.InitBuffer(outQueue, 1, kvDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->taskId == 0) {
            ProcessK();
        } else if (this->taskId == 1) {
            ProcessQV();
        }
    }

private:
    __aicore__ inline void ProcessQV()
    {
        // copy in cos/sin
        LocalTensor<half> qInUb = inQueue.AllocTensor<half>();
        LocalTensor<half> vInUb = qInUb[this->qHeadNum * this->hiddenSize];
        LocalTensor<half> cosUb = qInUb[this->qHeadNum * this->hiddenSize + this->kvHeadNum * this->hiddenSize];
        LocalTensor<half> sinUb = cosUb[this->hiddenSize];
        LocalTensor<float> quantScaleUb = cosUb[this->hiddenSize * ROPE_TENSOR_NUM].ReinterpretCast<float>();
        LocalTensor<float> quantOffsetUb = quantScaleUb[this->hiddenSize];
        LocalTensor<int32_t> quantOffsetUbOri = quantOffsetUb.ReinterpretCast<int32_t>();

        DataCopy(quantScaleUb, quantScaleGm, this->hiddenSize);
        DataCopy(quantOffsetUbOri, quantOffsetGm, this->hiddenSize);
        DataCopy(cosUb, cosGm, this->hiddenSize);
        DataCopy(sinUb, sinGm, this->hiddenSize);
        DataCopy(qInUb, inputGm, this->qHeadNum * this->hiddenSize);
        DataCopy(vInUb, inputGm[this->qHiddenSize + this->kHiddenSize], this->kvHeadNum * this->hiddenSize);
        inQueue.EnQue(qInUb);
        qInUb = inQueue.DeQue<half>();
        Muls(sinUb, sinUb, (half)-1.0, this->hiddenSize / ROPE_LAST_DIM_SPLIT);
        Cast(quantOffsetUb, quantOffsetUbOri, RoundMode::CAST_NONE, this->hiddenSize);
        PipeBarrier<PIPE_V>();

        // caculate q
        LocalTensor<half> qOutUb = outQueue.AllocTensor<half>();
        LocalTensor<half> vOutUb = qOutUb[this->qHeadNum * this->hiddenSize];
        // step 1
        if (this->hiddenSize <= FP16_ONE_REPEAT_NUM) {
            uint64_t halfSize = this->hiddenSize / ROPE_LAST_DIM_SPLIT;
            uint8_t repeatStride = this->hiddenSize / FP16_ONE_BLOCK_NUM;
            Mul(qOutUb, qInUb[halfSize], sinUb, halfSize, this->qHeadNum, {1, 1, 1, repeatStride, repeatStride, 0});
            Mul(qOutUb[halfSize], qInUb, sinUb[halfSize], halfSize, this->qHeadNum,
                {1, 1, 1, repeatStride, repeatStride, 0});
        } else {
            uint64_t halfSize = this->hiddenSize / ROPE_LAST_DIM_SPLIT;
            for (uint64_t r = 0; r < this->qHeadNum; r++) {
                uint64_t rowOffset = r * this->hiddenSize;
                Mul(qOutUb[rowOffset], qInUb[rowOffset + halfSize], sinUb, halfSize);
                Mul(qOutUb[rowOffset + halfSize], qInUb[rowOffset], sinUb[halfSize], halfSize);
            }
        }
        LocalTensor<float> vOutUbF32 = vOutUb.ReinterpretCast<float>();
        Cast(vOutUbF32, vInUb, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        // step 2
        if (this->hiddenSize <= FP16_ONE_REPEAT_NUM) {
            uint8_t repeatStride = this->hiddenSize / FP16_ONE_BLOCK_NUM;
            Mul(qInUb, qInUb, cosUb, this->hiddenSize, this->qHeadNum, {1, 1, 1, repeatStride, repeatStride, 0});
        } else {
            for (uint64_t r = 0; r < this->qHeadNum; r++) {
                uint64_t rowOffset = r * this->hiddenSize;
                Mul(qInUb[rowOffset], qInUb[rowOffset], cosUb, this->hiddenSize);
            }
        }
        for (uint64_t r = 0; r < this->kvHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
            Div(vOutUbF32[rowOffset], vOutUbF32[rowOffset], quantScaleUb, this->hiddenSize);
        }
        PipeBarrier<PIPE_V>();
        // step 3
        Add(qOutUb, qOutUb, qInUb, this->qHeadNum * this->hiddenSize);
        for (uint64_t r = 0; r < this->kvHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
            Add(vOutUbF32[rowOffset], vOutUbF32[rowOffset], quantOffsetUb, this->hiddenSize);
        }

        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopy(outputGm, qOutUb, this->qHeadNum * this->hiddenSize);

        // caculate v
        PipeBarrier<PIPE_V>();
        LocalTensor<int16_t> vOutUbS16 = vOutUb.ReinterpretCast<int16_t>();
        Cast(vOutUbS16, vOutUbF32, RoundMode::CAST_RINT, this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> vOutUbF16 = vOutUb.ReinterpretCast<half>();
        Cast(vOutUbF16, vOutUbS16, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> vOutUbS8 = vOutUb.ReinterpretCast<int8_t>();
        Cast(vOutUbS8, vOutUbF16, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopy(vCacheGm, vOutUbS8, this->kvHeadNum * this->hiddenSize);
    }

    __aicore__ inline void ProcessK()
    {
        // copy in cos/sin/scale/offset
        LocalTensor<half> kInUb = inQueue.AllocTensor<half>();
        LocalTensor<half> cosUb = kInUb[this->kvHeadNum * this->hiddenSize];
        LocalTensor<half> sinUb = cosUb[this->hiddenSize];
        LocalTensor<float> quantScaleUb = cosUb[this->hiddenSize * ROPE_TENSOR_NUM].ReinterpretCast<float>();
        LocalTensor<float> quantOffsetUb = quantScaleUb[this->hiddenSize];
        LocalTensor<int32_t> quantOffsetUbOri = quantOffsetUb.ReinterpretCast<int32_t>();

        DataCopy(quantScaleUb, quantScaleGm, this->hiddenSize);
        DataCopy(quantOffsetUbOri, quantOffsetGm, this->hiddenSize);
        DataCopy(cosUb, cosGm, this->hiddenSize);
        DataCopy(sinUb, sinGm, this->hiddenSize);
        DataCopy(kInUb, inputGm, this->kvHeadNum * this->hiddenSize);

        inQueue.EnQue(kInUb);
        kInUb = inQueue.DeQue<half>();

        Muls(sinUb, sinUb, (half)-1.0, this->hiddenSize / ROPE_LAST_DIM_SPLIT);
        Cast(quantOffsetUb, quantOffsetUbOri, RoundMode::CAST_NONE, this->hiddenSize);
        PipeBarrier<PIPE_V>();

        LocalTensor<half> kOutUb = outQueue.AllocTensor<half>();
        if (this->hiddenSize <= FP16_ONE_REPEAT_NUM) {
            uint64_t halfSize = this->hiddenSize / ROPE_LAST_DIM_SPLIT;
            uint8_t repeatStride = this->hiddenSize / FP16_ONE_BLOCK_NUM;
            Mul(kOutUb, kInUb[halfSize], sinUb, halfSize, this->kvHeadNum, {1, 1, 1, repeatStride, repeatStride, 0});
            Mul(kOutUb[halfSize], kInUb, sinUb[halfSize], halfSize, this->kvHeadNum,
                {1, 1, 1, repeatStride, repeatStride, 0});
        } else {
            uint64_t halfSize = this->hiddenSize / ROPE_LAST_DIM_SPLIT;
            for (uint64_t r = 0; r < this->kvHeadNum; r++) {
                uint64_t rowOffset = r * this->hiddenSize;
                Mul(kOutUb[rowOffset], kInUb[rowOffset + halfSize], sinUb, halfSize);
                Mul(kOutUb[rowOffset + halfSize], kInUb[rowOffset], sinUb[halfSize], halfSize);
            }
        }
        PipeBarrier<PIPE_V>();

        for (uint64_t r = 0; r < this->kvHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
            Mul(kInUb[rowOffset], kInUb[rowOffset], cosUb, this->hiddenSize);
        }
        PipeBarrier<PIPE_V>();

        Add(kInUb, kOutUb, kInUb, this->hiddenSize * this->kvHeadNum);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> kOutUbF32 = kOutUb.ReinterpretCast<float>();
        Cast(kOutUbF32, kInUb, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();

        for (uint64_t r = 0; r < this->kvHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
            Div(kOutUbF32[rowOffset], kOutUbF32[rowOffset], quantScaleUb, this->hiddenSize);
        }
        PipeBarrier<PIPE_V>();

        for (uint64_t r = 0; r < this->kvHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
            Add(kOutUbF32[rowOffset], kOutUbF32[rowOffset], quantOffsetUb, this->hiddenSize);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<int16_t> kOutUbS16 = kOutUb.ReinterpretCast<int16_t>();
        Cast(kOutUbS16, kOutUbF32, RoundMode::CAST_RINT, this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> kOutUbF16 = kOutUb.ReinterpretCast<half>();
        Cast(kOutUbF16, kOutUbS16, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> kOutUbS8 = kOutUb.ReinterpretCast<int8_t>();
        Cast(kOutUbS8, kOutUbF16, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);

        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

        DataCopy(kCacheGm, kOutUbS8, this->kvHeadNum * this->hiddenSize);
    }

    /* global memory address */
    GlobalTensor<half> inputGm;

    GlobalTensor<half> cosGm;
    GlobalTensor<half> sinGm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<int32_t> quantOffsetGm;

    GlobalTensor<half> outputGm;
    GlobalTensor<int8_t> kCacheGm;
    GlobalTensor<int8_t> vCacheGm;

    GlobalTensor<int32_t> indiceGm;

    /* variable */
    uint64_t cacheSeqlen;
    uint64_t qHeadNum;
    uint64_t kvHeadNum;
    uint64_t hiddenSize;
    uint64_t qHiddenSize;
    uint64_t kHiddenSize;
    uint64_t vHiddenSize;
    uint64_t taskId;

    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> othersQueue;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
};
} // namespace RopeQuantKvcacheND
