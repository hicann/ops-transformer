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
 * \file kv_rms_norm_rope_cache_comm.h
 * \brief
 */
#ifndef _KV_RMS_NORM_ROPE_CACHE_COMM_H_
#define _KV_RMS_NORM_ROPE_CACHE_COMM_H_
#include "kv_rms_norm_rope_cache_platform.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;

constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t NUM_SIXTEEN = 16;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t NUM_SIXTY_FOUR = 64;
constexpr int64_t MASK_CONFIG = 0xffffffff;

constexpr int64_t GAMMA_BUFFER_COUNT = 2;
constexpr int64_t X_BUFFER_COUNT = 2;
constexpr int64_t OUT_BUFFER_COUNT = 2;
constexpr int64_t WORKSPACE_BUFFER_MULTIPLIER = 3;

// Preset D sizes for KV cache
constexpr uint32_t METHOD_TYPES = 2;
constexpr int32_t METHOD_V1 = 0;
constexpr int32_t METHOD_V2 = 1;
constexpr int64_t K_CACHE_SIZES[METHOD_TYPES] = {64, 192};
constexpr int64_t V_CACHE_SIZES[METHOD_TYPES] = {512, 128};
constexpr int64_t KERNEL_ROPE_LENGTH[METHOD_TYPES] = {64, 64};
// Deduction Consts
constexpr int64_t KERNEL_RMS_NORM_LENGTH[METHOD_TYPES] = {V_CACHE_SIZES[METHOD_V1], K_CACHE_SIZES[METHOD_V2]};
constexpr int64_t KERNEL_KV_TOTAL_LENGTH[METHOD_TYPES] = {
    V_CACHE_SIZES[METHOD_V1] + K_CACHE_SIZES[METHOD_V1], // KV[V1] = 512 + 64 = 576
    V_CACHE_SIZES[METHOD_V2] + K_CACHE_SIZES[METHOD_V2], // KV[V1] = 128 + 192 = 320
};

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

// Template Branch Levels:
//  1) K/V TYPE Consistency
//  2) Quant Mode Availability: Differentiated Quant Implementation according to Tensor Fractal and Quant Inputs.

template <bool isPagedAttention, typename KV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KernelKvRmsNormRopeCacheBase {
public:
    __aicore__ inline KernelKvRmsNormRopeCacheBase(TPipe *pipe, const KvRmsNormRopeCacheTilingData *tiling)
        : pipe_(pipe),
          tilingData_(tiling)
    {}

    __aicore__ inline int32_t InitSharedData(int32_t methodMode)
    {
        if (methodMode >= METHOD_TYPES) {
            // Throw exception state for INVALID method modes
            return -1;
        }

        this->RMS_NORM_LENGTH = KERNEL_RMS_NORM_LENGTH[methodMode];
        this->ROPE_LENGTH = KERNEL_ROPE_LENGTH[methodMode];
        this->V_LENGTH = V_CACHE_SIZES[methodMode];
        return 0;
    }

protected:
    // Shared Shape Dims
    // In V1: D = Dv(RMS_NORM_LENGTH) + Dk(ROPE_LENGTH)
    // In V2: Dk(RMS_NORM_LENGTH) = ROPE_LENGTH + REM_K_LENGTH
    int64_t RMS_NORM_LENGTH{512};
    int64_t ROPE_LENGTH{64};
    int64_t V_LENGTH{128};
    int64_t batchIdx{0};
    int64_t seqIdx{0};
    int64_t numHeadIdx{0};

    // Sahred members
    TPipe *pipe_ = nullptr;
    const KvRmsNormRopeCacheTilingData *tilingData_;

    GlobalTensor<KV_DTYPE> kvGm, gammaGm, cosGm, sinGm, vGm;
    GlobalTensor<K_DTYPE> kCacheGm;
    GlobalTensor<V_DTYPE> vCacheGm;
    GlobalTensor<int64_t> indexGm;

    TQue<QuePosition::VECIN, 1> inQueueX;      // Kv
    TQue<QuePosition::VECIN, 1> inQueueV;      // vOptional
    TQue<QuePosition::VECIN, 1> inQueueGamma;  // Quant Scale
    TQue<QuePosition::VECIN, 1> inQueueOffset; // Quant Offset
};

template <bool isPagedAttention, typename KV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KernelKvRmsNormRopeCacheMTP : public KernelKvRmsNormRopeCacheBase<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE> {
    // Only 1 sub class: kv_rms_norm_rope_cache_b16_mtp.h
    // MTP is a simplified template for ONLY inputs in NORM and with no quant terms.
public:
    __aicore__ inline KernelKvRmsNormRopeCacheMTP(TPipe *pipe, const KvRmsNormRopeCacheTilingData *tiling)
        : KernelKvRmsNormRopeCacheBase<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE>(pipe, tiling)
    {}

protected:
    TQue<QuePosition::VECOUT, 1> outQueueK, outQueueV;
    TBuf<TPosition::VECCALC> bufferXFp32, bufferXSquare, bufferSum;
};

template <bool isPagedAttention, typename KV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KernelKvRmsNormRopeCacheMTPQuant
    : public KernelKvRmsNormRopeCacheMTP<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE> {
public:
    __aicore__ inline KernelKvRmsNormRopeCacheMTPQuant(TPipe *pipe, const KvRmsNormRopeCacheTilingData *tiling)
        : KernelKvRmsNormRopeCacheMTP<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE>(pipe, tiling)
    {}

protected:
    TBuf<TPosition::VECCALC> wsBuffer;
    GlobalTensor<KV_DTYPE> kCacheGmNd, vCacheGmNd;
    GlobalTensor<float> kRopeScaleGm, cKvScaleGm;   // SYMMETRIC QUANT
    GlobalTensor<float> kRopeOffsetGm, cKvOffsetGm; // ASYMMETRIC QUANT: Missing implementation
};

template <bool isPagedAttention, typename KV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KernelKvRmsNormRopeCacheCutBS
    : public KernelKvRmsNormRopeCacheBase<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE> {
public:
    __aicore__ inline KernelKvRmsNormRopeCacheCutBS(TPipe *pipe, const KvRmsNormRopeCacheTilingData *tiling)
        : KernelKvRmsNormRopeCacheBase<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE>(pipe, tiling)
    {}

protected:
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> wsBuffer;

    int64_t ubLoop{1};
    int64_t ubFactor{8};
    int64_t ubTail{0};
};

template <bool isPagedAttention, typename KV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KernelKvRmsNormRopeCacheCutBSScatter
    : public KernelKvRmsNormRopeCacheCutBS<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE> {
public:
    __aicore__ inline KernelKvRmsNormRopeCacheCutBSScatter(TPipe *pipe, const KvRmsNormRopeCacheTilingData *tiling)
        : KernelKvRmsNormRopeCacheCutBS<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE>(pipe, tiling)
    {}

protected:
    GlobalTensor<KV_DTYPE> kCacheGmNd, vCacheGmNd;
};

template <bool isPagedAttention, typename KV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KernelKvRmsNormRopeCacheCutBSQuant
    : public KernelKvRmsNormRopeCacheCutBSScatter<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE> {
public:
    __aicore__ inline KernelKvRmsNormRopeCacheCutBSQuant(TPipe *pipe, const KvRmsNormRopeCacheTilingData *tiling)
        : KernelKvRmsNormRopeCacheCutBSScatter<isPagedAttention, KV_DTYPE, K_DTYPE, V_DTYPE>(pipe, tiling)
    {}

protected:
    GlobalTensor<float> kRopeScaleGm, cKvScaleGm;   // SYMMETRIC QUANT
    GlobalTensor<float> kRopeOffsetGm, cKvOffsetGm; // ASYMMETRIC QUANT: Missing implementation
};

} // namespace KvRmsNormRopeCache
#endif // _KV_RMS_NORM_ROPE_CACHE_COMM_H_
