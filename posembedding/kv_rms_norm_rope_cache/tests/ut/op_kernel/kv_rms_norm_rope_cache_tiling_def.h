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
 * \file kv_rms_norm_rope_cache_tiling_def.h
 * \brief
 */

#ifndef _KV_RMS_NORM_ROPE_CACHE_TILING_H_
#define _KV_RMS_NORM_ROPE_CACHE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct KvRmsNormRopeCacheTilingData {
    int64_t blockDim = 16;
    int64_t rowsPerBlock = 4;
    int64_t cacheLength = 288;
    int64_t batchSize = 64;
    int64_t numHead = 1;
    int64_t seqLength = 1;
    int64_t blockSize = 1;
    int64_t blockFactor = 1;
    int64_t ubFactor = 1;
    float epsilon = 1e-5;
    float reciprocal = 1.0f / 512.0f;
    int8_t isOutputKv = true;
    int8_t isKQuant = 1;
    int8_t isVQuant = 1;
    int64_t methodMode = 0;
};
#define DTYPE_KV half
#define DTYPE_K_CACHE int8_t
#define DTYPE_CKV_CACHE int8_t
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, KvRmsNormRopeCacheTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(KvRmsNormRopeCacheTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, KvRmsNormRopeCacheTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(KvRmsNormRopeCacheTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer = \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer) \
    KvRmsNormRopeCacheTilingData tilingData; \
    INIT_TILING_DATA(KvRmsNormRopeCacheTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).blockDim = tilingDataPointer->blockDim; \
    (tilingData).rowsPerBlock = tilingDataPointer->rowsPerBlock; \
    (tilingData).cacheLength = tilingDataPointer->cacheLength; \
    (tilingData).batchSize = tilingDataPointer->batchSize; \
    (tilingData).numHead = tilingDataPointer->numHead; \
    (tilingData).seqLength = tilingDataPointer->seqLength; \
    (tilingData).blockSize = tilingDataPointer->blockSize; \
    (tilingData).blockFactor = tilingDataPointer->blockFactor; \
    (tilingData).ubFactor = tilingDataPointer->ubFactor; \
    (tilingData).epsilon = tilingDataPointer->epsilon; \
    (tilingData).reciprocal = tilingDataPointer->reciprocal; \
    (tilingData).isOutputKv = tilingDataPointer->isOutputKv; \
    (tilingData).isKQuant = tilingDataPointer->isKQuant; \
    (tilingData).isVQuant = tilingDataPointer->isVQuant; \
    (tilingData).methodMode = tilingDataPointer->methodMode;
#endif

#ifndef _KV_RMS_NORM_ROPE_CACHE_TILING_H_
#define _KV_RMS_NORM_ROPE_CACHE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct KvRmsNormRopeCacheTilingData {
    int64_t blockDim = 16;
    int64_t rowsPerBlock = 4;
    int64_t cacheLength = 288;
    int64_t batchSize = 64;
    int64_t numHead = 1;
    int64_t seqLength = 1;
    int64_t blockSize = 1;
    int64_t blockFactor = 1;
    int64_t ubFactor = 1;
    float epsilon = 1e-5;
    float reciprocal = 1.0f / 512.0f;
    int8_t isOutputKv = true;
    int8_t isKQuant = 1;
    int8_t isVQuant = 1;
    int64_t methodMode = 0;
};
#define DTYPE_KV half
#define DTYPE_K_CACHE int8_t
#define DTYPE_CKV_CACHE int8_t
#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer = \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer) \
    KvRmsNormRopeCacheTilingData tilingData; \
    INIT_TILING_DATA(KvRmsNormRopeCacheTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).blockDim = tilingDataPointer->blockDim; \
    (tilingData).rowsPerBlock = tilingDataPointer->rowsPerBlock; \
    (tilingData).cacheLength = tilingDataPointer->cacheLength; \
    (tilingData).batchSize = tilingDataPointer->batchSize; \
    (tilingData).numHead = tilingDataPointer->numHead; \
    (tilingData).seqLength = tilingDataPointer->seqLength; \
    (tilingData).blockSize = tilingDataPointer->blockSize; \
    (tilingData).blockFactor = tilingDataPointer->blockFactor; \
    (tilingData).ubFactor = tilingDataPointer->ubFactor; \
    (tilingData).epsilon = tilingDataPointer->epsilon; \
    (tilingData).reciprocal = tilingDataPointer->reciprocal; \
    (tilingData).isOutputKv = tilingDataPointer->isOutputKv; \
    (tilingData).isKQuant = tilingDataPointer->isKQuant; \
    (tilingData).isVQuant = tilingDataPointer->isVQuant; \
    (tilingData).methodMode = tilingDataPointer->methodMode;
#endif
