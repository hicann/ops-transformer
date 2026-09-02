/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qkv_rms_norm_rope_cache_tiling_def.h
 * \brief
 */

#ifndef _QKV_RMS_NORM_ROPE_CACHE_TILING_H_
#define _QKV_RMS_NORM_ROPE_CACHE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

#define DTYPE_QKV half
#define DTYPE_K_CACHE int8_t
#define DTYPE_V_CACHE int8_t

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, QkvRmsNormRopeCacheTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(QkvRmsNormRopeCacheTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, QkvRmsNormRopeCacheTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(QkvRmsNormRopeCacheTilingData));
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
    QkvRmsNormRopeCacheTilingData tilingData; \
    INIT_TILING_DATA(QkvRmsNormRopeCacheTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).batchSize = tilingDataPointer->batchSize; \
    (tilingData).seqLength = tilingDataPointer->seqLength; \
    (tilingData).numHead = tilingDataPointer->numHead; \
    (tilingData).qkvDim = tilingDataPointer->qkvDim; \
    (tilingData).ropeRange = tilingDataPointer->ropeRange; \
    (tilingData).numHeadQ = tilingDataPointer->numHeadQ; \
    (tilingData).numHeadK = tilingDataPointer->numHeadK; \
    (tilingData).numHeadV = tilingDataPointer->numHeadV; \
    (tilingData).blockNum = tilingDataPointer->blockNum; \
    (tilingData).blockSize = tilingDataPointer->blockSize; \
    (tilingData).epsilon = tilingDataPointer->epsilon; \
    (tilingData).blockFactor = tilingDataPointer->blockFactor; \
    (tilingData).blockFactorQ = tilingDataPointer->blockFactorQ; \
    (tilingData).blockFactorK = tilingDataPointer->blockFactorK; \
    (tilingData).blockFactorV = tilingDataPointer->blockFactorV; \
    (tilingData).blockDim = tilingDataPointer->blockDim; \
    (tilingData).blockDimQ = tilingDataPointer->blockDimQ; \
    (tilingData).blockDimK = tilingDataPointer->blockDimK; \
    (tilingData).blockDimV = tilingDataPointer->blockDimV; \
    (tilingData).ubFactor = tilingDataPointer->ubFactor; \
    (tilingData).ubFactorQ = tilingDataPointer->ubFactorQ; \
    (tilingData).ubFactorK = tilingDataPointer->ubFactorK; \
    (tilingData).ubFactorV = tilingDataPointer->ubFactorV; \
    (tilingData).reciprocal = tilingDataPointer->reciprocal; \
    (tilingData).isOutputQkv = tilingDataPointer->isOutputQkv; \
    (tilingData).isKQuant = tilingDataPointer->isKQuant; \
    (tilingData).isVQuant = tilingDataPointer->isVQuant;
#endif
