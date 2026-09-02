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
 * \file test_gather_pa_kv_cache.h
 * \brief
 */
#ifndef _APPLY_ADAM_W_QUANT_TILING_H_
#define _APPLY_ADAM_W_QUANT_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct GatherPaKvCacheTilingDataTest {
    int32_t blockSize;
    int32_t numTokens;
    int32_t numblkTabCol;
    int32_t tokenSizeK;
    int32_t tokenSizeV;
    int32_t typeByte;
    int32_t hasSeqStarts;
    int32_t isSeqLensCumsum;
};

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer = \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer) \
    GatherPaKvCacheTilingDataTest tilingData; \
    INIT_TILING_DATA(GatherPaKvCacheTilingDataTest, tilingDataPointer, tilingPointer); \
    (tilingData).blockSize = tilingDataPointer->blockSize; \
    (tilingData).numTokens = tilingDataPointer->numTokens; \
    (tilingData).numblkTabCol = tilingDataPointer->numblkTabCol; \
    (tilingData).tokenSizeK = tilingDataPointer->tokenSizeK; \
    (tilingData).tokenSizeV = tilingDataPointer->tokenSizeV; \
    (tilingData).typeByte = tilingDataPointer->typeByte; \
    (tilingData).hasSeqStarts = tilingDataPointer->hasSeqStarts; \
    (tilingData).isSeqLensCumsum = tilingDataPointer->isSeqLensCumsum;

#define GET_TILING_DATA_WITH_STRUCT(TilingData, tilingData, tilingPointer) \
    TilingData tilingData; \
    INIT_TILING_DATA(TilingData, tilingDataPointer, tilingPointer); \
    (tilingData).blockSize = tilingDataPointer->blockSize; \
    (tilingData).numTokens = tilingDataPointer->numTokens; \
    (tilingData).numblkTabCol = tilingDataPointer->numblkTabCol; \
    (tilingData).tokenSizeK = tilingDataPointer->tokenSizeK; \
    (tilingData).tokenSizeV = tilingDataPointer->tokenSizeV; \
    (tilingData).typeByte = tilingDataPointer->typeByte; \
    (tilingData).hasSeqStarts = tilingDataPointer->hasSeqStarts; \
    (tilingData).isSeqLensCumsum = tilingDataPointer->isSeqLensCumsum;

#endif
