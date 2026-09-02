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
 * \file test_aglu.h
 * \brief
 */

#ifndef _SWIN_ATTENTION_SCORE_QUANT_H_
#define _SWIN_ATTENTION_SCORE_QUANT_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct SwinAttentionScoreQuantTilingData {
    uint32_t coreLoops = 0;
    uint32_t dimB = 0;
    uint32_t dimN = 0;
    uint32_t dimS = 0;
    uint32_t dimH = 0;
    uint32_t qSize = 0;
    uint32_t kSize = 0;
    uint32_t pSize = 0;
    uint32_t vSize = 0;
    uint32_t cubeSharedUbSize = 0;
    uint32_t vecSharedUbSize = 0;
    TCubeTiling qkBmmTilingData;
    TCubeTiling pvBmmTilingData;
    SoftMaxTiling softmaxTilingData;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer = \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer) \
    SwinAttentionScoreQuantTilingData tilingData; \
    INIT_TILING_DATA(SwinAttentionScoreQuantTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).coreLoops = tilingDataPointer->coreLoops; \
    (tilingData).dimB = tilingDataPointer->dimB; \
    (tilingData).dimN = tilingDataPointer->dimN; \
    (tilingData).dimS = tilingDataPointer->dimS; \
    (tilingData).dimH = tilingDataPointer->dimH; \
    (tilingData).qSize = tilingDataPointer->qSize; \
    (tilingData).kSize = tilingDataPointer->kSize; \
    (tilingData).pSize = tilingDataPointer->pSize; \
    (tilingData).vSize = tilingDataPointer->vSize; \
    (tilingData).cubeSharedUbSize = tilingDataPointer->cubeSharedUbSize; \
    (tilingData).vecSharedUbSize = tilingDataPointer->vecSharedUbSize; \
    (tilingData).qkBmmTilingData = tilingDataPointer->qkBmmTilingData; \
    (tilingData).pvBmmTilingData = tilingDataPointer->pvBmmTilingData; \
    (tilingData).softmaxTilingData = tilingDataPointer->softmaxTilingData;
#endif
