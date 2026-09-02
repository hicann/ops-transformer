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
 * \file rope_quant_kvcache_tiling.h
 * \brief
 */
#ifndef _ROPE_QUANT_KVCACHE_TILING_H_
#define _ROPE_QUANT_KVCACHE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct RopeQuantKvcacheTilingData {
    uint64_t qHeadNum;
    uint64_t kvHeadNum;
    uint64_t hiddenSize;
    uint64_t cacheSeqlen;
    uint64_t qHiddenSize;
    uint64_t kHiddenSize;
    uint64_t vHiddenSize;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer = \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer) \
    RopeQuantKvcacheTilingData tilingData; \
    INIT_TILING_DATA(RopeQuantKvcacheTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).qHeadNum = tilingDataPointer->qHeadNum; \
    (tilingData).kvHeadNum = tilingDataPointer->kvHeadNum; \
    (tilingData).hiddenSize = tilingDataPointer->hiddenSize; \
    (tilingData).cacheSeqlen = tilingDataPointer->cacheSeqlen; \
    (tilingData).qHiddenSize = tilingDataPointer->qHiddenSize; \
    (tilingData).kHiddenSize = tilingDataPointer->kHiddenSize; \
    (tilingData).vHiddenSize = tilingDataPointer->vHiddenSize;
#endif
