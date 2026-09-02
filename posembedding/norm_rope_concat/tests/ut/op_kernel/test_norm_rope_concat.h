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
 * \file test_interleave_rope.h
 * \brief
 */

#ifndef _INTERLEAVE_ROPE_TILING_H_
#define _INTERLEAVE_ROPE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer = \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer) \
    NormRopeConcatTilingData tilingData; \
    INIT_TILING_DATA(NormRopeConcatTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).batch = tilingDataPointer->batch; \
    (tilingData).querySeq = tilingDataPointer->querySeq; \
    (tilingData).keySeq = tilingDataPointer->keySeq; \
    (tilingData).valueSeq = tilingDataPointer->valueSeq; \
    (tilingData).encoderQuerySeq = tilingDataPointer->encoderQuerySeq; \
    (tilingData).encoderKeySeq = tilingDataPointer->encoderKeySeq; \
    (tilingData).encoderValueSeq = tilingDataPointer->encoderValueSeq; \
    (tilingData).totalQuerySeq = tilingDataPointer->totalQuerySeq; \
    (tilingData).totalKeySeq = tilingDataPointer->totalKeySeq; \
    (tilingData).totalValueSeq = tilingDataPointer->totalValueSeq; \
    (tilingData).ropeActualSeq = tilingDataPointer->ropeActualSeq; \
    (tilingData).splitHeadNum = tilingDataPointer->splitHeadNum; \
    (tilingData).avgHeads = tilingDataPointer->avgHeads; \
    (tilingData).tailHeads = tilingDataPointer->tailHeads; \
    (tilingData).normDim = tilingDataPointer->normDim; \
    (tilingData).ropeDim = tilingDataPointer->ropeDim; \
    (tilingData).headNum = tilingDataPointer->headNum; \
    (tilingData).headDim = tilingDataPointer->headDim; \
    (tilingData).usedCore = tilingDataPointer->usedCore; \
    (tilingData).eps = tilingDataPointer->eps; \
    (tilingData).scale = tilingDataPointer->scale;

#endif
