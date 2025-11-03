/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                                   \
    InterleaveRopeTilingData tilingData;                                                             \
    INIT_TILING_DATA(InterleaveRopeTilingData, tilingDataPointer, tilingPointer);                    \
    (tilingData).blockDim = tilingDataPointer->blockDim;                                             \
    (tilingData).splitAxis = tilingDataPointer->splitAxis;                                           \
    (tilingData).batchSize = tilingDataPointer->batchSize;                                           \
    (tilingData).numHead = tilingDataPointer->numHead;                                               \
    (tilingData).seqLength = tilingDataPointer->seqLength;                                           \
    (tilingData).hiddenDim = tilingDataPointer->hiddenDim;                                           \
    (tilingData).batchsPerBlock = tilingDataPointer->batchsPerBlock;                                 \
    (tilingData).batchsLastBlock = tilingDataPointer->batchsLastBlock;                               \
    (tilingData).batchLoops = tilingDataPointer->batchLoops;                                         \
    (tilingData).batchPerLoop = tilingDataPointer->batchPerLoop;                                     \
    (tilingData).batchLastLoop = tilingDataPointer->batchLastLoop;                                   \
    (tilingData).hiddenDimCountPerBlock = tilingDataPointer->hiddenDimCountPerBlock;                 \
    (tilingData).hiddenDimCountLastBlock = tilingDataPointer->hiddenDimCountLastBlock;               \
    (tilingData).hiddenDimLoopsPerBlock = tilingDataPointer->hiddenDimLoopsPerBlock;                 \
    (tilingData).hiddenDimCountPerLoopPerBlock = tilingDataPointer->hiddenDimCountPerLoopPerBlock;   \
    (tilingData).hiddenDimCountLastLoopPerBlock = tilingDataPointer->hiddenDimCountLastLoopPerBlock; \
    (tilingData).hiddenDimLoopsLastBlock = tilingDataPointer->hiddenDimLoopsLastBlock;               \
    (tilingData).hiddenDimCountPerLoopLastBlock = tilingDataPointer->hiddenDimCountPerLoopLastBlock; \
    (tilingData).hiddenDimCountLastLoopLastBlock = tilingDataPointer->hiddenDimCountLastLoopLastBlock;

#define DTYPE_X half

#endif