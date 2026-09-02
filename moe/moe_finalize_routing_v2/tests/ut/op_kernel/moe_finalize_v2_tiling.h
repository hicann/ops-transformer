/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _MOE_TUTEL_COMBINE_GATES_TILING_v2_H_
#define _MOE_TUTEL_COMBINE_GATES_TILING_v2_H_

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer = \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer) \
    MoeFinalizeRoutingV2TilingData tilingData; \
    INIT_TILING_DATA(MoeFinalizeRoutingV2TilingData, tilingDataPointer, tilingPointer); \
    (tilingData).totalCoreNum = tilingDataPointer->totalCoreNum; \
    (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum; \
    (tilingData).biasRowNum = tilingDataPointer->biasRowNum; \
    (tilingData).totalRowNum = tilingDataPointer->totalRowNum; \
    (tilingData).H = tilingDataPointer->H; \
    (tilingData).K = tilingDataPointer->K; \
    (tilingData).normalCoreHandleNum = tilingDataPointer->normalCoreHandleNum; \
    (tilingData).normalCoreLoopNum = tilingDataPointer->normalCoreLoopNum; \
    (tilingData).normalCoreHandleNumPerLoop = tilingDataPointer->normalCoreHandleNumPerLoop; \
    (tilingData).normalCoreHandleNumTailLoop = tilingDataPointer->normalCoreHandleNumTailLoop; \
    (tilingData).tailCoreHandleNum = tilingDataPointer->tailCoreHandleNum; \
    (tilingData).tailCoreLoopNum = tilingDataPointer->tailCoreLoopNum; \
    (tilingData).tailCoreHandleNumPerLoop = tilingDataPointer->tailCoreHandleNumPerLoop; \
    (tilingData).tailCoreHandleNumTailLoop = tilingDataPointer->tailCoreHandleNumTailLoop; \
    (tilingData).tilingKey = tilingDataPointer->tilingKey; \
    (tilingData).skip2IsNull = tilingDataPointer->skip2IsNull; \
    (tilingData).normalH = tilingDataPointer->normalH; \
    (tilingData).unnormalH = tilingDataPointer->unnormalH; \
    (tilingData).hSliceNum = tilingDataPointer->hSliceNum; \
    (tilingData).normalK = tilingDataPointer->normalK; \
    (tilingData).unnormalK = tilingDataPointer->unnormalK; \
    (tilingData).kSliceNum = tilingDataPointer->kSliceNum; \
    (tilingData).skip1IsNull = tilingDataPointer->skip1IsNull;
#endif // _MOE_FINALIZE_ROUTING_TILING_V2_H_
