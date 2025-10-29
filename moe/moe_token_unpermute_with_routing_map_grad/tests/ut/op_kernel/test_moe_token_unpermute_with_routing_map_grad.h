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
 * \file test_moe_token_unpermute_with_routing_map_grad.h
 * \brief
 */
#ifndef _MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H_
#define _MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                             \
    MoeTokenUnpermuteWithRoutingMapGradTilingData tilingData;                                                 \
    INIT_TILING_DATA(MoeTokenUnpermuteWithRoutingMapGradTilingData, tilingDataPointer, tilingPointer);        \
    (tilingData).tokensNum = tilingDataPointer->tokensNum;                               \
    (tilingData).topK = tilingDataPointer->topK;                                 \
    (tilingData).capacity = tilingDataPointer->capacity;                                   \
    (tilingData).numExpert = tilingDataPointer->numExpert;                                 \
    (tilingData).hiddenSize = tilingDataPointer->hiddenSize;                                                     \
    (tilingData).numOutTokens = tilingDataPointer->numOutTokens;                                                     \
    (tilingData).formerCoreNum = tilingDataPointer->formerCoreNum;                 \
    (tilingData).tailCoreNum = tilingDataPointer->tailCoreNum;                     \
    (tilingData).tokenNumEachCore = tilingDataPointer->tokenNumEachCore;   \
    (tilingData).tokenNumTailCore = tilingDataPointer->tokenNumTailCore; \
    (tilingData).rowIdMapEachCore = tilingDataPointer->rowIdMapEachCore;                     \
    (tilingData).rowIdMapTailCore = tilingDataPointer->rowIdMapTailCore;                         \
    (tilingData).hiddenSizeAlign = tilingDataPointer->hiddenSizeAlign;       \
    (tilingData).hiddenSizeLoopTimes = tilingDataPointer->hiddenSizeLoopTimes;     \
    (tilingData).hiddenSizeLoopTimesAlign = tilingDataPointer->hiddenSizeLoopTimesAlign;                                     \
    (tilingData).hiddenSizeTail = tilingDataPointer->hiddenSizeTail;                                 \
    (tilingData).inputReserveNum = tilingDataPointer->inputReserveNum;                                         \
    (tilingData).indicesReserveNum = tilingDataPointer->indicesReserveNum;                                     \
    (tilingData).indicesReserveNumAlign = tilingDataPointer->indicesReserveNumAlign;                                     \
    (tilingData).numExpertAlign = tilingDataPointer->numExpertAlign;                                         \
    (tilingData).totalUbSize = tilingDataPointer->totalUbSize;
#endif // _MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H_