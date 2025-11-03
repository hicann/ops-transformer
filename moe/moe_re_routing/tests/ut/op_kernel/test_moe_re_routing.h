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
 * \file test_moe_re_routing.h
 * \brief
 */

#ifndef _MOE_RE_ROUTING_TILING_H_
#define _MOE_RE_ROUTING_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

// struct MoeReRoutingTilingData {
//     int64_t coreNum = 48;
//     int64_t ubFactor = 12;
//     int64_t tokensNum = 16384;
//     int64_t tokensSize = 7168;
//     int64_t rankNum = 16;
//     int64_t expertNumPerRank = 16;
//     int64_t hasScale = 1;
// };
#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                              \
    MoeReRoutingTilingData tilingData;                                          \
    INIT_TILING_DATA(MoeReRoutingTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).coreNum = tilingDataPointer->coreNum;                          \
    (tilingData).ubFactor = tilingDataPointer->ubFactor;                        \
    (tilingData).tokensNum = tilingDataPointer->tokensNum;                      \
    (tilingData).tokensSize = tilingDataPointer->tokensSize;                    \
    (tilingData).rankNum = tilingDataPointer->rankNum;                          \
    (tilingData).expertNumPerRank = tilingDataPointer->expertNumPerRank;        \
    (tilingData).hasScale = tilingDataPointer->hasScale;

#endif
