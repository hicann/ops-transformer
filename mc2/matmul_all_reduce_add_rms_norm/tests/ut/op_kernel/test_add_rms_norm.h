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
 * \file test_add_rms_norm.h
 * \brief
 */

#ifndef _ADD_RMS_NORM_TILING_H_
#define _ADD_RMS_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddRMSNormTilingData {
    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor;
    uint32_t rowDactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer = \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer) \
    AddRMSNormTilingData tilingData; \
    INIT_TILING_DATA(AddRMSNormTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).numRow = tilingDataPointer->numRow; \
    (tilingData).numCol = tilingDataPointer->numCol; \
    (tilingData).blockFactor = tilingDataPointer->blockFactor; \
    (tilingData).rowDactor = tilingDataPointer->rowDactor; \
    (tilingData).ubFactor = tilingDataPointer->ubFactor; \
    (tilingData).epsilon = tilingDataPointer->epsilon; \
    (tilingData).avgFactor = tilingDataPointer->avgFactor;
#endif
