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
 * \file quant_grouped_matmul_inplace_add_tiling_data.h
 * \brief
 */

#ifndef QUANT_GROUPED_MATMUL_INPLACE_ADD_TILING_DATA_H
#define QUANT_GROUPED_MATMUL_INPLACE_ADD_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace QuantGroupedMatmulInplaceAdd {
#pragma pack(push, 8)
struct QGmmInplaceAddParams {
    uint32_t groupNum = 0;
    uint8_t groupListType = 0;
    uint8_t reserved1 = 0;
    uint16_t reserved2 = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct QGmmInplaceAddTilingDataParams {
    QGmmInplaceAddParams quantGmmInplaceAddParams;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct QGmmInplaceAddBasicApiParams {
    uint32_t groupNum = 0;
    uint32_t aQuantMode = 0;
    uint32_t bQuantMode = 0;
    uint8_t groupListType = 0;
    uint8_t reserved1 = 0;
    uint16_t reserved2 = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct QGmmInplaceAddBasicApiMMTiling {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t kAL1 = 0;
    uint32_t kBL1 = 0;
    uint32_t scaleKAL1 = 0;
    uint32_t scaleKBL1 = 0;
    uint8_t dbL0C = 0;
    uint8_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t reserved3 = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct QGmmInplaceAddBasicApiTilingData {
    QGmmInplaceAddBasicApiParams gmmQuantParams;
    QGmmInplaceAddBasicApiMMTiling mmTilingData;
};
#pragma pack(pop)

} // namespace QuantGroupedMatmulInplaceAdd
#endif
