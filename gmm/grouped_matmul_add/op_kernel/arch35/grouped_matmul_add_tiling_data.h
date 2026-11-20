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
 * \file grouped_matmul_add_tiling_data.h
 * \brief
 */

#ifndef __GROUPED_MATMUL_ADD_TILING_DATA_H
#define __GROUPED_MATMUL_ADD_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace GroupedMatmulAdd {
#pragma pack(push, 8)
struct GmmAddParams {
    uint32_t groupNum = 0;
    uint32_t groupListType = 0;
    uint32_t mTailCnt = 0;
    uint32_t nTailCnt = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GmmAddTilingDataParams {
    GmmAddParams gmmAddParams;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)
} // namespace GroupedMatmulAdd
#endif
