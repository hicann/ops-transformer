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
 * \file quant_batch_matmul_v3_tiling_def.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V3_TILING_DEF_H
#define QUANT_BATCH_MATMUL_V3_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/quant_matmul_all_reduce_tiling_data.h"

constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;

inline void InitQuantMatmulAllReduceTilingData(uint8_t *tiling, Mc2Tiling::QuantMatmulAllReduceTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(Mc2Tiling::QuantMatmulAllReduceTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    Mc2Tiling::QuantMatmulAllReduceTilingData tiling_data; \
    InitQuantMatmulAllReduceTilingData(tiling_arg, &tiling_data)
#endif // QUANT_BATCH_MATMUL_V3_TILING_DEF_H
