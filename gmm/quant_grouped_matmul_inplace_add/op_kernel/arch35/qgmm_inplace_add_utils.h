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
 * \file qgmm_inplace_add_utils.h
 * \brief
 */
#ifndef QGMM_INPLACE_ADD_UTILS_H
#define QGMM_INPLACE_ADD_UTILS_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#if defined(ORIG_DTYPE_SCALE2) && defined(DT_FLOAT8_E8M0) && ORIG_DTYPE_SCALE2 == DT_FLOAT8_E8M0
#define V310_QGMM_QUANT_MX
#else
#define V310_QGMM_QUANT_MIX
#endif

#if defined(CONST_TILING)
#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling) \
    GET_TILING_DATA_MEMBER(tilingType, member, obj, tiling); \
    const int32_t *(var) = (const int32_t *)((const uint8_t *)&obj);
#else
#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling) \
    size_t offset##var = (size_t)(&((tilingType *)0)->member); \
    __gm__ int32_t *(var) = (__gm__ int32_t *)((tiling) + (offset##var));
#endif

#endif // QGMM_INPLACE_ADD_UTILS_H
