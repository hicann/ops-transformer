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
 * \file weight_quant_batch_matmul_v2_tiling.h
 * \brief
 */

#ifndef __WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H__
#define __WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/weight_quant_matmul_all_reduce_tiling_data.h"
#ifdef __CCE_KT_TEST__
#include "kernel_operator.h"
#endif

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    Mc2Tiling::WeightQuantMatmulAllReduceTilingData tiling_data; \
    InitTilingData<Mc2Tiling::WeightQuantMatmulAllReduceTilingData>(tiling_arg, &tiling_data);

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);

#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling) \
    auto typeVar##var = ((tiling_type *)0)->member; \
    decltype(typeVar##var) var; \
    size_t offset##var = (size_t)(&((tiling_type *)0)->member); \
    InitTilingData<decltype(typeVar##var)>(tiling + offset##var, &var);

#endif
