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
 * \file matmul_all_reduce_tiling_def.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_TILING_DEF_H
#define MATMUL_ALL_REDUCE_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "weight_quant_batch_matmul_v2_tiling.h"
#include "quant_batch_matmul_v3_tiling_def.h"
#include "../../../op_kernel/unquant_matmul_all_reduce_tiling_data.h"
#if __has_include("../../../../common/utils/hccl_stub.h")
#include "../../../../../tests/ut/framework_normal/common/hccl_stub.h"
#endif

#ifdef __CCE_KT_TEST__
#include "kernel_operator.h"
#endif

inline void InitMatmulAllReduceTilingData(uint8_t *tiling, Mc2Tiling::MatmulAllReduceTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(Mc2Tiling::MatmulAllReduceTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    Mc2Tiling::MatmulAllReduceTilingData tiling_data; \
    InitMatmulAllReduceTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);

#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling) \
    auto var = ((tiling_type *)((uint8_t *)AscendC::GmAlloc(1024)))->member; \
    size_t offset##var = (size_t)(&((tiling_type *)0)->member); \
    InitTilingData<decltype(var)>(tiling + offset##var, &var);

#endif // FOREACH_MINIMUM_SCALAR_TILING_DEF_H
