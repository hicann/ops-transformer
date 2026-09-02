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
 * \file matmul_all_reduce_add_rms_norm_tiling_def.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_DEF_H
#define MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "test_add_rms_norm.h"
#include "../../../../matmul_all_reduce/tests/ut/op_kernel/weight_quant_batch_matmul_v2_tiling.h"
#include "../../../op_kernel/matmul_all_reduce_add_rms_norm_tiling_data.h"
#if __has_include("../../../../common/utils/hccl_stub.h")
#include "../../../../../tests/ut/framework_normal/common/hccl_stub.h"
#endif

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
    uint32_t rowFactor;
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
    (tilingData).rowFactor = tilingDataPointer->rowFactor; \
    (tilingData).ubFactor = tilingDataPointer->ubFactor; \
    (tilingData).epsilon = tilingDataPointer->epsilon; \
    (tilingData).avgFactor = tilingDataPointer->avgFactor;
#endif

#ifdef __CCE_KT_TEST__
#include "kernel_operator.h"
#endif

constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;

inline void InitMatmulAllReduceAddRmsNormTilingData(uint8_t *tiling,
                                                    Mc2Tiling::MatmulAllReduceAddRmsNormTilingData *constData)
{
    memcpy(constData, tiling, sizeof(Mc2Tiling::MatmulAllReduceAddRmsNormTilingData));
}

#define GET_TILING_DATA(tilingData, tilingArg) \
    MatmulAllReduceAddRmsNormTilingData tilingData; \
    InitMatmulAllReduceAddRmsNormTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA_MEMBER(tilingType, member, var, tiling) \
    auto var = ((tilingType *)((uint8_t *)AscendC::GmAlloc(1024)))->member; \
    size_t offset##var = (size_t)(&((tilingType *)0)->member); \
    InitTilingData<decltype(var)>(tiling + offset##var, &var);

#endif // FOREACH_MINIMUM_SCALAR_TILING_DEF_H
