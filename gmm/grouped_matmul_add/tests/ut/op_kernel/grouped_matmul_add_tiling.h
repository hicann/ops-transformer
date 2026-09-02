/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __GROUPED_MATMUL_ADD_TILING_DEF_H__
#define __GROUPED_MATMUL_ADD_TILING_DEF_H__

#include "kernel_tiling/kernel_tiling.h"
// #include "tiling/tiling_api.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

constexpr uint16_t GMM_MAX_TENSOR_LIST_SIZE = 128;

struct GmmBaseParams {
    int64_t groupNum;
    int64_t coreNum;
    int64_t groupType; // 分组类型，仅支持2 --- K分组
    int64_t groupListType;
};

struct GroupedMatmulAddTilingData {
    GmmBaseParams gmmBaseParams;
    TCubeTiling mmTilingData;
};
#ifdef __NPU_TILING__
inline void InitGroupedMatmulAddTilingData(uint8_t *tiling, GroupedMatmulAddTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(GroupedMatmulAddTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitGroupedMatmulAddTilingData(uint8_t *tiling, GroupedMatmulAddTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(GroupedMatmulAddTilingData));
}
#endif

#define COPY_ARR(arrA, arrB, count) \
    for (uint16_t i = 0; i < count; i++) { \
        arrA[i] = arrB[i]; \
    }

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer = \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer) \
    GroupedMatmulAddTilingData tilingData; \
    InitGroupedMatmulAddTilingData(tilingPointer, &tilingData)

#endif
