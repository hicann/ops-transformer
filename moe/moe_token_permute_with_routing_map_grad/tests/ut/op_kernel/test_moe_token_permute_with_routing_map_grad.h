/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H_
#define _MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

#define DTYPE_PERMUTED_TOKEN_OUTPUT_GRAD bfloat16_t

// struct MoeTokenPermuteWithRoutingMapGradTilingData {
//     int64_t hidden_size = 0;
//     int64_t top_k = 0;
//     int64_t num_out_tokens = 0;
//     int64_t hidden_splited_length = 0;
//     int64_t hidden_splited_num = 0;
//     int64_t hidden_splited_remain = 0;
//     int64_t tokens_core_length = 0;
//     int64_t tokens_core_remain = 0;
//     int64_t tokens_splited_length = 0;
//     int64_t tokens_splited_num = 0;
//     int64_t tokens_splited_remain = 0;
//     int64_t buffer_num = 0;
// };

// struct MoeTokenpermuteWithRoutingMapDropPadTilingData {
//     int64_t capacity;
//     int64_t expertNum;
//     int64_t tokenNum;
//     int64_t singleCoreLen;
//     int64_t lastCoreLen;
// };

// struct MoeTokenPermuteWithRoutingMapGradTilingData {
//     struct MoeTokenPermuteWithRoutingMapGradTilingData moeTokenPermuteWithRoutingMapGradUnpermuteTilingData;
//     struct MoeTokenpermuteWithRoutingMapDropPadTilingData moeTokenpermuteWithRoutingMapDropPadTilingData;
// };

// inline void InitTilingData(uint8_t* tiling, MoeTokenPermuteWithRoutingMapGradTilingData* const_data)
// {
//     memcpy(const_data, tiling, sizeof(MoeTokenPermuteWithRoutingMapGradTilingData));
// }

// inline void InitTilingData(uint8_t* tiling, MoeTokenpermuteWithRoutingMapDropPadTilingData* const_data)
// {
//     memcpy(const_data, tiling, sizeof(MoeTokenpermuteWithRoutingMapDropPadTilingData));
// }

// template <class T>
// inline __aicore__ void InitTilingData(const uint8_t* p_tilingdata, T* tilingdata)
// {
//     memcpy(tilingdata, p_tilingdata, sizeof(T));
// }

#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling)           \
    auto var = ((tiling_type*)((uint8_t*)AscendC::GmAlloc(1024)))->member; \
    size_t offset##var = (size_t)(&((tiling_type*)0)->member);             \
    InitTilingData<decltype(var)>(tiling + offset##var, &var);

#define GET_TILING_DATA(tilingData, tilingPointer) \
    MoeTokenPermuteWithRoutingMapGradTilingData tilingData;        \
    InitMoeTokenUnpermuteTilingData(tilingPointer, &tilingData)
#endif