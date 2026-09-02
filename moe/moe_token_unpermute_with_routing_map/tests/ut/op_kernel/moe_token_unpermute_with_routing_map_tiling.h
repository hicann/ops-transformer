/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H_
#define _MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define DTYPE_PROBS float
#define DTYPE_PERMUTE_PROBS half
#define __CCE_UT_TEST__

struct MaskedSelectMTUTilingData {
    uint64_t formerNum;
    uint64_t formerLength;
    uint64_t formertileNum;
    uint64_t formertileLength;
    uint64_t formerlasttileLength;
    uint64_t tailNum;
    uint64_t tailLength;
    uint64_t tailtileNum;
    uint64_t tailtileLength;
    uint64_t taillasttileLength;
    uint64_t tokenNum;
    int64_t needCoreNum;
};

struct MoeTokenUnpermuteWithRoutingMapPadTilingData {
    uint64_t core_num_use;
    uint64_t num_tokens;
    uint64_t num_experts;
    uint64_t capacity;
    uint64_t front_core;
    uint64_t loop_time_each_front_core;
    uint64_t num_tokens_each_front_core;
    uint64_t num_tokens_front_core_each_loop;
    uint64_t num_tokens_front_core_last_loop;
    uint64_t tail_core;
    uint64_t loop_time_each_tail_core;
    uint64_t num_tokens_each_tail_core;
    uint64_t num_tokens_tail_core_each_loop;
    uint64_t num_tokens_tail_core_last_loop;
};

struct MoeTokenUnpermuteWithRoutingMapTilingData {
    int64_t hidden_size = 0;
    int64_t top_k = 0;
    int64_t num_out_tokens = 0;
    int64_t hidden_splited_length = 0;
    int64_t hidden_splited_num = 0;
    int64_t hidden_splited_remain = 0;
    int64_t tokens_core_length = 0;
    int64_t tokens_core_remain = 0;
    int64_t tokens_splited_length = 0;
    int64_t tokens_splited_num = 0;
    int64_t tokens_splited_remain = 0;
    int64_t used_core_num = 0;
    int64_t buffer_num = 0;
    MaskedSelectMTUTilingData maskedSelectParamsOp;
    MoeTokenUnpermuteWithRoutingMapPadTilingData moeTokenUnpermuteWithRoutingMapPadTilingData;
};

inline void InitMoeTokenUnpermuteTilingData(uint8_t *tiling, MoeTokenUnpermuteWithRoutingMapTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeTokenUnpermuteWithRoutingMapTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    MoeTokenUnpermuteWithRoutingMapTilingData tilingData; \
    InitMoeTokenUnpermuteTilingData(tilingPointer, &tilingData)
#endif
