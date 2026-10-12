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
 * \file attention_worker_combine.cpp
 * \brief
 */

#include "attention_worker_combine_split_bs.h"
#include "attention_worker_combine_split_h.h"
#include "attention_worker_combine_split_k.h"
#include "kernel_operator.h"

#define TILING_KEY_DIVIDE_BS_FP16 10000UL
#define TILING_KEY_DIVIDE_BS_BF16 10001UL
#define TILING_KEY_DIVIDE_H_FP16 10010UL
#define TILING_KEY_DIVIDE_H_BF16 10011UL
#define TILING_KEY_DIVIDE_K_FP16 10020UL
#define TILING_KEY_DIVIDE_K_BF16 10021UL

extern "C" __global__ __aicore__ void attention_worker_combine(GM_ADDR schedule_context, GM_ADDR expert_scales,
                                                               GM_ADDR layer_id, GM_ADDR y, GM_ADDR next_layer_id,
                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(TILING_KEY_DIVIDE_BS_FP16)) {
        KernelAttentionWorkerCombineSplitBS<half> op(&pipe, &tiling_data);
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_BS_BF16)) {
        KernelAttentionWorkerCombineSplitBS<bfloat16_t> op(&pipe, &tiling_data);
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_H_FP16)) {
        KernelAttentionWorkerCombineSplitH<half> op(&pipe, &tiling_data);
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_H_BF16)) {
        KernelAttentionWorkerCombineSplitH<bfloat16_t> op(&pipe, &tiling_data);
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_K_FP16)) {
        KernelAttentionWorkerCombineSplitK<half> op(&pipe, &tiling_data);
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_K_BF16)) {
        KernelAttentionWorkerCombineSplitK<bfloat16_t> op(&pipe, &tiling_data);
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id);
        op.Process();
    }
}
