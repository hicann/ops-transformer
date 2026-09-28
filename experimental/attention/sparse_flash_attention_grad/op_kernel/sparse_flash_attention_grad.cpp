/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_grad.cpp
 * \brief
 */
#include "sparse_flash_attention_grad_ksplit_det.h"
#include "sparse_flash_attention_grad_post.h"
#include "kernel_operator.h"
using namespace AscendC;

constexpr static uint32_t ND = 0;
constexpr static uint32_t NZ = 1;
constexpr static uint32_t TND = 3;

// 切 K 确定性路径（2000 系列，仅 TND）
#define INVOKE_SELECTED_ATTENTION_KSPLIT_DET_IMPL(INPUT_TYPE, ATTEN_ENABLE, HAS_ROPE) \
    do { \
        __gm__ uint8_t *user = GetUserWorkspace(workspace); \
        GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionGradBasicTilingData, tiling_data_in, tiling_data); \
        const SparseFlashAttentionGradBasicTilingData *__restrict tilingData = &tiling_data_in; \
        SFAG_BASIC::SelectedAttentionGradKsplitDet<SFAG_BASIC::SFAG_TYPE< \
            SparseFlashAttentionGradBasicTilingData, INPUT_TYPE, TND, ATTEN_ENABLE, HAS_ROPE, false>> \
            op; \
        op.Process(query, key, value, out, d_out, softmax_max, softmax_sum, sparse_indices, cur_seq_qlen, \
                   cur_seq_kvlen, query_rope, key_rope, dq, dk, dv, dq_rope, dk_rope, user, tilingData); \
    } while (0)

extern "C" __global__ __aicore__ void sparse_flash_attention_grad(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparse_indices,
    __gm__ uint8_t *d_out, __gm__ uint8_t *out, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
    __gm__ uint8_t *cur_seq_qlen, __gm__ uint8_t *cur_seq_kvlen, __gm__ uint8_t *query_rope, __gm__ uint8_t *key_rope,
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dq_rope, __gm__ uint8_t *dk_rope,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    if (TILING_KEY_IS(2000)) {
        INVOKE_SELECTED_ATTENTION_KSPLIT_DET_IMPL(half, false, false);
        return;
    } else if (TILING_KEY_IS(2100)) {
        INVOKE_SELECTED_ATTENTION_KSPLIT_DET_IMPL(half, true, false);
        return;
    } else if (TILING_KEY_IS(2010)) {
        INVOKE_SELECTED_ATTENTION_KSPLIT_DET_IMPL(half, false, true);
        return;
    } else if (TILING_KEY_IS(2110)) {
        INVOKE_SELECTED_ATTENTION_KSPLIT_DET_IMPL(half, true, true);
        return;
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16)
    if (TILING_KEY_IS(2000)) {
        INVOKE_SELECTED_ATTENTION_KSPLIT_DET_IMPL(bfloat16_t, false, false);
        return;
    } else if (TILING_KEY_IS(2100)) {
        INVOKE_SELECTED_ATTENTION_KSPLIT_DET_IMPL(bfloat16_t, true, false);
        return;
    } else if (TILING_KEY_IS(2010)) {
        INVOKE_SELECTED_ATTENTION_KSPLIT_DET_IMPL(bfloat16_t, false, true);
        return;
    } else if (TILING_KEY_IS(2110)) {
        INVOKE_SELECTED_ATTENTION_KSPLIT_DET_IMPL(bfloat16_t, true, true);
        return;
    }
#endif
}
