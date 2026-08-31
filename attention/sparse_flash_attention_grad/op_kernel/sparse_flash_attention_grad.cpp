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
 * \file sparse_flash_attention_grad.cpp
 * \brief
 */

#if __CCE_AICORE__ == 310
#include "arch35/sparse_flash_attention_grad_template_tiling_key.h"
#include "arch35/sparse_flash_attention_grad_tiling_data_regbase.h"
#include "arch35/sparse_flash_attention_grad_entry_regbase.h"
#else
#include "arch22/sparse_flash_attention_grad_bs1_basic.h"
#include "arch22/sparse_flash_attention_grad_post.h"
#endif

#include "kernel_operator.h"

using namespace AscendC;
constexpr static uint32_t ND = 0;
constexpr static uint32_t NZ = 1;
constexpr static uint32_t TND = 3;

#if __CCE_AICORE__ == 310

template <uint8_t inputDType, bool isTnd, uint16_t gTemplateType, uint16_t s2TemplateType, uint16_t dTemplateType,
          bool isRope, bool deterministic, bool kvMerge, bool isSinks>
__global__ __aicore__ void sparse_flash_attention_grad(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *sparse_indices, __gm__ uint8_t *d_out,
    __gm__ uint8_t *out, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum, __gm__ uint8_t *sinks,
    __gm__ uint8_t *value, __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
    __gm__ uint8_t *query_rope, __gm__ uint8_t *key_rope, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
    __gm__ uint8_t *dq_rope, __gm__ uint8_t *dk_rope, __gm__ uint8_t *d_sinks, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling_data)
{
    REGISTER_TILING_DEFAULT(optiling::sfag::SparseFlashAttentionGradTilingDataRegbase);
    RegbaseSFAG<inputDType, isTnd, gTemplateType, s2TemplateType, dTemplateType, isRope, deterministic, kvMerge,
                isSinks>(query, key, value, sparse_indices, d_out, out, softmax_max, softmax_sum, actual_seq_qlen,
                         actual_seq_kvlen, query_rope, key_rope, dq, dk, dv, dq_rope, dk_rope, sinks, d_sinks,
                         workspace, tiling_data);
}

#else
#define INVOKE_SELECTED_ATTENTION_BASIC_IMPL(INPUT_TYPE, ATTEN_ENABLE, HAS_ROPE, IS_BSND, IS_DETERMINISTIC, KV_MERGE) \
    do { \
        __gm__ uint8_t *user = GetUserWorkspace(workspace); \
        GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionGradBasicTilingData, tiling_data_in, tiling_data); \
        const SparseFlashAttentionGradBasicTilingData *__restrict tilingData = &tiling_data_in; \
        SFAG_BASIC::SelectedAttentionGradBasic< \
            SFAG_BASIC::SFAG_TYPE<SparseFlashAttentionGradBasicTilingData, INPUT_TYPE, TND, ATTEN_ENABLE, HAS_ROPE, \
                                  IS_BSND, IS_DETERMINISTIC, KV_MERGE>> \
            op; \
        __gm__ uint8_t *valueIn = KV_MERGE ? key : value; \
        op.Process(query, key, valueIn, out, d_out, softmax_max, softmax_sum, sparse_indices, actual_seq_qlen, \
                   actual_seq_kvlen, query_rope, key_rope, dq, dk, dv, dq_rope, dk_rope, user, tilingData); \
    } while (0)

#define SFAG_DISPATCH_KEY(KEY, INPUT_TYPE, ATTEN_ENABLE, HAS_ROPE, IS_BSND, IS_DETERMINISTIC, KV_MERGE) \
    if (TILING_KEY_IS(KEY)) { \
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(INPUT_TYPE, ATTEN_ENABLE, HAS_ROPE, IS_BSND, IS_DETERMINISTIC, KV_MERGE); \
        return; \
    }

#define SFAG_DISPATCH_ALL(INPUT_TYPE) \
    SFAG_DISPATCH_KEY(100000, INPUT_TYPE, false, false, false, false, false) \
    SFAG_DISPATCH_KEY(100001, INPUT_TYPE, false, false, false, false, true) \
    SFAG_DISPATCH_KEY(100010, INPUT_TYPE, false, false, false, true, false) \
    SFAG_DISPATCH_KEY(100011, INPUT_TYPE, false, false, false, true, true) \
    SFAG_DISPATCH_KEY(100100, INPUT_TYPE, false, false, true, false, false) \
    SFAG_DISPATCH_KEY(100101, INPUT_TYPE, false, false, true, false, true) \
    SFAG_DISPATCH_KEY(100110, INPUT_TYPE, false, false, true, true, false) \
    SFAG_DISPATCH_KEY(100111, INPUT_TYPE, false, false, true, true, true) \
    SFAG_DISPATCH_KEY(101000, INPUT_TYPE, false, true, false, false, false) \
    SFAG_DISPATCH_KEY(101001, INPUT_TYPE, false, true, false, false, true) \
    SFAG_DISPATCH_KEY(101010, INPUT_TYPE, false, true, false, true, false) \
    SFAG_DISPATCH_KEY(101011, INPUT_TYPE, false, true, false, true, true) \
    SFAG_DISPATCH_KEY(101100, INPUT_TYPE, false, true, true, false, false) \
    SFAG_DISPATCH_KEY(101101, INPUT_TYPE, false, true, true, false, true) \
    SFAG_DISPATCH_KEY(101110, INPUT_TYPE, false, true, true, true, false) \
    SFAG_DISPATCH_KEY(101111, INPUT_TYPE, false, true, true, true, true) \
    SFAG_DISPATCH_KEY(110000, INPUT_TYPE, true, false, false, false, false) \
    SFAG_DISPATCH_KEY(110001, INPUT_TYPE, true, false, false, false, true) \
    SFAG_DISPATCH_KEY(110010, INPUT_TYPE, true, false, false, true, false) \
    SFAG_DISPATCH_KEY(110011, INPUT_TYPE, true, false, false, true, true) \
    SFAG_DISPATCH_KEY(110100, INPUT_TYPE, true, false, true, false, false) \
    SFAG_DISPATCH_KEY(110101, INPUT_TYPE, true, false, true, false, true) \
    SFAG_DISPATCH_KEY(110110, INPUT_TYPE, true, false, true, true, false) \
    SFAG_DISPATCH_KEY(110111, INPUT_TYPE, true, false, true, true, true) \
    SFAG_DISPATCH_KEY(111000, INPUT_TYPE, true, true, false, false, false) \
    SFAG_DISPATCH_KEY(111001, INPUT_TYPE, true, true, false, false, true) \
    SFAG_DISPATCH_KEY(111010, INPUT_TYPE, true, true, false, true, false) \
    SFAG_DISPATCH_KEY(111011, INPUT_TYPE, true, true, false, true, true) \
    SFAG_DISPATCH_KEY(111100, INPUT_TYPE, true, true, true, false, false) \
    SFAG_DISPATCH_KEY(111101, INPUT_TYPE, true, true, true, false, true) \
    SFAG_DISPATCH_KEY(111110, INPUT_TYPE, true, true, true, true, false) \
    SFAG_DISPATCH_KEY(111111, INPUT_TYPE, true, true, true, true, true)

extern "C" __global__ __aicore__ void sparse_flash_attention_grad(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *sparse_indices, __gm__ uint8_t *d_out,
    __gm__ uint8_t *out, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum, __gm__ uint8_t *sinks,
    __gm__ uint8_t *value, __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
    __gm__ uint8_t *query_rope, __gm__ uint8_t *key_rope, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
    __gm__ uint8_t *dq_rope, __gm__ uint8_t *dk_rope, __gm__ uint8_t *d_sinks, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    SFAG_DISPATCH_ALL(half)
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16)
    SFAG_DISPATCH_ALL(bfloat16_t)
#endif
}
#endif
