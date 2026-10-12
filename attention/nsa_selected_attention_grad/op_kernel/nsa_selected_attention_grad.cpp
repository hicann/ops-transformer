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
 * \file nsa_selected_attention_grad.cpp
 * \brief
 */
#include "nsa_selected_attention_grad_bs1.h"
#include "nsa_selected_attention_grad_bs1_basic.h"
#include "nsa_selected_attention_grad_post.h"
#include "kernel_operator.h"
using namespace AscendC;

constexpr static uint32_t ND = 0;
constexpr static uint32_t NZ = 1;
constexpr static uint32_t TND = 3;

#define INVOKE_SELECTED_ATTENTION_IMPL(INPUT_TYPE, ATTEN_ENABLE) \
    do { \
        TPipe pipeIn; \
        SetMaskNorm(); \
        __gm__ uint8_t* user = GetUserWorkspace(workspace); \
        GET_TILING_DATA_WITH_STRUCT(NsaSelectedAttentionGradTilingData, tiling_data_in, tiling_data); \
        const NsaSelectedAttentionGradTilingData* __restrict tilingData = &tiling_data_in; \
        const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1TilingData); \
        const TCubeTiling* __restrict bmm2tiling = &(tilingData->mm2TilingData); \
        const TCubeTiling* __restrict bmm31tiling = &(tilingData->mm31TilingData); \
        const TCubeTiling* __restrict bmm4tiling = &(tilingData->mm4TilingData); \
        const TCubeTiling* __restrict bmm5tiling = &(tilingData->mm5TilingData); \
        SelectedAttentionGrad< \
            NSAG_BASIC::NSAG_TYPE<NsaSelectedAttentionGradTilingData, INPUT_TYPE, TND, ATTEN_ENABLE>> \
            op; \
        REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm2tiling, op.mm3, bmm31tiling, \
                          op.mm4, bmm4tiling, op.mm5, bmm5tiling); \
        op.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices, \
                actual_seq_qlen, actual_seq_kvlen, atten_mask, dq, dk, dv, user, tilingData, &pipeIn); \
        op.Process(); \
        pipeIn.Destroy(); \
        TPipe pipeCast; \
        NsaSelectedAttentionGradPost<INPUT_TYPE, NsaSelectedAttentionGradTilingData, true, TND, ND> opCast; \
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeCast); \
        opCast.Process(); \
    } while (0)

#define INVOKE_SELECTED_ATTENTION_DETERMINISTIC_IMPL(INPUT_TYPE, ATTEN_ENABLE) \
    do { \
        TPipe pipeIn; \
        SetMaskNorm(); \
        __gm__ uint8_t* user = GetUserWorkspace(workspace); \
        GET_TILING_DATA_WITH_STRUCT(NsaSelectedAttentionGradTilingData, tiling_data_in, tiling_data); \
        const NsaSelectedAttentionGradTilingData* __restrict tilingData = &tiling_data_in; \
        const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1TilingData); \
        const TCubeTiling* __restrict bmm2tiling = &(tilingData->mm2TilingData); \
        const TCubeTiling* __restrict bmm31tiling = &(tilingData->mm31TilingData); \
        const TCubeTiling* __restrict bmm4tiling = &(tilingData->mm4TilingData); \
        const TCubeTiling* __restrict bmm5tiling = &(tilingData->mm5TilingData); \
        SelectedAttentionGrad< \
            NSAG_BASIC::NSAG_TYPE<NsaSelectedAttentionGradTilingData, INPUT_TYPE, TND, ATTEN_ENABLE>> \
            op; \
        REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm2tiling, op.mm3, bmm31tiling, \
                          op.mm4, bmm4tiling, op.mm5, bmm5tiling); \
        op.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices, \
                actual_seq_qlen, actual_seq_kvlen, atten_mask, dq, dk, dv, user, tilingData, &pipeIn); \
        op.DeterministicProcess(); \
        pipeIn.Destroy(); \
        TPipe pipeCast; \
        NsaSelectedAttentionGradPost<INPUT_TYPE, NsaSelectedAttentionGradTilingData, true, TND, ND> opCast; \
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeCast); \
        opCast.Process(); \
    } while (0)

#define INVOKE_SELECTED_ATTENTION_BASIC_IMPL(INPUT_TYPE, ATTEN_ENABLE) \
    do { \
        __gm__ uint8_t* user = GetUserWorkspace(workspace); \
        GET_TILING_DATA_WITH_STRUCT(NsaSelectedAttentionGradBasicTilingData, tiling_data_in, tiling_data); \
        const NsaSelectedAttentionGradBasicTilingData* __restrict tilingData = &tiling_data_in; \
        NSAG_BASIC::SelectedAttentionGradBasic< \
            NSAG_BASIC::NSAG_TYPE<NsaSelectedAttentionGradBasicTilingData, INPUT_TYPE, TND, ATTEN_ENABLE>> \
            op; \
        op.Process(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices, \
                   actual_seq_qlen, actual_seq_kvlen, atten_mask, dq, dk, dv, user, tilingData); \
    } while (0)

extern "C" __global__ __aicore__ void nsa_selected_attention_grad(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* attention_out,
    __gm__ uint8_t* attention_out_grad, __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum,
    __gm__ uint8_t* topk_indices, __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen,
    __gm__ uint8_t* atten_mask, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* workspace,
    __gm__ uint8_t* tiling_data)
{
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        INVOKE_SELECTED_ATTENTION_IMPL(half, false);
        return;
    } else if (TILING_KEY_IS(1)) {
        INVOKE_SELECTED_ATTENTION_IMPL(half, true);
        return;
    } else if (TILING_KEY_IS(10)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(half, false);
        return;
    } else if (TILING_KEY_IS(11)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(half, true);
    } else if (TILING_KEY_IS(2)) {
        INVOKE_SELECTED_ATTENTION_DETERMINISTIC_IMPL(half, false);
        return;
    } else if (TILING_KEY_IS(3)) {
        INVOKE_SELECTED_ATTENTION_DETERMINISTIC_IMPL(half, true);
        return;
    }

#endif

#if (ORIG_DTYPE_QUERY == DT_BF16)
    if (TILING_KEY_IS(0)) {
        INVOKE_SELECTED_ATTENTION_IMPL(bfloat16_t, false);
        return;
    } else if (TILING_KEY_IS(1)) {
        INVOKE_SELECTED_ATTENTION_IMPL(bfloat16_t, true);
        return;
    } else if (TILING_KEY_IS(10)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(bfloat16_t, false);
        return;
    } else if (TILING_KEY_IS(11)) {
        INVOKE_SELECTED_ATTENTION_BASIC_IMPL(bfloat16_t, true);
    } else if (TILING_KEY_IS(2)) {
        INVOKE_SELECTED_ATTENTION_DETERMINISTIC_IMPL(bfloat16_t, false);
        return;
    } else if (TILING_KEY_IS(3)) {
        INVOKE_SELECTED_ATTENTION_DETERMINISTIC_IMPL(bfloat16_t, true);
        return;
    }
#endif
}
