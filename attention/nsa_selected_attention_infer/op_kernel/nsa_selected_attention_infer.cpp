/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file nsa_selected_attention_infer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "nsa_selected_attention_infer.h"
#include "nsa_public_define.h"

#define INVOKE_NSA_NO_KFC_OP_IMPL(templateClass, ...)                                                       \
    do {                                                                                                    \
        templateClass<NSAType<__VA_ARGS__>> op;                                                             \
        COPY_TILING_DATA_ALL(tiling);                                                                       \
        op.Init(query, key, value, topkIndices, attenMask, blocktable, actualQSeqLengths, actualKVSeqLengths,attentionOut,    \
                user, tiling_data, tiling, &tPipe);                                                         \
        op.Process();                                                                                       \
    } while (0)

#define COPY_TILING_DATA_ALL(tiling)                                                                        \
    GET_TILING_DATA_MEMBER(NsaSelectAttentionInferTilingDataV2, tilingBase, tiling_data_in, tiling);        \
    const NsaSelectAttentionInferTilingData *__restrict tiling_data = &tiling_data_in;                      \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                     \
    const TCubeTiling *__restrict bmm2tiling = nullptr;

extern "C" __global__ __aicore__ void
nsa_selected_attention_infer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                           __gm__ uint8_t *topkIndices, __gm__ uint8_t *attenMask, __gm__ uint8_t *blocktable,
                           __gm__ uint8_t *actualQSeqLengths, __gm__ uint8_t *actualKVSeqLengths,
                           __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    #if (ORIG_DTYPE_QUERY == DT_FLOAT16)
        if (TILING_KEY_IS(0)) {
            INVOKE_NSA_NO_KFC_OP_IMPL(NsaSelectAttentionInfer, half, half, half, half, false, false,
                LAYOUT::BSND);
        } else if (TILING_KEY_IS(1)) {
            INVOKE_NSA_NO_KFC_OP_IMPL(NsaSelectAttentionInfer, half, half, half, half, false, false,
                LAYOUT::TND);
    }
    #endif
    #if (ORIG_DTYPE_QUERY == DT_BF16)
        if (TILING_KEY_IS(0)) {
            INVOKE_NSA_NO_KFC_OP_IMPL(NsaSelectAttentionInfer, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false,
                LAYOUT::BSND);
        } else if (TILING_KEY_IS(1)) {
            INVOKE_NSA_NO_KFC_OP_IMPL(NsaSelectAttentionInfer, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false,
                LAYOUT::TND);
    }
    #endif
}
