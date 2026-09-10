/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "generic_block_sparse_attention_tilingkey.h"
#include "generic_block_sparse_attention_kernel_interface.cpp"

// W8A8 pseudo-quantization only (arch22 / ascend910b):
// FP16/BF16 query + INT8 KV (PA_NZ), no attenMask / pQuantScale / softmaxLse.
extern "C" __global__ __aicore__ void generic_block_sparse_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseBlockIdx,
    __gm__ uint8_t *sparseBlockCount, __gm__ uint8_t *metaData, __gm__ uint8_t *qDequantScale,
    __gm__ uint8_t *kDequantScale, __gm__ uint8_t *vDequantScale, __gm__ uint8_t *cuSeqLengths,
    __gm__ uint8_t *cuSeqLengthsKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    if (TILING_KEY_VAR >= GSA_BASE_ARCH22_TILING) {
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 220)
        TILING_KEY_IS(GSA_FP16_INT8_ARCH22_TILING);
        TILING_KEY_IS(GSA_BF16_INT8_ARCH22_TILING);

#if TILING_KEY_VAR == GSA_FP16_INT8_ARCH22_TILING
        // W8A8 pseudo-quantization: FP16 query + INT8 KV (online antiquant, MSD)
        GsaInferIntfAntiquantArch22<half, float>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, qDequantScale, kDequantScale, vDequantScale, attentionOut, nullptr, user, tiling);
#elif TILING_KEY_VAR == GSA_BF16_INT8_ARCH22_TILING
        // W8A8 pseudo-quantization: BF16 query + INT8 KV (online antiquant, MSD)
        GsaInferIntfAntiquantArch22<bfloat16_t, float>(
            query, key, value, sparseBlockIdx, sparseBlockCount, metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ,
            sequsedKv, blockTable, qDequantScale, kDequantScale, vDequantScale, attentionOut, nullptr, user, tiling);
#endif
#endif
    }
}
