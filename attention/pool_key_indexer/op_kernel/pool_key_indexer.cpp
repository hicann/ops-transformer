/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "pool_key_indexer_template_tiling_key.h"

#if (__CCE_AICORE__ == 310)
#include "arch35/pool_key_indexer_kernel_arch35.h"
#else
#include "arch22/pool_key_indexer_kernel_arch22.h"
#endif

using namespace PkiKernel;

#define INVOKE_PKI_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<PkiType<__VA_ARGS__>> op; \
        GET_TILING_DATA_WITH_STRUCT(PoolKeyIndexerTilingData, tiling_data_in, tiling); \
        const PoolKeyIndexerTilingData *__restrict tiling_data = &tiling_data_in; \
        op.Init(query, poolKey, weights, poolTailK, actualSeqQ, actualSeqK, blockTable, qDescale, kDescale, \
                sparseIndices, sparseValues, user, tiling_data, &tPipe); \
        op.Process(); \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int LAYOUT_Q, int LAYOUT_K, int MASK_MODE, int QUANT_MODE, int RETURN_VALUE>
__global__ __aicore__ void pool_key_indexer(__gm__ uint8_t *query, __gm__ uint8_t *poolKey, __gm__ uint8_t *weights,
                                            __gm__ uint8_t *poolTailK, __gm__ uint8_t *actualSeqQ,
                                            __gm__ uint8_t *actualSeqK, __gm__ uint8_t *blockTable,
                                            __gm__ uint8_t *qDescale, __gm__ uint8_t *kDescale,
                                            __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                            __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 310)
    // arch35 (A5/950): radix/histogram TopK, dual-dst Fixpipe, shared UB
    // NOTE: must use `if constexpr`, NOT plain `if`: ORIG_DTYPE_* are -D macros
    // (per-bin constants), and a plain `if` still instantiates ALL dtype
    // branches, including fp8_e4m3fn_t. Instantiating the fp8 branch for
    // bf16/fp16 bins drags scalar-fp8 semantics into codegen, which the
    // bisheng backend rejects with
    //   "fp8eXmY/... type only supports pointer operations, scalar float type
    //    semantics are not supported" (exit code 70).
    // QUANT_MODE tpl 值: 0=none / 1=fp8 per-token(scale=float, vector 融合) /
    // 2=mxFP8(scale=e8m0, LoadData-Mx 硬件反量化)
    if constexpr (ORIG_DTYPE_QUERY == DT_BF16) {
        INVOKE_PKI_OP_IMPL(PoolKeyIndexerKernel, bfloat16_t, bfloat16_t, int32_t, PKI_LAYOUT(LAYOUT_Q),
                           PKI_LAYOUT(LAYOUT_K));
    } else if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16) {
        INVOKE_PKI_OP_IMPL(PoolKeyIndexerKernel, half, half, int32_t, PKI_LAYOUT(LAYOUT_Q), PKI_LAYOUT(LAYOUT_K));
    } else if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN) {
        // 量化场景 weights 支持 FP16/BF16(规格+tiling 均允许), kernel 按
        // ORIG_DTYPE_WEIGHTS 分发 WEIGHTS_T(参考 quant_lightning_indexer 的双 dtype 模式)。
        // 注意模板参数顺序: ..., DT_W_FLAG, SCALE_T, WEIGHTS_T —— scale 在前, weights 在后
        if constexpr (ORIG_DTYPE_WEIGHTS == DT_BF16) {
            if constexpr (QUANT_MODE == PKI_TPL_QUANT_MXFP8) {
                INVOKE_PKI_OP_IMPL(PoolKeyIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, PKI_LAYOUT(LAYOUT_Q),
                                   PKI_LAYOUT(LAYOUT_K), false, fp8_e8m0_t, bfloat16_t);
            } else {
                // per-token(float scale)与兜底(quantMode=-1 已被 tiling 拒绝)
                INVOKE_PKI_OP_IMPL(PoolKeyIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, PKI_LAYOUT(LAYOUT_Q),
                                   PKI_LAYOUT(LAYOUT_K), false, float, bfloat16_t);
            }
        } else {
            if constexpr (QUANT_MODE == PKI_TPL_QUANT_MXFP8) {
                INVOKE_PKI_OP_IMPL(PoolKeyIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, PKI_LAYOUT(LAYOUT_Q),
                                   PKI_LAYOUT(LAYOUT_K), false, fp8_e8m0_t, half);
            } else {
                INVOKE_PKI_OP_IMPL(PoolKeyIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, PKI_LAYOUT(LAYOUT_Q),
                                   PKI_LAYOUT(LAYOUT_K), false, float, half);
            }
        }
    }
#else
    // arch22 (A2/A3): Sort+MrgSort TopK, single-dst Fixpipe, GM workspace
    if constexpr (DT_Q == PKI_TPL_FP16 && DT_K == PKI_TPL_FP16) {
        INVOKE_PKI_OP_IMPL(PoolKeyIndexerKernel, half, half, int32_t, PKI_LAYOUT(LAYOUT_Q), PKI_LAYOUT(LAYOUT_K));
    } else {
        INVOKE_PKI_OP_IMPL(PoolKeyIndexerKernel, bfloat16_t, bfloat16_t, int32_t, PKI_LAYOUT(LAYOUT_Q),
                           PKI_LAYOUT(LAYOUT_K));
    }
#endif
}
