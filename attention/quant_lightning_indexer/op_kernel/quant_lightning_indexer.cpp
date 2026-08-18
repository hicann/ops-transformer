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
 * \file quant_lightning_indexer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_lightning_indexer_template_tiling_key.h"

#if (__CCE_AICORE__ == 310)
    #include "arch35/quant_lightning_indexer_kernel_arch35.h"

#else
    #include "arch22/quant_lightning_indexer_kernel_arch22.h"
#endif

using namespace QLIKernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                         \
    do {                                                                                                     \
        templateClass<QLIType<__VA_ARGS__>> op;                                                              \
        GET_TILING_DATA_WITH_STRUCT(QLITilingData, tiling_data_in, tiling);                                  \
        const QLITilingData *__restrict tiling_data = &tiling_data_in;                                       \
        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \
                sparseIndices, user, tiling_data, &tPipe);                                                   \
        op.Process();                                                                                        \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void quant_lightning_indexer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                                   __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                                   __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                                   __gm__ uint8_t *blocktable, __gm__ uint8_t *sparseIndices,
                                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 310)
    if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN) {
        if (ORIG_DTYPE_WEIGHTS == DT_FLOAT16) {
            INVOKE_LI_NO_KFC_OP_IMPL(QuantLightningIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                half, half, float32_t, uint16_t);
        } else {
            INVOKE_LI_NO_KFC_OP_IMPL(QuantLightningIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                bfloat16_t, float32_t, float32_t, uint16_t);
        }
    } else if (ORIG_DTYPE_QUERY == DT_HIFLOAT8) {
        INVOKE_LI_NO_KFC_OP_IMPL(QuantLightningIndexerKernel, hifloat8, hifloat8, int32_t,
                                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                                bfloat16_t, float32_t, float32_t, uint16_t);
    } else {
        INVOKE_LI_NO_KFC_OP_IMPL(QuantLightningIndexerKernel, int8_t, int8_t, int32_t,
                                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                                half, half, int32_t, uint32_t);
    }
#else
    INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, int8_t, int8_t, int32_t,
                                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
#endif
}
