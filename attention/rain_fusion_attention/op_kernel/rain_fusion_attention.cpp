/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "rain_fusion_attention_tilingkey.h"
#include "rain_fusion_attention_interface.cpp"

// ============================================================================
// Kernel Entry Point
// ============================================================================

/**
 * @brief RainFusionAttention算子的kernel入口函数
 *
 * 这是NPU上执行的入口点，使用extern "C"以避免C++名称修饰，
 * 使用__global__和__aicore__修饰符标识为全局kernel函数
 */
extern "C" __global__ __aicore__ void rain_fusion_attention(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* selectIdx,
    __gm__ uint8_t* selectNumIdx, __gm__ uint8_t* blockShape, __gm__ uint8_t* mask, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKv, __gm__ uint8_t* blockTable, __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    if (TILING_KEY_VAR >= RFA_BASE_TILING) {
        __gm__ uint8_t* user = GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        // 读取tilingKey进行kernel分发
        __gm__ RainFusionAttentionTilingData* tilingDataPtr =
            reinterpret_cast<__gm__ RainFusionAttentionTilingData*>(tiling);
        uint64_t tilingKey = tilingDataPtr->tilingKey;

        TILING_KEY_IS(QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING);
        TILING_KEY_IS(QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_RFA_TILING);
        TILING_KEY_IS(QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING);
        TILING_KEY_IS(QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING);
        TILING_KEY_IS(QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_RFA_TILING);
        TILING_KEY_IS(QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING);

#if TILING_KEY_VAR == QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING
        RainFusion::RainFusionAttentionInfer<half, float, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockShape,
            selectIdx, selectNumIdx, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_RFA_TILING
        RainFusion::RainFusionAttentionInfer<half, half, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockShape,
            selectIdx, selectNumIdx, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING
        RainFusion::RainFusionAttentionInfer<bfloat16_t, float, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockShape,
            selectIdx, selectNumIdx, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING
        RainFusion::RainFusionAttentionInfer<half, float, Epilogue::LseMode::NONE, 1, 1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockShape,
            selectIdx, selectNumIdx, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_RFA_TILING
        RainFusion::RainFusionAttentionInfer<half, half, Epilogue::LseMode::NONE, 1, 1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockShape,
            selectIdx, selectNumIdx, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING
        RainFusion::RainFusionAttentionInfer<bfloat16_t, float, Epilogue::LseMode::NONE, 1, 1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockShape,
            selectIdx, selectNumIdx, user, softmaxLse, tiling);
#else
        RainFusion::RainFusionAttentionInfer<half, half, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockShape,
            selectIdx, selectNumIdx, user, softmaxLse, tiling);
#endif
    }
}
