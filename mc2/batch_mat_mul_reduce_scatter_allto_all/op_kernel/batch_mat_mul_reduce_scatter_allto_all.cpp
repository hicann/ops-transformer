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
 * \file batch_mat_mul_reduce_scatter_allto_all.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "batch_mat_mul_reduce_scatter_allto_all.h"
#include "batch_mat_mul_reduce_scatter_allto_all_shard_zero.h"

using namespace AscendC;

template <typename X_T, typename BIAS_T, const bool NEED_BIAS, const int64_t Y_SHARD_TYPE, const bool IS_TRANS,
          const bool IS_LITE>
struct BMMRSATAType { // Batch_Mat_Mul_Reduce_Scatter_All_to_All_Type
    using xType = X_T;
    using biasType = BIAS_T;
    static constexpr bool needBias = NEED_BIAS;
    static constexpr int64_t yShardType = Y_SHARD_TYPE;
    static constexpr bool transposeWeight = IS_TRANS;
    static constexpr bool isLite = IS_LITE;
};

#define INVOKE_BMMRSATA_OP_IMPL(templateClass, ...)                             \
    do {                                                                        \
        templateClass<BMMRSATAType<__VA_ARGS__>> op;                            \
        op.Init(xGM, weightGM, biasGM, yGM, userWorkspace, &tilingData, &pipe); \
        op.Process();                                                           \
    } while (0)

extern "C" __global__ __aicore__ void batch_mat_mul_reduce_scatter_allto_all(GM_ADDR xGM, GM_ADDR weightGM,
                                                                             GM_ADDR biasGM, GM_ADDR yGM,
                                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    #ifdef __CCE_KT_TEST__
        REGISTER_TILING_DEFAULT(BatchMatMulReduceScatterAlltoAllTilingData);
    #endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // 强制kernelCV核配比1:2
    if (workspaceGM == nullptr) {return;}
    GM_ADDR userWorkspace = GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {return;}
    GET_TILING_DATA(tilingData, tilingGM);
    TPipe pipe;

#if (ORIG_DTYPE_X == DT_FLOAT16)
    using X_TYPE = half;
    using BIAS_TYPE = half;
    // x fp16
    // yShardType = 1
    if (TILING_KEY_IS(1000000000000000001)) { // no bias, no transpose_weight
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, false, 1, false, false);
    } else if (TILING_KEY_IS(1000000000000000101)) { // bias fp16, no transpose_weight
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, true, 1, false, false);
    } else if (TILING_KEY_IS(1000000000000000011)) { // no bias, transpose_weight
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, false, 1, true, false);
    } else if (TILING_KEY_IS(1000000000000000111)) { // bias fp16, transpose_weight
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, true, 1, true, false);
    } else if (TILING_KEY_IS(1000000000000001001)) { // no bias, no transpose_weight, lite mode
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, false, 1, false, true);
    } else if (TILING_KEY_IS(1000000000000001101)) { // bias fp16, no transpose_weight, lite mode
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, true, 1, false, true);
    } else if (TILING_KEY_IS(1000000000000001011)) { // no bias, transpose_weight, lite mode
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, false, 1, true, true);
    } else if (TILING_KEY_IS(1000000000000001111)) { // bias fp16, transpose_weight, lite mode
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, true, 1, true, true);
    // yShardType = 0
    } else if (TILING_KEY_IS(1000000000000000000)) { // no bias, no transpose_weight, shard0
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAllShard0, X_TYPE, BIAS_TYPE, false, 0, false, false);
    } else if (TILING_KEY_IS(1000000000000000100)) { // bias fp16, no transpose_weight, shard0
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAllShard0, X_TYPE, BIAS_TYPE, true, 0, false, false);
    } else if (TILING_KEY_IS(1000000000000000010)) { // no bias, transpose_weight, shard0
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAllShard0, X_TYPE, BIAS_TYPE, false, 0, true, false);
    } else if (TILING_KEY_IS(1000000000000000110)) { // bias, transpose_weight, shard0
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAllShard0, X_TYPE, BIAS_TYPE, true, 0, true, false);
    }
#endif

#if (ORIG_DTYPE_X == DT_BF16)
    using X_TYPE = bfloat16_t;
    using BIAS_TYPE = float;
    // x bf16
    // yShardType = 1
    if (TILING_KEY_IS(1000000000000000001)) { // no bias, no transpose_weight
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, false, 1, false, false);
    } else if (TILING_KEY_IS(1000000000000000101)) { // bias fp32, no transpose_weight
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, true, 1, false, false);
    } else if (TILING_KEY_IS(1000000000000000011)) { // no bias, transpose_weight
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, false, 1, true, false);
    } else if (TILING_KEY_IS(1000000000000000111)) { // bias fp32, transpose_weight
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, true, 1, true, false);
    } else if (TILING_KEY_IS(1000000000000001001)) { // no bias, no transpose_weight, lite mode
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, false, 1, false, true);
    } else if (TILING_KEY_IS(1000000000000001101)) { // bias fp32, no transpose_weight, lite mode
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, true, 1, false, true);
    } else if (TILING_KEY_IS(1000000000000001011)) { // no bias, transpose_weight, lite mode
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, false, 1, true, true);
    } else if (TILING_KEY_IS(1000000000000001111)) { // bias fp32, transpose_weight, lite mode
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, true, 1, true, true);
    // yShardType = 0
    } else if (TILING_KEY_IS(1000000000000000000)) { // no bias, no transpose_weight, shard0
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAllShard0, X_TYPE, BIAS_TYPE, false, 0, false, false);
    } else if (TILING_KEY_IS(1000000000000000100)) { // bias fp16, no transpose_weight, shard0
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAllShard0, X_TYPE, BIAS_TYPE, true, 0, false, false);
    } else if (TILING_KEY_IS(1000000000000000010)) { // no bias, transpose_weight, shard0
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAllShard0, X_TYPE, BIAS_TYPE, false, 0, true, false);
    } else if (TILING_KEY_IS(1000000000000000110)) { // bias, transpose_weight, shard0
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAllShard0, X_TYPE, BIAS_TYPE, true, 0, true, false);
    }
#endif
}
