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
 * \file batch_mat_mul_reduce_scatter_allto_all.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../batch_mat_mul_reduce_scatter_allto_all.h"
#include "../batch_mat_mul_reduce_scatter_allto_all_tiling_key.h"
#include "../batch_mat_mul_reduce_scatter_allto_all_shard_zero.h"

using namespace AscendC;

#ifdef __CCE_KT_TEST__
namespace {
REGISTER_TILING_DEFAULT(BatchMatMulReduceScatterAlltoAllTilingData);
}
#endif

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

#define INVOKE_BMMRSATA_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<BMMRSATAType<__VA_ARGS__>> op; \
        op.Init(xGM, weightGM, biasGM, yGM, userWorkspace, &tilingData, &pipe, hcclInitTiling, reduceScatterCcTiling, \
                alltoAllCcTiling); \
        op.Process(); \
    } while (0)

template <int YShard, bool WeightTranspose, bool IsBias, bool LiteMode>
__global__ __aicore__ void batch_mat_mul_reduce_scatter_allto_all(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM,
                                                                  GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // 强制kernelCV核配比1:2
    if (workspaceGM == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tilingGM);
    REGISTER_TILING_DEFAULT(BatchMatMulReduceScatterAlltoAllTilingData);
    auto tiling = (__gm__ BatchMatMulReduceScatterAlltoAllTilingData *)tilingGM;
    __gm__ void *hcclInitTiling = (__gm__ void *)(&(tiling->hcclInitTiling));
    __gm__ void *reduceScatterCcTiling = (__gm__ void *)(&(tiling->reduceScatterCcTiling));
    __gm__ void *alltoAllCcTiling = (__gm__ void *)(&(tiling->alltoAllCcTiling));
    TPipe pipe;
#if (ORIG_DTYPE_X == DT_FLOAT16)
    using X_TYPE = half;
    using BIAS_TYPE = half;
    // yShardType = 1
    if constexpr (YShard == 1) {
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, IsBias, YShard, WeightTranspose,
                                LiteMode);
    }
    // yShardType = 0
    else if constexpr (YShard == 0) {
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAllShard0, X_TYPE, BIAS_TYPE, IsBias, YShard,
                                WeightTranspose, LiteMode);
    }
#endif

#if (ORIG_DTYPE_X == DT_BF16)
    using X_TYPE = bfloat16_t;
    using BIAS_TYPE = float;
    // yShardType = 1
    if constexpr (YShard == 1) {
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAll, X_TYPE, BIAS_TYPE, IsBias, YShard, WeightTranspose,
                                LiteMode);
    }
    // yShardType = 0
    else if constexpr (YShard == 0) {
        INVOKE_BMMRSATA_OP_IMPL(BatchMatMulReduceScatterAlltoAllShard0, X_TYPE, BIAS_TYPE, IsBias, YShard,
                                WeightTranspose, LiteMode);
    }
#endif
}
