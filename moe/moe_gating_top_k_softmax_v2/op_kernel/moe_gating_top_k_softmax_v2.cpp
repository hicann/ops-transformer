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
 * \file moe_gating_top_k_softmax_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "moe_gating_top_k_softmax_v2_ek_fullload.h"
#include "moe_gating_top_k_softmax_v2_k_fullload.h"
#include "moe_gating_top_k_softmax_v2_k_renorm.h"
#include "moe_gating_top_k_softmax_v2_perf.h"

using namespace MoeGatingTopKSoftmaxV2;

#define MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(INPUT_TYPE, BUFFER_NUM, RENORM) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKSoftmaxV2EKFullLoadTilingData, tiling_data_in, tiling); \
        const MoeGatingTopKSoftmaxV2EKFullLoadTilingData* __restrict tilingData = &tiling_data_in; \
        MoeGatingTopKSoftmaxV2EKFullLoad<INPUT_TYPE, BUFFER_NUM, RENORM> op; \
        op.Init(gating, finished, out, indicesOut, softmaxOut, workspace, tilingData); \
        op.Process(); \
    } while (0)

#define MOE_GATING_TOP_K_SOFTMAX_V2_K_FULL_LOAD_IMPL(INPUT_TYPE, RENORM) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKSoftmaxV2KFullLoadTilingData, tiling_data_in, tiling); \
        const MoeGatingTopKSoftmaxV2KFullLoadTilingData* __restrict tilingData = &tiling_data_in; \
        MoeGatingTopKSoftmaxV2KFullLoad<INPUT_TYPE, RENORM> op; \
        op.Init(gating, finished, out, indicesOut, softmaxOut, workspace, tilingData); \
        op.Process(); \
    } while (0)

#define MOE_GATING_TOP_K_SOFTMAX_V2_K_RENORM_IMPL(INPUT_TYPE, RENORM) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKSoftmaxV2KFullLoadTilingData, tiling_data_in, tiling); \
        const MoeGatingTopKSoftmaxV2KFullLoadTilingData* __restrict tilingData = &tiling_data_in; \
        MoeGatingTopKSoftmaxV2KRenorm<INPUT_TYPE, RENORM> op; \
        op.Init(gating, finished, out, indicesOut, softmaxOut, workspace, tilingData); \
        op.Process(); \
    } while (0)

#define MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(INPUT_TYPE, RENORM, COL_RANGE) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKSoftmaxV2PerfTilingData, tiling_data_in, tiling); \
        const MoeGatingTopKSoftmaxV2PerfTilingData* __restrict tilingData = &tiling_data_in; \
        MoeGatingTopKSoftmaxV2Perf<INPUT_TYPE, RENORM, COL_RANGE> op; \
        op.Init(gating, finished, out, indicesOut, softmaxOut, workspace, tilingData); \
        op.Process(tilingData); \
    } while (0)

extern "C" __global__ __aicore__ void moe_gating_top_k_softmax_v2(GM_ADDR gating, GM_ADDR finished, GM_ADDR out,
                                                                  GM_ADDR indicesOut, GM_ADDR softmaxOut,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (g_coreType == AIC) {
        return;
    }

    if (TILING_KEY_IS(101010)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(float, 1, 0);
        return;
    } else if (TILING_KEY_IS(101011)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(float, 2, 0);
        return;
    } else if (TILING_KEY_IS(101020)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(half, 1, 0);
        return;
    } else if (TILING_KEY_IS(101021)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(half, 2, 0);
        return;
    } else if (TILING_KEY_IS(101030)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(bfloat16_t, 1, 0);
        return;
    } else if (TILING_KEY_IS(101031)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(bfloat16_t, 2, 0);
        return;
    } else if (TILING_KEY_IS(101110)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(float, 1, 1);
        return;
    } else if (TILING_KEY_IS(101111)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(float, 2, 1);
        return;
    } else if (TILING_KEY_IS(101120)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(half, 1, 1);
        return;
    } else if (TILING_KEY_IS(101121)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(half, 2, 1);
        return;
    } else if (TILING_KEY_IS(101130)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(bfloat16_t, 1, 1);
        return;
    } else if (TILING_KEY_IS(101131)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_EK_FULL_LOAD_IMPL(bfloat16_t, 2, 1);
        return;
    } else if (TILING_KEY_IS(102011)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_K_FULL_LOAD_IMPL(float, 0);
        return;
    } else if (TILING_KEY_IS(102021)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_K_FULL_LOAD_IMPL(half, 0);
        return;
    } else if (TILING_KEY_IS(102031)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_K_FULL_LOAD_IMPL(bfloat16_t, 0);
        return;
    } else if (TILING_KEY_IS(102111)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_K_RENORM_IMPL(float, 1);
        return;
    } else if (TILING_KEY_IS(102121)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_K_RENORM_IMPL(half, 1);
        return;
    } else if (TILING_KEY_IS(102131)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_K_RENORM_IMPL(bfloat16_t, 1);
        return;
    } else if (TILING_KEY_IS(103011)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(float, 0, ColRangeEnum::SMALLER_THAN_8);
        return;
    } else if (TILING_KEY_IS(103021)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(half, 0, ColRangeEnum::SMALLER_THAN_8);
        return;
    } else if (TILING_KEY_IS(103031)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(bfloat16_t, 0, ColRangeEnum::SMALLER_THAN_8);
        return;
    } else if (TILING_KEY_IS(103111)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(float, 1, ColRangeEnum::SMALLER_THAN_8);
        return;
    } else if (TILING_KEY_IS(103121)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(half, 1, ColRangeEnum::SMALLER_THAN_8);
        return;
    } else if (TILING_KEY_IS(103131)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(bfloat16_t, 1, ColRangeEnum::SMALLER_THAN_8);
        return;
    } else if (TILING_KEY_IS(103012)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(float, 0, ColRangeEnum::FROM_8_TO_64);
        return;
    } else if (TILING_KEY_IS(103022)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(half, 0, ColRangeEnum::FROM_8_TO_64);
        return;
    } else if (TILING_KEY_IS(103032)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(bfloat16_t, 0, ColRangeEnum::FROM_8_TO_64);
        return;
    } else if (TILING_KEY_IS(103112)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(float, 1, ColRangeEnum::FROM_8_TO_64);
        return;
    } else if (TILING_KEY_IS(103122)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(half, 1, ColRangeEnum::FROM_8_TO_64);
        return;
    } else if (TILING_KEY_IS(103132)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(bfloat16_t, 1, ColRangeEnum::FROM_8_TO_64);
        return;
    } else if (TILING_KEY_IS(103013)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(float, 0, ColRangeEnum::BIGGER_THAN_64);
        return;
    } else if (TILING_KEY_IS(103023)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(half, 0, ColRangeEnum::BIGGER_THAN_64);
        return;
    } else if (TILING_KEY_IS(103033)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(bfloat16_t, 0, ColRangeEnum::BIGGER_THAN_64);
        return;
    } else if (TILING_KEY_IS(103113)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(float, 1, ColRangeEnum::BIGGER_THAN_64);
        return;
    } else if (TILING_KEY_IS(103123)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(half, 1, ColRangeEnum::BIGGER_THAN_64);
        return;
    } else if (TILING_KEY_IS(103133)) {
        MOE_GATING_TOP_K_SOFTMAX_V2_PERF_IMPL(bfloat16_t, 1, ColRangeEnum::BIGGER_THAN_64);
        return;
    }
}
