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
 * \file lightning_indexer_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lightning_indexer_grad_template_tiling_key.h"
#include "lightning_indexer_grad_kernel_pre.h"
#include "lightning_indexer_grad_kernel.h"
#include "lightning_indexer_grad_kernel_post.h"
#include "lightning_indexer_grad_tiling.h"

using namespace LigKernel;
#define INVOKE_LIG_NO_KFC_OP_IMPL(...) \
    do { \
        LIG_COPY_TILING_DATA(tiling); \
        TPipe pipePre; \
        LIGVectorPre<LIGType<__VA_ARGS__>> opPre; \
        opPre.Init(&pipePre, dq, dweights, user, tilingData, actual_seq_lengths_query); \
        opPre.Process(); \
        opPre.SyncALLCores(); \
        pipePre.Destroy(); \
        LIGKernel<LIGType<__VA_ARGS__>> op; \
        TPipe pipeOp; \
        op.Init(query, key, dy, sparse_indices, weights, actual_seq_lengths_query, actual_seq_lengths_key, dq, dk, \
                dweights, user, tilingData, &pipeOp); \
        op.Process(); \
        pipeOp.Destroy(); \
        TPipe pipePost; \
        LIGVectorPost<LIGType<__VA_ARGS__>> opPost; \
        opPost.Init(&pipePost, dk, user, tilingData); \
        opPost.Process(); \
        pipePost.Destroy(); \
    } while (0)

#define LIG_COPY_TILING_DATA(tiling) \
    REGISTER_TILING_DEFAULT(optiling::LIGTilingData); \
    GET_TILING_DATA_WITH_STRUCT(optiling::LIGTilingData, tilingDataIn, tiling); \
    const optiling::LIGTilingData* __restrict tilingData = &tilingDataIn;

template <int DATATYPE, int LAYOUT>
__global__ __aicore__ void lightning_indexer_grad(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* dy,
                                                  __gm__ uint8_t* sparse_indices, __gm__ uint8_t* weights,
                                                  __gm__ uint8_t* actual_seq_lengths_query,
                                                  __gm__ uint8_t* actual_seq_lengths_key, __gm__ uint8_t* dq,
                                                  __gm__ uint8_t* dk, __gm__ uint8_t* dweights,
                                                  __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
#if (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
#else
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    REGISTER_TILING_DEFAULT(optiling::LIGTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if constexpr (DATATYPE == LIG_TPL_FP16 && LAYOUT == LIG_LAYOUT_BSND) {
        INVOKE_LIG_NO_KFC_OP_IMPL(half, LIG_LAYOUT::BSND);
    } else if constexpr (DATATYPE == LIG_TPL_FP16 && LAYOUT == LIG_LAYOUT_TND) {
        INVOKE_LIG_NO_KFC_OP_IMPL(half, LIG_LAYOUT::TND);
    } else if constexpr (DATATYPE == LIG_TPL_BF16 && LAYOUT == LIG_LAYOUT_BSND) {
        INVOKE_LIG_NO_KFC_OP_IMPL(bfloat16_t, LIG_LAYOUT::BSND);
    } else if constexpr (DATATYPE == LIG_TPL_BF16 && LAYOUT == LIG_LAYOUT_TND) {
        INVOKE_LIG_NO_KFC_OP_IMPL(bfloat16_t, LIG_LAYOUT::TND);
    }
#endif
}
