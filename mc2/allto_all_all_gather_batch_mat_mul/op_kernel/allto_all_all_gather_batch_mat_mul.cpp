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
 * \file allto_all_all_gather_batch_mat_mul.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "allto_all_all_gather_batch_mat_mul.h"
#include "allto_all_all_gather_batch_mat_mul_shard_h.h"

using namespace AscendC;

#if ORIG_DTYPE_X == DT_FLOAT16
using DT_X = half;
#else
using DT_X = bfloat16_t;
#endif

#if ORIG_DTYPE_X == DT_FLOAT16
using DT_BIAS = half;
#else
using DT_BIAS = float;
#endif

#define AlltoAllAllGatherBatchMatMul_IMPL_CLASS(...)                                     \
    do {                                                                                 \
        TPipe pipe;                                                                      \                                                         
        AlltoAllAllGatherBatchMatMul<DT_X, DT_BIAS, __VA_ARGS__> op;                     \
        op.Init(xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspaceGM, &pipe, &tilingData);  \
        op.Process();                                                                    \
    } while (0)

#define AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(...)                                     \
    do {                                                                                 \
        TPipe pipe;                                                                      \                                                         
        AlltoAllAllGatherBatchMatMulShardH<DT_X, DT_BIAS, __VA_ARGS__> op;                     \
        op.Init(xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspaceGM, &pipe, &tilingData);  \
        op.Process();                                                                    \
    } while (0)

extern "C" __global__ __aicore__ void allto_all_all_gather_batch_mat_mul(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                                         GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    #ifdef __CCE_KT_TEST__
        REGISTER_TILING_DEFAULT(AlltoAllAllGatherBatchMatMulTilingData);
    #endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tilingGM);
/*
Tiling key
1. 个位shardType 0/1
2. 十位transposeweight 0/1
3. 百位bias 0/1
4. 千位y2/y3 0/1/2/3
*/

    if (TILING_KEY_IS(1000000000000000001)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, false, false, false, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000000011)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, true, false, false, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000000101)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, false, true, false, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000000111)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, true, true, false, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000001001)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, false, false, true, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000001011)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, true, false, true, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000001101)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, false, true, true, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000001111)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, true, true, true, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000002001)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, false, false, false, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000002011)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, true, false, false, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000002101)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, false, true, false, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000002111)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, true, true, false, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000003001)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, false, false, true, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000003011)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, true, false, true, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000003101)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, false, true, true, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000003111)) {
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(1, true, true, true, true);
        return;
    }
    // shard 0 tilingKey
        if (TILING_KEY_IS(1000000000000000000)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, false, false, false, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000000010)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, true, false, false, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000000100)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, false, true, false, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000000110)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, true, true, false, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000001000)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, false, false, true, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000001010)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, true, false, true, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000001100)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, false, true, true, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000001110)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, true, true, true, false);
        return;
    }

    if (TILING_KEY_IS(1000000000000002000)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, false, false, false, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000002010)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, true, false, false, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000002100)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, false, true, false, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000002110)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, true, true, false, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000003000)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, false, false, true, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000003010)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, true, false, true, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000003100)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, false, true, true, true);
        return;
    }

    if (TILING_KEY_IS(1000000000000003110)) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(0, true, true, true, true);
        return;
    }
}