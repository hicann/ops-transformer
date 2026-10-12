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
 * \file ring_attention_update_apt.cpp
 * \brief
 */
#include "arch35/ring_attention_update_regbase_tnd.h"
#include "arch35/ring_attention_update_regbase_softmax_tnd.h"
#include "arch35/ring_attention_update_regbase_sbh.h"

using namespace RingAttentionUpdateRegbase;
using namespace AscendC;

#define TILINGKEY_REGBASE_SBH 100
#define TILINGKEY_REGBASE_SOFTMAX_TND 110
#define TILINGKEY_REGBASE_TND 120

extern "C" __global__ __aicore__ void ring_attention_update(GM_ADDR prevAttnOut, GM_ADDR prevSoftmaxMax,
                                                            GM_ADDR prevSoftmaxSum, GM_ADDR curAttnOut,
                                                            GM_ADDR curSoftmaxMax, GM_ADDR curSoftmaxSum,
                                                            GM_ADDR actualSeqQlen, GM_ADDR attnOut, GM_ADDR softmaxMax,
                                                            GM_ADDR softmaxSum, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if (TILING_KEY_IS(TILINGKEY_REGBASE_SBH)) {
        GET_TILING_DATA_WITH_STRUCT(RingAttentionUpdateRegbaseSBHTilingData, tilingData, tiling);
        KernelRingAttentionUpdateRegbaseSBH<DTYPE_PREV_ATTN_OUT> op;
        op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, actualSeqQlen,
                attnOut, softmaxMax, softmaxSum, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_REGBASE_SOFTMAX_TND)) {
        GET_TILING_DATA_WITH_STRUCT(RingAttentionUpdateRegbaseSoftmaxTNDTilingData, tilingData, tiling);
        KernelRingAttentionUpdateRegbaseSoftmaxTND<DTYPE_PREV_ATTN_OUT> op;
        op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, actualSeqQlen,
                attnOut, softmaxMax, softmaxSum, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_REGBASE_TND)) {
        GET_TILING_DATA_WITH_STRUCT(RingAttentionUpdateRegbaseTNDTilingData, tilingData, tiling);
        KernelRingAttentionUpdateRegbaseTND<DTYPE_PREV_ATTN_OUT> op;
        op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, actualSeqQlen,
                attnOut, softmaxMax, softmaxSum, workspace, &tilingData, &pipe);
        op.Process();
    }
}
