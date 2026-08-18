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
 * \file ring_attention_update.cpp
 * \brief
 */

#include "ring_attention_update_kernel.h"
#include "ring_attention_update_tnd.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void ring_attention_update(
  GM_ADDR prevAttnOut, GM_ADDR prevSoftmaxMax, GM_ADDR prevSoftmaxSum,
  GM_ADDR curAttnOut, GM_ADDR curSoftmaxMax, GM_ADDR curSoftmaxSum,
  GM_ADDR actualSeqQlen,
  GM_ADDR attnOut, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
  GM_ADDR workspace, GM_ADDR tiling)
{
  TPipe pipe;
  GET_TILING_DATA(tilingDataIn, tiling);
  const RingAttentionUpdateTilingData* __restrict tilingData = &tilingDataIn;
  GM_ADDR userWorkspace = GetUserWorkspace(workspace);

  if (TILING_KEY_IS(0)) {
    KernelRingAttentionUpdate<half> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(1)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    KernelRingAttentionUpdate<bfloat16_t> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
#endif
  } else if (TILING_KEY_IS(2)) {
    KernelRingAttentionUpdate<float> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(10)) {
    KernelRingAttentionUpdateTND<half> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(11)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    KernelRingAttentionUpdateTND<bfloat16_t> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
#endif
  } else if (TILING_KEY_IS(12)) {
    KernelRingAttentionUpdateTND<float> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
  }
}
