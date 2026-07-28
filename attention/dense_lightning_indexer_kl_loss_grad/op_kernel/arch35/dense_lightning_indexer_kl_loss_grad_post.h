/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dense_lightning_indexer_kl_loss_grad_post.h
 * \brief
 */
#ifndef DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_POST_H_
#define DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_POST_H_
#include "kernel_operator.h"

using namespace AscendC;

TEMPLATES_DEF
class DenseLightningIndexerKLLossGradPost {
public:
    __aicore__ inline DenseLightningIndexerKLLossGradPost(){};
    __aicore__ inline void Init(__gm__ uint8_t *dk, __gm__ uint8_t *workspace,
                                const optiling::DenseLightningIndexerGradRegBaseTilingData *__restrict ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void Process();

    uint32_t REGBASE_POST_BASE = 128 * 128;
    uint32_t VALUE_DIM = 128;
    TPipe *pipe;
    const optiling::DenseLightningIndexerGradRegBaseTilingData *tilingData;
    TQue<QuePosition::VECIN, 1> inQueuePing;
    TQue<QuePosition::VECOUT, 1> outQueuePing;
    TQue<QuePosition::VECIN, 1> inQueuePong;
    TQue<QuePosition::VECOUT, 1> outQueuePong;
    GlobalTensor<OUT_T> dkGm;
    GlobalTensor<float> dkWorkspace;
    uint32_t vBlockIdx;
    uint32_t blockSize;
    uint32_t kPostBlockFactor;
    uint32_t kPostBlockTotal;
    uint32_t kPostBlockTail;
};
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLightningIndexerKLLossGradPost<TEMPLATE_ARGS>::Init(
    __gm__ uint8_t *dk, __gm__ uint8_t *workspace,
    const optiling::DenseLightningIndexerGradRegBaseTilingData *__restrict tilingData, TPipe *pipe_in)
{
    vBlockIdx = GetBlockIdx();
    tilingData = tilingData;
    pipe = pipe_in;

    dkGm.SetGlobalBuffer((__gm__ OUT_T *)dk);

    kPostBlockFactor = tilingData->multiCoreParams.kPostBlockFactor;
    kPostBlockTotal = tilingData->multiCoreParams.kPostBlockTotal;
    kPostBlockTail = tilingData->multiCoreParams.kPostBlockTail;
    blockSize = (vBlockIdx == kPostBlockTotal - 1 ? kPostBlockTail : kPostBlockFactor) * VALUE_DIM;

    dkWorkspace.SetGlobalBuffer((__gm__ float *)workspace + tilingData->baseParams.mm3ResWorkSpaceOffset / sizeof(T));

    pipe->InitBuffer(inQueuePing, 1, REGBASE_POST_BASE * sizeof(T));
    pipe->InitBuffer(outQueuePing, 1, REGBASE_POST_BASE * sizeof(OUT_T));
    pipe->InitBuffer(inQueuePong, 1, REGBASE_POST_BASE * sizeof(T));
    pipe->InitBuffer(outQueuePong, 1, REGBASE_POST_BASE * sizeof(OUT_T));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLightningIndexerKLLossGradPost<TEMPLATE_ARGS>::Process()
{
    if (g_coreType != AIV || vBlockIdx >= kPostBlockTotal) {
        return;
    }
    int64_t beginIdx = vBlockIdx * kPostBlockFactor * VALUE_DIM;
    int64_t endIdx = beginIdx + blockSize;
    for (uint64_t pingIdx = beginIdx; pingIdx < endIdx; pingIdx = pingIdx + (REGBASE_POST_BASE << 1)) {
        LocalTensor<float> vecInPing = inQueuePing.AllocTensor<float>();
        uint64_t pongIdx = pingIdx + REGBASE_POST_BASE;
        bool neeedPong = pongIdx < endIdx;
        uint32_t pingSize = neeedPong ? REGBASE_POST_BASE : endIdx - pingIdx;
        DataCopy(vecInPing, dkWorkspace[pingIdx], (pingSize + 7) >> 3 << 3);
        inQueuePing.EnQue(vecInPing);
        inQueuePing.DeQue<float>();

        LocalTensor<OUT_T> vecOutPing = outQueuePing.AllocTensor<OUT_T>();
        Cast(vecOutPing, vecInPing, RoundMode::CAST_ROUND, pingSize);
        uint32_t pongSize;
        uint32_t seqPongSize;
        LocalTensor<float> vecInPong;
        if (neeedPong) {
            vecInPong = inQueuePong.AllocTensor<float>();
            pongSize = pongIdx + REGBASE_POST_BASE < endIdx ? REGBASE_POST_BASE : endIdx - pongIdx;
            DataCopy(vecInPong, dkWorkspace[pongIdx], (pongSize + 7) >> 3 << 3);
        }
        outQueuePing.EnQue(vecOutPing);
        outQueuePing.DeQue<OUT_T>();
        inQueuePing.FreeTensor(vecInPing);
        DataCopy(dkGm[pingIdx], vecOutPing, (pingSize + 15) >> 4 << 4);
        outQueuePing.FreeTensor(vecOutPing);
        LocalTensor<OUT_T> vecOutPong;
        if (neeedPong) {
            vecOutPong = outQueuePong.AllocTensor<OUT_T>();
            inQueuePong.EnQue(vecInPong);
            inQueuePong.DeQue<float>();
            Cast(vecOutPong, vecInPong, RoundMode::CAST_ROUND, pongSize);
            outQueuePong.EnQue(vecOutPong);
            outQueuePong.DeQue<OUT_T>();
            inQueuePong.FreeTensor(vecInPong);
            DataCopy(dkGm[pongIdx], vecOutPong, (pongSize + 15) >> 4 << 4);
            outQueuePong.FreeTensor(vecOutPong);
        }
    }
}

#endif // SPARSE_FLASH_ATTENTION_GRAD_POST_KERNEL_REGBASE_H_
