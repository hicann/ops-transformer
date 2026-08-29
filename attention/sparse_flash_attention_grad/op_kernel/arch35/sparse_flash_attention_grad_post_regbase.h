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
 * \file sparse_flash_attention_grad_post_regbase.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_GRAD_POST_KERNEL_REGBASE_H_
#define SPARSE_FLASH_ATTENTION_GRAD_POST_KERNEL_REGBASE_H_
#include "kernel_operator.h"

using namespace AscendC;

template <typename T1, typename T2, typename OUTDTYPE = T1, const bool IS_ROPE = false, const bool IS_TND = 0,
          const bool KV_MERGE = false>
class SparseFlashAttentionGradPostRegbase {
public:
    __aicore__ inline SparseFlashAttentionGradPostRegbase(){};
    __aicore__ inline void Init(
        __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dqRope, __gm__ uint8_t *dkRope,
        __gm__ uint8_t *workspace,
        const optiling::sfag::SparseFlashAttentionGradTilingDataRegbase *__restrict ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();

    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    uint32_t REGBASE_POST_BASE = 128 * 128;
    uint32_t ROPE_DIM = 64;
    uint32_t VALUE_DIM = 512;
    TPipe *pipe;
    const optiling::sfag::SparseFlashAttentionGradTilingDataRegbase *__restrict tilingData;
    TQue<QuePosition::VECIN, 1> inQueuePing;
    TQue<QuePosition::VECOUT, 1> outQueuePing;
    TQue<QuePosition::VECIN, 1> inQueuePong;
    TQue<QuePosition::VECOUT, 1> outQueuePong;
    GlobalTensor<OUTDTYPE> dqkv[3];
    GlobalTensor<OUTDTYPE> dqkRope[2];
    GlobalTensor<float> dqkvWorkspace[3];
    GlobalTensor<float> deterGm[2];
    uint32_t vBlockIdx;
    uint32_t loop;
    uint32_t inputTotalSize;
    uint32_t qPostTailNum;
};

template <typename T1, typename T2, typename OUTDTYPE, bool IS_ROPE, const bool IS_TND, const bool KV_MERGE>
__aicore__ inline void SparseFlashAttentionGradPostRegbase<T1, T2, OUTDTYPE, IS_ROPE, IS_TND, KV_MERGE>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dqRope, __gm__ uint8_t *dkRope,
    __gm__ uint8_t *workspace,
    const optiling::sfag::SparseFlashAttentionGradTilingDataRegbase *__restrict ordTilingData, TPipe *pipe_in)
{
    vBlockIdx = GetBlockIdx();
    tilingData = ordTilingData;
    pipe = pipe_in;

    dqkv[0].SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dqkv[1].SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    if constexpr (!KV_MERGE) {
        dqkv[2].SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    }

    dqkRope[0].SetGlobalBuffer((__gm__ OUTDTYPE *)dqRope);
    dqkRope[1].SetGlobalBuffer((__gm__ OUTDTYPE *)dkRope);

    loop = tilingData->postTilingData.qPostBlockFactor;
    inputTotalSize = tilingData->postTilingData.qPostBlockTotal;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;

    dqkvWorkspace[0].SetGlobalBuffer((__gm__ float *)workspace +
                                     tilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
    dqkvWorkspace[1].SetGlobalBuffer((__gm__ float *)workspace +
                                     tilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
    if constexpr (!KV_MERGE) {
        dqkvWorkspace[2].SetGlobalBuffer((__gm__ float *)workspace +
                                         tilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));
    }

    REGBASE_POST_BASE = tilingData->postTilingData.qPostBaseNum;
    uint32_t bufElems = REGBASE_POST_BASE;
    if constexpr (!KV_MERGE) {
        uint32_t vPostBaseNum = tilingData->postTilingData.vPostBaseNum;
        bufElems = bufElems > vPostBaseNum ? bufElems : vPostBaseNum;
    }

    pipe->InitBuffer(inQueuePing, 1, bufElems * sizeof(T2));
    pipe->InitBuffer(outQueuePing, 1, bufElems * sizeof(OUTDTYPE));
    pipe->InitBuffer(inQueuePong, 1, bufElems * sizeof(T2));
    pipe->InitBuffer(outQueuePong, 1, bufElems * sizeof(OUTDTYPE));
}

template <typename T1, typename T2, typename OUTDTYPE, bool IS_ROPE, const bool IS_TND, const bool KV_MERGE>
__aicore__ inline void SparseFlashAttentionGradPostRegbase<T1, T2, OUTDTYPE, IS_ROPE, IS_TND, KV_MERGE>::Process()
{
    if (g_coreType != AIV) {
        return;
    }
    constexpr int QKV_NUM = KV_MERGE ? 2 : 3;
    constexpr uint32_t packedDim = 64 + 512; // ROPE_DIM + VALUE_DIM
    for (int qkvIdx = 0; qkvIdx < QKV_NUM; qkvIdx++) {
        if (qkvIdx == 0) {
            REGBASE_POST_BASE = tilingData->postTilingData.qPostBaseNum;
        } else if (qkvIdx == 1) {
            loop = tilingData->postTilingData.kPostBlockFactor;
            inputTotalSize = tilingData->postTilingData.kPostBlockTotal;
            qPostTailNum = tilingData->postTilingData.kPostTailNum;
            REGBASE_POST_BASE = tilingData->postTilingData.kPostBaseNum;
        } else if (qkvIdx == 2) {
            loop = tilingData->postTilingData.vPostBlockFactor;
            inputTotalSize = tilingData->postTilingData.vPostBlockTotal;
            qPostTailNum = tilingData->postTilingData.vPostTailNum;
            REGBASE_POST_BASE = tilingData->postTilingData.vPostBaseNum;
        }
        uint64_t blockCore = static_cast<uint64_t>(loop) * REGBASE_POST_BASE;
        uint64_t begin = static_cast<uint64_t>(vBlockIdx) * blockCore;
        uint64_t end = begin + blockCore;

        if (end > inputTotalSize) {
            end = inputTotalSize;
        }

        uint32_t tokensPerBase = IS_ROPE ? (REGBASE_POST_BASE / packedDim) : 0;
        uint64_t dqPingGmOffset = static_cast<uint64_t>(vBlockIdx) * loop * tokensPerBase * VALUE_DIM;
        for (uint64_t pingIdx = begin; pingIdx < end;
             pingIdx = pingIdx + (static_cast<uint64_t>(REGBASE_POST_BASE) << 1)) {
            LocalTensor<float> vecInPing = inQueuePing.AllocTensor<float>();
            uint64_t pongIdx = pingIdx + REGBASE_POST_BASE;
            uint32_t pingSize = pongIdx < inputTotalSize ? REGBASE_POST_BASE : qPostTailNum;
            uint64_t dqPongGmOffset = dqPingGmOffset + static_cast<uint64_t>(tokensPerBase) * VALUE_DIM;
            uint32_t seqPingSize = pingSize / packedDim;
            DataCopy(vecInPing, dqkvWorkspace[qkvIdx][pingIdx], (pingSize + 7) >> 3 << 3);
            inQueuePing.EnQue(vecInPing);
            inQueuePing.DeQue<float>();
            if (qkvIdx < 1) {
                Muls(vecInPing, vecInPing, (float)tilingData->baseParams.scaleValue, pingSize);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<OUTDTYPE> vecOutPing = outQueuePing.AllocTensor<OUTDTYPE>();
            Cast(vecOutPing, vecInPing, RoundMode::CAST_ROUND, pingSize);
            uint32_t pongSize;
            uint32_t seqPongSize;
            LocalTensor<float> vecInPong;
            bool neeedPong = loop > 1 && pongIdx < end;
            if (neeedPong) {
                vecInPong = inQueuePong.AllocTensor<float>();
                pongSize = pongIdx + REGBASE_POST_BASE < inputTotalSize ? REGBASE_POST_BASE : qPostTailNum;
                seqPongSize = pongSize / packedDim;
                DataCopy(vecInPong, dqkvWorkspace[qkvIdx][pongIdx], (pongSize + 7) >> 3 << 3);
            }
            outQueuePing.EnQue(vecOutPing);
            outQueuePing.DeQue<OUTDTYPE>();
            inQueuePing.FreeTensor(vecInPing);
            if constexpr (IS_ROPE) {
                if (qkvIdx == 2) {
                    DataCopy(dqkv[qkvIdx][pingIdx], vecOutPing, pingSize);
                } else {
                    DataCopyParams dataCopyParams;
                    dataCopyParams.blockCount = seqPingSize;
                    // 512 * sizeof(OUTDTYPE) / 32B
                    dataCopyParams.blockLen = 16 * sizeof(OUTDTYPE);
                    // 64 * sizeof(OUTDTYPE) / 32B
                    dataCopyParams.srcStride = 2 * sizeof(OUTDTYPE);
                    DataCopy(dqkv[qkvIdx][dqPingGmOffset], vecOutPing, dataCopyParams);

                    dataCopyParams.blockLen = 2 * sizeof(OUTDTYPE);
                    dataCopyParams.srcStride = 16 * sizeof(OUTDTYPE);
                    DataCopy(dqkRope[qkvIdx][dqPingGmOffset / 8], vecOutPing[VALUE_DIM], dataCopyParams);
                }
                dqPingGmOffset = dqPingGmOffset + static_cast<uint64_t>(seqPingSize) * VALUE_DIM;
            } else {
                DataCopy(dqkv[qkvIdx][pingIdx], vecOutPing, (pingSize + 15) >> 4 << 4);
            }
            outQueuePing.FreeTensor(vecOutPing);
            LocalTensor<OUTDTYPE> vecOutPong;
            if (neeedPong) {
                vecOutPong = outQueuePong.AllocTensor<OUTDTYPE>();
                inQueuePong.EnQue(vecInPong);
                inQueuePong.DeQue<float>();
                if (qkvIdx < 1) {
                    Muls(vecInPong, vecInPong, (float)tilingData->baseParams.scaleValue, pongSize);
                    PipeBarrier<PIPE_V>();
                }
                Cast(vecOutPong, vecInPong, RoundMode::CAST_ROUND, pongSize);
                outQueuePong.EnQue(vecOutPong);
                outQueuePong.DeQue<OUTDTYPE>();
                inQueuePong.FreeTensor(vecInPong);
                if constexpr (IS_ROPE) {
                    if (qkvIdx == 2) {
                        DataCopy(dqkv[qkvIdx][pongIdx], vecOutPong, pongSize);
                    } else {
                        DataCopyParams dataCopyParams;
                        dataCopyParams.blockCount = seqPongSize;
                        dataCopyParams.blockLen = 16 * sizeof(OUTDTYPE);
                        dataCopyParams.srcStride = 2 * sizeof(OUTDTYPE);
                        DataCopy(dqkv[qkvIdx][dqPongGmOffset], vecOutPong, dataCopyParams);

                        dataCopyParams.blockLen = 2 * sizeof(OUTDTYPE);
                        dataCopyParams.srcStride = 16 * sizeof(OUTDTYPE);
                        DataCopy(dqkRope[qkvIdx][dqPongGmOffset / 8], vecOutPong[VALUE_DIM], dataCopyParams);
                    }
                    dqPingGmOffset = dqPingGmOffset + static_cast<uint64_t>(seqPongSize) * VALUE_DIM;
                } else {
                    DataCopy(dqkv[qkvIdx][pongIdx], vecOutPong, (pongSize + 15) >> 4 << 4);
                }
                outQueuePong.FreeTensor(vecOutPong);
            }
        }
    }
}

#endif // SPARSE_FLASH_ATTENTION_GRAD_POST_KERNEL_REGBASE_H_
