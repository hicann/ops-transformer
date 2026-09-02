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
 * \file sparse_flash_mla_grad_post_regbase.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_GRAD_POST_KERNEL_REGBASE_H_
#define SPARSE_FLASH_MLA_GRAD_POST_KERNEL_REGBASE_H_
#include "kernel_operator.h"
#include "sparse_flash_mla_grad_arch35_common.h"

using namespace AscendC;

template <typename T1, typename T2, typename OUTDTYPE = T1, const bool IS_TND = 0, const bool isOriKVExist = 0,
          const bool isCmpKVExist = 0>
class SparseFlashMlaGradPostRegbase {
public:
    __aicore__ inline SparseFlashMlaGradPostRegbase(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dori_kv, __gm__ uint8_t *dcmp_kv,
                                __gm__ uint8_t *dsinks, __gm__ uint8_t *workspace,
                                const optiling::smlag::SparseFlashMlaGradTilingDataRegbase *__restrict ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessSink();
    __aicore__ inline void ProcessDqkv();

    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    uint32_t REGBASE_POST_BASE = 128 * 128;
    uint32_t VALUE_DIM = 512;
    uint32_t POST_S_BASE = 18;
    TPipe *pipe;
    const optiling::smlag::SparseFlashMlaGradTilingDataRegbase *__restrict tilingData;
    TQue<QuePosition::VECIN, 1> inQueuePing;
    TQue<QuePosition::VECOUT, 1> outQueuePing;
    TQue<QuePosition::VECIN, 1> inQueuePong;
    TQue<QuePosition::VECOUT, 1> outQueuePong;
    TBuf<> dsinkResBuf;
    GlobalTensor<OUTDTYPE> dqkv[3];
    GlobalTensor<float> dqkvWorkspace[3];
    GlobalTensor<float> dsinkWorkspace;
    GlobalTensor<float> dsinkGm;
    uint32_t vBlockIdx;
    uint64_t loop;
    uint64_t inputTotalSize;
    uint64_t qPostTailNum;
};

template <typename T1, typename T2, typename OUTDTYPE, const bool IS_TND, const bool isOriKVExist,
          const bool isCmpKVExist>
__aicore__ inline void SparseFlashMlaGradPostRegbase<T1, T2, OUTDTYPE, IS_TND, isOriKVExist, isCmpKVExist>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dori_kv, __gm__ uint8_t *dcmp_kv, __gm__ uint8_t *dsinks,
    __gm__ uint8_t *workspace, const optiling::smlag::SparseFlashMlaGradTilingDataRegbase *__restrict ordTilingData,
    TPipe *pipe_in)
{
    vBlockIdx = GetBlockIdx();
    tilingData = ordTilingData;
    pipe = pipe_in;

    dqkv[0].SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dqkvWorkspace[0].SetGlobalBuffer((__gm__ float *)workspace +
                                     tilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
    if constexpr (isOriKVExist) {
        dqkv[1].SetGlobalBuffer((__gm__ OUTDTYPE *)dori_kv);
        dqkvWorkspace[1].SetGlobalBuffer((__gm__ float *)workspace +
                                         tilingData->postTilingData.dOriKVWorkSpaceOffset / sizeof(T2));
    }
    if constexpr (isCmpKVExist) {
        dqkv[2].SetGlobalBuffer((__gm__ OUTDTYPE *)dcmp_kv);
        dqkvWorkspace[2].SetGlobalBuffer((__gm__ float *)workspace +
                                         tilingData->postTilingData.dCmpKVWorkSpaceOffset / sizeof(T2));
    }

    loop = tilingData->postTilingData.qPostBlockFactor;
    inputTotalSize = tilingData->postTilingData.qPostBlockTotal;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;

    if (unlikely(tilingData->baseParams.isSink)) {
        dsinkWorkspace.SetGlobalBuffer((__gm__ float *)workspace +
                                       tilingData->baseParams.dSinkWorkSpaceOffset / sizeof(float));
        dsinkGm.SetGlobalBuffer((__gm__ float *)dsinks);
        pipe->InitBuffer(dsinkResBuf, tilingData->baseParams.g * sizeof(float));
    }

    pipe->InitBuffer(inQueuePing, 1, REGBASE_POST_BASE * sizeof(T2));
    pipe->InitBuffer(outQueuePing, 1, REGBASE_POST_BASE * sizeof(OUTDTYPE));
    pipe->InitBuffer(inQueuePong, 1, REGBASE_POST_BASE * sizeof(T2));
    pipe->InitBuffer(outQueuePong, 1, REGBASE_POST_BASE * sizeof(OUTDTYPE));
}

template <typename T1, typename T2, typename OUTDTYPE, const bool IS_TND, const bool isOriKVExist,
          const bool isCmpKVExist>
__aicore__ inline void
SparseFlashMlaGradPostRegbase<T1, T2, OUTDTYPE, IS_TND, isOriKVExist, isCmpKVExist>::ProcessSink()
{
    if (vBlockIdx != 0) {
        return;
    }
    event_t eventIDMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
    uint16_t usedCoreNum = tilingData->baseParams.usedCoreNum;
    uint16_t gSize = tilingData->baseParams.g;
    LocalTensor<float> dsinkResTensor = dsinkResBuf.Get<float>();
    for (uint16_t vIdx = 0; vIdx < usedCoreNum; vIdx++) {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = gSize * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPadExtParams<float> copyPadParams;
        copyPadParams.isPad = false;
        copyPadParams.leftPadding = 0;
        copyPadParams.rightPadding = 0;
        copyPadParams.paddingValue = 0;
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
        DataCopyPad(dsinkResTensor, dsinkWorkspace[vIdx * gSize], copyParams, copyPadParams);
        SetFlag<HardEvent::MTE2_MTE3>(eventIDMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDMte2ToMte3);

        SetAtomicAdd<float>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = (gSize * sizeof(float) + 31) / 32;
        dataCopyParams.srcGap = 0;
        dataCopyParams.dstGap = 0;
        DataCopy(dsinkGm, dsinkResTensor, dataCopyParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
        SetAtomicNone();
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(eventIDMte2ToMte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
}
template <typename T1, typename T2, typename OUTDTYPE, const bool IS_TND, const bool isOriKVExist,
          const bool isCmpKVExist>
__aicore__ inline void
SparseFlashMlaGradPostRegbase<T1, T2, OUTDTYPE, IS_TND, isOriKVExist, isCmpKVExist>::ProcessDqkv()
{
    for (int qkvIdx = 0; qkvIdx < 3; qkvIdx++) {
        if constexpr (!isOriKVExist) {
            if (qkvIdx == 1) {
                continue;
            }
        }
        if constexpr (!isCmpKVExist) {
            if (qkvIdx == 2) {
                continue;
            }
        }

        if (qkvIdx == 1) {
            loop = tilingData->postTilingData.oriKVPostBlockFactor;
            inputTotalSize = tilingData->postTilingData.oriKVPostBlockTotal;
            qPostTailNum = tilingData->postTilingData.oriKVPostTailNum;
        } else if (qkvIdx == 2) {
            loop = tilingData->postTilingData.cmpKVPostBlockFactor;
            inputTotalSize = tilingData->postTilingData.cmpKVPostBlockTotal;
            qPostTailNum = tilingData->postTilingData.cmpKVPostTailNum;
        }

        uint64_t blockCore = loop * REGBASE_POST_BASE;
        uint64_t begin = vBlockIdx * blockCore;
        uint64_t end = begin + blockCore;

        if (end > inputTotalSize) {
            end = inputTotalSize;
        }

        for (uint64_t pingIdx = begin; pingIdx < end; pingIdx = pingIdx + (REGBASE_POST_BASE << 1)) {
            LocalTensor<float> vecInPing = inQueuePing.AllocTensor<float>();
            uint64_t pongIdx = pingIdx + REGBASE_POST_BASE;
            uint32_t pingSize = pongIdx < inputTotalSize ? REGBASE_POST_BASE : qPostTailNum;
            DataCopy(vecInPing, dqkvWorkspace[qkvIdx][pingIdx], (pingSize + 7) >> 3 << 3);
            inQueuePing.EnQue(vecInPing);
            inQueuePing.DeQue<float>();
            if (qkvIdx < 1) {
                Muls(vecInPing, vecInPing, (float)tilingData->baseParams.scaleValue, pingSize);
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
                seqPongSize = pongSize / VALUE_DIM;
                DataCopy(vecInPong, dqkvWorkspace[qkvIdx][pongIdx], (pongSize + 7) >> 3 << 3);
            }
            outQueuePing.EnQue(vecOutPing);
            outQueuePing.DeQue<OUTDTYPE>();
            inQueuePing.FreeTensor(vecInPing);
            DataCopy(dqkv[qkvIdx][pingIdx], vecOutPing, (pingSize + 15) >> 4 << 4);
            outQueuePing.FreeTensor(vecOutPing);
            LocalTensor<OUTDTYPE> vecOutPong;
            if (neeedPong) {
                vecOutPong = outQueuePong.AllocTensor<OUTDTYPE>();
                inQueuePong.EnQue(vecInPong);
                inQueuePong.DeQue<float>();
                if (qkvIdx < 1) {
                    Muls(vecInPong, vecInPong, (float)tilingData->baseParams.scaleValue, pongSize);
                }
                Cast(vecOutPong, vecInPong, RoundMode::CAST_ROUND, pongSize);
                outQueuePong.EnQue(vecOutPong);
                outQueuePong.DeQue<OUTDTYPE>();
                inQueuePong.FreeTensor(vecInPong);
                DataCopy(dqkv[qkvIdx][pongIdx], vecOutPong, (pongSize + 15) >> 4 << 4);
                outQueuePong.FreeTensor(vecOutPong);
            }
        }
    }
}
template <typename T1, typename T2, typename OUTDTYPE, const bool IS_TND, const bool isOriKVExist,
          const bool isCmpKVExist>
__aicore__ inline void SparseFlashMlaGradPostRegbase<T1, T2, OUTDTYPE, IS_TND, isOriKVExist, isCmpKVExist>::Process()
{
    if (g_coreType != AIV) {
        return;
    }
    ProcessDqkv();
    if (unlikely(tilingData->baseParams.isSink)) {
        ProcessSink();
    }
}
#endif // SPARSE_FLASH_MLA_GRAD_POST_KERNEL_REGBASE_H_
