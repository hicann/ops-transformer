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
 * \file sparse_flash_attention_grad_pre_regbase.h
 * \brief 结构对齐 FAG presfmg：Duplicate 一次 + DoSoftmaxGrad 圈内 CopyIn/VF/CopyDqkv 重叠。
 *        SFAG 仅保留 D=512 连续行、KV_MERGE、sinks。
 */
#ifndef SPARSE_FLASH_ATTENTION_GRAD_PRE_KERNEL_REGBASE_H_
#define SPARSE_FLASH_ATTENTION_GRAD_PRE_KERNEL_REGBASE_H_
#include "kernel_operator.h"
#include "sparse_flash_attention_grad_common.h"
#include "vector_api/vf_softmax_grad_front_cast_sfag.h"

using namespace AscendC;

constexpr uint32_t PRE_INIT_UB_SIZE = 32 * 1024;
constexpr uint32_t SFMG_HEAD_DIM = 512;

template <typename T1, typename T2, const uint32_t IS_TND = 0, bool IS_SINKS = false, bool IS_DETER = false,
          bool KV_MERGE = false>
class FlashAttentionGradPreRegbase {
public:
    __aicore__ inline FlashAttentionGradPreRegbase(){};
    __aicore__ inline void Init(__gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dSinks,
                                __gm__ uint8_t* dy, __gm__ uint8_t* y, __gm__ uint8_t* actual_seq_kvlen,
                                __gm__ uint8_t* workspace,
                                const optiling::sfag::SparseFlashAttentionGradTilingDataRegbase92* ordTilingData,
                                TPipe* pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    __aicore__ inline void CopyInSfmg(uint64_t nBurst, uint64_t sfmgOutputOffset);
    __aicore__ inline void CalculateSoftmaxGrad(uint64_t taskId, uint64_t sfmgOutputOffset, uint64_t curNBurst);
    __aicore__ inline void DoSoftmaxGrad();
    template <uint8_t GM_IDX>
    __aicore__ inline void CopyDqkvToGm(const uint64_t& loopIdx);
    __aicore__ inline uint64_t Ceil(uint64_t a, uint64_t b);
    __aicore__ inline uint64_t Min(uint64_t a, uint64_t b);

    TPipe* pipe;
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    GlobalTensor<T1> dqGm, dkGm, dvGm;
    GlobalTensor<float> dSinksGm, dSinkWorkSpaceGm;
    GlobalTensor<T1> dyGm, yGm;
    GlobalTensor<float> softmaxGradWorkSpaceGm;

    const optiling::sfag::SparseFlashAttentionGradTilingDataRegbase92* TilingData;

    uint32_t vBlockIdx;
    uint32_t usedCoreNum;
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t kPreBlockFactor;
    uint32_t kPreBlockTotal;
    uint32_t kPreBlockTail;
    uint32_t vPreBlockFactor;
    uint32_t vPreBlockTotal;
    uint32_t vPreBlockTail;

    uint64_t initdqSize;
    uint64_t dqOffset;
    uint64_t initdkSize;
    uint64_t dkOffset;
    uint64_t initdvSize;
    uint64_t dvOffset;
    uint64_t dqLoopNum;
    uint64_t dkLoopNum;
    uint64_t dvLoopNum;
    uint64_t dqBlockTail;
    uint64_t dkBlockTail;
    uint64_t dvBlockTail;
    uint64_t singleInitSize;
    event_t eventIDVToMte3;

    LocalTensor<T1> input1Buf;
    LocalTensor<T1> input2Buf;
    TBuf<> dqInitBuf;
    TBuf<> dkInitBuf;
    TBuf<> dvInitBuf;
    LocalTensor<T2> dqTensor;
    LocalTensor<T2> dkTensor;
    LocalTensor<T2> dvTensor;

    TQue<QuePosition::VECIN, 1> input1Que[2];
    TQue<QuePosition::VECIN, 1> input2Que[2];
    TQue<QuePosition::VECOUT, 1> out1Que;
};

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline uint64_t FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::Ceil(uint64_t a,
                                                                                                            uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline uint64_t FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::Min(uint64_t a,
                                                                                                           uint64_t b)
{
    return a < b ? a : b;
}

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline void FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::Init(
    __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dSinks, __gm__ uint8_t* dy,
    __gm__ uint8_t* y, __gm__ uint8_t* actual_seq_kvlen, __gm__ uint8_t* workspace,
    const optiling::sfag::SparseFlashAttentionGradTilingDataRegbase92* orgTilingData, TPipe* pipe_in)
{
    if (g_coreType == AIV) {
        vBlockIdx = GetBlockIdx();
        TilingData = orgTilingData;
        pipe = pipe_in;

        qPreBlockFactor = TilingData->preTilingData.qPreBlockFactor;
        qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
        qPreBlockTail = TilingData->preTilingData.qPreBlockTail;
        kPreBlockFactor = TilingData->preTilingData.kPreBlockFactor;
        kPreBlockTotal = TilingData->preTilingData.kPreBlockTotal;
        kPreBlockTail = TilingData->preTilingData.kPreBlockTail;
        vPreBlockFactor = TilingData->preTilingData.vPreBlockFactor;
        vPreBlockTotal = TilingData->preTilingData.vPreBlockTotal;
        vPreBlockTail = TilingData->preTilingData.vPreBlockTail;
        usedCoreNum = TilingData->preTilingData.sfmgUsedCoreNum;

        dqGm.SetGlobalBuffer((__gm__ T1*)dq);
        dkGm.SetGlobalBuffer((__gm__ T1*)dk);
        if constexpr (!KV_MERGE) {
            dvGm.SetGlobalBuffer((__gm__ T1*)dv);
        }

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                      TilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                      TilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                      TilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));
        softmaxGradWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                               TilingData->baseParams.softmaxGradWorkSpaceOffset / sizeof(T2));
        dyGm.SetGlobalBuffer((__gm__ T1*)dy);
        yGm.SetGlobalBuffer((__gm__ T1*)y);

        if constexpr (IS_SINKS) {
            dSinksGm.SetGlobalBuffer((__gm__ float*)dSinks);
            dSinkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                             TilingData->postTilingData.dSinkWorkSpaceOffset / sizeof(T2));
        }

        initdqSize = vBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        dqOffset = ((uint64_t)vBlockIdx) * qPreBlockFactor;
        initdkSize = vBlockIdx == kPreBlockTotal - 1 ? kPreBlockTail : kPreBlockFactor;
        dkOffset = ((uint64_t)vBlockIdx) * kPreBlockFactor;
        initdvSize = 0;
        dvOffset = 0;
        if constexpr (!KV_MERGE) {
            initdvSize = vBlockIdx == vPreBlockTotal - 1 ? vPreBlockTail : vPreBlockFactor;
            dvOffset = ((uint64_t)vBlockIdx) * vPreBlockFactor;
        }
        if (vBlockIdx >= qPreBlockTotal) {
            initdqSize = 0;
            dqOffset = 0;
        }
        if (vBlockIdx >= kPreBlockTotal) {
            initdkSize = 0;
            dkOffset = 0;
        }
        if constexpr (!KV_MERGE) {
            if (vBlockIdx >= vPreBlockTotal) {
                initdvSize = 0;
                dvOffset = 0;
            }
        }

        singleInitSize = PRE_INIT_UB_SIZE / sizeof(T2);
        dqLoopNum = Ceil(initdqSize, singleInitSize);
        dkLoopNum = Ceil(initdkSize, singleInitSize);
        dvLoopNum = Ceil(initdvSize, singleInitSize);
        dqBlockTail = initdqSize % singleInitSize == 0 ? singleInitSize : initdqSize % singleInitSize;
        dkBlockTail = initdkSize % singleInitSize == 0 ? singleInitSize : initdkSize % singleInitSize;
        dvBlockTail = initdvSize % singleInitSize == 0 ? singleInitSize : initdvSize % singleInitSize;
        pipe->InitBuffer(dqInitBuf, PRE_INIT_UB_SIZE);
        pipe->InitBuffer(dkInitBuf, PRE_INIT_UB_SIZE);
        if constexpr (!KV_MERGE) {
            pipe->InitBuffer(dvInitBuf, PRE_INIT_UB_SIZE);
        }
        dqTensor = dqInitBuf.template Get<T2>();
        dkTensor = dkInitBuf.template Get<T2>();
        if constexpr (!KV_MERGE) {
            dvTensor = dvInitBuf.template Get<T2>();
        }
        eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        pipe->InitBuffer(input1Que[0], 1, static_cast<uint32_t>(TilingData->preTilingData.sfmgYBufferLen));
        pipe->InitBuffer(input1Que[1], 1, static_cast<uint32_t>(TilingData->preTilingData.sfmgYBufferLen));
        pipe->InitBuffer(input2Que[0], 1, static_cast<uint32_t>(TilingData->preTilingData.sfmgDyBufferLen));
        pipe->InitBuffer(input2Que[1], 1, static_cast<uint32_t>(TilingData->preTilingData.sfmgDyBufferLen));
        pipe->InitBuffer(out1Que, 2, static_cast<uint32_t>(TilingData->preTilingData.sfmgOutputBufferLen));
    }
}

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline void FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::Process()
{
    if (g_coreType == AIV && vBlockIdx < usedCoreNum) {
        Duplicate<T2>(dqTensor, 0, singleInitSize);
        Duplicate<T2>(dkTensor, 0, singleInitSize);
        if constexpr (!KV_MERGE) {
            Duplicate<T2>(dvTensor, 0, singleInitSize);
        }
        SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        DoSoftmaxGrad();
        // FAG presfmg 无 sink（有 sink 时走另一条 pre + InitOutput）。SFAG 必须清：g≤128，核 0 一次 InitOutput。
        if constexpr (IS_SINKS) {
            if (vBlockIdx == 0) {
                InitOutput<float>(dSinksGm, TilingData->baseParams.g, 0);
                if constexpr (IS_DETER) {
                    InitOutput<float>(dSinkWorkSpaceGm, TilingData->baseParams.usedCoreNum * optiling::sfag::MAX_N1, 0);
                }
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline void FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::CopyInSfmg(
    uint64_t nBurst, uint64_t sfmgOutputOffset)
{
    if (nBurst == 0) {
        return;
    }
    // D=512 且 dy/y 为 [T1, N1, 512] 连续行，一次 DataCopy，不走 FAG 的 InitIndex/借轴。
    DataCopy(input1Buf, yGm[sfmgOutputOffset * SFMG_HEAD_DIM], nBurst * SFMG_HEAD_DIM);
    DataCopy(input2Buf, dyGm[sfmgOutputOffset * SFMG_HEAD_DIM], nBurst * SFMG_HEAD_DIM);
}

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline void FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::CalculateSoftmaxGrad(
    uint64_t taskId, uint64_t sfmgOutputOffset, uint64_t curNBurst)
{
    LocalTensor<T1> yInTensor = input1Que[taskId & 1].DeQue<T1>();
    LocalTensor<T1> dxInTensor = input2Que[taskId & 1].DeQue<T1>();
    auto output1Buf = out1Que.AllocTensor<T2>();
    MySoftmaxGradFrontCast<T1, T2, SFMG_HEAD_DIM, SFMG_HEAD_DIM>(output1Buf, yInTensor, dxInTensor,
                                                                 static_cast<uint32_t>(curNBurst), SFMG_HEAD_DIM);
    out1Que.EnQue(output1Buf);
    out1Que.DeQue<T2>();
    DataCopyPad(softmaxGradWorkSpaceGm[sfmgOutputOffset], output1Buf,
                {1, static_cast<uint32_t>(curNBurst * sizeof(T2)), 0, 0, 0});
    out1Que.FreeTensor(output1Buf);
}

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
template <uint8_t GM_IDX>
__aicore__ inline void FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::CopyDqkvToGm(
    const uint64_t& loopIdx)
{
    if constexpr (GM_IDX == DQ_IDX) {
        if (loopIdx == dqLoopNum - 1) {
            DataCopyPad(dqWorkSpaceGm[dqOffset + loopIdx * singleInitSize], dqTensor,
                        {1, static_cast<uint32_t>(dqBlockTail * sizeof(T2)), 0, 0, 0});
        } else {
            DataCopy(dqWorkSpaceGm[dqOffset + loopIdx * singleInitSize], dqTensor, singleInitSize);
        }
    } else if constexpr (GM_IDX == DK_IDX) {
        if (loopIdx == dkLoopNum - 1) {
            DataCopyPad(dkWorkSpaceGm[dkOffset + loopIdx * singleInitSize], dkTensor,
                        {1, static_cast<uint32_t>(dkBlockTail * sizeof(T2)), 0, 0, 0});
        } else {
            DataCopy(dkWorkSpaceGm[dkOffset + loopIdx * singleInitSize], dkTensor, singleInitSize);
        }
    } else {
        if constexpr (!KV_MERGE) {
            if (loopIdx == dvLoopNum - 1) {
                DataCopyPad(dvWorkSpaceGm[dvOffset + loopIdx * singleInitSize], dvTensor,
                            {1, static_cast<uint32_t>(dvBlockTail * sizeof(T2)), 0, 0, 0});
            } else {
                DataCopy(dvWorkSpaceGm[dvOffset + loopIdx * singleInitSize], dvTensor, singleInitSize);
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline void FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::DoSoftmaxGrad()
{
    if (vBlockIdx < usedCoreNum) {
        uint64_t singleCoreLoopTimes;
        uint64_t singleCoreLastLoopNBurstNum;
        if (vBlockIdx == usedCoreNum - 1) {
            singleCoreLoopTimes = TilingData->preTilingData.tailCoreLoopTimes;
            singleCoreLastLoopNBurstNum = TilingData->preTilingData.tailCoreLastLoopNBurstNum;
        } else {
            singleCoreLoopTimes = TilingData->preTilingData.normalCoreLoopTimes;
            singleCoreLastLoopNBurstNum = TilingData->preTilingData.normalCoreLastLoopNBurstNum;
        }

        uint64_t sfmgOutputOffset = vBlockIdx * TilingData->preTilingData.normalCoreNBurstNums;
        uint64_t nBurst = TilingData->preTilingData.singleLoopNBurstNum;
        uint64_t taskId = 0;
        uint64_t commonLoopTimes;
        if constexpr (KV_MERGE) {
            commonLoopTimes = Min(Min(dqLoopNum, dkLoopNum), singleCoreLoopTimes);
        } else {
            commonLoopTimes = Min(Min(Min(dqLoopNum, dkLoopNum), dvLoopNum), singleCoreLoopTimes);
        }
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);
        for (uint64_t i = 0; i < commonLoopTimes; i++) {
            if (i == singleCoreLoopTimes - 1) {
                nBurst = singleCoreLastLoopNBurstNum;
            }

            input1Buf = input1Que[taskId & 1].AllocTensor<T1>();
            input2Buf = input2Que[taskId & 1].AllocTensor<T1>();
            CopyInSfmg(nBurst, sfmgOutputOffset);
            input1Que[taskId & 1].EnQue(input1Buf);
            input2Que[taskId & 1].EnQue(input2Buf);
            CalculateSoftmaxGrad(taskId, sfmgOutputOffset, nBurst);
            CopyDqkvToGm<DQ_IDX>(i);
            CopyDqkvToGm<DK_IDX>(i);
            if constexpr (!KV_MERGE) {
                CopyDqkvToGm<DV_IDX>(i);
            }

            input1Que[taskId & 1].FreeTensor(input1Buf);
            input2Que[taskId & 1].FreeTensor(input2Buf);
            sfmgOutputOffset += TilingData->preTilingData.singleLoopNBurstNum;
            taskId++;
        }
        for (uint64_t i = commonLoopTimes; i < singleCoreLoopTimes; i++) {
            if (i == singleCoreLoopTimes - 1) {
                nBurst = singleCoreLastLoopNBurstNum;
            }

            input1Buf = input1Que[taskId & 1].AllocTensor<T1>();
            input2Buf = input2Que[taskId & 1].AllocTensor<T1>();
            CopyInSfmg(nBurst, sfmgOutputOffset);
            input1Que[taskId & 1].EnQue(input1Buf);
            input2Que[taskId & 1].EnQue(input2Buf);
            CalculateSoftmaxGrad(taskId, sfmgOutputOffset, nBurst);
            input1Que[taskId & 1].FreeTensor(input1Buf);
            input2Que[taskId & 1].FreeTensor(input2Buf);
            sfmgOutputOffset += TilingData->preTilingData.singleLoopNBurstNum;
            taskId++;
        }
        for (uint64_t i = commonLoopTimes; i < dqLoopNum; i++) {
            CopyDqkvToGm<DQ_IDX>(i);
        }
        for (uint64_t i = commonLoopTimes; i < dkLoopNum; i++) {
            CopyDqkvToGm<DK_IDX>(i);
        }
        if constexpr (!KV_MERGE) {
            for (uint64_t i = commonLoopTimes; i < dvLoopNum; i++) {
                CopyDqkvToGm<DV_IDX>(i);
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_TND, bool IS_SINKS, bool IS_DETER, bool KV_MERGE>
__aicore__ inline void FlashAttentionGradPreRegbase<T1, T2, IS_TND, IS_SINKS, IS_DETER, KV_MERGE>::SyncALLCores()
{
    SyncAll<false>();
}
#endif // _SPARSE_FLASH_ATTENTION_GRAD_PRE_KERNEL_REGBASE_H_
