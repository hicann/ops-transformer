/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_kl_loss_grad_vector2.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR2_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR2_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_lightning_indexer_kl_loss_grad_common.h"
#include "sparse_lightning_indexer_kl_loss_grad_tiling.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SLIT>
class SLIKLLossVector2Service {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SLIT::inputQT;
    using KV_T = typename SLIT::inputKT;
    using OUT_T = typename SLIT::outputT;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using MM12_OUT_T = T;
    using MM3_OUT_T = T;

    static constexpr SLILayout LAYOUT_T = SLIT::inputQLayout;
    static constexpr SLILayout KV_LAYOUT_T = SLIT::inputKLayout;
    static constexpr bool deterministic = SLIT::deterministic;
    static constexpr bool privateScatter = SLIT::privateScatter;

    using scatterAddGmType = typename std::conditional<deterministic, GlobalTensor<MM3_OUT_T>, int8_t>::type;

    __aicore__ inline SLIKLLossVector2Service(){};
    __aicore__ inline void InitParams(
        const struct SLIKLLossGradConstInfo &vecConstInfo,
        const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    // =============== vector 2 functions ==============
    __aicore__ inline void InitVector2GM(const GlobalTensor<MM3_OUT_T> &scatterAddBank0,
                                         const GlobalTensor<int32_t> &topK, const GlobalTensor<OUT_T> &dKeyIndexGm,
                                         GlobalTensor<int32_t> &actualSeqLengthsQ,
                                         GlobalTensor<int32_t> &actualSeqLengthsKV,
                                         scatterAddGmType scatterAddBanks[SLI_DETER_SCATTER_BANK_NUM]);
    __aicore__ inline void ProcessVector2();

private:
    // =============== vector 2 functions ==============

    // =============== vector2 variable ==============
    SLIKLLossGradConstInfo constInfo;
    const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tilingData;

    // global tensor
    GlobalTensor<MM3_OUT_T> bmm3ResGm;
    GlobalTensor<int32_t> topKGm;
    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<OUT_T> dKeyIndexGm;
    scatterAddGmType scatterAddResGmBanks[SLI_DETER_SCATTER_BANK_NUM];

    // local tensor
    TBuf<> mm3TBuf; // 64K, 64 * 128 * 4 * 2(DB)
    TBuf<> scatterAddPongTBuf;
    TBuf<> topKTBuf;    // 16KB, 2048 * 4 * 2(DB)
    TBuf<> castOutTBuf; // 32KB, 64 * 128 * 4 * 2(DB)
    LocalTensor<MM3_OUT_T> mm3ResUb;
    LocalTensor<MM3_OUT_T> scatterAddPongUb;
    LocalTensor<int32_t> topKUb;
    LocalTensor<OUT_T> castOutUb;

    static constexpr uint64_t SYNC_SCATTER_BUF_FLAG = 0;
    static constexpr uint64_t SYNC_SCATTER_BUF_PONG_FLAG = 1;
    static constexpr int64_t UB_ROW_SIZE = 64;
};

template <typename SLIT>
__aicore__ inline void SLIKLLossVector2Service<SLIT>::InitParams(
    const struct SLIKLLossGradConstInfo &vecConstInfo,
    const optiling::SparseLightningIndexerKLLossGradTilingData *__restrict tilingData)
{
    this->constInfo = vecConstInfo;
    this->tilingData = tilingData;
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVector2Service<SLIT>::InitVector2GM(
    const GlobalTensor<MM3_OUT_T> &scatterAddBank0, const GlobalTensor<int32_t> &topK,
    const GlobalTensor<OUT_T> &dKeyIndexGm, GlobalTensor<int32_t> &actualSeqLengthsQ,
    GlobalTensor<int32_t> &actualSeqLengthsKV, scatterAddGmType scatterAddBanks[SLI_DETER_SCATTER_BANK_NUM])
{
    this->bmm3ResGm = scatterAddBank0;
    this->topKGm = topK;
    this->actualSeqLengthsQGm = actualSeqLengthsQ;
    this->actualSeqLengthsKVGm = actualSeqLengthsKV;
    this->dKeyIndexGm = dKeyIndexGm;
    if constexpr (deterministic) {
        for (int32_t bank = 0; bank < SLI_DETER_SCATTER_BANK_NUM; ++bank) {
            this->scatterAddResGmBanks[bank] = scatterAddBanks[bank];
        }
    }
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVector2Service<SLIT>::InitBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(mm3TBuf, SLIKLLossGradConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    if constexpr (deterministic) {
        pipe->InitBuffer(scatterAddPongTBuf, SLIKLLossGradConstInfo::BUFFER_SIZE_BYTE_32K * 2);
    }
    pipe->InitBuffer(castOutTBuf, SLIKLLossGradConstInfo::BUFFER_SIZE_BYTE_16K * 2); // 2:pingpong

    mm3ResUb = mm3TBuf.Get<MM3_OUT_T>();
    if constexpr (deterministic) {
        scatterAddPongUb = scatterAddPongTBuf.Get<MM3_OUT_T>();
    }
    castOutUb = castOutTBuf.Get<OUT_T>();
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVector2Service<SLIT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_PONG_FLAG);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVector2Service<SLIT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_PONG_FLAG);
}

template <typename SLIT>
__aicore__ inline void SLIKLLossVector2Service<SLIT>::ProcessVector2()
{
    // 输入数据[T2,Nidx2,D]
    // T2切多核，执行前需要做全核同步

    int64_t totalCost = 0;
    if constexpr (KV_LAYOUT_T == SLILayout::TND) {
        totalCost = actualSeqLengthsKVGm.GetValue(constInfo.bSize);
    } else {
        totalCost = constInfo.bSize * constInfo.s2Size;
    }

    int64_t totalCoreNum = GetBlockNum() * GetTaskRation();
    int64_t avgCost = CeilDiv(totalCost, totalCoreNum);

    int32_t t2Start = Min(constInfo.aivIdx * avgCost, totalCost);
    int32_t t2End = Min(t2Start + avgCost, totalCost);

    int32_t t2ProcessSize = UB_ROW_SIZE;
    int32_t tailSize = (t2End - t2Start) % UB_ROW_SIZE;
    int32_t t2TailSize = (!tailSize) ? UB_ROW_SIZE : (tailSize);
    int32_t pingPongIdx = 0;

    LocalTensor<MM3_OUT_T> copyInUb;
    LocalTensor<MM3_OUT_T> copyInPongUb;
    LocalTensor<OUT_T> copyOutUb;

    for (int32_t t2Idx = t2Start; t2Idx < t2End; t2Idx += UB_ROW_SIZE) {
        if (t2Idx + UB_ROW_SIZE >= t2End) {
            t2ProcessSize = t2TailSize;
        }

        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
        copyInUb = mm3ResUb[pingPongIdx * (UB_ROW_SIZE * constInfo.dSizeQueryIndex)];
        if constexpr (privateScatter) {
            int32_t scatterBankNum = static_cast<int32_t>(GetBlockNum());
            if (scatterBankNum <= 0) {
                scatterBankNum = 1;
            }
            int64_t bankElems = totalCost * constInfo.dSizeQueryIndex;
            __gm__ MM3_OUT_T *basePtr = (__gm__ MM3_OUT_T *)bmm3ResGm.GetPhyAddr();
            GlobalTensor<MM3_OUT_T> bank0Gm;
            bank0Gm.SetGlobalBuffer(basePtr);
            DataCopy(copyInUb, bank0Gm[t2Idx * constInfo.dSizeQueryIndex], t2ProcessSize * constInfo.dSizeQueryIndex);
            copyInPongUb = scatterAddPongUb[pingPongIdx * (UB_ROW_SIZE * constInfo.dSizeQueryIndex)];
            for (int32_t bank = 1; bank < scatterBankNum; ++bank) {
                if (bank > 1) {
                    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
                }
                GlobalTensor<MM3_OUT_T> bankGm;
                bankGm.SetGlobalBuffer(basePtr + static_cast<uint64_t>(bank) * bankElems);
                DataCopy(copyInPongUb, bankGm[t2Idx * constInfo.dSizeQueryIndex],
                         t2ProcessSize * constInfo.dSizeQueryIndex);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
                Add(copyInUb, copyInUb, copyInPongUb, t2ProcessSize * constInfo.dSizeQueryIndex);
                PipeBarrier<PIPE_V>();
            }
            if (scatterBankNum <= 1) {
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
            }
        } else if constexpr (deterministic) {
            DataCopy(copyInUb, scatterAddResGmBanks[0][t2Idx * constInfo.dSizeQueryIndex],
                     t2ProcessSize * constInfo.dSizeQueryIndex);
            copyInPongUb = scatterAddPongUb[pingPongIdx * (UB_ROW_SIZE * constInfo.dSizeQueryIndex)];
            for (int32_t bank = 1; bank < SLI_DETER_SCATTER_BANK_NUM; ++bank) {
                if (bank > 1) {
                    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
                }
                DataCopy(copyInPongUb, scatterAddResGmBanks[bank][t2Idx * constInfo.dSizeQueryIndex],
                         t2ProcessSize * constInfo.dSizeQueryIndex);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
                Add(copyInUb, copyInUb, copyInPongUb, t2ProcessSize * constInfo.dSizeQueryIndex);
                PipeBarrier<PIPE_V>();
            }
        } else {
            DataCopy(copyInUb, bmm3ResGm[t2Idx * constInfo.dSizeQueryIndex], t2ProcessSize * constInfo.dSizeQueryIndex);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
        }

        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
        copyOutUb = castOutUb[pingPongIdx * (UB_ROW_SIZE * constInfo.dSizeQueryIndex)];
        Cast(copyOutUb, copyInUb, RoundMode::CAST_ROUND, t2ProcessSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SCATTER_BUF_FLAG + pingPongIdx); // 释放copyInUb

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_SCATTER_BUF_FLAG + pingPongIdx);

        DataCopy(dKeyIndexGm[t2Idx * constInfo.dSizeQueryIndex], copyOutUb, t2ProcessSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_SCATTER_BUF_FLAG + pingPongIdx); // 释放copyOutUb

        pingPongIdx = 1 - pingPongIdx;
    }
}

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR2_H
