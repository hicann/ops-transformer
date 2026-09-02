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
 * \file dense_lightning_indexer_grad_kl_loss_vector2.h
 * \brief
 */

#ifndef DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR2_H
#define DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR2_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "dense_lightning_indexer_grad_kl_loss_common.h"
#include "dense_lightning_indexer_grad_kl_loss_tiling_data.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename DLIT>
class DLIKLLossVector2Service {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using OUT_T = typename DLIT::outputT;
    using MM3_OUT_T = T;
    using MM4_OUT_T = T;

    DLIGradKLLossConstInfo constInfo;
    const optiling::DenseLightningIndexerGradKLLossTilingData *__restrict tilingData;

    __aicore__ inline DLIKLLossVector2Service(){};

    __aicore__ inline void InitParams(const struct DLIGradKLLossConstInfo &vecConstInfo,
                                      const optiling::DenseLightningIndexerGradKLLossTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitVector2GM(const GlobalTensor<MM3_OUT_T> &dKeyIndexGmIn,
                                         const GlobalTensor<OUT_T> &dKeyIndexGmOut);
    __aicore__ inline void InitVector2DeterGM(const GlobalTensor<T> &lossGm, const GlobalTensor<T> &lossDeterGmFloat);
    __aicore__ inline void ProcessVectorDk();
    __aicore__ inline void DeterSumLoss();

private:
    TBuf<> uBuf_;
    GlobalTensor<MM3_OUT_T> dKeyIndexGmIn;
    GlobalTensor<OUT_T> dKeyIndexGmOut;
    GlobalTensor<T> lossGm;
    GlobalTensor<T> lossDeterGmFloat;

    LocalTensor<MM4_OUT_T> ubInFloatPing_;
    LocalTensor<OUT_T> ubOutHalfPing_;
    LocalTensor<MM4_OUT_T> ubInFloatPong_;
    LocalTensor<OUT_T> ubOutHalfPong_;

    // deter 相关
    LocalTensor<T> ubLossIn_;
    LocalTensor<T> ubLossOut_;
    LocalTensor<uint8_t> tmpUb_;

    event_t eventId = EVENT_ID0;
    int32_t pingPongFlag = 0;
};

template <typename DLIT>
__aicore__ inline void DLIKLLossVector2Service<DLIT>::InitParams(
    const struct DLIGradKLLossConstInfo &vecConstInfo,
    const optiling::DenseLightningIndexerGradKLLossTilingData *__restrict tilingData)
{
    this->constInfo = vecConstInfo;
    this->tilingData = tilingData;
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVector2Service<DLIT>::InitVector2GM(const GlobalTensor<MM3_OUT_T> &dKeyIndexGmIn,
                                                                    const GlobalTensor<OUT_T> &dKeyIndexGmOut)
{
    this->dKeyIndexGmIn = dKeyIndexGmIn;
    this->dKeyIndexGmOut = dKeyIndexGmOut;
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVector2Service<DLIT>::InitVector2DeterGM(const GlobalTensor<T> &lossGm,
                                                                         const GlobalTensor<T> &lossDeterGmFloat)
{
    this->lossGm = lossGm;
    this->lossDeterGmFloat = lossDeterGmFloat;
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVector2Service<DLIT>::InitBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(this->uBuf_, DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_192K);
    uint32_t ubOffset = 0;

    ubInFloatPing_ = uBuf_.GetWithOffset<T>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K, ubOffset);
    ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_64K;
    ubInFloatPong_ = uBuf_.GetWithOffset<T>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K, ubOffset);
    ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_64K;

    ubOutHalfPing_ = uBuf_.GetWithOffset<OUT_T>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K, ubOffset);
    ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_32K;
    ubOutHalfPong_ = uBuf_.GetWithOffset<OUT_T>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K, ubOffset);
    ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_32K;

    // deter 相关
    ubLossIn_ = ubInFloatPing_;
    ubLossOut_ = ubInFloatPong_;
    tmpUb_ = ubOutHalfPong_.template ReinterpretCast<uint8_t>();
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVector2Service<DLIT>::ProcessVectorDk()
{
    if (constInfo.dKeySingleCoreSize <= 0) {
        return;
    }

    uint32_t loopTimes = CeilDiv(constInfo.dKeySingleCoreSize, DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K);

    DataCopyParams dataCopyOutParams;
    dataCopyOutParams.blockCount = 1;
    dataCopyOutParams.dstStride = 0;
    dataCopyOutParams.srcStride = 0;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        uint32_t dKeyGmOffsetCur = constInfo.dKeyGmOffset + loopIdx * DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K;
        uint32_t processNum = DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K;
        uint32_t processNumPad = processNum;
        if (loopIdx == loopTimes - 1) {
            processNum = constInfo.dKeySingleCoreSize - DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_16K * loopIdx;
            processNumPad = (processNum + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        }

        eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;

        LocalTensor<MM4_OUT_T> dKeyIndexUbIn = pingPongFlag ? ubInFloatPong_ : ubInFloatPing_;
        LocalTensor<OUT_T> dKeyIndexUbOut = pingPongFlag ? ubOutHalfPong_ : ubOutHalfPing_;

        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        DataCopy(dKeyIndexUbIn, dKeyIndexGmIn[dKeyGmOffsetCur], processNumPad);
        SetFlag<HardEvent::MTE2_V>(eventId);

        WaitFlag<HardEvent::MTE2_V>(eventId);
        Cast(dKeyIndexUbOut, dKeyIndexUbIn, RoundMode::CAST_ROUND, processNumPad);
        SetFlag<HardEvent::V_MTE3>(eventId);

        WaitFlag<HardEvent::V_MTE3>(eventId);
        dataCopyOutParams.blockLen = processNum * sizeof(OUT_T);
        DataCopyPad(dKeyIndexGmOut[dKeyGmOffsetCur], dKeyIndexUbOut, dataCopyOutParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);

        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVector2Service<DLIT>::DeterSumLoss()
{
    if (constInfo.aivIdx == 0) {
        event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());

        int64_t count = constInfo.aivNum;
        int64_t countAlign = BlockAlign<T>(count);
        SumParams sumParams;
        sumParams.outter = 1;
        sumParams.n = count;
        sumParams.inner = countAlign;

        DataCopyParams dataCopyParams;
        DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        dataCopyParams.blockLen = count * sizeof(T);
        dataCopyPadParams.rightPadding = countAlign - count;
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(ubLossIn_, lossDeterGmFloat, dataCopyParams, dataCopyPadParams);

        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);

        AscendC::Sum(ubLossOut_, ubLossIn_, tmpUb_, sumParams);

        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);

        AscendC::DataCopyPad(lossGm, ubLossOut_,
                             {static_cast<uint32_t>(1), static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(0),
                              static_cast<uint32_t>(0)});

        SetFlag<HardEvent::MTE3_V>(mte3ToV);
        WaitFlag<HardEvent::MTE3_V>(mte3ToV);

        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV);
    }
}

#endif // DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR2_H
