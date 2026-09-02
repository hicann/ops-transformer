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
 * \file dense_lightning_indexer_grad_kl_loss_vector.h
 * \brief
 */

#ifndef DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_H
#define DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "dense_lightning_indexer_grad_kl_loss_common.h"
#include "dense_lightning_indexer_grad_kl_loss_tiling_data.h"

using namespace AscendC;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename DLIT>
class DLIKLLossVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename DLIT::inputQT;
    using KV_T = typename DLIT::inputKT;
    using W_T = typename DLIT::inputWT;
    using OUT_T = typename DLIT::outputT;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using MM12_OUT_T = T;
    using MM5_OUT_T = T;
    using MM3_OUT_T = T;
    using MM4_OUT_T = T;
    using INFO_INT_64_T = int64_t;

    static constexpr DLILayout LAYOUT_T = DLIT::inputQLayout;
    static constexpr DLILayout KV_LAYOUT_T = DLIT::inputKLayout;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr bool deterministic = DLIT::deterministic;

    __aicore__ inline DLIKLLossVectorService(){};
    __aicore__ inline void InitParams(const DLIGradKLLossConstInfo &vecConstInfo,
                                      const optiling::DenseLightningIndexerGradKLLossTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitVector1GM(
        const GlobalTensor<T> &softmaxMax, const GlobalTensor<T> &softmaxSum, const GlobalTensor<T> &softmaxMaxIndex,
        const GlobalTensor<T> &softmaxSumIndex, const GlobalTensor<MM12_OUT_T> &bmm1Res,
        const GlobalTensor<MM12_OUT_T> &bmm2Res, const GlobalTensor<W_T> &weight, const GlobalTensor<T> pSync,
        const GlobalTensor<T> sySync, const GlobalTensor<T> &loss, const GlobalTensor<T> &dWeightFloat,
        const GlobalTensor<T> &reluGm, const GlobalTensor<KV_T> &reluGradRes, const GlobalTensor<W_T> &dWeight,
        const GlobalTensor<MM4_OUT_T> &dQueryIndexFloat, const GlobalTensor<OUT_T> &dQueryIndex);
    __aicore__ inline void InitVector1DeterGM(const GlobalTensor<INFO_INT_64_T> &deterCoreInfoGm,
                                              const GlobalTensor<T> &dKeyIndexDeterGmFloat,
                                              const GlobalTensor<T> &dKeyIndexGmFloat,
                                              const GlobalTensor<T> &lossGmDeterFloat);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void ProcessVector1(DLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void CastOutWeightGrad(DLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void CastOutQIndexGrad(DLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void DeterAddKIndexGrad(DLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void FreeEventID();
    __aicore__ inline void SaveDeterRunInfo(const DLIGradKLLossRunInfo &info);
    __aicore__ inline void SaveDeterRunInfoInvalid();
    __aicore__ inline void GetDeterRunInfo(int64_t aicIdx);

private:
    __aicore__ inline void PreloadWeight(DLIGradKLLossRunInfo &runInfo, uint32_t s1InnerIdx, uint32_t curS1InnerSize,
                                         uint32_t pingpongFlag);
    __aicore__ inline void VectorP(DLIGradKLLossRunInfo &runInfo, uint32_t s1InnerIdx, uint32_t curS1InnerSize,
                                   uint32_t pingpongFlag);
    __aicore__ inline void VectorSy(DLIGradKLLossRunInfo &runInfo, uint32_t s1InnerIdx, uint32_t curS1InnerSize,
                                    uint32_t pingpongFlag);
    __aicore__ inline void VectorLoss(DLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void VectorDwDqDk(DLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void VectorCopyInDwDqDk(DLIGradKLLossRunInfo &runInfo);
    __aicore__ inline void CopyInWithSparseMask(DLIGradKLLossRunInfo &runInfo, LocalTensor<T> dstUbTensor,
                                                GlobalTensor<T> srcGmTensor, uint32_t rowSize,
                                                int32_t actualLenFirstRow);

    DLIGradKLLossConstInfo constInfo;
    const optiling::DenseLightningIndexerGradKLLossTilingData *__restrict tilingData;

    __aicore__ inline void DataCopyReluRes(LocalTensor<T> &reluResTensor, GlobalTensor<T> &reluGm,
                                           DLIGradKLLossRunInfo &runInfo, int64_t reluResOffset, int64_t count);

    __aicore__ inline void ReLUGrad(LocalTensor<KV_T> &reluGradOutTensor, LocalTensor<T> &ReLuTensor,
                                    LocalTensor<T> &subResTensor, DLIGradKLLossRunInfo &runInfo);

    // global tensor
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyIndexGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<W_T> weightGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<T> softmaxMaxIndexGm;
    GlobalTensor<T> softmaxSumIndexGm;
    GlobalTensor<int64_t> actualSeqLengthsQGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    GlobalTensor<T> lossGm;
    GlobalTensor<T> lossGmDeterFloat;
    GlobalTensor<T> dWeightGmFloat;
    GlobalTensor<OUT_T> dKeyIndexGm;

    GlobalTensor<W_T> dWeightGmOut;
    GlobalTensor<MM4_OUT_T> dQueryIndexGmIn;
    GlobalTensor<OUT_T> dQueryIndexGmOut;

    GlobalTensor<INFO_INT_64_T> deterCoreInfoGm;
    GlobalTensor<T> dKeyIndexDeterGmFloat;
    GlobalTensor<T> dKeyIndexGmFloat;

    // workspace
    GlobalTensor<MM12_OUT_T> bmm1ResGm[2];
    GlobalTensor<MM12_OUT_T> bmm2ResGm[2];
    GlobalTensor<T> pSyncGm[2], sySyncGm[2];
    GlobalTensor<T> reluGm;
    GlobalTensor<KV_T> reluGradResGm[2];
    GlobalTensor<MM5_OUT_T> bmm5ResGm;

    // local tensor
    TBuf<> mm1Tbuf;
    TBuf<> mm2TBuf; // 复用 -> mm4 scatterAdd reluGrad
    TBuf<> sharedTBuf;
    TBuf<> reduceSumPTBuf;
    TBuf<> reduceSumYTBuf;
    TBuf<> v1TmpTBuf;
    TBuf<> reluGradTBuf;
    TBuf<> weightTBuf;
    TBuf<> lossSumTBuf;
    TBuf<> dwTBuf;
    TBuf<> maskBuf;
    TBuf<> uBuf_;

    // =============== vector1 variable ==============
    LocalTensor<Q_T> weightHalfUb_[2];
    LocalTensor<T> weightUb_[2];
    LocalTensor<T> mm1ResUb_;
    LocalTensor<T> mm2ResUb_;
    LocalTensor<T> pReduceUb_;
    LocalTensor<T> syReduceUb_;
    LocalTensor<T> softmaxMaxUb_;
    LocalTensor<T> softmaxSumUb_;
    LocalTensor<T> softmaxMaxBrdUb_;
    LocalTensor<T> softmaxSumBrdUb_;
    LocalTensor<uint8_t> softmaxTmpUb_;
    LocalTensor<uint8_t> softmaxIndexTmpUb_;

    LocalTensor<T> reduceSumYResTmpBuffer;
    LocalTensor<T> reduceSumYResUb;
    LocalTensor<uint8_t> reduceSumTmpBuffer;
    LocalTensor<uint8_t> softmaxTmpBuffer;

    // deter 相关
    LocalTensor<int64_t> deterCoreInfoUb_;
    LocalTensor<int64_t> deterCoreInfoUbConsumer_;

    event_t eventIdMte2ToV4P;
    event_t eventIdVToMte34P;
    event_t eventIdVToMte24P;
    event_t eventIdVToMte24Sm;
    event_t eventIdMte3ToV4P;

    event_t eventIdMte2ToV4SY;
    event_t eventIdVToMte34SY;
    event_t eventIdVToMte24SY;
    event_t eventIdVToMte24SmIndex;
    event_t eventIdMte3ToV4SY;

    event_t eventIdMTE3ToVTmp;
    event_t eventIdVToMte2Weight[2];

    // =============== vector4 variable ==============
    LocalTensor<T> reduceSumPUb_;
    LocalTensor<T> reduceSumYUb_;
    LocalTensor<T> tmpResultUb_;
    LocalTensor<T> reluResUb_;
    LocalTensor<T> mulLeftUb_;
    LocalTensor<KV_T> reluGradUb_;
    LocalTensor<uint8_t> maskUb_;
    LocalTensor<T> reduceSumResTensor_;
    LocalTensor<KV_T> dwOutTensor_;
    LocalTensor<uint8_t> tmpUb_;

    // ====================cast dwdq out variable =========
    LocalTensor<MM4_OUT_T> ubInPing_;
    LocalTensor<OUT_T> ubOutPing_;
    LocalTensor<MM4_OUT_T> ubInPong_;
    LocalTensor<OUT_T> ubOutPong_;
};

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::InitParams(
    const DLIGradKLLossConstInfo &vecConstInfo,
    const optiling::DenseLightningIndexerGradKLLossTilingData *__restrict tilingData)
{
    this->constInfo = vecConstInfo;
    this->tilingData = tilingData;
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::InitVector1GM(
    const GlobalTensor<T> &softmaxMax, const GlobalTensor<T> &softmaxSum, const GlobalTensor<T> &softmaxMaxIndex,
    const GlobalTensor<T> &softmaxSumIndex, const GlobalTensor<MM12_OUT_T> &bmm1Res,
    const GlobalTensor<MM12_OUT_T> &bmm2Res, const GlobalTensor<W_T> &weight, const GlobalTensor<T> pSync,
    const GlobalTensor<T> sySync, const GlobalTensor<T> &loss, const GlobalTensor<T> &dWeightFloat,
    const GlobalTensor<T> &reluGm, const GlobalTensor<KV_T> &reluGradRes, const GlobalTensor<W_T> &dWeight,
    const GlobalTensor<MM4_OUT_T> &dQueryIndexFloat, const GlobalTensor<OUT_T> &dQueryIndex)
{
    this->bmm1ResGm[0] = bmm1Res;
    this->bmm1ResGm[1] = bmm1Res[constInfo.n1Size * S1_BASE_STEP * S2_BASE_STEP];
    this->bmm2ResGm[0] = bmm2Res;
    this->bmm2ResGm[1] = bmm2Res[constInfo.n1IndexSize * S1_BASE_STEP * S2_BASE_STEP];
    this->softmaxMaxGm = softmaxMax;
    this->softmaxSumGm = softmaxSum;
    this->softmaxMaxIndexGm = softmaxMaxIndex;
    this->softmaxSumIndexGm = softmaxSumIndex;
    this->weightGm = weight;
    this->pSyncGm[0] = pSync;
    this->pSyncGm[1] = pSync[AIC_AIV_RATIO * S1_VEC_SIZE_8 * S2_BASE_STEP];
    this->sySyncGm[0] = sySync;
    this->sySyncGm[1] = sySync[AIC_AIV_RATIO * S1_VEC_SIZE_8 * S2_BASE_STEP];

    this->lossGm = loss;
    this->dWeightGmFloat = dWeightFloat;
    this->reluGm = reluGm;
    this->reluGradResGm[0] = reluGradRes;
    this->reluGradResGm[1] = reluGradRes[constInfo.n1IndexSize * S1_BASE_STEP * S2_BASE_STEP];

    this->dWeightGmOut = dWeight;
    this->dQueryIndexGmIn = dQueryIndexFloat;
    this->dQueryIndexGmOut = dQueryIndex;
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::InitVector1DeterGM(
    const GlobalTensor<INFO_INT_64_T> &deterCoreInfoGm, const GlobalTensor<T> &dKeyIndexDeterGmFloat,
    const GlobalTensor<T> &dKeyIndexGmFloat, const GlobalTensor<T> &lossGmDeterFloat)
{
    this->deterCoreInfoGm = deterCoreInfoGm;
    this->dKeyIndexDeterGmFloat = dKeyIndexDeterGmFloat;
    this->dKeyIndexGmFloat = dKeyIndexGmFloat;
    this->lossGmDeterFloat = lossGmDeterFloat;
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(this->uBuf_, DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_192K);

    // vec1/2
    uint32_t weightSize = S1_VEC_SIZE_8 * constInfo.n1IndexSize;
    uint32_t mm12ResultSize = S1_VEC_SIZE_8 * S2_BASE_STEP;
    uint32_t softmaxMaxSumSize = S1_VEC_SIZE_8;
    uint32_t ubOffset = 0;

    weightHalfUb_[0] = uBuf_.GetWithOffset<Q_T>(weightSize, ubOffset);
    ubOffset += weightSize * sizeof(Q_T);
    weightHalfUb_[1] = uBuf_.GetWithOffset<Q_T>(weightSize, ubOffset);
    ubOffset += weightSize * sizeof(Q_T);

    weightUb_[0] = uBuf_.GetWithOffset<T>(weightSize, ubOffset);
    ubOffset += weightSize * sizeof(T);
    weightUb_[1] = uBuf_.GetWithOffset<T>(weightSize, ubOffset);
    ubOffset += weightSize * sizeof(T);

    mm1ResUb_ = uBuf_.GetWithOffset<T>(mm12ResultSize, ubOffset);
    ubOffset += mm12ResultSize * sizeof(T);

    mm2ResUb_ = uBuf_.GetWithOffset<T>(mm12ResultSize, ubOffset);
    ubOffset += mm12ResultSize * sizeof(T);

    pReduceUb_ = uBuf_.GetWithOffset<T>(mm12ResultSize, ubOffset);
    ubOffset += mm12ResultSize * sizeof(T);

    syReduceUb_ = uBuf_.GetWithOffset<T>(mm12ResultSize, ubOffset);
    ubOffset += mm12ResultSize * sizeof(T);

    softmaxMaxUb_ = uBuf_.GetWithOffset<T>(softmaxMaxSumSize, ubOffset);
    ubOffset += softmaxMaxSumSize * sizeof(T);

    softmaxSumUb_ = uBuf_.GetWithOffset<T>(softmaxMaxSumSize, ubOffset);
    ubOffset += softmaxMaxSumSize * sizeof(T);

    softmaxMaxBrdUb_ = uBuf_.GetWithOffset<T>(softmaxMaxSumSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += softmaxMaxSumSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);

    softmaxSumBrdUb_ = uBuf_.GetWithOffset<T>(softmaxMaxSumSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += softmaxMaxSumSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);

    softmaxTmpUb_ = uBuf_.GetWithOffset<uint8_t>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K, ubOffset);
    ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K;

    softmaxIndexTmpUb_ = uBuf_.GetWithOffset<uint8_t>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K, ubOffset);
    ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K;

    // V4 UB
    uint32_t mmReluResultSize = S1_VEC_SIZE_8 * S2_BASE_STEP;
    uint32_t reluResSize = S2_BASE_STEP;
    uint32_t s2StepMaskSize = S2_BASE_STEP_MASK_V3V4;
    uint32_t tempSize = TEMP_VEC_SIZE_V3V4;

    ubOffset = 0;

    ubOffset += weightSize * sizeof(Q_T) * 6;
    reduceSumPUb_ = uBuf_.GetWithOffset<T>(mmReluResultSize, ubOffset);
    ubOffset += mmReluResultSize * sizeof(T);
    reduceSumYUb_ = uBuf_.GetWithOffset<T>(mmReluResultSize, ubOffset);
    ubOffset += mmReluResultSize * sizeof(T);
    tmpResultUb_ = uBuf_.GetWithOffset<T>(mmReluResultSize, ubOffset);
    ubOffset += mmReluResultSize * sizeof(T);
    reluResUb_ = uBuf_.GetWithOffset<T>(reluResSize, ubOffset);
    ubOffset += reluResSize * sizeof(T);
    mulLeftUb_ = uBuf_.GetWithOffset<T>(reluResSize, ubOffset);
    ubOffset += reluResSize * sizeof(T);
    reluGradUb_ = uBuf_.GetWithOffset<KV_T>(reluResSize, ubOffset);
    ubOffset += reluResSize * sizeof(KV_T);
    maskUb_ = uBuf_.GetWithOffset<uint8_t>(s2StepMaskSize, ubOffset);
    ubOffset += s2StepMaskSize * sizeof(uint8_t);
    reduceSumResTensor_ = uBuf_.GetWithOffset<T>(weightSize, ubOffset);
    ubOffset += weightSize * sizeof(T);
    dwOutTensor_ = uBuf_.GetWithOffset<KV_T>(weightSize, ubOffset);
    ubOffset += weightSize * sizeof(KV_T);
    tmpUb_ = uBuf_.GetWithOffset<uint8_t>(tempSize, ubOffset);
    ubOffset += tempSize * sizeof(uint8_t);

    // cast dwdq out UB
    ubOffset = 0;
    ubInPing_ = uBuf_.GetWithOffset<T>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K, ubOffset);
    ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K * sizeof(T);
    ubInPong_ = uBuf_.GetWithOffset<T>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K, ubOffset);
    ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K * sizeof(T);

    ubOutPing_ = uBuf_.GetWithOffset<OUT_T>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K, ubOffset);
    ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K * sizeof(OUT_T);
    ubOutPong_ = uBuf_.GetWithOffset<OUT_T>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K, ubOffset);
    ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K * sizeof(OUT_T);

    // deter 相关
    if constexpr (deterministic) {
        deterCoreInfoUb_ = uBuf_.GetWithOffset<int64_t>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_128, ubOffset);
        ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_128;
        deterCoreInfoUbConsumer_ = uBuf_.GetWithOffset<int64_t>(DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_128, ubOffset);
        ubOffset += DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_128;
    }
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::AllocEventID()
{
    eventIdMTE3ToVTmp = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    eventIdMte3ToV4P = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    eventIdMte3ToV4SY = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());

    eventIdVToMte24P = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte24Sm = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte24SY = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte24SmIndex = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte2Weight[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte2Weight[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());

    eventIdMte2ToV4SY = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    eventIdVToMte34SY = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    eventIdMte2ToV4P = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    eventIdVToMte34P = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMTE3ToVTmp);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV4P);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV4SY);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24P);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24Sm);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SmIndex);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[0]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[1]);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::FreeEventID()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMTE3ToVTmp);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV4P);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV4SY);

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24P);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24Sm);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SmIndex);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[0]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[1]);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMTE3ToVTmp);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV4P);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV4SY);

    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte24P);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte24Sm);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte24SY);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte24SmIndex);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2Weight[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2Weight[1]);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV4SY);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV4P);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte34SY);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte34P);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::CopyInWithSparseMask(DLIGradKLLossRunInfo &runInfo,
                                                                          LocalTensor<T> dstUbTensor,
                                                                          GlobalTensor<T> srcGmTensor, uint32_t rowSize,
                                                                          int32_t actualLenFirstRow)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, 0, SOFTMAX_MIN_NUM};

    if (actualLenFirstRow >= static_cast<int32_t>(S2_BASE_STEP)) {
        copyParams.blockLen = rowSize * S2_BASE_STEP * sizeof(T);
        padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad<T>(dstUbTensor, srcGmTensor, copyParams, padParams);
        PipeBarrier<PIPE_V>();
        return;
    }

    for (uint32_t rowIdx = 0; rowIdx < rowSize; rowIdx++) {
        int32_t actualLenThisRow = actualLenFirstRow + rowIdx;
        if (actualLenThisRow < static_cast<int32_t>(0)) {
            // 会有当前s2Loop竖切mask对角线的情况，这样前几行不需要处理
            actualLenThisRow = 0;
        }
        actualLenThisRow = actualLenThisRow >= S2_BASE_STEP ? S2_BASE_STEP : actualLenThisRow;
        uint32_t actualLenThisRowAlign8 = CeilDiv(actualLenThisRow, FLOAT_DATA_BLOCK_NUM) * FLOAT_DATA_BLOCK_NUM;
        uint32_t padNum = actualLenThisRowAlign8 - actualLenThisRow;
        if (actualLenThisRow > 0) {
            copyParams.blockLen = actualLenThisRow * sizeof(T);
            padParams.rightPadding = (uint8_t)padNum;
            AscendC::DataCopyPad<T>(dstUbTensor[rowIdx * runInfo.curS2StepSizeAlign8],
                                    srcGmTensor[rowIdx * runInfo.curS2StepSize], copyParams, padParams);
        }

        // 填充剩余的-inf
        padNum = runInfo.curS2StepSizeAlign8 - actualLenThisRowAlign8;
        AscendC::Duplicate(dstUbTensor[rowIdx * runInfo.curS2StepSizeAlign8 + actualLenThisRowAlign8], SOFTMAX_MIN_NUM,
                           padNum);
    }

    PipeBarrier<PIPE_V>();
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::PreloadWeight(DLIGradKLLossRunInfo &runInfo, uint32_t s1InnerIdx,
                                                                   uint32_t curS1InnerSize, uint32_t pingpongFlag)
{
    uint32_t weightSize = curS1InnerSize * constInfo.n1IndexSize;
    uint32_t weightGmOffset = runInfo.weightTensorOffset + s1InnerIdx * S1_VEC_SIZE_8 * constInfo.n1IndexSize;
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());

    // weight 可以常驻, 所以直接搬运, 减少搬运切片
    if constexpr (IsSameType<W_T, float>::value) {
        // n1Index最小为8, 一定对齐搬运
        AscendC::DataCopy(weightUb_[pingpongFlag], weightGm[weightGmOffset], weightSize);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[pingpongFlag]);
        if (weightSize % C0_SIZE != 0) {
            uint32_t padNum = (weightSize + C0_SIZE - 1) / C0_SIZE * C0_SIZE - weightSize;
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(weightSize * sizeof(Q_T)), 0, 0, 0};
            DataCopyPadExtParams<KV_T> copyPadParams = {true, 0, (uint8_t)(padNum), 0.0};
            AscendC::DataCopyPad(weightHalfUb_[pingpongFlag], weightGm[weightGmOffset], copyParams, copyPadParams);
        } else {
            AscendC::DataCopy(weightHalfUb_[pingpongFlag], weightGm[weightGmOffset], weightSize);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::Cast(weightUb_[pingpongFlag], weightHalfUb_[pingpongFlag], AscendC::RoundMode::CAST_NONE, weightSize);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte2Weight[pingpongFlag]);

        PipeBarrier<PIPE_V>();
    }

    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::VectorP(DLIGradKLLossRunInfo &runInfo, uint32_t s1InnerIdx,
                                                             uint32_t curS1InnerSize, uint32_t pingpongFlag)
{
    uint32_t mm1ResGmOffsetS1 = constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO * runInfo.curS2StepSize +
                                s1InnerIdx * S1_VEC_SIZE_8 * runInfo.curS2StepSize;
    uint32_t processNumPerHead = curS1InnerSize * runInfo.curS2StepSizeAlign8;

    for (uint32_t n1Idx = 0; n1Idx < constInfo.n1Size; n1Idx++) {
        int32_t actualLenFirstRow = runInfo.sparseMaskStartIdx +
                                    constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO +
                                    s1InnerIdx * S1_VEC_SIZE_8 + 1;
        uint32_t mm1ResGmOffset = mm1ResGmOffsetS1 + n1Idx * runInfo.curS1Size * runInfo.curS2StepSize;
        auto mm1SrcGm = bmm1ResGm[runInfo.taskIdMod2][mm1ResGmOffset];

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24P);
        CopyInWithSparseMask(runInfo, mm1ResUb_, mm1SrcGm, curS1InnerSize, actualLenFirstRow);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4P);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4P);

        AscendC::Muls(mm1ResUb_, mm1ResUb_, constInfo.scaleValue, processNumPerHead);
        PipeBarrier<PIPE_V>();

        // simple softmax
        uint32_t softmaxGmOffset =
            runInfo.softmaxTensorOffset + s1InnerIdx * S1_VEC_SIZE_8 + n1Idx * constInfo.softmaxHeadOffset;
        DataCopyExtParams softmaxCopyParams = {static_cast<uint16_t>(1),
                                               static_cast<uint32_t>(curS1InnerSize * sizeof(T)), // 8 * 4 = 32Bytes
                                               static_cast<uint32_t>(0), 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24Sm);
        AscendC::DataCopyPad(softmaxMaxUb_, softmaxMaxGm[softmaxGmOffset], softmaxCopyParams, padParams);
        AscendC::DataCopyPad(softmaxSumUb_, softmaxSumGm[softmaxGmOffset], softmaxCopyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4P);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4P);
        AscendC::Brcb(softmaxMaxBrdUb_, softmaxMaxUb_, S1_VEC_SIZE_8 / FLOAT_DATA_BLOCK_NUM, {1, 8});
        AscendC::Brcb(softmaxSumBrdUb_, softmaxSumUb_, S1_VEC_SIZE_8 / FLOAT_DATA_BLOCK_NUM, {1, 8});
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24Sm);
        PipeBarrier<PIPE_V>();
        SoftMaxTiling simpleSoftMaxTilingInfo = tilingData->vectorParams.simpleSoftmaxPTilingData;
        AscendC::SimpleSoftMax<T>(
            mm1ResUb_, softmaxSumBrdUb_, softmaxMaxBrdUb_, mm1ResUb_, softmaxTmpUb_, simpleSoftMaxTilingInfo,
            {static_cast<uint32_t>(curS1InnerSize), static_cast<uint32_t>(runInfo.curS2StepSizeAlign8),
             static_cast<uint32_t>(curS1InnerSize), static_cast<uint32_t>(runInfo.curS2StepSize)});
        PipeBarrier<PIPE_V>();

        if (n1Idx == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV4P);
            AscendC::DataCopy(pReduceUb_, mm1ResUb_, processNumPerHead);
        } else {
            AscendC::Add(pReduceUb_, pReduceUb_, mm1ResUb_, processNumPerHead);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24P);
    }

    PipeBarrier<PIPE_V>();
    float normScale = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.n1Size));
    AscendC::Muls(pReduceUb_, pReduceUb_, normScale, processNumPerHead);

    uint32_t dstPGmOffset = constInfo.subBlockIdx * S1_VEC_SIZE_8 * S2_BASE_STEP;
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(processNumPerHead * sizeof(T)), 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte34P);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte34P);
    AscendC::DataCopyPad(pSyncGm[pingpongFlag][dstPGmOffset], pReduceUb_, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV4P);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::VectorSy(DLIGradKLLossRunInfo &runInfo, uint32_t s1InnerIdx,
                                                              uint32_t curS1InnerSize, uint32_t pingpongFlag)
{
    // weight UB cast To weight.GetValue and release eventID
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);

    uint32_t mm2ResGmOffsetS1 = constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO * runInfo.curS2StepSize +
                                s1InnerIdx * S1_VEC_SIZE_8 * runInfo.curS2StepSize;
    uint32_t processNumPerHead = curS1InnerSize * runInfo.curS2StepSizeAlign8;

    for (uint32_t n1IndexIdx = 0; n1IndexIdx < constInfo.n1IndexSize; n1IndexIdx++) {
        int32_t actualLenFirstRow = runInfo.sparseMaskStartIdx +
                                    constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO +
                                    s1InnerIdx * S1_VEC_SIZE_8 + 1;
        uint32_t mm2ResGmOffset = mm2ResGmOffsetS1 + n1IndexIdx * runInfo.curS1Size * runInfo.curS2StepSize;
        auto mm2SrcGm = bmm2ResGm[runInfo.taskIdMod2][mm2ResGmOffset];

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
        CopyInWithSparseMask(runInfo, mm2ResUb_, mm2SrcGm, curS1InnerSize, actualLenFirstRow);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);

        for (uint32_t rowIdx = 0; rowIdx < curS1InnerSize; rowIdx++) {
            float weightValue = weightUb_[pingpongFlag].GetValue(rowIdx * constInfo.n1IndexSize + n1IndexIdx);
            uint32_t mulOffset = rowIdx * runInfo.curS2StepSizeAlign8;

            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);

            int32_t processNumThisRow = actualLenFirstRow + rowIdx;
            if (processNumThisRow > static_cast<int32_t>(S2_BASE_STEP)) {
                processNumThisRow = static_cast<int32_t>(S2_BASE_STEP);
            }
            if (processNumThisRow > 0) {
                AscendC::Muls<T>(mm2ResUb_[mulOffset], mm2ResUb_[mulOffset], weightValue, processNumThisRow);
            }
        }

        PipeBarrier<PIPE_V>();

        // Reduce N
        if (n1IndexIdx == 0) {
            AscendC::DataCopy(syReduceUb_, mm2ResUb_, processNumPerHead);
        } else {
            AscendC::Add(syReduceUb_, syReduceUb_, mm2ResUb_, processNumPerHead);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SY);
    }

    PipeBarrier<PIPE_V>();

    // simple softmax
    uint32_t softmaxGmOffset = runInfo.softmaxIndexTensorOffset + s1InnerIdx * S1_VEC_SIZE_8;
    DataCopyExtParams softmaxCopyParams = {static_cast<uint16_t>(1),
                                           static_cast<uint32_t>(curS1InnerSize * sizeof(T)), // 8 * 4 = 32Bytes
                                           static_cast<uint32_t>(0), 0, 0};
    DataCopyPadExtParams<T> softmaxCopyPadParams = {false, 0, 0, 0};
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SmIndex);
    AscendC::DataCopyPad(softmaxMaxUb_, softmaxMaxIndexGm[softmaxGmOffset], softmaxCopyParams, softmaxCopyPadParams);
    AscendC::DataCopyPad(softmaxSumUb_, softmaxSumIndexGm[softmaxGmOffset], softmaxCopyParams, softmaxCopyPadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV4SY);
    AscendC::Brcb(softmaxMaxBrdUb_, softmaxMaxUb_, S1_VEC_SIZE_8 / FLOAT_DATA_BLOCK_NUM, {1, 8});
    AscendC::Brcb(softmaxSumBrdUb_, softmaxSumUb_, S1_VEC_SIZE_8 / FLOAT_DATA_BLOCK_NUM, {1, 8});
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMte24SmIndex);
    PipeBarrier<PIPE_V>();
    SoftMaxTiling simpleSoftMaxTilingInfo = tilingData->vectorParams.simpleSoftmaxPTilingData;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV4SY);
    AscendC::SimpleSoftMax<T>(
        syReduceUb_, softmaxSumBrdUb_, softmaxMaxBrdUb_, syReduceUb_, softmaxIndexTmpUb_, simpleSoftMaxTilingInfo,
        {static_cast<uint32_t>(curS1InnerSize), static_cast<uint32_t>(runInfo.curS2StepSizeAlign8),
         static_cast<uint32_t>(curS1InnerSize), static_cast<uint32_t>(runInfo.curS2StepSize)});
    PipeBarrier<PIPE_V>();

    uint32_t dstSyGmOffset = constInfo.subBlockIdx * S1_VEC_SIZE_8 * S2_BASE_STEP;
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(processNumPerHead * sizeof(T)), 0, 0, 0};
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte34SY);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte34SY);
    AscendC::DataCopyPad(sySyncGm[pingpongFlag][dstSyGmOffset], syReduceUb_, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV4SY);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::VectorCopyInDwDqDk(DLIGradKLLossRunInfo &runInfo)
{
    // 拷入P和SY
    int64_t copyPSYOffset = constInfo.subBlockIdx * S1_VEC_SIZE_8 * S2_BASE_STEP;

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.dstStride = 0;
    dataCopyParams.srcStride = 0;
    dataCopyParams.blockLen = runInfo.curS1InnerSizeV1V2 * runInfo.curS2StepSizeAlign8 / FLOAT_DATA_BLOCK_NUM;

    DataCopy(reduceSumPUb_, pSyncGm[runInfo.pingPongFlagV1V2][copyPSYOffset], dataCopyParams);
    DataCopy(reduceSumYUb_, sySyncGm[runInfo.pingPongFlagV1V2][copyPSYOffset], dataCopyParams);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::DataCopyReluRes(LocalTensor<T> &reluResTensor,
                                                                     GlobalTensor<T> &reluGm,
                                                                     DLIGradKLLossRunInfo &runInfo,
                                                                     int64_t reluResOffset, int64_t count)
{
    int64_t countAlign = BlockAlign<T>(count);
    if (likely(countAlign == count)) {
        DataCopy(reluResTensor, reluGm[reluResOffset], count);
    } else {
        DataCopyParams dataCopyParams;
        DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        dataCopyParams.blockLen = count * sizeof(T);
        dataCopyPadParams.rightPadding = countAlign - count;
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(reluResTensor, reluGm[reluResOffset], dataCopyParams, dataCopyPadParams);
    }
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::ReLUGrad(LocalTensor<KV_T> &reluGradOutTensor,
                                                              LocalTensor<T> &reLuTensor, LocalTensor<T> &subResTensor,
                                                              DLIGradKLLossRunInfo &runInfo)
{
    uint32_t curS2StepSizeAlign64 =
        (runInfo.curS2StepSize + FLOAT_REPEAT_NUM - 1) / FLOAT_REPEAT_NUM * FLOAT_REPEAT_NUM;
    CompareScalar(maskUb_, subResTensor, static_cast<T>(0.0), AscendC::CMPMODE::GT, curS2StepSizeAlign64);
    PipeBarrier<PIPE_V>();
    Select(reLuTensor, maskUb_, reLuTensor, static_cast<T>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
           runInfo.curS2StepSizeAlign8);
    PipeBarrier<PIPE_V>();
    Cast(reluGradOutTensor, reLuTensor, AscendC::RoundMode::CAST_ROUND, runInfo.curS2StepSizeAlign8);
    PipeBarrier<PIPE_V>();
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::VectorLoss(DLIGradKLLossRunInfo &runInfo)
{
    event_t vToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    const float MIN_VALUE = 1e-8;
    SumParams sumParams;
    sumParams.outter = 1;
    sumParams.inner = runInfo.curS2StepSizeAlign8;
    sumParams.n = runInfo.curS2StepSize;
    int64_t calcCount = runInfo.curS1InnerSizeV1V2 * runInfo.curS2StepSizeAlign8;

    VectorCopyInDwDqDk(runInfo);

    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);

    AscendC::Maxs<T>(tmpResultUb_, reduceSumPUb_, MIN_VALUE, calcCount);
    PipeBarrier<PIPE_V>();
    AscendC::Maxs<T>(reduceSumYUb_, reduceSumYUb_, MIN_VALUE, calcCount);
    PipeBarrier<PIPE_V>();

    AscendC::Log(tmpResultUb_, tmpResultUb_, calcCount);
    AscendC::Log(reduceSumYUb_, reduceSumYUb_, calcCount);
    PipeBarrier<PIPE_V>();

    AscendC::Sub(reduceSumYUb_, tmpResultUb_, reduceSumYUb_, calcCount);
    PipeBarrier<PIPE_V>();
    AscendC::Mul(reduceSumYUb_, reduceSumYUb_, reduceSumPUb_, calcCount);
    PipeBarrier<PIPE_V>();

    for (int64_t rowIdx = 0; rowIdx < runInfo.curS1InnerSizeV1V2; rowIdx++) {
        if (rowIdx == 0) {
            AscendC::Sum(reduceSumResTensor_, reduceSumYUb_[rowIdx * runInfo.curS2StepSizeAlign8], tmpUb_, sumParams);
            PipeBarrier<PIPE_V>();
        } else {
            AscendC::Sum(mulLeftUb_, reduceSumYUb_[rowIdx * runInfo.curS2StepSizeAlign8], tmpUb_, sumParams);
            PipeBarrier<PIPE_V>();

            AscendC::Add(reduceSumResTensor_, reduceSumResTensor_, mulLeftUb_, 8);
            PipeBarrier<PIPE_V>();
        }
    }

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);

    AscendC::SetAtomicAdd<float>();
    if constexpr (deterministic) {
        AscendC::DataCopyPad(lossGmDeterFloat[constInfo.aivIdx], reduceSumResTensor_,
                             {static_cast<uint32_t>(1), static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(0),
                              static_cast<uint32_t>(0)});
    } else {
        AscendC::DataCopyPad(lossGm, reduceSumResTensor_,
                             {static_cast<uint32_t>(1), static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(0),
                              static_cast<uint32_t>(0)});
    }
    SetAtomicNone();

    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vToMte2);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::VectorDwDqDk(DLIGradKLLossRunInfo &runInfo)
{
    event_t vToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t mte3ToVDw = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());

    SumParams sumParams;
    sumParams.outter = 1;
    sumParams.inner = runInfo.curS2StepSizeAlign8;
    sumParams.n = runInfo.curS2StepSize;

    DataCopyParams dataCopyReluGradParams;
    dataCopyReluGradParams.blockLen = runInfo.curS2StepSize * sizeof(KV_T);
    dataCopyReluGradParams.srcStride = 0;
    dataCopyReluGradParams.blockCount = 1;
    dataCopyReluGradParams.dstStride = 0;
    DataCopyPadParams dataCopyReluGradPadParams;
    dataCopyReluGradPadParams.rightPadding = runInfo.curS2StepSizeAlign8 - runInfo.curS2StepSize;
    dataCopyReluGradPadParams.paddingValue = 0;

    DataCopyParams dataCopyDwParams;
    dataCopyDwParams.blockLen = sizeof(T);
    dataCopyDwParams.srcStride = 0;
    dataCopyDwParams.blockCount = constInfo.n1IndexSize;
    dataCopyDwParams.dstStride = 0;

    uint32_t mm2ResGmOffsetS1 = constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO * runInfo.curS2StepSize +
                                runInfo.s1InnerIdxV1V2 * S1_VEC_SIZE_8 * runInfo.curS2StepSize;

    VectorCopyInDwDqDk(runInfo);

    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);

    Sub(reduceSumPUb_, reduceSumYUb_, reduceSumPUb_, runInfo.curS1InnerSizeV1V2 * runInfo.curS2StepSizeAlign8);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    SetFlag<HardEvent::MTE3_V>(mte3ToVDw);
    SetFlag<HardEvent::V_MTE2>(vToMte2);
    for (int64_t rowIdx = 0; rowIdx < runInfo.curS1InnerSizeV1V2; rowIdx++) {
        WaitFlag<HardEvent::MTE3_V>(mte3ToVDw);
        for (int64_t nIdx = 0; nIdx < constInfo.n1IndexSize; nIdx++) {
            WaitFlag<HardEvent::V_MTE2>(vToMte2);

            int64_t reluResOffset =
                nIdx * runInfo.curS1Size * runInfo.curS2StepSize + mm2ResGmOffsetS1 + rowIdx * runInfo.curS2StepSize;
            int64_t reluResCount = runInfo.curS2StepSize;
            DataCopyReluRes(reluResUb_, bmm2ResGm[runInfo.taskIdMod2], runInfo, reluResOffset, reluResCount);

            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);

            AscendC::Mul(mulLeftUb_, reduceSumPUb_[rowIdx * runInfo.curS2StepSizeAlign8], reluResUb_,
                         runInfo.curS2StepSizeAlign8);
            PipeBarrier<PIPE_V>();

            AscendC::Sum(reduceSumResTensor_[nIdx * 8], mulLeftUb_, tmpUb_, sumParams);
            PipeBarrier<PIPE_V>();

            float weightValue = weightUb_[runInfo.pingPongFlagV1V2].GetValue(rowIdx * constInfo.n1IndexSize + nIdx);
            AscendC::Muls(mulLeftUb_, reduceSumPUb_[rowIdx * runInfo.curS2StepSizeAlign8], weightValue,
                          runInfo.curS2StepSizeAlign8);
            PipeBarrier<PIPE_V>();

            WaitFlag<HardEvent::MTE3_V>(mte3ToV);
            ReLUGrad(reluGradUb_, mulLeftUb_, reluResUb_, runInfo);
            SetFlag<HardEvent::V_MTE2>(vToMte2);

            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);

            DataCopyPad(reluGradResGm[runInfo.taskIdMod2][reluResOffset], reluGradUb_, dataCopyReluGradParams);
            SetFlag<HardEvent::MTE3_V>(mte3ToV);
        }

        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);

        // dwOutTensor拷出到dWeightGmFloat
        if (runInfo.s2Idx > 0) {
            AscendC::SetAtomicAdd<float>();
        }

        if constexpr (!IsSameType<W_T, float>::value) {
            int64_t dWeightGmOffset =
                constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO * constInfo.n1IndexSize +
                (runInfo.s1InnerIdxV1V2 * S1_VEC_SIZE_8 + rowIdx) * constInfo.n1IndexSize;
            DataCopyPad(dWeightGmFloat[dWeightGmOffset], reduceSumResTensor_, dataCopyDwParams);
        } else {
            int64_t dWeightGmOffset =
                runInfo.accumS1Idx * constInfo.n1IndexSize +
                constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO * constInfo.n1IndexSize +
                (runInfo.s1InnerIdxV1V2 * S1_VEC_SIZE_8 + rowIdx) * constInfo.n1IndexSize;
            DataCopyPad(dWeightGmOut[dWeightGmOffset], reduceSumResTensor_, dataCopyDwParams);
        }
        if (runInfo.s2Idx > 0) {
            SetAtomicNone();
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVDw);
    }
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVDw);
    WaitFlag<HardEvent::V_MTE2>(vToMte2);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vToMte2);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToVDw);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::ProcessVector1(DLIGradKLLossRunInfo &runInfo)
{
    if (runInfo.curS1SizeVec == 0) {
        return;
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMTE3ToVTmp);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

    uint32_t s1InnerLoopTimes = CeilDiv(runInfo.curS1SizeVec, S1_VEC_SIZE_8);
    uint32_t pingpongFlag = 0;

    for (uint32_t s1InnerIdx = 0; s1InnerIdx < s1InnerLoopTimes; s1InnerIdx++) {
        uint32_t curS1InnerSize =
            (s1InnerIdx == s1InnerLoopTimes - 1) ? (runInfo.curS1SizeVec - s1InnerIdx * S1_VEC_SIZE_8) : S1_VEC_SIZE_8;
        PreloadWeight(runInfo, s1InnerIdx, curS1InnerSize, pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        VectorSy(runInfo, s1InnerIdx, curS1InnerSize, pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        VectorP(runInfo, s1InnerIdx, curS1InnerSize, pingpongFlag);

        runInfo.curS1InnerSizeV1V2 = curS1InnerSize;
        runInfo.pingPongFlagV1V2 = pingpongFlag;
        runInfo.s1InnerIdxV1V2 = s1InnerIdx;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);

        VectorDwDqDk(runInfo);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        VectorLoss(runInfo);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        pingpongFlag = (pingpongFlag + 1) & 1;
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMTE3ToVTmp);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::CastOutWeightGrad(DLIGradKLLossRunInfo &runInfo)
{
    if (runInfo.curS1SizeVec == 0) {
        return;
    }

    if constexpr (!IsSameType<W_T, float>::value) {
        // cast dw
        int64_t dwGMOutOffset = runInfo.accumS1Idx * constInfo.n1IndexSize +
                                constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO * constInfo.n1IndexSize;
        int64_t dwGMInOffset = constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO * constInfo.n1IndexSize;
        int64_t dWeightCount = runInfo.curS1SizeVec * constInfo.n1IndexSize;
        DataCopy(ubInPing_, dWeightGmFloat[dwGMInOffset], dWeightCount);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        Cast(ubOutPing_, ubInPing_, RoundMode::CAST_ROUND, dWeightCount);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        if ((dWeightCount * sizeof(W_T)) % VEC_ALIGN_SIZE != 0) {
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(dWeightCount * sizeof(W_T)), 0, 0, 0};
            AscendC::DataCopyPad(dWeightGmOut[dwGMOutOffset], ubOutPing_, copyParams);
        } else {
            DataCopy(dWeightGmOut[dwGMOutOffset], ubOutPing_, dWeightCount);
        }

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::CastOutQIndexGrad(DLIGradKLLossRunInfo &runInfo)
{
    if (runInfo.curS1SizeVec == 0) {
        return;
    }
    // cast dq
    event_t eventId = EVENT_ID2;
    int32_t pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    for (uint32_t n1IndexId = 0; n1IndexId < constInfo.n1IndexSize; n1IndexId++) {
        uint32_t dqGmInOffset = n1IndexId * runInfo.curS1Size * constInfo.dSizeQueryIndex +
                                constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO * constInfo.dSizeQueryIndex;
        uint32_t dqGmOutOffset = runInfo.accumS1Idx * constInfo.n1IndexSize * constInfo.dSizeQueryIndex +
                                 constInfo.subBlockIdx * runInfo.curS1Size / AIC_AIV_RATIO * constInfo.n1IndexSize *
                                     constInfo.dSizeQueryIndex +
                                 n1IndexId * constInfo.dSizeQueryIndex;
        uint32_t dqCount = runInfo.curS1SizeVec * constInfo.dSizeQueryIndex;

        eventId = pingPongFlag ? EVENT_ID2 : EVENT_ID3;
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        LocalTensor<MM4_OUT_T> dQueryIndexUbIn = pingPongFlag ? ubInPong_ : ubInPing_;
        LocalTensor<OUT_T> dQueryIndexUbOut = pingPongFlag ? ubOutPong_ : ubOutPing_;

        DataCopy(dQueryIndexUbIn, dQueryIndexGmIn[dqGmInOffset], dqCount);
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        Cast(dQueryIndexUbOut, dQueryIndexUbIn, RoundMode::CAST_ROUND, dqCount);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(runInfo.curS1SizeVec),
            static_cast<uint32_t>(constInfo.dSizeQueryIndex * sizeof(OUT_T)), 0,
            static_cast<uint32_t>((constInfo.n1IndexSize - 1) * constInfo.dSizeQueryIndex * sizeof(OUT_T)), 0};
        DataCopyPad(dQueryIndexGmOut[dqGmOutOffset], dQueryIndexUbOut, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);

        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::DeterAddKIndexGrad(DLIGradKLLossRunInfo &runInfo)
{
    for (int aicIdx = 0; aicIdx < constInfo.aicNum; aicIdx++) {
        GetDeterRunInfo(aicIdx);
        int64_t bIdx = deterCoreInfoUbConsumer_.GetValue(0);
        int64_t s2StartIdx = deterCoreInfoUbConsumer_.GetValue(1);
        int64_t curS2Size = deterCoreInfoUbConsumer_.GetValue(2);
        if (bIdx < 0 || s2StartIdx < 0 || curS2Size <= 0) {
            continue;
        }
        int64_t totalSize = curS2Size * constInfo.n2IndexSize * constInfo.dSizeQueryIndex;
        int64_t useCoreNum = constInfo.aivNum;
        int64_t splitLine = CeilDiv(S2_BASE_STEP, useCoreNum);

        useCoreNum = Min(constInfo.aivNum, CeilDiv(curS2Size, splitLine));
        if (constInfo.aivIdx >= useCoreNum) {
            continue;
        }

        int64_t srcGmOffset = aicIdx * constInfo.dKeyDeterGmLength +
                              constInfo.aivIdx * splitLine * constInfo.n2IndexSize * constInfo.dSizeQueryIndex;
        GlobalTensor<T> srcGm = dKeyIndexDeterGmFloat[srcGmOffset];

        int64_t dstGmOffset = s2StartIdx * constInfo.n2IndexSize * constInfo.dSizeQueryIndex +
                              constInfo.aivIdx * splitLine * constInfo.n2IndexSize * constInfo.dSizeQueryIndex;
        GlobalTensor<T> dstGm = dKeyIndexGmFloat[dstGmOffset];

        int64_t perCoreS2Line = Min(splitLine, curS2Size - constInfo.aivIdx * splitLine);
        if (perCoreS2Line <= 0) {
            continue;
        }
        int64_t onelineSize = constInfo.dSizeQueryIndex * sizeof(float);
        int64_t linesOneLoop = DLIGradKLLossConstInfo::BUFFER_SIZE_BYTE_8K / onelineSize;
        uint32_t loopTimes = CeilDiv(perCoreS2Line, linesOneLoop);

        event_t eventId = EVENT_ID0;
        int32_t pingPongFlag = 0;

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int64_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
            int64_t dKeyGmOffsetCur = loopIdx * linesOneLoop * constInfo.n2IndexSize * constInfo.dSizeQueryIndex;
            int64_t processNum = linesOneLoop * constInfo.n2IndexSize * constInfo.dSizeQueryIndex;
            if (loopIdx == loopTimes - 1) {
                processNum =
                    (perCoreS2Line - loopIdx * linesOneLoop) * constInfo.n2IndexSize * constInfo.dSizeQueryIndex;
            }
            eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
            LocalTensor<T> dKeyIndexUbIn = pingPongFlag ? ubInPong_ : ubInPing_;

            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            AscendC::DataCopy(dKeyIndexUbIn, srcGm[dKeyGmOffsetCur], processNum);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);

            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            AscendC::SetAtomicAdd<T>();
            AscendC::DataCopy(dstGm[dKeyGmOffsetCur], dKeyIndexUbIn, processNum);
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_MTE2>(eventId);

            pingPongFlag = 1 - pingPongFlag;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::SaveDeterRunInfo(const DLIGradKLLossRunInfo &info)
{
    deterCoreInfoUb_.SetValue(0, info.bIdx);
    deterCoreInfoUb_.SetValue(1, info.accumS2Idx);
    deterCoreInfoUb_.SetValue(2, static_cast<int64_t>(info.curS2StepSize));

    DataCopy(deterCoreInfoGm[constInfo.aicIdx * 16], deterCoreInfoUb_, 4);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::SaveDeterRunInfoInvalid()
{
    deterCoreInfoUb_.SetValue(0, DETER_INVALID_RUNINFO_VALUE);
    deterCoreInfoUb_.SetValue(1, DETER_INVALID_RUNINFO_VALUE);
    deterCoreInfoUb_.SetValue(2, DETER_INVALID_RUNINFO_VALUE);

    DataCopy(deterCoreInfoGm[constInfo.aicIdx * 16], deterCoreInfoUb_, (uint32_t)4);
}

template <typename DLIT>
__aicore__ inline void DLIKLLossVectorService<DLIT>::GetDeterRunInfo(int64_t aicIdx)
{
    event_t mte2ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
    AscendC::DataCopy(deterCoreInfoUbConsumer_, deterCoreInfoGm[aicIdx * 16], (uint32_t)4);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2ToS);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2ToS);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_S>(mte2ToS);
}
#endif // DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_VECTOR_H
