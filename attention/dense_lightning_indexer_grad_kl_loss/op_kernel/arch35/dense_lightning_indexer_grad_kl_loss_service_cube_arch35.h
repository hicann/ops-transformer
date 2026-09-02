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
 * \file dense_lightning_indexer_grad_kl_loss_service_cube_arch35.h
 * \brief
 */
#ifndef DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SERVICE_CUBE_ARCH35_H
#define DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SERVICE_CUBE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "dense_lightning_indexer_grad_kl_loss_common_arch35.h"

template <typename DLIT>
class DenseLightningIndexerGradKLLossServiceCube {
public:
    // 中间计算数据类型为float, 高精度模式
    using T = float;
    using Q_T = typename DLIT::inputQT;
    using KV_T = typename DLIT::inputKT;
    using OUT_T = typename DLIT::outputT;
    using MM12_OUT_T = T;

    __aicore__ inline DenseLightningIndexerGradKLLossServiceCube(){};
    __aicore__ inline void InitParams(const DLIGradKLLossConstInfo &constInfo);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> &queryGm, GlobalTensor<KV_T> &keyGm,
                                               GlobalTensor<Q_T> &qRopeGm, GlobalTensor<KV_T> &keyRopeGm,
                                               GlobalTensor<int64_t> &actualSeqLengthsQueryGm,
                                               GlobalTensor<int64_t> &actualSeqLengthsKeyGm,
                                               GlobalTensor<MM12_OUT_T> &bmm1Res, GM_ADDR mm1Res);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<Q_T> &queryIndex, GlobalTensor<KV_T> &keyIndex,
                                               GlobalTensor<T> &mm2Res);
    __aicore__ inline void InitMm5GlobalTensor(GlobalTensor<Q_T> &reluGradRes, GlobalTensor<MM12_OUT_T> &bmm5Res);
    __aicore__ inline void InitMm5DeterGlobalTensor(GlobalTensor<MM12_OUT_T> &dKeyIndexDeterGmFloat);
    __aicore__ inline void InitMm6GlobalTensor(GlobalTensor<T> &bmm6Res);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void ComputeMm1(const DLIGradKLLossRunInfo &info);
    __aicore__ inline void ComputeMm2(const DLIGradKLLossRunInfo &info);
    __aicore__ inline void ComputeMm34(const DLIGradKLLossRunInfo &info);
    __aicore__ inline void ComputeMm34Deter(const DLIGradKLLossRunInfo &info);
    __aicore__ inline void ComputeMm4(const DLIGradKLLossRunInfo &info);
    __aicore__ inline void FreeEventID();

private:
    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(const DLIGradKLLossRunInfo &runInfo, uint32_t qHeadIdx);
    __aicore__ inline void CopyInMm1BToL1(const DLIGradKLLossRunInfo &runInfo, uint32_t kvHeadIdx, uint32_t s2InnerIdx,
                                          uint32_t actualS2Len, uint8_t pingpongFlag);
    __aicore__ inline void CopyInMm2AToL1(const DLIGradKLLossRunInfo &runInfo, uint32_t n1IndexId,
                                          uint32_t pingpongFlag);
    __aicore__ inline void CopyInMm2BToL1(const DLIGradKLLossRunInfo &runInfo, uint32_t n2IndexId, uint32_t s2InnerIdx,
                                          uint32_t actualS2Len, uint8_t pingpongFlag);
    __aicore__ inline void CopyInReluGradToL1A(const DLIGradKLLossRunInfo &runInfo, uint32_t n1IndexId);
    __aicore__ inline void CopyInMm4BToL1(const DLIGradKLLossRunInfo &runInfo, uint32_t n2IndexId, uint32_t s2InnerIdx,
                                          uint32_t actualS2Len, uint8_t pingpongFlag);
    __aicore__ inline void MmadInner(struct MMParam &mmParams, LocalTensor<Q_T> &l1QTensor,
                                     LocalTensor<KV_T> &l1KTensor, LocalTensor<MM12_OUT_T> l0cTensor,
                                     uint8_t l1PingPongFlag, uint8_t l0abPingPongFlag, uint8_t l0cPingPongFlag);
    __aicore__ inline void LoadDataMmA(LocalTensor<Q_T> &aL0Tensor, LocalTensor<Q_T> &aL1Tensor,
                                       struct MMParam &mmParam);
    __aicore__ inline void LoadDataMmAWithTranspose(LocalTensor<Q_T> &aL0Tensor, LocalTensor<Q_T> &aL1Tensor,
                                                    struct MMParam &mmParam);
    __aicore__ inline void LoadDataMmB(LocalTensor<KV_T> &bL0Tensor, LocalTensor<KV_T> &bL1Tensor,
                                       struct MMParam &mmParam);
    __aicore__ inline void LoadDataMmBWithTranspose(LocalTensor<KV_T> &bL0Tensor, LocalTensor<KV_T> &bL1Tensor,
                                                    struct MMParam &mmParam);
    __aicore__ inline void FixCopyOut(struct MMParam &mmParams, GlobalTensor<MM12_OUT_T> &resGm,
                                      LocalTensor<MM12_OUT_T> l0cTensor, uint8_t l0cPingPongFlag);
    __aicore__ inline void FixOutAtomicAdd(struct MMParam &mmParams, GlobalTensor<MM12_OUT_T> &resGm,
                                           LocalTensor<MM12_OUT_T> l0cTensor, uint8_t l0cPingPongFlag);

    static constexpr bool HAS_ROPE = DLIT::hasRope;
    static constexpr DLILayout INPUT_LAYOUT = DLIT::inputQLayout;

    // m <> mte1 EventID
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;

    // mte2 <> mte1 EventID
    // L1 3buf, 使用3个eventId
    static constexpr uint32_t SYNC_MTE21_FLAG_Q = EVENT_ID0;
    static constexpr uint32_t SYNC_MTE21_FLAG_RELU_GRAD = EVENT_ID1;
    static constexpr uint32_t SYNC_MTE21_FLAG_KEY[2] = {EVENT_ID2, EVENT_ID3};
    static constexpr uint32_t SYNC_MTE21_FLAG_Q_INDEX[2] = {EVENT_ID4, EVENT_ID5};
    static constexpr uint32_t SYNC_MTE21_FLAG_KEY_INDEX[2] = {EVENT_ID6, EVENT_ID7};

    static constexpr uint32_t SYNC_MTE1MM_FLAG[2] = {EVENT_ID4, EVENT_ID5};
    static constexpr uint32_t SYNC_MMFIX_FLAG[2] = {EVENT_ID6, EVENT_ID7};

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    DLIGradKLLossConstInfo constInfo{};

    // GM
    GlobalTensor<int64_t> actualSeqLengthsQueryGm, actualSeqLengthsKeyGm;
    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<Q_T> qRopeGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<MM12_OUT_T> mm1ResGm[2];
    // mm2
    GlobalTensor<Q_T> queryIndexGm;
    GlobalTensor<KV_T> keyIndexGm;
    GlobalTensor<MM12_OUT_T> mm2ResGm[2];
    // mm5
    GlobalTensor<MM12_OUT_T> mm5ResGm;
    GlobalTensor<OUT_T> dKeyIndexGm;
    // deter 相关
    GlobalTensor<MM12_OUT_T> dKeyIndexDeterGmFloat;
    // mm6
    GlobalTensor<MM12_OUT_T> mm6ResGm;
    GlobalTensor<Q_T> reluGradRes[2];

    TBuf<TPosition::A1> queryBuf;
    TBuf<TPosition::A1> queryIndexBuf[2];
    TBuf<TPosition::A1> keyBuf[2];
    TBuf<TPosition::A1> reluGradOutBuf;
    TBuf<TPosition::A2> tmpBufL0A[2];
    TBuf<TPosition::B2> tmpBufL0B[2];
    TBuf<TPosition::CO1> tmpBufL0C[2];

    LocalTensor<Q_T> l1QueryTensor;
    LocalTensor<Q_T> l1QueryRopeTensor;
    LocalTensor<Q_T> l1QIndexTensor[2];
    LocalTensor<KV_T> l1KeyTensor[2];
    LocalTensor<KV_T> l1KeyRopeTensor[2];
    LocalTensor<Q_T> aL0TensorPingPong[2];
    LocalTensor<KV_T> bL0TensorPingPong[2];
    LocalTensor<MM12_OUT_T> cL0TensorPingPong[2];
    LocalTensor<Q_T> l1ReLuGradTensor;
};

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::InitParams(
    const DLIGradKLLossConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::InitMm1GlobalTensor(
    GlobalTensor<Q_T> &queryGm, GlobalTensor<KV_T> &keyGm, GlobalTensor<Q_T> &qRopeGm, GlobalTensor<KV_T> &keyRopeGm,
    GlobalTensor<int64_t> &actualSeqLengthsQueryGm, GlobalTensor<int64_t> &actualSeqLengthsKeyGm,
    GlobalTensor<MM12_OUT_T> &bmm1Res, GM_ADDR mm1Res)
{
    this->queryGm = queryGm;
    this->qRopeGm = qRopeGm;
    this->keyGm = keyGm;
    this->keyRopeGm = keyRopeGm;
    this->actualSeqLengthsQueryGm = actualSeqLengthsQueryGm;
    this->actualSeqLengthsKeyGm = actualSeqLengthsKeyGm;
    this->mm1ResGm[0] = bmm1Res;
    this->mm1ResGm[1] = bmm1Res[constInfo.n1Size * S1_BASE_STEP * S2_BASE_STEP];
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::InitMm2GlobalTensor(
    GlobalTensor<Q_T> &queryIndex, GlobalTensor<KV_T> &keyIndex, GlobalTensor<T> &mm2Res)
{
    queryIndexGm = queryIndex;
    keyIndexGm = keyIndex;
    mm2ResGm[0] = mm2Res;
    mm2ResGm[1] = mm2Res[constInfo.n1IndexSize * S1_BASE_STEP * S2_BASE_STEP];
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::InitMm5GlobalTensor(
    GlobalTensor<Q_T> &reluGradRes, GlobalTensor<MM12_OUT_T> &bmm5Res)
{
    this->reluGradRes[0] = reluGradRes;
    this->reluGradRes[1] = reluGradRes[constInfo.n1IndexSize * S1_BASE_STEP * S2_BASE_STEP];
    this->mm5ResGm = bmm5Res;
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::InitMm5DeterGlobalTensor(
    GlobalTensor<T> &dKeyIndexDeterGmFloat)
{
    this->dKeyIndexDeterGmFloat = dKeyIndexDeterGmFloat;
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::InitMm6GlobalTensor(GlobalTensor<T> &bmm6Res)
{
    this->mm6ResGm = bmm6Res;
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::InitBuffers(TPipe *pipe)
{
    // L1
    pipe->InitBuffer(queryBuf, 128 * 192 * sizeof(Q_T));         // 48k
    pipe->InitBuffer(keyBuf[0], 128 * 192 * sizeof(KV_T));       // 48k
    pipe->InitBuffer(keyBuf[1], 128 * 192 * sizeof(KV_T));       // 48k
    pipe->InitBuffer(queryIndexBuf[0], 128 * 128 * sizeof(Q_T)); // 32k
    pipe->InitBuffer(queryIndexBuf[1], 128 * 128 * sizeof(Q_T)); // 32k
    pipe->InitBuffer(reluGradOutBuf, 128 * 1024 * sizeof(KV_T)); // 256k

    l1QueryTensor = queryBuf.Get<Q_T>();
    l1QueryRopeTensor = l1QueryTensor[CUBE_BASE_BLOCK * constInfo.dSizeQuery];
    l1QIndexTensor[0] = queryIndexBuf[0].Get<Q_T>();
    l1QIndexTensor[1] = queryIndexBuf[1].Get<Q_T>();
    l1KeyTensor[0] = keyBuf[0].Get<KV_T>();
    l1KeyTensor[1] = keyBuf[1].Get<KV_T>();
    l1KeyRopeTensor[0] = l1KeyTensor[0][CUBE_BASE_BLOCK * constInfo.dSizeQuery];
    l1KeyRopeTensor[1] = l1KeyTensor[1][CUBE_BASE_BLOCK * constInfo.dSizeQuery];
    // l1KeyIndexTensor[0] = keyIndexBuf[0].Get<KV_T>();
    // l1KeyIndexTensor[1] = keyIndexBuf[1].Get<KV_T>();
    l1ReLuGradTensor = reluGradOutBuf.Get<Q_T>();

    // L0A/B/C
    pipe->InitBuffer(tmpBufL0A[0], L0A_PP_SIZE); // 32K
    pipe->InitBuffer(tmpBufL0A[1], L0A_PP_SIZE); // 32K
    pipe->InitBuffer(tmpBufL0B[0], L0B_PP_SIZE); // 32K
    pipe->InitBuffer(tmpBufL0B[1], L0B_PP_SIZE); // 32K
    pipe->InitBuffer(tmpBufL0C[0], L0C_PP_SIZE); // 64K
    pipe->InitBuffer(tmpBufL0C[1], L0C_PP_SIZE); // 64K

    aL0TensorPingPong[0] = tmpBufL0A[0].Get<KV_T>();
    aL0TensorPingPong[1] = tmpBufL0A[1].Get<KV_T>();
    bL0TensorPingPong[0] = tmpBufL0B[0].Get<KV_T>();
    bL0TensorPingPong[1] = tmpBufL0B[1].Get<KV_T>();
    cL0TensorPingPong[0] = tmpBufL0C[0].Get<MM12_OUT_T>();
    cL0TensorPingPong[1] = tmpBufL0C[1].Get<MM12_OUT_T>();
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_RELU_GRAD);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[0]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[1]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q_INDEX[0]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q_INDEX[1]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY_INDEX[0]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY_INDEX[1]);

    SetFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[0]);
    SetFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[1]);
    SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[0]);
    SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[1]);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_RELU_GRAD);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[0]);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[1]);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q_INDEX[0]);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q_INDEX[1]);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY_INDEX[0]);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY_INDEX[1]);

    WaitFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[0]);
    WaitFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[1]);
    WaitFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[0]);
    WaitFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[1]);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::CopyGmToL1(LocalTensor<KV_T> &l1Tensor,
                                                                                    GlobalTensor<KV_T> &gmSrcTensor,
                                                                                    uint32_t srcN, uint32_t srcD,
                                                                                    uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::CopyInMm1AToL1(
    const DLIGradKLLossRunInfo &runInfo, uint32_t n1Idx)
{
    uint32_t headOffset = n1Idx * constInfo.dSizeQuery;
    auto querySrcGm = queryGm[runInfo.queryTensorOffset + headOffset];
    CopyGmToL1(l1QueryTensor, querySrcGm, runInfo.curS1Size, constInfo.dSizeQuery,
               constInfo.dSizeQuery * constInfo.n1Size);

    if constexpr (HAS_ROPE) {
        headOffset = n1Idx * constInfo.dSizeQueryRope;
        auto queryRopeSrcGm = qRopeGm[runInfo.queryRopeTensorOffset + headOffset];
        CopyGmToL1(l1QueryRopeTensor, queryRopeSrcGm, runInfo.curS1Size, constInfo.dSizeQueryRope,
                   constInfo.dSizeQueryRope * constInfo.n1Size);
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::CopyInMm1BToL1(
    const DLIGradKLLossRunInfo &runInfo, uint32_t n2Idx, uint32_t s2InnerIdx, uint32_t actualS2Len,
    uint8_t pingpongFlag)
{
    uint32_t keyGmOffset = runInfo.keyTensorOffset +
                           s2InnerIdx * constInfo.s2BaseBlk * constInfo.n2Size * constInfo.dSizeQuery +
                           n2Idx * constInfo.dSizeQuery;
    auto keySrcGm = keyGm[keyGmOffset];
    CopyGmToL1(l1KeyTensor[pingpongFlag], keySrcGm, actualS2Len, constInfo.dSizeQuery,
               constInfo.dSizeQuery * constInfo.n2Size);

    if constexpr (HAS_ROPE) {
        keyGmOffset = runInfo.keyRopeTensorOffset +
                      s2InnerIdx * constInfo.s2BaseBlk * constInfo.n2Size * constInfo.dSizeQueryRope +
                      n2Idx * constInfo.dSizeQueryRope;
        keySrcGm = keyRopeGm[keyGmOffset];
        CopyGmToL1(l1KeyRopeTensor[pingpongFlag], keySrcGm, actualS2Len, constInfo.dSizeQueryRope,
                   constInfo.dSizeQueryRope * constInfo.n2Size);
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::CopyInMm2AToL1(
    const DLIGradKLLossRunInfo &runInfo, uint32_t n1IndexId, uint32_t pingpongFlag)
{
    uint32_t headOffset = n1IndexId * constInfo.dSizeQueryIndex;
    auto querySrcGm = queryIndexGm[runInfo.queryIndexTensorOffset + headOffset];
    CopyGmToL1(l1QIndexTensor[pingpongFlag], querySrcGm, runInfo.curS1Size, constInfo.dSizeQueryIndex,
               constInfo.dSizeQueryIndex * constInfo.n1IndexSize);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::CopyInMm2BToL1(
    const DLIGradKLLossRunInfo &runInfo, uint32_t n2IndexId, uint32_t s2InnerIdx, uint32_t actualS2Len,
    uint8_t pingpongFlag)
{
    uint32_t keyGmOffset = runInfo.keyIndexTensorOffset +
                           s2InnerIdx * constInfo.s2BaseBlk * constInfo.n2IndexSize * constInfo.dSizeQueryIndex +
                           n2IndexId * constInfo.dSizeQueryIndex;
    auto keySrcGm = keyIndexGm[keyGmOffset];
    CopyGmToL1(l1KeyTensor[pingpongFlag], keySrcGm, actualS2Len, constInfo.dSizeQueryIndex,
               constInfo.dSizeQueryIndex * constInfo.n2IndexSize);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::CopyInReluGradToL1A(
    const DLIGradKLLossRunInfo &runInfo, uint32_t n1IndexId)
{
    uint32_t reluGradGmOffset = n1IndexId * runInfo.curS1Size * runInfo.curS2StepSize;
    auto reluGradWs = reluGradRes[runInfo.taskIdMod2][reluGradGmOffset];

    CopyGmToL1(l1ReLuGradTensor, reluGradWs, runInfo.curS1Size, runInfo.curS2StepSize, runInfo.curS2StepSize);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::LoadDataMmA(LocalTensor<Q_T> &aL0Tensor,
                                                                                     LocalTensor<Q_T> &aL1Tensor,
                                                                                     struct MMParam &mmParam)
{
    uint32_t alignM = AlignTo(mmParam.singleM, static_cast<uint32_t>(C0_SIZE));
    uint32_t alignK = AlignTo(mmParam.singleK, static_cast<uint32_t>(C0_SIZE));

    LoadData2DParamsV2 loadData2DParams;
    loadData2DParams.mStartPosition = 0;
    loadData2DParams.kStartPosition = 0;
    loadData2DParams.mStep = alignM / C0_SIZE;
    loadData2DParams.kStep = alignK / C0_SIZE;
    loadData2DParams.srcStride = alignM / C0_SIZE;
    loadData2DParams.dstStride = alignM / C0_SIZE;
    loadData2DParams.ifTranspose = false;
    LoadData(aL0Tensor, aL1Tensor, loadData2DParams);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::LoadDataMmAWithTranspose(
    LocalTensor<Q_T> &aL0Tensor, LocalTensor<Q_T> &aL1Tensor, struct MMParam &mmParam)
{
    uint32_t alignM = AlignTo(mmParam.singleM, static_cast<uint32_t>(C0_SIZE));
    uint32_t alignK = AlignTo(mmParam.singleK, static_cast<uint32_t>(C0_SIZE));

    LoadData2DParamsV2 loadData2DParams;
    loadData2DParams.mStartPosition = 0;
    loadData2DParams.kStartPosition = 0;
    loadData2DParams.mStep = alignK / C0_SIZE;
    loadData2DParams.kStep = alignM / C0_SIZE;
    loadData2DParams.srcStride = alignK / C0_SIZE;
    loadData2DParams.dstStride = alignM / C0_SIZE;
    loadData2DParams.ifTranspose = true;
    LoadData(aL0Tensor, aL1Tensor, loadData2DParams);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::LoadDataMmB(LocalTensor<KV_T> &l0Tensor,
                                                                                     LocalTensor<KV_T> &l1Tensor,
                                                                                     struct MMParam &mmParam)
{
    uint32_t alignK = AlignTo(mmParam.singleK, static_cast<uint32_t>(C0_SIZE));
    uint32_t alignN = AlignTo(mmParam.singleN, static_cast<uint32_t>(C0_SIZE));

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = alignK * alignN / CUBE_MATRIX_SIZE;
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;

    LoadData(l0Tensor, l1Tensor, loadData2DParams);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::LoadDataMmBWithTranspose(
    LocalTensor<KV_T> &l0Tensor, LocalTensor<KV_T> &l1Tensor, struct MMParam &mmParam)
{
    uint32_t alignK = AlignTo(mmParam.singleK, static_cast<uint32_t>(C0_SIZE));
    uint32_t alignN = AlignTo(mmParam.singleN, static_cast<uint32_t>(C0_SIZE));

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = alignN / C0_SIZE;
    loadData2DParams.srcStride = alignK / C0_SIZE;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = true;

    for (int32_t i = 0; i < alignK / C0_SIZE; i++) {
        LoadData(l0Tensor[i * alignN * C0_SIZE], l1Tensor[i * CUBE_MATRIX_SIZE], loadData2DParams);
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::MmadInner(
    struct MMParam &mmParams, LocalTensor<Q_T> &l1QTensor, LocalTensor<KV_T> &l1KTensor,
    LocalTensor<MM12_OUT_T> l0cTensor, uint8_t l1PingPongFlag, uint8_t l0abPingPongFlag, uint8_t l0cPingPongFlag)
{
    MmadParams mmadParams;
    mmadParams.m = mmParams.singleM == 1 ? 16 : mmParams.singleM;
    mmadParams.n = mmParams.singleN;
    mmadParams.k = mmParams.singleK;
    mmadParams.cmatrixInitVal = mmParams.isL0CInit;

    // l0
    LocalTensor<Q_T> l0aTensor = aL0TensorPingPong[l0abPingPongFlag & 1];
    LocalTensor<KV_T> l0bTensor = bL0TensorPingPong[l0abPingPongFlag & 1];
    if (mmParams.isRightReuse) {
        l0bTensor = bL0TensorPingPong[l1PingPongFlag & 1];
    }

    WaitFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[l0abPingPongFlag & 1]); // Wait before the next copy
    if (mmParams.isLeftTranspose) {
        LoadDataMmAWithTranspose(l0aTensor, l1QTensor, mmParams);
    } else {
        LoadDataMmA(l0aTensor, l1QTensor, mmParams);
    }
    if (mmParams.needCopyRight) {
        if (mmParams.isRightTranspose) {
            LoadDataMmBWithTranspose(l0bTensor, l1KTensor, mmParams);
        } else {
            LoadDataMmB(l0bTensor, l1KTensor, mmParams);
        }
    }
    SetFlag<AscendC::HardEvent::MTE1_M>(SYNC_MTE1MM_FLAG[l0abPingPongFlag & 1]);  // L0A/L0B copy complete
    WaitFlag<AscendC::HardEvent::MTE1_M>(SYNC_MTE1MM_FLAG[l0abPingPongFlag & 1]); // Wait for MTE1
    WaitFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);    // Wait for FixPipe

    Mmad(l0cTensor, l0aTensor, l0bTensor, mmadParams);
    if (mmParams.needAccumL0C) {
        PipeBarrier<PIPE_M>();
    }
    SetFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[l0abPingPongFlag & 1]);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::FixCopyOut(struct MMParam &mmParams,
                                                                                    GlobalTensor<MM12_OUT_T> &resGm,
                                                                                    LocalTensor<MM12_OUT_T> l0cTensor,
                                                                                    uint8_t l0cPingPongFlag)
{
    if (mmParams.isFixOut) {
        SetFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
        WaitFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = mmParams.singleN;
        fixpipeParams.mSize = mmParams.singleM;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = mmParams.dstFixStride;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        fixpipeParams.reluEn = mmParams.isFixRelu;

        Fixpipe<MM12_OUT_T, MM12_OUT_T>(resGm, l0cTensor, fixpipeParams); // 将matmul结果从L0C搬运到UB
    }
    SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[l0cPingPongFlag & 1]);
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::FixOutAtomicAdd(
    struct MMParam &mmParams, GlobalTensor<MM12_OUT_T> &resGm, LocalTensor<MM12_OUT_T> l0cTensor,
    uint8_t l0cPingPongFlag)
{
    if (mmParams.atomicFlag) {
        if constexpr (DLIT::deterministic) {
            // A5 requires the DMA atomic state to be explicitly reset before
            // enabling an atomic Fixpipe write.
            AscendC::DisableDmaAtomic();
        }
        AscendC::SetAtomicAdd<MM12_OUT_T>();
    }

    FixCopyOut(mmParams, resGm, l0cTensor, l0cPingPongFlag);

    if (mmParams.atomicFlag) {
        if constexpr (DLIT::deterministic) {
            // Fixpipe is asynchronous. Keep atomic mode enabled until the
            // atomic Fixpipe write has completed.
            AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::DisableDmaAtomic();
        } else {
            SetAtomicNone();
        }
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::ComputeMm1(const DLIGradKLLossRunInfo &info)
{
    uint8_t l1PingPongFlag = 0;
    uint8_t l0abPingPongFlag = 0;
    uint8_t l0cPingPongFlag = 0;

    MMParam mmParams;
    mmParams.singleM = info.curS1Size;
    mmParams.dstFixStride = info.curS2StepSize;
    mmParams.needAccumL0C = constInfo.dLoopTimesCube > 1;

    for (uint32_t qHeadIdx = 0; qHeadIdx < this->constInfo.n1Size; qHeadIdx++) {
        uint32_t kvHeadIdx = qHeadIdx / constInfo.gSizeQuery;
        uint32_t dstHeadOffset = qHeadIdx * info.curS1Size * info.curS2StepSize;
        WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q);
        CopyInMm1AToL1(info, qHeadIdx);

        for (uint32_t s2InnerIdx = 0; s2InnerIdx < info.curS2LoopTimes; s2InnerIdx++) {
            mmParams.dstFixOffset = dstHeadOffset + s2InnerIdx * constInfo.s2BaseBlk;
            uint32_t curS2BlkSize = (s2InnerIdx == info.curS2LoopTimes - 1) ?
                                        (info.curS2StepSize - s2InnerIdx * constInfo.s2BaseBlk) :
                                        constInfo.s2BaseBlk;
            mmParams.singleN = curS2BlkSize;

            WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[l1PingPongFlag & 1]);
            CopyInMm1BToL1(info, kvHeadIdx, s2InnerIdx, curS2BlkSize, l1PingPongFlag);

            SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_KEY[l1PingPongFlag & 1]);
            WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_KEY[l1PingPongFlag & 1]);

            for (uint32_t kIdx = 0; kIdx < constInfo.dLoopTimesCube; kIdx++) {
                mmParams.singleK = (kIdx == constInfo.dLoopTimesCube - 1) ?
                                       (constInfo.dSizeActual - kIdx * CUBE_BASE_BLOCK) :
                                       CUBE_BASE_BLOCK;
                mmParams.isFixOut = (kIdx == constInfo.dLoopTimesCube - 1);
                mmParams.isL0CInit = (kIdx == 0);

                auto mm1DstGm = mm1ResGm[info.taskIdMod2][mmParams.dstFixOffset];
                auto l1SrcQTensor = l1QueryTensor[kIdx * CUBE_BASE_BLOCK * CUBE_BASE_BLOCK];
                auto l1SrcKTensor = l1KeyTensor[l1PingPongFlag & 1][kIdx * CUBE_BASE_BLOCK * CUBE_BASE_BLOCK];
                LocalTensor<MM12_OUT_T> l0cTensor = cL0TensorPingPong[l0cPingPongFlag & 1];
                MmadInner(mmParams, l1SrcQTensor, l1SrcKTensor, l0cTensor, l1PingPongFlag, l0abPingPongFlag,
                          l0cPingPongFlag);
                FixCopyOut(mmParams, mm1DstGm, l0cTensor, l0cPingPongFlag);
                l0abPingPongFlag = (l0abPingPongFlag + 1) & 1;
            }

            // L0B copy complete; allow MTE2 to refill L1 for the next iteration.
            SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[l1PingPongFlag & 1]);
            l0cPingPongFlag = (l0cPingPongFlag + 1) & 1;
            l1PingPongFlag = (l1PingPongFlag + 1) & 1;
        }
        SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q);
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::ComputeMm2(const DLIGradKLLossRunInfo &info)
{
    uint8_t l1PingPongFlag = 0;
    uint8_t l0PingPongFlag = 0;

    MMParam mmParams;
    mmParams.singleM = info.curS1Size;
    mmParams.singleK = constInfo.dSizeQueryIndex;
    mmParams.dstFixStride = info.curS2StepSize;
    mmParams.isFixOut = true;
    mmParams.isFixRelu = true;
    mmParams.isL0CInit = true;

    for (uint32_t n1IndexId = 0; n1IndexId < this->constInfo.n1IndexSize; n1IndexId++) {
        uint32_t n2IndexId = n1IndexId / constInfo.gSizeQueryIndex;
        uint32_t dstHeadOffset = n1IndexId * info.curS1Size * info.curS2StepSize;
        WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q_INDEX[l1PingPongFlag & 1]);
        CopyInMm2AToL1(info, n1IndexId, l1PingPongFlag);

        if (info.curS2LoopTimes > 0) {
            uint32_t curS2BlkSize = (info.curS2LoopTimes == 1) ? info.curS2StepSize : constInfo.s2BaseBlk;
            WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[l0PingPongFlag & 1]);
            CopyInMm2BToL1(info, n2IndexId, 0, curS2BlkSize, l0PingPongFlag);
            SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_KEY_INDEX[l0PingPongFlag & 1]);
        }

        for (uint32_t s2InnerIdx = 0; s2InnerIdx < info.curS2LoopTimes; s2InnerIdx++) {
            mmParams.dstFixOffset = dstHeadOffset + s2InnerIdx * constInfo.s2BaseBlk;
            uint32_t curS2BlkSize = (s2InnerIdx == info.curS2LoopTimes - 1) ?
                                        (info.curS2StepSize - s2InnerIdx * constInfo.s2BaseBlk) :
                                        constInfo.s2BaseBlk;
            mmParams.singleN = curS2BlkSize;

            uint8_t nextL0PingPongFlag = (l0PingPongFlag + 1) & 1;
            if (s2InnerIdx + 1 < info.curS2LoopTimes) {
                uint32_t nextS2InnerIdx = s2InnerIdx + 1;
                uint32_t nextS2BlkSize = (nextS2InnerIdx == info.curS2LoopTimes - 1) ?
                                             (info.curS2StepSize - nextS2InnerIdx * constInfo.s2BaseBlk) :
                                             constInfo.s2BaseBlk;
                WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[nextL0PingPongFlag]);
                CopyInMm2BToL1(info, n2IndexId, nextS2InnerIdx, nextS2BlkSize, nextL0PingPongFlag);
                SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_KEY_INDEX[nextL0PingPongFlag]);
            }

            WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_KEY_INDEX[l0PingPongFlag & 1]);

            auto mm2DstGm = mm2ResGm[info.taskIdMod2][mmParams.dstFixOffset];
            auto l1SrcQTensor = l1QIndexTensor[l1PingPongFlag & 1];
            auto l1SrcKTensor = l1KeyTensor[l0PingPongFlag & 1];
            LocalTensor<MM12_OUT_T> l0cTensor = cL0TensorPingPong[l0PingPongFlag & 1];
            MmadInner(mmParams, l1SrcQTensor, l1SrcKTensor, l0cTensor, l1PingPongFlag, l0PingPongFlag, l0PingPongFlag);
            SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[l0PingPongFlag & 1]);
            FixCopyOut(mmParams, mm2DstGm, l0cTensor, l0PingPongFlag);
            l0PingPongFlag = (l0PingPongFlag + 1) & 1;
        }
        SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q_INDEX[l1PingPongFlag & 1]);
        l1PingPongFlag = (l1PingPongFlag + 1) & 1;
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::ComputeMm34(const DLIGradKLLossRunInfo &info)
{
    uint8_t l1PingPongFlag = 0;
    uint8_t l0abPingPongFlag = 0;
    uint8_t l0cPingPongFlag = 0;

    MMParam mm3Params;
    mm3Params.singleK = info.curS1Size;
    mm3Params.singleN = constInfo.dSizeQueryIndex;
    mm3Params.dstFixStride = constInfo.dSizeQueryIndex;
    mm3Params.isFixOut = true;
    mm3Params.isL0CInit = true;
    mm3Params.isLeftTranspose = true;
    mm3Params.isRightTranspose = true; // NZ -> ZN
    // mm3Params.isRightReuse = true;

    MMParam mm4Params;
    mm4Params.singleM = info.curS1Size;
    mm4Params.singleN = constInfo.dSizeQueryIndex;
    mm4Params.dstFixStride = constInfo.dSizeQueryIndex;
    mm4Params.isLeftTranspose = false;
    mm4Params.isRightTranspose = true;
    mm4Params.needAccumL0C = true;

    for (uint32_t n1IndexId = 0; n1IndexId < this->constInfo.n1IndexSize; n1IndexId++) {
        uint32_t n2IndexId = n1IndexId / constInfo.gSizeQueryIndex;
        // 一个head拷贝一次, mm3和mm4复用
        WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_RELU_GRAD);
        CopyInReluGradToL1A(info, n1IndexId);

        // matmul 3
        WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q_INDEX[l1PingPongFlag & 1]);
        CopyInMm2AToL1(info, n1IndexId, l1PingPongFlag);
        SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_Q_INDEX[l1PingPongFlag & 1]);
        WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_Q_INDEX[l1PingPongFlag & 1]);

        for (uint32_t s2InnerIdx = 0; s2InnerIdx < info.curS2LoopTimes; s2InnerIdx++) {
            uint32_t curS2BlkSize = (s2InnerIdx == info.curS2LoopTimes - 1) ?
                                        (info.curS2StepSize - s2InnerIdx * constInfo.s2BaseBlk) :
                                        constInfo.s2BaseBlk;
            mm3Params.singleM = curS2BlkSize;
            // mm3Params.needCopyRight = (s2InnerIdx == 0);

            uint32_t l1ReluGradOffset = s2InnerIdx * constInfo.s2BaseBlk * info.curS1SizeAlign16;
            auto l1aTensor = l1ReLuGradTensor[l1ReluGradOffset];
            auto l1bTensor = l1QIndexTensor[l1PingPongFlag & 1];
            LocalTensor<MM12_OUT_T> l0cTensor = cL0TensorPingPong[l0abPingPongFlag & 1];

            uint32_t dKeyGmOffset =
                info.keyIndexTensorOffset +
                s2InnerIdx * constInfo.s2BaseBlk * constInfo.n2IndexSize * constInfo.dSizeQueryIndex +
                n2IndexId * constInfo.dSizeQueryIndex;
            auto dstGm = mm5ResGm[dKeyGmOffset];
            MmadInner(mm3Params, l1aTensor, l1bTensor, l0cTensor, l1PingPongFlag, l0abPingPongFlag, l0abPingPongFlag);
            FixOutAtomicAdd(mm3Params, dstGm, l0cTensor, l0abPingPongFlag);

            l0abPingPongFlag = (l0abPingPongFlag + 1) & 1;
        }
        SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q_INDEX[l1PingPongFlag & 1]);
        // 这个必须加, 因为mm3是L0B reuse，使用l1PingPongFlag去控制L0B的，而接下来的mm4是使用l0abPingPongFlag来控制的
        l0abPingPongFlag = l1PingPongFlag;

        // matmul 4
        uint32_t dQueryGmOffset = n1IndexId * info.curS1Size * constInfo.dSizeQueryIndex;
        LocalTensor<MM12_OUT_T> mm4L0cTensor = cL0TensorPingPong[l0cPingPongFlag & 1];

        for (uint32_t s2InnerIdx = 0; s2InnerIdx < info.curS2LoopTimes; s2InnerIdx++) {
            uint32_t curS2BlkSize = (s2InnerIdx == info.curS2LoopTimes - 1) ?
                                        (info.curS2StepSize - s2InnerIdx * constInfo.s2BaseBlk) :
                                        constInfo.s2BaseBlk;
            mm4Params.singleK = curS2BlkSize;
            mm4Params.isFixOut = (s2InnerIdx == info.curS2LoopTimes - 1);
            mm4Params.isL0CInit = (s2InnerIdx == 0);
            WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[l0abPingPongFlag & 1]);
            CopyInMm2BToL1(info, n2IndexId, s2InnerIdx, curS2BlkSize, l0abPingPongFlag);

            SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_KEY_INDEX[l0abPingPongFlag & 1]);
            WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_KEY_INDEX[l0abPingPongFlag & 1]);

            uint32_t l1ReluGradOffset = s2InnerIdx * constInfo.s2BaseBlk * info.curS1SizeAlign16;
            auto l1aTensor = l1ReLuGradTensor[l1ReluGradOffset];
            auto l1bTensor = l1KeyTensor[l0abPingPongFlag & 1];

            auto dstGm = mm6ResGm[dQueryGmOffset];
            MmadInner(mm4Params, l1aTensor, l1bTensor, mm4L0cTensor, l1PingPongFlag, l0abPingPongFlag, l0cPingPongFlag);
            SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[l0abPingPongFlag & 1]);
            mm4Params.atomicFlag = mm4Params.isFixOut && (info.s2Idx > 0);
            FixOutAtomicAdd(mm4Params, dstGm, mm4L0cTensor, l0cPingPongFlag);

            l0abPingPongFlag = (l0abPingPongFlag + 1) & 1;
        }
        SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_RELU_GRAD);
        l0cPingPongFlag = (l0cPingPongFlag + 1) & 1;
        // 和mm3结束赋值l0PingPongFlag的理由一样
        l0abPingPongFlag = (l0abPingPongFlag + 1) & 1;
        l1PingPongFlag = l0abPingPongFlag;
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::ComputeMm4(const DLIGradKLLossRunInfo &info)
{}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossServiceCube<DLIT>::ComputeMm34Deter(
    const DLIGradKLLossRunInfo &info)
{
    uint8_t l1PingPongFlag = 0;
    uint8_t l0abPingPongFlag = 0;
    uint8_t l0cPingPongFlag = 0;

    MMParam mm3Params;
    mm3Params.singleK = info.curS1Size;
    mm3Params.singleN = constInfo.dSizeQueryIndex;
    mm3Params.dstFixStride = constInfo.dSizeQueryIndex;
    mm3Params.isFixOut = true;
    mm3Params.isL0CInit = true;
    mm3Params.isLeftTranspose = true;
    mm3Params.isRightTranspose = true; // NZ -> ZN

    MMParam mm4Params;
    mm4Params.singleM = info.curS1Size;
    mm4Params.singleN = constInfo.dSizeQueryIndex;
    mm4Params.dstFixStride = constInfo.dSizeQueryIndex;
    mm4Params.isLeftTranspose = false;
    mm4Params.isRightTranspose = true;
    mm4Params.needAccumL0C = true;

    for (uint32_t n1IndexId = 0; n1IndexId < this->constInfo.n1IndexSize; n1IndexId++) {
        uint32_t n2IndexId = n1IndexId / constInfo.gSizeQueryIndex;
        // 一个head拷贝一次, mm3和mm4复用
        WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_RELU_GRAD);
        CopyInReluGradToL1A(info, n1IndexId);

        // matmul 3
        WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q_INDEX[l1PingPongFlag & 1]);
        CopyInMm2AToL1(info, n1IndexId, l1PingPongFlag);
        SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_Q_INDEX[l1PingPongFlag & 1]);
        WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_Q_INDEX[l1PingPongFlag & 1]);

        // GM初始清零，不累加通过atomicFlag控制
        mm3Params.atomicFlag = (n1IndexId != 0);
        for (uint32_t s2InnerIdx = 0; s2InnerIdx < info.curS2LoopTimes; s2InnerIdx++) {
            uint32_t curS2BlkSize = (s2InnerIdx == info.curS2LoopTimes - 1) ?
                                        (info.curS2StepSize - s2InnerIdx * constInfo.s2BaseBlk) :
                                        constInfo.s2BaseBlk;
            mm3Params.singleM = curS2BlkSize;
            uint32_t l1ReluGradOffset = s2InnerIdx * constInfo.s2BaseBlk * info.curS1SizeAlign16;
            auto l1aTensor = l1ReLuGradTensor[l1ReluGradOffset];
            auto l1bTensor = l1QIndexTensor[l1PingPongFlag & 1];
            LocalTensor<MM12_OUT_T> l0cTensor = cL0TensorPingPong[l0abPingPongFlag & 1];

            uint32_t dKeyGmOffset =
                constInfo.dKeyDeterGmOffset +
                s2InnerIdx * constInfo.s2BaseBlk * constInfo.n2IndexSize * constInfo.dSizeQueryIndex +
                n2IndexId * constInfo.dSizeQueryIndex;
            auto dstGm = dKeyIndexDeterGmFloat[dKeyGmOffset];
            MmadInner(mm3Params, l1aTensor, l1bTensor, l0cTensor, l1PingPongFlag, l0abPingPongFlag, l0abPingPongFlag);
            FixOutAtomicAdd(mm3Params, dstGm, l0cTensor, l0abPingPongFlag);
            l0abPingPongFlag = (l0abPingPongFlag + 1) & 1;
        }
        SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q_INDEX[l1PingPongFlag & 1]);
        // 这个必须加, 因为mm3是L0B reuse，使用l1PingPongFlag去控制L0B的，而接下来的mm4是使用l0abPingPongFlag来控制的
        l0abPingPongFlag = l1PingPongFlag;

        // matmul 4
        uint32_t dQueryGmOffset = n1IndexId * info.curS1Size * constInfo.dSizeQueryIndex;
        LocalTensor<MM12_OUT_T> mm4L0cTensor = cL0TensorPingPong[l0cPingPongFlag & 1];

        for (uint32_t s2InnerIdx = 0; s2InnerIdx < info.curS2LoopTimes; s2InnerIdx++) {
            uint32_t curS2BlkSize = (s2InnerIdx == info.curS2LoopTimes - 1) ?
                                        (info.curS2StepSize - s2InnerIdx * constInfo.s2BaseBlk) :
                                        constInfo.s2BaseBlk;
            mm4Params.singleK = curS2BlkSize;
            mm4Params.isFixOut = (s2InnerIdx == info.curS2LoopTimes - 1);
            mm4Params.isL0CInit = (s2InnerIdx == 0);
            WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[l0abPingPongFlag & 1]);
            CopyInMm2BToL1(info, n2IndexId, s2InnerIdx, curS2BlkSize, l0abPingPongFlag);

            SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_KEY_INDEX[l0abPingPongFlag & 1]);
            WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_KEY_INDEX[l0abPingPongFlag & 1]);

            uint32_t l1ReluGradOffset = s2InnerIdx * constInfo.s2BaseBlk * info.curS1SizeAlign16;
            auto l1aTensor = l1ReLuGradTensor[l1ReluGradOffset];
            auto l1bTensor = l1KeyTensor[l0abPingPongFlag & 1];

            auto dstGm = mm6ResGm[dQueryGmOffset];
            MmadInner(mm4Params, l1aTensor, l1bTensor, mm4L0cTensor, l1PingPongFlag, l0abPingPongFlag, l0cPingPongFlag);
            SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_KEY[l0abPingPongFlag & 1]);
            mm4Params.atomicFlag = mm4Params.isFixOut && (info.s2Idx > 0);
            FixOutAtomicAdd(mm4Params, dstGm, mm4L0cTensor, l0cPingPongFlag);

            l0abPingPongFlag = (l0abPingPongFlag + 1) & 1;
        }
        SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_RELU_GRAD);
        l0cPingPongFlag = (l0cPingPongFlag + 1) & 1;
        // 和mm3结束赋值l0PingPongFlag的理由一样
        l0abPingPongFlag = (l0abPingPongFlag + 1) & 1;
        l1PingPongFlag = l0abPingPongFlag;
    }
    AscendC::PipeBarrier<PIPE_FIX>();
}

#endif // DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_SERVICE_CUBE_ARCH35_H
