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
 * \file moe_gating_top_k_e_k_fullload_regbase.h
 * \brief Arch35 register-based kernel for MoE Gating TopK DeepSeekV3 high-performance scenario.
 */
#ifndef MOE_GATING_TOP_K_E_K_FULLLOAD_REGBASE_H
#define MOE_GATING_TOP_K_E_K_FULLLOAD_REGBASE_H

#include <cmath>
#include "common.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"

namespace MoeGatingTopK {
using namespace AscendC;
using MicroAPI::RegTensor;

constexpr MicroAPI::DivSpecificMode EK_DIV_MODE = {MicroAPI::MaskMergeMode::ZEROING, true};

template <typename T>
class MoeGatingTopKEKFullloadRegbase {
public:
    __aicore__ inline MoeGatingTopKEKFullloadRegbase(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKRegbaseTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBias();
    __aicore__ inline void CopyInX(int64_t progress);
    __aicore__ inline void ComputeX();
    __aicore__ inline void SortInGroup();
    __aicore__ inline void SelectTopKGroupIndex();
    __aicore__ inline void SelectTopKInGroup(LocalTensor<float> sortedInGroupTensor,
                                             LocalTensor<float> top2InGroupTensor, uint16_t groupCount0,
                                             uint32_t perGroupExpertCountAlign0, uint32_t padNegInfNum);
    __aicore__ inline void SelectTopKAfterSort(LocalTensor<float> sortedGroupTensor,
                                               LocalTensor<float> top2InGroupTensor, uint32_t size, int32_t kGroup0,
                                               LocalTensor<float> tmpLocal);
    __aicore__ inline void FinalSortByKGroup();
    __aicore__ inline void SelectTopKExpertScore();
    __aicore__ inline void CopyOut(int64_t progress);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;

    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<QuePosition::VECCALC> xBiasBuf_;
    TBuf<QuePosition::VECCALC> xSigmoidBuf_;
    TBuf<QuePosition::VECCALC> groupBuf_;
    TBuf<QuePosition::VECCALC> sortedInGroupBuf_;
    TBuf<QuePosition::VECCALC> sortedGroupBuf_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    TBuf<TPosition::VECCALC> finalSortBuffer_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;

    LocalTensor<uint32_t> indexTensor;
    LocalTensor<float> sortedInGroupTensor;
    LocalTensor<float> sortedGroupTensor;
    LocalTensor<float> mrgSortTensor;

    int64_t blockIdx_;
    int64_t curCoreRowCount_;
    int64_t expertCount_;
    int64_t k_;
    int64_t kGroup_;
    int64_t groupCount_;
    float routedScalingFactor_;
    float eps_;

    int64_t perGroupExpertCount_;
    int64_t perGroupExpertCountAlign_;
    const MoeGatingTopKRegbaseTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::CopyInBias()
{
    LocalTensor<T> biasTensor = biasBuf_.Get<T>();

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = groupCount_;
    dataCopyParams.blockLen = perGroupExpertCount_ * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (perGroupExpertCountAlign_ - perGroupExpertCount_) * sizeof(T) / BLOCK_BYTES;
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};

    DataCopyPad(biasTensor, biasGm_, dataCopyParams, dataCopyPadParams);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::CopyInX(int64_t row)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = groupCount_;
    dataCopyParams.blockLen = perGroupExpertCount_ * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (perGroupExpertCountAlign_ - perGroupExpertCount_) * sizeof(T) / BLOCK_BYTES;
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    LocalTensor<T> xInLocalTensor = xInQueue_.AllocTensor<T>();
    DataCopyPad(xInLocalTensor, xGm_[row * expertCount_], dataCopyParams, dataCopyPadParams);
    xInQueue_.EnQue(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::ComputeX()
{
    LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.Get<float>();
    LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
    indexTensor = indexBuffer_.Get<uint32_t>();

    LocalTensor<T> xInLocalTensor = xInQueue_.DeQue<T>();

    uint32_t size = perGroupExpertCountAlign_ * groupCount_;
    uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, VL_FLOAT_SIZE));
    LocalTensor<T> biasTensor = biasBuf_.Get<T>();

    __ubuf__ T *inputAddr = (__ubuf__ T *)xInLocalTensor.GetPhyAddr();
    __ubuf__ float *sigmoidOutAddr = (__ubuf__ float *)xSigmoidTensor.GetPhyAddr();
    __ubuf__ int32_t *indexOutAddr = (__ubuf__ int32_t *)indexTensor.GetPhyAddr();
    __ubuf__ float *addBiasOutAddr = (__ubuf__ float *)xBiasTensor.GetPhyAddr();
    __ubuf__ T *biasAddr = (__ubuf__ T *)biasTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> vregBiasFp32;
        RegTensor<int32_t> vregIndex;
        RegTensor<float> vregSigmoidResult;
        RegTensor<float> vregBiasResult;
        RegTensor<float> vregOne;
        RegTensor<float> vregInFp32;
        RegTensor<float> vreg1;
        RegTensor<float> vreg2;
        RegTensor<float> vreg3;
        MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
        MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::ZEROING, float>(vregOne, static_cast<float>(1), preg0);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = MicroAPI::UpdateMask<float>(size);
            ops::LoadTwoTensorForDtypeT<T>(inputAddr, biasAddr, vregInFp32, vregBiasFp32, preg0, preg0,
                                           i * VL_FLOAT_SIZE, i * VL_FLOAT_SIZE);
            MicroAPI::Muls(vreg1, vregInFp32, static_cast<float>(-1), preg0);
            MicroAPI::Exp(vreg2, vreg1, preg0);
            MicroAPI::Adds(vreg3, vreg2, static_cast<float>(1), preg0);
            MicroAPI::Div<float, &EK_DIV_MODE>(vregSigmoidResult, vregOne, vreg3, preg0);
            MicroAPI::Add(vregBiasResult, vregSigmoidResult, vregBiasFp32, preg0);
            MicroAPI::Arange(vregIndex, static_cast<int32_t>(i * VL_FLOAT_SIZE));
            MicroAPI::StoreAlign(sigmoidOutAddr + i * VL_FLOAT_SIZE, vregSigmoidResult, preg0);
            MicroAPI::StoreAlign(indexOutAddr + i * VL_FLOAT_SIZE, vregIndex, preg0);
            MicroAPI::StoreAlign(addBiasOutAddr + i * VL_FLOAT_SIZE, vregBiasResult, preg0);
        }
    }

    xInQueue_.FreeTensor(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::SortInGroup()
{
    LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
    sortedInGroupTensor = sortedInGroupBuf_.Get<float>();

    Sort32(sortedInGroupTensor, xBiasTensor, indexTensor, groupCount_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::SelectTopKInGroup(LocalTensor<float> sortedInGroupTensor,
                                                                            LocalTensor<float> top2InGroupTensor,
                                                                            uint16_t groupCount0,
                                                                            uint32_t perGroupExpertCountAlign0,
                                                                            uint32_t padNegInfNum)
{
    __VEC_SCOPE__
    {
        RegTensor<float> vreg0;
        RegTensor<float> vreg1;
        RegTensor<float> vreg2;
        RegTensor<float> vregPad;
        MicroAPI::UnalignRegForStore u0;
        MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
        __ubuf__ float *inputAddr = (__ubuf__ float *)sortedInGroupTensor.GetPhyAddr();
        __ubuf__ float *outputAddr = (__ubuf__ float *)top2InGroupTensor.GetPhyAddr();
        MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
        for (uint16_t i = 0; i < groupCount0; i++) {
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vreg0, vreg1, inputAddr + i * perGroupExpertCountAlign0 * 2);
            Reg::PairReduceElem<Reg::PairReduce::SUM>(vreg2, vreg0, preg0);
            MicroAPI::StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputAddr, vreg2, u0, 1);
        }
        MicroAPI::StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputAddr, vregPad, u0, padNegInfNum);
        MicroAPI::StoreUnAlignPost(outputAddr, u0, 0);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::SelectTopKAfterSort(LocalTensor<float> sortedGroupTensor,
                                                                              LocalTensor<float> top2InGroupTensor,
                                                                              uint32_t size, int32_t kGroup0,
                                                                              LocalTensor<float> tmpLocal)
{
    int32_t kGroupNumAlign = (kGroup0 + 31) / 32 * 32;
    uint32_t padkGroupNum = kGroupNumAlign - kGroup0;
    __VEC_SCOPE__
    {
        RegTensor<int32_t> vreg0;
        RegTensor<int32_t> vreg1;
        RegTensor<float> vregPad;
        MicroAPI::UnalignRegForStore u0;

        __ubuf__ int32_t *inputAddr = (__ubuf__ int32_t *)sortedGroupTensor.GetPhyAddr();
        __ubuf__ int32_t *outputAddr = (__ubuf__ int32_t *)top2InGroupTensor.GetPhyAddr();
        MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<int32_t>();
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, VL_FLOAT_SIZE));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = MicroAPI::UpdateMask<int32_t>(size);
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                              inputAddr + i * 2 * VL_FLOAT_SIZE);
            MicroAPI::StoreAlign(outputAddr + i * VL_FLOAT_SIZE, vreg1, preg0);
        }
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_STORE>();
        MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
        outputAddr = outputAddr + kGroup0;
        MicroAPI::StoreUnAlign(outputAddr, (RegTensor<int32_t> &)vregPad, u0, padkGroupNum);
        MicroAPI::StoreUnAlignPost(outputAddr, u0, 0);
    }
    Sort<float, true>(sortedGroupTensor, top2InGroupTensor, indexTensor, tmpLocal,
                      kGroupNumAlign / ONE_REPEAT_SORT_NUM);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::SelectTopKGroupIndex()
{
    sortedInGroupTensor = sortedInGroupBuf_.Get<float>();
    LocalTensor<float> top2InGroupTensor = groupBuf_.Get<float>();
    LocalTensor<float> tmpLocal = xBiasBuf_.Get<float>();
    sortedGroupTensor = sortedGroupBuf_.Get<float>();

    uint16_t groupCount0 = groupCount_;
    uint32_t perGroupExpertCountAlign0 = perGroupExpertCountAlign_;
    int32_t groupCountNumAlign = (groupCount_ + 31) / 32 * 32;
    uint32_t padNegInfNum = groupCountNumAlign - groupCount_;

    SelectTopKInGroup(sortedInGroupTensor, top2InGroupTensor, groupCount0, perGroupExpertCountAlign0, padNegInfNum);

    Sort<float, true>(sortedGroupTensor, top2InGroupTensor, indexTensor, tmpLocal,
                      groupCountNumAlign / ONE_REPEAT_SORT_NUM);

    uint32_t size = groupCountNumAlign;
    int32_t kGroup0 = kGroup_;
    SelectTopKAfterSort(sortedGroupTensor, top2InGroupTensor, size, kGroup0, tmpLocal);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::FinalSortByKGroup()
{
    mrgSortTensor = finalSortBuffer_.Get<float>();
    LocalTensor<uint32_t> tmpLocal = sortedGroupTensor.template ReinterpretCast<uint32_t>();
    uint32_t offset[MRG_SORT_ELEMENT_LEN] = {0, 0, 0, 0};

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    uint16_t lenArr[CONSTANT_FOUR] = {
        static_cast<uint16_t>(perGroupExpertCount_), static_cast<uint16_t>(perGroupExpertCount_),
        static_cast<uint16_t>(perGroupExpertCount_), static_cast<uint16_t>(perGroupExpertCount_)};
    MrgSort4Info params{lenArr, false, 0b1111, 1};
    MrgSortSrcList<float> srcList;

    int32_t i = kGroup_ - 1;
    offset[0] = tmpLocal.GetValue(i * 2) * perGroupExpertCountAlign_ * 2;
    offset[1] = tmpLocal.GetValue((i - 1) * 2) * perGroupExpertCountAlign_ * 2;
    offset[CONSTANT_TWO] = tmpLocal.GetValue((i - 2) * 2) * perGroupExpertCountAlign_ * 2;
    offset[3] = tmpLocal.GetValue((i - 3) * 2) * perGroupExpertCountAlign_ * 2;
    srcList.src1 = sortedInGroupTensor[offset[0]];
    srcList.src2 = sortedInGroupTensor[offset[1]];
    srcList.src3 = sortedInGroupTensor[offset[CONSTANT_TWO]];
    srcList.src4 = sortedInGroupTensor[offset[3]];
    MrgSort(mrgSortTensor, srcList, params);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::SelectTopKExpertScore()
{
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> mrgSortTensorInt32 = mrgSortTensor.ReinterpretCast<int32_t>();

    LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.Get<float>();
    LocalTensor<T> yTensor = yOutQueue_.AllocTensor<T>();

    __ubuf__ float *inputAddr = (__ubuf__ float *)xSigmoidTensor.GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T *)yTensor.GetPhyAddr();
    __ubuf__ uint32_t *mrgSortAddr = (__ubuf__ uint32_t *)mrgSortTensorInt32.GetPhyAddr();
    __ubuf__ uint32_t *expertIdxAddr = (__ubuf__ uint32_t *)expertIdxTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> vreg2;
        RegTensor<float> vreg3;
        RegTensor<float> vreg4;
        RegTensor<float> vregOutput;
        RegTensor<uint32_t> vreg0;
        RegTensor<uint32_t> vreg1;

        uint32_t kU32 = static_cast<uint32_t>(k_);
        MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(kU32);

        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1, mrgSortAddr);
        MicroAPI::Gather(vreg2, inputAddr, vreg1, preg0);

        Reg::Reduce<Reg::ReduceType::SUM>(vreg3, vreg2, preg0);
        MicroAPI::Adds(vreg3, vreg3, eps_, preg0);
        MicroAPI::Duplicate(vreg4, vreg3, preg0);
        MicroAPI::Div(vreg4, vreg2, vreg4, preg0);
        MicroAPI::Muls(vregOutput, vreg4, routedScalingFactor_, preg0);

        ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOutput, preg0, 0);
        MicroAPI::StoreAlign(expertIdxAddr, vreg1, preg0);
    }

    yOutQueue_.EnQue(yTensor);
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::CopyOut(int64_t row)
{
    LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm_[row * k_], yOutTensor, dataCopyParams);

    dataCopyParams.blockLen = k_ * sizeof(int32_t);
    DataCopyPad(expertIdxGm_[row * k_], expertIdxTensor, dataCopyParams);

    expertIdxOutQueue_.FreeTensor(expertIdxTensor);
    yOutQueue_.FreeTensor(yOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
                                                               GM_ADDR out, GM_ADDR workspace,
                                                               const MoeGatingTopKRegbaseTilingData *tilingData,
                                                               TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreRowCount_ = tilingData_->lastCoreRowCount;
    } else {
        curCoreRowCount_ = tilingData_->perCoreRowCount;
    }
    expertCount_ = tilingData_->expertCount;
    k_ = tilingData_->k;
    kGroup_ = tilingData_->kGroup;
    groupCount_ = tilingData_->groupCount;
    perGroupExpertCount_ = tilingData_->perGroupExpertCount;
    perGroupExpertCountAlign_ = tilingData_->perGroupExpertCountAlign;
    routedScalingFactor_ = tilingData_->routedScalingFactor;
    eps_ = tilingData_->eps;

    xGm_.SetGlobalBuffer((__gm__ T *)x + tilingData_->perCoreRowCount * expertCount_ * blockIdx_, expertCount_);
    biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);
    yGm_.SetGlobalBuffer((__gm__ T *)y + tilingData_->perCoreRowCount * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + tilingData_->perCoreRowCount * k_ * blockIdx_, k_);

    int32_t expertGroupAlign = groupCount_ * perGroupExpertCountAlign_;
    int32_t groupAlign = static_cast<int32_t>(CeilAlign(groupCount_, ONE_REPEAT_SORT_NUM));
    pipe_->InitBuffer(xInQueue_, CONSTANT_TWO, expertGroupAlign * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(yOutQueue_, CONSTANT_TWO, AlignBytes(k_, sizeof(T)));
    pipe_->InitBuffer(expertIdxOutQueue_, CONSTANT_TWO, AlignBytes(k_, sizeof(int32_t)));

    pipe_->InitBuffer(biasBuf_, expertGroupAlign * sizeof(T));
    pipe_->InitBuffer(xSigmoidBuf_, expertGroupAlign * sizeof(float));
    pipe_->InitBuffer(xBiasBuf_, expertGroupAlign * sizeof(float));
    pipe_->InitBuffer(indexBuffer_, expertGroupAlign * sizeof(int32_t));
    pipe_->InitBuffer(sortedInGroupBuf_, expertGroupAlign * sizeof(float) * CONSTANT_TWO);
    pipe_->InitBuffer(finalSortBuffer_, expertGroupAlign * sizeof(float) * CONSTANT_TWO);
    pipe_->InitBuffer(groupBuf_, groupAlign * sizeof(float));
    pipe_->InitBuffer(sortedGroupBuf_, groupAlign * sizeof(float) * CONSTANT_TWO);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullloadRegbase<T>::Process()
{
    CopyInBias();
    for (int64_t row = 0; row < curCoreRowCount_; row++) {
        CopyInX(row);
        ComputeX();
        SortInGroup();
        SelectTopKGroupIndex();
        FinalSortByKGroup();
        SelectTopKExpertScore();
        CopyOut(row);
    }
}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_E_K_FULLLOAD_REGBASE_H
