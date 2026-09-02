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
 * \file moe_gating_top_k_without_group_regbase.h
 * \brief Arch35 register-based kernel for MoE Gating TopK without group.
 */
#ifndef MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_H
#define MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_H

#include <cmath>
#include "common.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "basic_api/kernel_operator_utils_intf.h"

namespace MoeGatingTopK {
using namespace AscendC;
using MicroAPI::RegTensor;

constexpr int32_t WG_CONSTANT_TWO = 2;
constexpr int32_t WG_CONSTANT_TEN = 10;
constexpr int64_t DEFAULT_BATCH_ROWS = 4;
constexpr int64_t SINGLE_EXPERT_FALLBACK_BATCH = 512;
constexpr MicroAPI::DivSpecificMode WG_DIV_MODE = {MicroAPI::MaskMergeMode::ZEROING, true};

template <typename T>
class MoeGatingTopKWithoutGroupRegbase {
public:
    __aicore__ inline MoeGatingTopKWithoutGroupRegbase(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKRegbaseTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBias();
    __aicore__ inline void CopyInXBatch(int64_t batchIdx, int64_t rowsInBatch);
    __aicore__ inline void ComputeNorm(LocalTensor<float> xInTensor, int64_t rowInBatch);
    __aicore__ inline void ComputeNormSigmoid(__ubuf__ float *xRowAddr, __ubuf__ float *xNormAddr);
    __aicore__ inline void ComputeNormSoftMax(__ubuf__ float *xRowAddr, __ubuf__ float *xNormAddr,
                                              __ubuf__ float *xNormWithBiasAddr, __ubuf__ float *biasAddr,
                                              int64_t duplicateNum);
    __aicore__ inline void ComputeNormSoftplus(__ubuf__ float *xRowAddr, __ubuf__ float *xNormAddr);
    __aicore__ inline void ApplyBiasAndPad(LocalTensor<float> xNormWithBiasTensor, LocalTensor<float> xNormTensor,
                                           LocalTensor<float> biasTensor, int64_t duplicateNum, int64_t duplicateIndex);
    __aicore__ inline void CopyOutXNorm(int64_t globalRow);
    __aicore__ inline void SelectTopKAndScore(LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut,
                                              int64_t rowInBatch);
    __aicore__ inline void CopyOutBatch(int64_t globalBaseRow, int64_t rowsInBatch);

    __aicore__ inline void ProcessSingleExpert();

    __aicore__ inline void SelectTop1AndScoreNoNorm(LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut,
                                                    int64_t rowInBatch);
    __aicore__ inline void SelectTop1AndScoreWithNorm(LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut,
                                                      int64_t rowInBatch);
    __aicore__ inline void SelectTopKAndScoreNoNorm(LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut,
                                                    int64_t rowInBatch);
    __aicore__ inline void SelectTopKAndScoreWithNorm(LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut,
                                                      int64_t rowInBatch);

    __aicore__ inline void CopyInSingleExpertBatch(int64_t gmOffset, int64_t rowsInBatch, LocalTensor<T> &xInTensor,
                                                   LocalTensor<T> &yOutTensor, LocalTensor<int32_t> &expertIdxOut,
                                                   LocalTensor<float> &outOutTensor);
    __aicore__ inline void CopyOutSingleExpertBatch(int64_t gmOffset, int64_t rowsInBatch, LocalTensor<T> &xInTensor,
                                                    LocalTensor<T> &yOutTensor, LocalTensor<int32_t> &expertIdxOut,
                                                    LocalTensor<float> &outOutTensor);

    __aicore__ inline void ProcessSingleExpertSigmoid(int64_t batchSize, int64_t numBatches);
    __aicore__ inline void ProcessSingleExpertSoftMaxRenorm(int64_t batchSize, int64_t numBatches);
    __aicore__ inline void ProcessSingleExpertSoftMaxNoRenorm(int64_t batchSize, int64_t numBatches);
    __aicore__ inline void ProcessSingleExpertSoftplus(int64_t batchSize, int64_t numBatches);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;
    TQue<QuePosition::VECOUT, 1> outOutQueue_;

    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> xNormBuf_;
    TBuf<TPosition::VECCALC> xNormWithBiasBuf_;
    TBuf<TPosition::VECCALC> sortedBuf_;
    TBuf<TPosition::VECCALC> calcTmpBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<float> outGm_;

    int64_t blockIdx_;
    int64_t perCoreRowCount_;
    int64_t curCoreRowCount_;
    int64_t expertCount_;
    int64_t expertCountAlign_;
    int64_t k_;
    int64_t kStride_;
    int64_t batchRows_;
    int64_t normType_;
    int64_t renorm_;
    bool hasBias_;
    bool outFlag_;
    bool noNorm_;
    float routedScalingFactor_;
    float eps_;

    const MoeGatingTopKRegbaseTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::CopyInBias()
{
    if (!hasBias_) {
        return;
    }
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(biasTensor, biasGm_, dataCopyParams, dataCopyPadParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        DataCopyPad(biasTensor[expertCountAlign_].ReinterpretCast<T>(), biasGm_, dataCopyParams, dataCopyPadParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(biasTensor, biasTensor[expertCountAlign_].ReinterpretCast<T>(), RoundMode::CAST_NONE, expertCountAlign_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::CopyInXBatch(int64_t batchIdx, int64_t rowsInBatch)
{
    LocalTensor<float> xInTensor = xInQueue_.AllocTensor<float>();

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = static_cast<uint16_t>(rowsInBatch);
    dataCopyParams.blockLen = static_cast<uint32_t>(expertCount_ * sizeof(T));
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = static_cast<uint32_t>((expertCountAlign_ - expertCount_) * sizeof(T) / BLOCK_BYTES);
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};

    int64_t gmOffset = batchIdx * batchRows_ * expertCount_;
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(xInTensor, xGm_[gmOffset], dataCopyParams, dataCopyPadParams);
    } else {
        LocalTensor<T> rawTensor = xInTensor[batchRows_ * expertCountAlign_].ReinterpretCast<T>();
        DataCopyPad(rawTensor, xGm_[gmOffset], dataCopyParams, dataCopyPadParams);
    }
    xInQueue_.EnQue(xInTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ComputeNormSigmoid(__ubuf__ float *xRowAddr,
                                                                               __ubuf__ float *xNormAddr)
{
    uint32_t expertCountU32 = static_cast<uint32_t>(expertCountAlign_);
    uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(expertCountU32, VL_FLOAT_SIZE));

    __VEC_SCOPE__
    {
        RegTensor<float> vregIn;
        RegTensor<float> vregNorm;
        RegTensor<float> vregOne;
        RegTensor<float> vregNegInput;
        RegTensor<float> vregExpNeg;
        RegTensor<float> vregExpPlusOne;
        MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
        MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::ZEROING, float>(vregOne, static_cast<float>(1), preg0);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = MicroAPI::UpdateMask<float>(expertCountU32);
            ops::LoadOneTensorForDtypeT<float>(xRowAddr, vregIn, preg0, i * VL_FLOAT_SIZE);
            MicroAPI::Muls(vregNegInput, vregIn, static_cast<float>(-1), preg0);
            MicroAPI::Exp(vregExpNeg, vregNegInput, preg0);
            MicroAPI::Adds(vregExpPlusOne, vregExpNeg, static_cast<float>(1), preg0);
            MicroAPI::Div<float, &WG_DIV_MODE>(vregNorm, vregOne, vregExpPlusOne, preg0);
            MicroAPI::StoreAlign(xNormAddr + i * VL_FLOAT_SIZE, vregNorm, preg0);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ComputeNormSoftMax(__ubuf__ float *xRowAddr,
                                                                               __ubuf__ float *xNormAddr,
                                                                               __ubuf__ float *xNormWithBiasAddr,
                                                                               __ubuf__ float *biasAddr,
                                                                               int64_t duplicateNum)
{
    uint32_t size = static_cast<uint32_t>(expertCountAlign_);

    __VEC_SCOPE__
    {
        RegTensor<float> vregX;
        RegTensor<float> vregMax;
        RegTensor<float> vregMaxBcast;
        RegTensor<float> vregExp;
        RegTensor<float> vregSum;
        RegTensor<float> vregSumBcast;
        RegTensor<float> vregResult;
        MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(size);

        MicroAPI::LoadAlign(vregX, xRowAddr);
        Reg::Reduce<Reg::ReduceType::MAX>(vregMax, vregX, preg0);
        MicroAPI::Duplicate(vregMaxBcast, vregMax, preg0);
        MicroAPI::Sub(vregExp, vregX, vregMaxBcast, preg0);
        MicroAPI::Exp(vregExp, vregExp, preg0);
        Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregExp, preg0);
        MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
        MicroAPI::Div(vregResult, vregExp, vregSumBcast, preg0);
        MicroAPI::StoreAlign(xNormAddr, vregResult, preg0);
    }
    if (hasBias_) {
        __VEC_SCOPE__
        {
            RegTensor<float> vregResult;
            RegTensor<float> vregBias;
            RegTensor<float> vregBiasResult;
            MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(size);
            MicroAPI::LoadAlign(vregResult, xNormAddr);
            MicroAPI::LoadAlign(vregBias, biasAddr);
            MicroAPI::Add(vregBiasResult, vregResult, vregBias, preg0);
            MicroAPI::StoreAlign(xNormWithBiasAddr, vregBiasResult, preg0);
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> vregResult;
            MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(size);
            MicroAPI::LoadAlign(vregResult, xNormAddr);
            MicroAPI::StoreAlign(xNormWithBiasAddr, vregResult, preg0);
        }
    }
    if (duplicateNum > 0) {
        __VEC_SCOPE__
        {
            RegTensor<float> vregPad;
            MicroAPI::UnalignRegForStore u0;
            MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
            uint32_t padCount = static_cast<uint32_t>(expertCountAlign_ - expertCount_);
            auto padAddr = xNormWithBiasAddr + expertCount_;
            MicroAPI::StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(padAddr, vregPad, u0, padCount);
            MicroAPI::StoreUnAlignPost(padAddr, u0, 0);
        }
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ComputeNormSoftplus(__ubuf__ float *xRowAddr,
                                                                                __ubuf__ float *xNormAddr)
{
    uint32_t expertCountU32 = static_cast<uint32_t>(expertCountAlign_);
    uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(expertCountU32, VL_FLOAT_SIZE));

    __VEC_SCOPE__
    {
        RegTensor<float> vregIn;
        RegTensor<float> vregNorm;
        RegTensor<float> vregExpInput;
        RegTensor<float> vregExpPlusOne;
        RegTensor<float> vregLnResult;
        MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = MicroAPI::UpdateMask<float>(expertCountU32);
            ops::LoadOneTensorForDtypeT<float>(xRowAddr, vregIn, preg0, i * VL_FLOAT_SIZE);
            MicroAPI::Exp(vregExpInput, vregIn, preg0);
            MicroAPI::Adds(vregExpPlusOne, vregExpInput, static_cast<float>(1), preg0);
            MicroAPI::Ln(vregLnResult, vregExpPlusOne, preg0);
            MicroAPI::Sqrt(vregNorm, vregLnResult, preg0);
            MicroAPI::StoreAlign(xNormAddr + i * VL_FLOAT_SIZE, vregNorm, preg0);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ComputeNorm(LocalTensor<float> xInTensor,
                                                                        int64_t rowInBatch)
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();

    int64_t rowOffset = rowInBatch * expertCountAlign_;

    if constexpr (!IsSameType<T, float>::value) {
        LocalTensor<T> rawTensor = xInTensor[batchRows_ * expertCountAlign_].ReinterpretCast<T>();
        Cast(xInTensor[rowOffset], rawTensor[rowOffset], RoundMode::CAST_NONE, expertCountAlign_);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> xRow = xInTensor[rowOffset];

    int64_t duplicateNum = expertCount_ % ONE_REPEAT_SORT_NUM;
    int64_t duplicateIndex = expertCount_ - duplicateNum;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(xRow.ReinterpretCast<int32_t>()[duplicateIndex], MIN_FP32, mask, 1, 1, 1);
        PipeBarrier<PIPE_V>();
    }

    __ubuf__ float *xRowAddr = (__ubuf__ float *)xRow.GetPhyAddr();
    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNormTensor.GetPhyAddr();
    __ubuf__ float *xNormWithBiasAddr = (__ubuf__ float *)xNormWithBiasTensor.GetPhyAddr();
    __ubuf__ float *biasAddr = (__ubuf__ float *)biasTensor.GetPhyAddr();

    if (normType_ == 1) {
        ComputeNormSigmoid(xRowAddr, xNormAddr);
    } else if (normType_ == 0) {
        ComputeNormSoftMax(xRowAddr, xNormAddr, xNormWithBiasAddr, biasAddr, duplicateNum);
    } else {
        ComputeNormSoftplus(xRowAddr, xNormAddr);
    }

    if (normType_ != 0) {
        ApplyBiasAndPad(xNormWithBiasTensor, xNormTensor, biasTensor, duplicateNum, duplicateIndex);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ApplyBiasAndPad(LocalTensor<float> xNormWithBiasTensor,
                                                                            LocalTensor<float> xNormTensor,
                                                                            LocalTensor<float> biasTensor,
                                                                            int64_t duplicateNum,
                                                                            int64_t duplicateIndex)
{
    if (hasBias_) {
        Add(xNormWithBiasTensor, xNormTensor, biasTensor, expertCountAlign_);
    } else {
        DataCopy(xNormWithBiasTensor, xNormTensor, expertCountAlign_);
    }
    PipeBarrier<PIPE_V>();

    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(xNormWithBiasTensor.ReinterpretCast<int32_t>()[duplicateIndex], MIN_FP32, mask, 1, 1, 1);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::CopyOutXNorm(int64_t globalRow)
{
    LocalTensor<float> outOutTensor = outOutQueue_.AllocTensor<float>();
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    DataCopy(outOutTensor, xNormTensor, expertCountAlign_);
    outOutQueue_.EnQue<float>(outOutTensor);
    outOutTensor = outOutQueue_.DeQue<float>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(outGm_[globalRow * expertCount_], outOutTensor, dataCopyParams);
    outOutQueue_.FreeTensor(outOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::SelectTop1AndScoreNoNorm(LocalTensor<T> yOutTensor,
                                                                                     LocalTensor<int32_t> expertIdxOut,
                                                                                     int64_t rowInBatch)
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> sortedScore = sortedBuf_.Get<float>();

    uint32_t expertCountU32 = static_cast<uint32_t>(expertCount_);
    __ubuf__ float *xNormWithBiasAddr = (__ubuf__ float *)xNormWithBiasTensor.GetPhyAddr();
    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNormTensor.GetPhyAddr();
    __ubuf__ float *sortedAddr = (__ubuf__ float *)sortedScore.GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr() + rowInBatch * kStride_;
    __ubuf__ uint32_t *expertIdxAddr = (__ubuf__ uint32_t *)expertIdxOut.GetPhyAddr() + rowInBatch * kStride_;

    __VEC_SCOPE__
    {
        RegTensor<float> valueAndIndexReg;
        MicroAPI::MaskReg maskForVL2 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL2>();
        MicroAPI::MaskReg maskForExpertCount = MicroAPI::UpdateMask<uint32_t>(expertCountU32);

        MicroAPI::LoadAlign(valueAndIndexReg, xNormWithBiasAddr);
        Reg::Reduce<Reg::ReduceType::MAX>(valueAndIndexReg, valueAndIndexReg, maskForExpertCount);
        MicroAPI::StoreAlign(sortedAddr, valueAndIndexReg, maskForVL2);
    }

    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValue;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGathered;
        RegTensor<float> vregOutput;
        uint32_t kU32 = static_cast<uint32_t>(k_);
        MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(kU32);

        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx,
                                                                           (__ubuf__ uint32_t *)sortedAddr);
        MicroAPI::Gather(vregGathered, xNormAddr, vregExpertIdx, preg0);
        MicroAPI::Muls(vregOutput, vregGathered, routedScalingFactor_, preg0);
        ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOutput, preg0, 0);
        MicroAPI::StoreAlign(expertIdxAddr, vregExpertIdx, preg0);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::SelectTop1AndScoreWithNorm(
    LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut, int64_t rowInBatch)
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> sortedScore = sortedBuf_.Get<float>();

    uint32_t expertCountU32 = static_cast<uint32_t>(expertCount_);
    __ubuf__ float *xNormWithBiasAddr = (__ubuf__ float *)xNormWithBiasTensor.GetPhyAddr();
    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNormTensor.GetPhyAddr();
    __ubuf__ float *sortedAddr = (__ubuf__ float *)sortedScore.GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr() + rowInBatch * kStride_;
    __ubuf__ uint32_t *expertIdxAddr = (__ubuf__ uint32_t *)expertIdxOut.GetPhyAddr() + rowInBatch * kStride_;

    __VEC_SCOPE__
    {
        RegTensor<float> valueAndIndexReg;
        MicroAPI::MaskReg maskForVL2 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL2>();
        MicroAPI::MaskReg maskForExpertCount = MicroAPI::UpdateMask<uint32_t>(expertCountU32);

        MicroAPI::LoadAlign(valueAndIndexReg, xNormWithBiasAddr);
        Reg::Reduce<Reg::ReduceType::MAX>(valueAndIndexReg, valueAndIndexReg, maskForExpertCount);
        MicroAPI::StoreAlign(sortedAddr, valueAndIndexReg, maskForVL2);
    }

    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValue;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGathered;
        RegTensor<float> vregSum;
        RegTensor<float> vregSumBcast;
        RegTensor<float> vregOutput;
        uint32_t kU32 = static_cast<uint32_t>(k_);
        MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(kU32);

        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx,
                                                                           (__ubuf__ uint32_t *)sortedAddr);
        MicroAPI::Gather(vregGathered, xNormAddr, vregExpertIdx, preg0);
        Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregGathered, preg0);
        MicroAPI::Adds(vregSum, vregSum, eps_, preg0);
        MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
        MicroAPI::Div(vregSumBcast, vregGathered, vregSumBcast, preg0);
        MicroAPI::Muls(vregOutput, vregSumBcast, routedScalingFactor_, preg0);
        ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOutput, preg0, 0);
        MicroAPI::StoreAlign(expertIdxAddr, vregExpertIdx, preg0);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::SelectTopKAndScoreNoNorm(LocalTensor<T> yOutTensor,
                                                                                     LocalTensor<int32_t> expertIdxOut,
                                                                                     int64_t rowInBatch)
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> sortedScore = sortedBuf_.Get<float>();
    LocalTensor<uint32_t> indexTensor = indexBuf_.Get<uint32_t>();
    LocalTensor<float> sortTmp = calcTmpBuf_.Get<float>();

    PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedScore, xNormWithBiasTensor, indexTensor, sortTmp, expertCountAlign_ / ONE_REPEAT_SORT_NUM);

    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNormTensor.GetPhyAddr();
    __ubuf__ uint32_t *sortedAddr = (__ubuf__ uint32_t *)sortedScore.ReinterpretCast<uint32_t>().GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr() + rowInBatch * kStride_;
    __ubuf__ uint32_t *expertIdxAddr = (__ubuf__ uint32_t *)expertIdxOut.GetPhyAddr() + rowInBatch * kStride_;
    uint32_t kU32 = static_cast<uint32_t>(k_);

    if (k_ <= static_cast<int64_t>(VL_FLOAT_SIZE)) {
        SmallKAlignEVFNoNorm<T>(xNormAddr, sortedAddr, outputAddr, expertIdxAddr, kU32, routedScalingFactor_);
    } else {
        LargeKAlignEVFNoNorm<T>(xNormAddr, sortedAddr, outputAddr, expertIdxAddr, kU32, routedScalingFactor_);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::SelectTopKAndScoreWithNorm(
    LocalTensor<T> yOutTensor, LocalTensor<int32_t> expertIdxOut, int64_t rowInBatch)
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> sortedScore = sortedBuf_.Get<float>();
    LocalTensor<uint32_t> indexTensor = indexBuf_.Get<uint32_t>();
    LocalTensor<float> sortTmp = calcTmpBuf_.Get<float>();

    PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedScore, xNormWithBiasTensor, indexTensor, sortTmp, expertCountAlign_ / ONE_REPEAT_SORT_NUM);

    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNormTensor.GetPhyAddr();
    __ubuf__ uint32_t *sortedAddr = (__ubuf__ uint32_t *)sortedScore.ReinterpretCast<uint32_t>().GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr() + rowInBatch * kStride_;
    __ubuf__ uint32_t *expertIdxAddr = (__ubuf__ uint32_t *)expertIdxOut.GetPhyAddr() + rowInBatch * kStride_;
    uint32_t kU32 = static_cast<uint32_t>(k_);

    if (k_ <= static_cast<int64_t>(VL_FLOAT_SIZE)) {
        SmallKAlignEVFWithNorm<T>(xNormAddr, sortedAddr, outputAddr, expertIdxAddr, kU32, eps_, routedScalingFactor_);
    } else {
        LargeKAlignEVFWithNorm<T>(xNormAddr, sortedAddr, outputAddr, expertIdxAddr, kU32, eps_, routedScalingFactor_);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::SelectTopKAndScore(LocalTensor<T> yOutTensor,
                                                                               LocalTensor<int32_t> expertIdxOut,
                                                                               int64_t rowInBatch)
{
    bool useTop1FastPath = (k_ == 1) && (expertCount_ <= static_cast<int64_t>(VL_FLOAT_SIZE));
    if (useTop1FastPath) {
        if (noNorm_) {
            SelectTop1AndScoreNoNorm(yOutTensor, expertIdxOut, rowInBatch);
        } else {
            SelectTop1AndScoreWithNorm(yOutTensor, expertIdxOut, rowInBatch);
        }
    } else {
        if (noNorm_) {
            SelectTopKAndScoreNoNorm(yOutTensor, expertIdxOut, rowInBatch);
        } else {
            SelectTopKAndScoreWithNorm(yOutTensor, expertIdxOut, rowInBatch);
        }
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::CopyOutBatch(int64_t globalBaseRow, int64_t rowsInBatch)
{
    LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
    LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams yCopyParams{static_cast<uint16_t>(rowsInBatch), static_cast<uint32_t>(k_ * sizeof(T)),
                                  static_cast<uint32_t>((kStride_ - k_) * sizeof(T) / BLOCK_BYTES), 0, 0};
    DataCopyPad(yGm_[globalBaseRow * k_], yOutTensor, yCopyParams);
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(rowsInBatch), static_cast<uint32_t>(k_ * sizeof(int32_t)),
                                    static_cast<uint32_t>((kStride_ - k_) * sizeof(int32_t) / BLOCK_BYTES), 0, 0};
    DataCopyPad(expertIdxGm_[globalBaseRow * k_], expertIdxOut, idxCopyParams);
    yOutQueue_.FreeTensor(yOutTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxOut);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::CopyInSingleExpertBatch(
    int64_t gmOffset, int64_t rowsInBatch, LocalTensor<T> &xInTensor, LocalTensor<T> &yOutTensor,
    LocalTensor<int32_t> &expertIdxOut, LocalTensor<float> &outOutTensor)
{
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    xInTensor = xInQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(T)), 0, 0, 0};
    DataCopyPad(xInTensor, xGm_[gmOffset], copyParams, dataCopyPadParams);
    xInQueue_.EnQue(xInTensor);
    xInTensor = xInQueue_.DeQue<T>();
    yOutTensor = yOutQueue_.AllocTensor<T>();
    expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();
    outOutTensor = outOutQueue_.AllocTensor<float>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::CopyOutSingleExpertBatch(
    int64_t gmOffset, int64_t rowsInBatch, LocalTensor<T> &xInTensor, LocalTensor<T> &yOutTensor,
    LocalTensor<int32_t> &expertIdxOut, LocalTensor<float> &outOutTensor)
{
    yOutQueue_.EnQue(yOutTensor);
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
    yOutTensor = yOutQueue_.DeQue<T>();
    expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams yCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm_[gmOffset], yOutTensor, yCopyParams);
    DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(expertIdxGm_[gmOffset], expertIdxOut, idxCopyParams);
    yOutQueue_.FreeTensor(yOutTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxOut);

    if (outFlag_) {
        outOutQueue_.EnQue<float>(outOutTensor);
        outOutTensor = outOutQueue_.DeQue<float>();
        DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(rowsInBatch * sizeof(float)), 0, 0, 0};
        DataCopyPad(outGm_[gmOffset], outOutTensor, outCopyParams);
    }
    outOutQueue_.FreeTensor(outOutTensor);
    xInQueue_.FreeTensor(xInTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ProcessSingleExpertSigmoid(int64_t batchSize,
                                                                                       int64_t numBatches)
{
    for (int64_t batch = 0; batch < numBatches; batch++) {
        int64_t rowsInBatch = Min(batchSize, curCoreRowCount_ - batch * batchSize);
        int64_t gmOffset = batch * batchSize;

        LocalTensor<T> xInTensor, yOutTensor;
        LocalTensor<int32_t> expertIdxOut;
        LocalTensor<float> outOutTensor;
        CopyInSingleExpertBatch(gmOffset, rowsInBatch, xInTensor, yOutTensor, expertIdxOut, outOutTensor);

        __ubuf__ T *inputAddr = (__ubuf__ T *)xInTensor.GetPhyAddr();
        __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr();
        __ubuf__ int32_t *expertIdxAddr = (__ubuf__ int32_t *)expertIdxOut.GetPhyAddr();
        __ubuf__ float *outAddr = (__ubuf__ float *)outOutTensor.GetPhyAddr();

        uint32_t rowsInBatchU32 = static_cast<uint32_t>(rowsInBatch);
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(rowsInBatchU32, VL_FLOAT_SIZE));

        __VEC_SCOPE__
        {
            RegTensor<float> vregIn;
            RegTensor<float> vregNorm;
            RegTensor<float> vregOut;
            RegTensor<float> vregOne;
            RegTensor<float> vregTmp1;
            RegTensor<float> vregTmp2;
            RegTensor<float> vregTmp3;
            RegTensor<int32_t> vregZeroIdx;
            MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
            MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::ZEROING, float>(vregOne, static_cast<float>(1), preg0);

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(rowsInBatchU32);
                ops::LoadOneTensorForDtypeT<T>(inputAddr, vregIn, preg0, i * VL_FLOAT_SIZE);
                MicroAPI::Muls(vregTmp1, vregIn, static_cast<float>(-1), preg0);
                MicroAPI::Exp(vregTmp2, vregTmp1, preg0);
                MicroAPI::Adds(vregTmp3, vregTmp2, static_cast<float>(1), preg0);
                MicroAPI::Div<float, &WG_DIV_MODE>(vregNorm, vregOne, vregTmp3, preg0);
                RegTensor<float> vregSum;
                Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregNorm, preg0);
                MicroAPI::Adds(vregSum, vregSum, eps_, preg0);
                RegTensor<float> vregSumBcast;
                MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
                MicroAPI::Div(vregOut, vregNorm, vregSumBcast, preg0);
                MicroAPI::Muls(vregOut, vregOut, routedScalingFactor_, preg0);
                MicroAPI::Duplicate(vregZeroIdx, static_cast<int32_t>(0), preg0);
                ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOut, preg0, i * VL_FLOAT_SIZE);
                MicroAPI::StoreAlign(expertIdxAddr + i * VL_FLOAT_SIZE, vregZeroIdx, preg0);
                MicroAPI::StoreAlign(outAddr + i * VL_FLOAT_SIZE, vregNorm, preg0);
            }
        }

        CopyOutSingleExpertBatch(gmOffset, rowsInBatch, xInTensor, yOutTensor, expertIdxOut, outOutTensor);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ProcessSingleExpertSoftMaxRenorm(int64_t batchSize,
                                                                                             int64_t numBatches)
{
    for (int64_t batch = 0; batch < numBatches; batch++) {
        int64_t rowsInBatch = Min(batchSize, curCoreRowCount_ - batch * batchSize);
        int64_t gmOffset = batch * batchSize;

        LocalTensor<T> xInTensor, yOutTensor;
        LocalTensor<int32_t> expertIdxOut;
        LocalTensor<float> outOutTensor;
        CopyInSingleExpertBatch(gmOffset, rowsInBatch, xInTensor, yOutTensor, expertIdxOut, outOutTensor);

        __ubuf__ T *inputAddr = (__ubuf__ T *)xInTensor.GetPhyAddr();
        __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr();
        __ubuf__ int32_t *expertIdxAddr = (__ubuf__ int32_t *)expertIdxOut.GetPhyAddr();
        __ubuf__ float *outAddr = (__ubuf__ float *)outOutTensor.GetPhyAddr();

        uint32_t rowsInBatchU32 = static_cast<uint32_t>(rowsInBatch);
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(rowsInBatchU32, VL_FLOAT_SIZE));

        __VEC_SCOPE__
        {
            RegTensor<float> vregIn;
            RegTensor<float> vregNorm;
            RegTensor<float> vregOut;
            RegTensor<int32_t> vregZeroIdx;
            MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(rowsInBatchU32);
                ops::LoadOneTensorForDtypeT<T>(inputAddr, vregIn, preg0, i * VL_FLOAT_SIZE);
                MicroAPI::Duplicate(vregNorm, static_cast<float>(1), preg0);
                RegTensor<float> vregSum;
                Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregNorm, preg0);
                MicroAPI::Adds(vregSum, vregSum, eps_, preg0);
                RegTensor<float> vregSumBcast;
                MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
                MicroAPI::Div(vregOut, vregNorm, vregSumBcast, preg0);
                MicroAPI::Muls(vregOut, vregOut, routedScalingFactor_, preg0);
                MicroAPI::Duplicate(vregZeroIdx, static_cast<int32_t>(0), preg0);
                ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOut, preg0, i * VL_FLOAT_SIZE);
                MicroAPI::StoreAlign(expertIdxAddr + i * VL_FLOAT_SIZE, vregZeroIdx, preg0);
                MicroAPI::StoreAlign(outAddr + i * VL_FLOAT_SIZE, vregNorm, preg0);
            }
        }

        CopyOutSingleExpertBatch(gmOffset, rowsInBatch, xInTensor, yOutTensor, expertIdxOut, outOutTensor);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ProcessSingleExpertSoftMaxNoRenorm(int64_t batchSize,
                                                                                               int64_t numBatches)
{
    for (int64_t batch = 0; batch < numBatches; batch++) {
        int64_t rowsInBatch = Min(batchSize, curCoreRowCount_ - batch * batchSize);
        int64_t gmOffset = batch * batchSize;

        LocalTensor<T> xInTensor, yOutTensor;
        LocalTensor<int32_t> expertIdxOut;
        LocalTensor<float> outOutTensor;
        CopyInSingleExpertBatch(gmOffset, rowsInBatch, xInTensor, yOutTensor, expertIdxOut, outOutTensor);

        __ubuf__ T *inputAddr = (__ubuf__ T *)xInTensor.GetPhyAddr();
        __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr();
        __ubuf__ int32_t *expertIdxAddr = (__ubuf__ int32_t *)expertIdxOut.GetPhyAddr();
        __ubuf__ float *outAddr = (__ubuf__ float *)outOutTensor.GetPhyAddr();

        uint32_t rowsInBatchU32 = static_cast<uint32_t>(rowsInBatch);
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(rowsInBatchU32, VL_FLOAT_SIZE));

        __VEC_SCOPE__
        {
            RegTensor<float> vregIn;
            RegTensor<float> vregNorm;
            RegTensor<float> vregOut;
            RegTensor<int32_t> vregZeroIdx;
            MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(rowsInBatchU32);
                ops::LoadOneTensorForDtypeT<T>(inputAddr, vregIn, preg0, i * VL_FLOAT_SIZE);
                MicroAPI::Duplicate(vregNorm, static_cast<float>(1), preg0);
                MicroAPI::Muls(vregOut, vregNorm, static_cast<float>(1), preg0);
                MicroAPI::Muls(vregOut, vregOut, routedScalingFactor_, preg0);
                MicroAPI::Duplicate(vregZeroIdx, static_cast<int32_t>(0), preg0);
                ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOut, preg0, i * VL_FLOAT_SIZE);
                MicroAPI::StoreAlign(expertIdxAddr + i * VL_FLOAT_SIZE, vregZeroIdx, preg0);
                MicroAPI::StoreAlign(outAddr + i * VL_FLOAT_SIZE, vregNorm, preg0);
            }
        }

        CopyOutSingleExpertBatch(gmOffset, rowsInBatch, xInTensor, yOutTensor, expertIdxOut, outOutTensor);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ProcessSingleExpertSoftplus(int64_t batchSize,
                                                                                        int64_t numBatches)
{
    for (int64_t batch = 0; batch < numBatches; batch++) {
        int64_t rowsInBatch = Min(batchSize, curCoreRowCount_ - batch * batchSize);
        int64_t gmOffset = batch * batchSize;

        LocalTensor<T> xInTensor, yOutTensor;
        LocalTensor<int32_t> expertIdxOut;
        LocalTensor<float> outOutTensor;
        CopyInSingleExpertBatch(gmOffset, rowsInBatch, xInTensor, yOutTensor, expertIdxOut, outOutTensor);

        __ubuf__ T *inputAddr = (__ubuf__ T *)xInTensor.GetPhyAddr();
        __ubuf__ T *outputAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr();
        __ubuf__ int32_t *expertIdxAddr = (__ubuf__ int32_t *)expertIdxOut.GetPhyAddr();
        __ubuf__ float *outAddr = (__ubuf__ float *)outOutTensor.GetPhyAddr();

        uint32_t rowsInBatchU32 = static_cast<uint32_t>(rowsInBatch);
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(rowsInBatchU32, VL_FLOAT_SIZE));

        __VEC_SCOPE__
        {
            RegTensor<float> vregIn;
            RegTensor<float> vregNorm;
            RegTensor<float> vregOut;
            RegTensor<float> vregTmp1;
            RegTensor<float> vregTmp2;
            RegTensor<float> vregTmp3;
            RegTensor<int32_t> vregZeroIdx;
            MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(rowsInBatchU32);
                ops::LoadOneTensorForDtypeT<T>(inputAddr, vregIn, preg0, i * VL_FLOAT_SIZE);
                MicroAPI::Exp(vregTmp1, vregIn, preg0);
                MicroAPI::Adds(vregTmp2, vregTmp1, static_cast<float>(1), preg0);
                MicroAPI::Ln(vregTmp3, vregTmp2, preg0);
                MicroAPI::Sqrt(vregNorm, vregTmp3, preg0);
                MicroAPI::Muls(vregOut, vregNorm, static_cast<float>(1), preg0);
                MicroAPI::Muls(vregOut, vregOut, routedScalingFactor_, preg0);
                MicroAPI::Duplicate(vregZeroIdx, static_cast<int32_t>(0), preg0);
                ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOut, preg0, i * VL_FLOAT_SIZE);
                MicroAPI::StoreAlign(expertIdxAddr + i * VL_FLOAT_SIZE, vregZeroIdx, preg0);
                MicroAPI::StoreAlign(outAddr + i * VL_FLOAT_SIZE, vregNorm, preg0);
            }
        }

        CopyOutSingleExpertBatch(gmOffset, rowsInBatch, xInTensor, yOutTensor, expertIdxOut, outOutTensor);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::ProcessSingleExpert()
{
    int64_t batchSize = Min(batchRows_, curCoreRowCount_);
    int64_t numBatches = CeilDiv(curCoreRowCount_, batchSize);

    if (normType_ == 1) {
        ProcessSingleExpertSigmoid(batchSize, numBatches);
    } else if (normType_ == 0) {
        if (renorm_ == 0) {
            ProcessSingleExpertSoftMaxNoRenorm(batchSize, numBatches);
        } else {
            ProcessSingleExpertSoftMaxRenorm(batchSize, numBatches);
        }
    } else {
        ProcessSingleExpertSoftplus(batchSize, numBatches);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
                                                                 GM_ADDR out, GM_ADDR workspace,
                                                                 const MoeGatingTopKRegbaseTilingData *tilingData,
                                                                 TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    perCoreRowCount_ = tilingData_->perCoreRowCount;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreRowCount_ = tilingData_->lastCoreRowCount;
    } else {
        curCoreRowCount_ = tilingData_->perCoreRowCount;
    }
    expertCount_ = tilingData_->expertCount;
    hasBias_ = (tilingData_->addBias == 1) && (bias != nullptr);
    k_ = tilingData_->k;
    normType_ = tilingData_->normType;
    renorm_ = tilingData_->renorm;
    outFlag_ = tilingData_->outFlag == 1;
    noNorm_ = (normType_ == 0 && renorm_ == 0);
    routedScalingFactor_ = tilingData_->routedScalingFactor;
    eps_ = tilingData_->eps;
    batchRows_ = tilingData_->batchRows > 0 ? tilingData_->batchRows : DEFAULT_BATCH_ROWS;

    expertCountAlign_ = Ceil(expertCount_, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    kStride_ = Ceil(k_, static_cast<int64_t>(BLOCK_BYTES) / static_cast<int64_t>(sizeof(T))) *
               static_cast<int64_t>(BLOCK_BYTES) / static_cast<int64_t>(sizeof(T));

    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    if (hasBias_) {
        biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);
    }
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    outGm_.SetGlobalBuffer((__gm__ float *)out + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);

    if (expertCount_ == 1) {
        int64_t singleExpertBatch = tilingData_->batchRows > 0 ? tilingData_->batchRows : SINGLE_EXPERT_FALLBACK_BATCH;
        batchRows_ = singleExpertBatch;
        pipe_->InitBuffer(xInQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(T));
        pipe_->InitBuffer(yOutQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(T));
        pipe_->InitBuffer(expertIdxOutQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(int32_t));
        pipe_->InitBuffer(outOutQueue_, WG_CONSTANT_TWO, singleExpertBatch * sizeof(float));
        return;
    }

    int64_t batchExpertAlign = batchRows_ * expertCountAlign_;
    pipe_->InitBuffer(xInQueue_, WG_CONSTANT_TWO, batchExpertAlign * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(yOutQueue_, WG_CONSTANT_TWO, batchRows_ * kStride_ * sizeof(float));
    pipe_->InitBuffer(expertIdxOutQueue_, WG_CONSTANT_TWO, batchRows_ * kStride_ * sizeof(int32_t));
    pipe_->InitBuffer(outOutQueue_, WG_CONSTANT_TWO, expertCountAlign_ * sizeof(float));

    pipe_->InitBuffer(biasBuf_, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(xNormBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(xNormWithBiasBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(sortedBuf_, expertCountAlign_ * sizeof(float) * WG_CONSTANT_TWO);
    pipe_->InitBuffer(calcTmpBuf_, expertCountAlign_ * sizeof(float) * WG_CONSTANT_TEN);
    pipe_->InitBuffer(indexBuf_, expertCountAlign_ * sizeof(uint32_t));
}

template <typename T>
__aicore__ inline void MoeGatingTopKWithoutGroupRegbase<T>::Process()
{
    if (expertCount_ == 1) {
        ProcessSingleExpert();
        return;
    }

    CopyInBias();

    LocalTensor<int32_t> indexTensor = indexBuf_.Get<int32_t>();
    ArithProgression(indexTensor, static_cast<int32_t>(0), static_cast<int32_t>(1), expertCountAlign_);

    int64_t numBatches = CeilDiv(curCoreRowCount_, batchRows_);

    for (int64_t batch = 0; batch < numBatches; batch++) {
        int64_t rowsInBatch = Min(batchRows_, curCoreRowCount_ - batch * batchRows_);
        CopyInXBatch(batch, rowsInBatch);

        LocalTensor<float> xInTensor = xInQueue_.DeQue<float>();
        LocalTensor<T> yOutTensor = yOutQueue_.AllocTensor<T>();
        LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();
        for (int64_t r = 0; r < rowsInBatch; r++) {
            int64_t globalRow = batch * batchRows_ + r;
            ComputeNorm(xInTensor, r);
            if (outFlag_) {
                CopyOutXNorm(globalRow);
            }
            SelectTopKAndScore(yOutTensor, expertIdxOut, r);
        }
        yOutQueue_.EnQue<T>(yOutTensor);
        expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
        CopyOutBatch(batch * batchRows_, rowsInBatch);
        xInQueue_.FreeTensor(xInTensor);
    }
}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_WITHOUT_GROUP_REGBASE_H
