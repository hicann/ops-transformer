/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_gather_dynamic_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_DYNAMIC_QUANT_H_REGBASE
#define MOE_V3_GATHER_DYNAMIC_QUANT_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "../../inc/load_store_utils.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;
constexpr int64_t GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM = 1;

template <typename T>
class MoeGatherOutDynamicQuant {
public:
    __aicore__ inline MoeGatherOutDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR expandedScale, GM_ADDR sortedExpertIdx,
                                const MoeInitRoutingV3TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyOutXQuantEH(int64_t progress);
    template <bool IS_INPUT_SCALE>
    __aicore__ inline void Compute(LocalTensor<float> &smoothLocal);
    __aicore__ inline void CopyOutPartialXQuantEH(int64_t progress);
    template <bool IS_INPUT_SCALE>
    __aicore__ inline float ComputeMax(LocalTensor<float> &inLocal, LocalTensor<float> &scaleLocal, int32_t srcIdx,
                                       int32_t expertIdx, int64_t j);
    __aicore__ inline void ComputeScale(LocalTensor<float> &inLocal, float scaleTemp, int64_t dstIndex, int64_t j);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> inputXInQueue_;
    TQue<QuePosition::VECIN, 1> smoothInQueue_;
    TQue<QuePosition::VECIN, 1> expandRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> calcQueue_;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> quantSmoothGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<float> quantTempGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;
    int64_t totalLength_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t currentLoopRowsAlign_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t colsTileLength_;
    int64_t perLoopCols_;
    int64_t perLoopColsAlign_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t isInputScale_;
    int64_t expertStart_;

    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;

    constexpr static MicroAPI::CastTrait castTraitF32ToF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    constexpr static MicroAPI::CastTrait castTraitF16ToI8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    DataCopyPad(indicesLocal[currentLoopRowsAlign_], expandedExpertIdxGm_[indicesOffset_], dataCopyParams,
                dataCopyPadParams);
    expandRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
template <bool IS_INPUT_SCALE>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::Compute(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

    __local_mem__ float *inUbAddr = (__local_mem__ float *)inLocal.GetPhyAddr();
    __local_mem__ float *scaleUbAddr = (__local_mem__ float *)scaleLocal.GetPhyAddr();
    __local_mem__ int8_t *outUbAddr = (__local_mem__ int8_t *)outLocal.GetPhyAddr();
    __local_mem__ T *inUbAddrCastT;
    if constexpr (!IsSameType<T, float>::value) {
        inUbAddrCastT = (__local_mem__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + perLoopColsAlign_;
    }
    __local_mem__ float *smoothUbAddr;
    if constexpr (IS_INPUT_SCALE) {
        smoothUbAddr = (__local_mem__ float *)smoothLocal.GetPhyAddr();
    }

    uint16_t repeatTimes = Ceil(cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, smoothReg, scaleValueReg;
        MicroAPI::Duplicate(scaleValueReg, 0.0f);
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;

        MicroAPI::MaskReg maskRegInLoop;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegVL1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        MicroAPI::MaskReg maskRegVL8 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL8>();

        sreg = static_cast<uint32_t>(cols_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            if constexpr (!IsSameType<T, float>::value) {
                ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            } else {
                MicroAPI::DataCopy(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            }
            if constexpr (IS_INPUT_SCALE) {
                MicroAPI::DataCopy(smoothReg, smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
                MicroAPI::Mul(inReg, inReg, smoothReg, maskRegInLoop);
            }
            if constexpr (!IsSameType<T, float>::value || IS_INPUT_SCALE) {
                MicroAPI::DataCopy(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegInLoop);
            }
            MicroAPI::Abs(inReg, inReg, maskRegInLoop);
            MicroAPI::Max(scaleValueReg, scaleValueReg, inReg, maskRegAll);
        }
        MicroAPI::ReduceMax(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / 127.0f, maskRegVL1);
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::DataCopy(scaleUbAddr, scaleValueReg, maskRegVL8);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        sreg = static_cast<uint32_t>(cols_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::DataCopy(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Div(inReg, inReg, scaleValueReg, maskRegInLoop);
            MicroAPI::Cast<half, float, castTraitF32ToF16>(outRegF16, inReg, maskRegInLoop);
            MicroAPI::Cast<int8_t, half, castTraitF16ToI8>(outRegI8, outRegF16, maskRegInLoop);
            MicroAPI::DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                            outRegI8, maskRegInLoop);
        }
    }

    inputXOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::CopyOutXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(int8_t)), 0, 0, 0};

    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    int32_t lastExpertIdx = -1;
    for (int64_t i = 0; i < currentLoopRows_; i++) {
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i) - expertStart_;
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, inputXGm_[srcIdx / k_ * cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[perLoopColsAlign_], inputXGm_[srcIdx / k_ * cols_], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);

        if (isInputScale_ && expertIdx != lastExpertIdx) {
            DataCopyPad(smoothLocal, quantSmoothGm_[expertIdx * cols_], smoothParams, {false, 0, 0, 0});
            smoothInQueue_.EnQue(smoothLocal);
            smoothLocal = smoothInQueue_.DeQue<float>();
            lastExpertIdx = expertIdx;
        }
        if (isInputScale_) {
            Compute<true>(smoothLocal);
        } else {
            Compute<false>(smoothLocal);
        }
        inputXInQueue_.FreeTensor(inLocal);

        LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();
        DataCopyPad(expandedScaleGm_[(rowOffset + i)], scaleLocal, {1, 4, 0, 0, 0});
        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        DataCopyPad(expandedXGm_[(rowOffset + i) * cols_], outLocal, copyOutParams);

        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }

    smoothInQueue_.FreeTensor(smoothLocal);
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
template <bool IS_INPUT_SCALE>
__aicore__ inline float MoeGatherOutDynamicQuant<T>::ComputeMax(LocalTensor<float> &inLocal,
                                                                LocalTensor<float> &scaleLocal, int32_t srcIdx,
                                                                int32_t expertIdx, int64_t j)
{
    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();

    DataCopyExtParams intriParamsT{1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams intriParamsFp32{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};

    if constexpr (!IsSameType<T, float>::value) {
        DataCopyPad(inLocal.ReinterpretCast<T>()[perLoopColsAlign_], inputXGm_[srcIdx * cols_ + j * perLoopCols_],
                    intriParamsT, {false, 0, 0, 0});
    } else {
        DataCopyPad(inLocal, inputXGm_[srcIdx * cols_ + j * perLoopCols_], intriParamsT, {false, 0, 0, 0});
    }

    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();

    if constexpr (IS_INPUT_SCALE) {
        DataCopyPad(smoothLocal, quantSmoothGm_[expertIdx * cols_ + j * perLoopCols_], intriParamsFp32,
                    {false, 0, 0, 0});
        smoothInQueue_.EnQue(smoothLocal);
        smoothLocal = smoothInQueue_.DeQue<float>();
    }

    __local_mem__ float *inUbAddr = (__local_mem__ float *)inLocal.GetPhyAddr();
    __local_mem__ float *scaleUbAddr = (__local_mem__ float *)scaleLocal.GetPhyAddr();
    __local_mem__ T *inUbAddrCastT;
    if constexpr (!IsSameType<T, float>::value) {
        inUbAddrCastT = (__local_mem__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + perLoopColsAlign_;
    }
    __local_mem__ float *smoothUbAddr;
    if constexpr (IS_INPUT_SCALE) {
        smoothUbAddr = (__local_mem__ float *)smoothLocal.GetPhyAddr();
    }

    uint16_t repeatTimes = Ceil(colsTileLength_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, scaleReg, smoothReg;
        MicroAPI::Duplicate(scaleReg, 0.0f);

        MicroAPI::MaskReg maskRegLoop;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegVL2 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL2>();

        sreg = static_cast<uint32_t>(colsTileLength_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
            if constexpr (!IsSameType<T, float>::value) {
                ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegLoop, i * FLOAT_REG_TENSOR_LENGTH);
            } else {
                MicroAPI::DataCopy(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            }
            if constexpr (IS_INPUT_SCALE) {
                MicroAPI::DataCopy(smoothReg, smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
                MicroAPI::Mul(inReg, inReg, smoothReg, maskRegLoop);
            }
            MicroAPI::DataCopy(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegLoop);
            MicroAPI::Abs(inReg, inReg, maskRegLoop);
            MicroAPI::Max(scaleReg, scaleReg, inReg, maskRegAll);
        }
        MicroAPI::ReduceMax(scaleReg, scaleReg, maskRegAll);
        MicroAPI::DataCopy(scaleUbAddr + 8, scaleReg, maskRegVL2);
    }

    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyPad(quantTempGm_[j * perLoopCols_], inLocal, intriParamsFp32);

    smoothInQueue_.FreeTensor(smoothLocal);

    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    return scaleLocal.GetValue(8);
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::ComputeScale(LocalTensor<float> &inLocal, float scaleTemp,
                                                                 int64_t dstIndex, int64_t j)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(int8_t)), 0, 0, 0};

    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();

    DataCopyPad(inLocal, quantTempGm_[j * perLoopCols_], copyInParams, {false, 0, 0, 0});
    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();

    __local_mem__ float *inUbAddr = (__local_mem__ float *)inLocal.GetPhyAddr();
    __local_mem__ int8_t *outUbAddr = (__local_mem__ int8_t *)outLocal.GetPhyAddr();

    uint16_t repeatTimes = Ceil(colsTileLength_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, tempReg;
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;

        MicroAPI::MaskReg maskRegLoop;

        sreg = static_cast<uint32_t>(colsTileLength_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::Duplicate(tempReg, scaleTemp, maskRegLoop);
            MicroAPI::DataCopy(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Div(tempReg, inReg, tempReg, maskRegLoop);
            MicroAPI::Cast<half, float, castTraitF32ToF16>(outRegF16, tempReg, maskRegLoop);
            MicroAPI::Cast<int8_t, half, castTraitF16ToI8>(outRegI8, outRegF16, maskRegLoop);
            MicroAPI::DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                            outRegI8, maskRegLoop);
        }
    }

    inputXOutQueue_.EnQue(outLocal);
    outLocal = inputXOutQueue_.DeQue<int8_t>();
    DataCopyPad(expandedXGm_[dstIndex * cols_ + j * perLoopCols_], outLocal, copyOutParams);

    inputXOutQueue_.FreeTensor(outLocal);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::CopyOutPartialXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    for (int64_t i = 0; i < currentLoopRows_; i++) {
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i) - expertStart_;

        LocalTensor<float> inLocal = inputXInQueue_.AllocTensor<float>();
        LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

        uint32_t tmp = 0xFF7FFFFF;
        float reduceMax = *((float *)&tmp);
        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            float tileMax;
            if (isInputScale_) {
                tileMax = ComputeMax<true>(inLocal, scaleLocal, srcIdx / k_, expertIdx, j);
            } else {
                tileMax = ComputeMax<false>(inLocal, scaleLocal, srcIdx / k_, expertIdx, j);
            }
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / 127.0f;
        Duplicate<float>(scaleLocal, scaleTemp, 8);
        scaleOutQueue_.EnQue(scaleLocal);
        scaleLocal = scaleOutQueue_.DeQue<float>();

        DataCopyPad(expandedScaleGm_[(rowOffset + i)], scaleLocal, {1, 4, 0, 0, 0});

        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            ComputeScale(inLocal, scaleTemp, rowOffset + i, j);
        }
        inputXInQueue_.FreeTensor(inLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR sortedExpertIdx,
                                                         GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                         GM_ADDR expandedScale,
                                                         const MoeInitRoutingV3TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    totalLength_ = n_ * k_;
    isInputScale_ = tilingData->isInputScale;
    expertStart_ = tilingData->expertStart;
    rowIdxType_ = tilingData->rowIdxType;

    // core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        1);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        expertTotalCountGm_);

    int64_t expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreRow_;

    // inner core split
    int64_t originPerLoopElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    } else {
        coreRows_ = perCoreRow_;
        originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    // cols split
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;

    perLoopColsAlign_ = Align(perLoopCols_, sizeof(T));

    inputXGm_.SetGlobalBuffer((__gm__ T *)inputX);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + blockIdx_ * perCoreRow_,
                                         Align(coreRows_, sizeof(int32_t)));

    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreRow_,
                                          Align(perCoreRow_, sizeof(int32_t)));
    } else {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + Align(n_ * k_, sizeof(int32_t)) +
                                              blockIdx_ * perCoreRow_,
                                          Align(perCoreRow_, sizeof(int32_t)));
    }

    if (isInputScale_) {
        quantSmoothGm_.SetGlobalBuffer((__gm__ float *)quantSmooth);
    }
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);

    if (colLoops_ > 1) {
        // cols非全载 smooth*x结果临时存储
        quantTempGm_.SetGlobalBuffer((__gm__ float *)sortedExpertIdx + Align(totalLength_, sizeof(int32_t)) * 2 +
                                         Align(actualExpertNum_, sizeof(int32_t)) + Align(1, sizeof(int32_t)) +
                                         blockIdx_ * cols_,
                                     cols_ * sizeof(float));
    }

    currentLoopRowsAlign_ = Align(perLoopRows_, sizeof(int32_t));

    int64_t perLoopColsAlignBytes = AlignBytes(perLoopCols_, sizeof(T));
    perLoopColsAlignBytes = Max(static_cast<int64_t>(perLoopColsAlignBytes * sizeof(float) / sizeof(T)),
                                static_cast<int64_t>(BLOCK_BYTES + BLOCK_BYTES));
    pipe_->InitBuffer(inputXInQueue_, GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM,
                      perLoopColsAlignBytes); //? percols * 2  * 4
    pipe_->InitBuffer(expandRowIdxInQueue_, GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM,
                      2 * AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(smoothInQueue_, GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM,
                      AlignBytes(perLoopCols_, sizeof(float)));                      // percols *  2 * 4
    pipe_->InitBuffer(calcQueue_, 1, AlignBytes(perLoopCols_, sizeof(float)));       // percols * 1 * 4
    pipe_->InitBuffer(inputXOutQueue_, 1, AlignBytes(perLoopCols_, sizeof(int8_t))); // percols * 1
    pipe_->InitBuffer(scaleOutQueue_, 1, BLOCK_BYTES + BLOCK_BYTES);                 // 32 + 32
}

template <typename T>
__aicore__ inline void MoeGatherOutDynamicQuant<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        if (colLoops_ > 1) {
            // 一行无法全载，需要workspace
            for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
                CopyInExpandedExpertIdx(loop);
                CopyOutPartialXQuantEH(loop);
            }
            currentLoopRows_ = lastLoopRows_;
            CopyInExpandedExpertIdx(rowLoops_ - 1);
            CopyOutPartialXQuantEH(rowLoops_ - 1);
        } else {
            // 一行可以全载
            for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
                CopyInExpandedExpertIdx(loop);
                CopyOutXQuantEH(loop);
            }
            currentLoopRows_ = lastLoopRows_;
            CopyInExpandedExpertIdx(rowLoops_ - 1);
            CopyOutXQuantEH(rowLoops_ - 1);
        }
    }
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_DYNAMIC_QUANT_H_REGBASE