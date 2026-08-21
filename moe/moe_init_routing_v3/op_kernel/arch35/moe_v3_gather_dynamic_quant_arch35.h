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
 * \file moe_v3_gather_dynamic_quant_arch35.h
 * \brief
 */
#ifndef MOE_V3_GATHER_DYNAMIC_QUANT_H_REGBASE_ARCH35
#define MOE_V3_GATHER_DYNAMIC_QUANT_H_REGBASE_ARCH35

#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;
constexpr int64_t GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM = 1;
constexpr int64_t GATHER_OUT_DYNAMIC_QUANT_INPUT_X_QUEUE_NUM = 3;
constexpr int64_t GATHER_OUT_DYNAMIC_QUANT_INPUT_QUEUE_NUM = 1;
constexpr int64_t GATHER_OUT_DYNAMIC_QUANT_OUTPUT_QUEUE_NUM = 1;

template <typename T, typename QuantT = int8_t>
class MoeGatherOutDynamicQuant {
public:
    __aicore__ inline MoeGatherOutDynamicQuant(){};
    __aicore__ inline void InitBaseData(GM_ADDR sortedExpertIdx, const MoeInitRoutingV3Arch35TilingData *tilingData,
                                        TPipe *tPipe);
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR expandedScale, GM_ADDR sortedExpertIdx,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void StoreInt4QuantOut(__ubuf__ uint8_t *outUbAddr, MicroAPI::RegTensor<float> &inReg,
                                             MicroAPI::MaskReg &maskRegInLoop, MicroAPI::MaskReg &maskRegHalf);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyOutXQuantEH(int64_t progress);
    template <bool IS_INPUT_SCALE>
    __aicore__ inline void Compute(LocalTensor<float> &smoothLocal);
    template <bool IS_INPUT_SCALE>
    __aicore__ inline void ComputeInt4SinglePass(LocalTensor<float> &smoothLocal);
    template <bool IS_INPUT_SCALE>
    __aicore__ inline void ComputeMultiPass(LocalTensor<float> &smoothLocal);
    __aicore__ inline void CopyOutPartialXQuantEH(int64_t progress);
    template <bool IS_INPUT_SCALE>
    __aicore__ inline float ComputeMax(LocalTensor<float> &inLocal, LocalTensor<float> &scaleLocal, int32_t srcIdx,
                                       int32_t expertIdx, int64_t j);
    __aicore__ inline void QuantizeToInt4(LocalTensor<float> &inLocal, LocalTensor<int8_t> &outLocal,
                                          float quantFactor);
    __aicore__ inline void QuantizeToInt8(LocalTensor<float> &inLocal, LocalTensor<int8_t> &outLocal, float scaleTemp);
    __aicore__ inline void ComputeScale(LocalTensor<float> &inLocal, float scaleTemp, int64_t dstIndex, int64_t j);
    __aicore__ inline void SetColTileParams(int64_t j);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, GATHER_OUT_DYNAMIC_QUANT_INPUT_X_QUEUE_NUM> inputXInQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_DYNAMIC_QUANT_INPUT_QUEUE_NUM> smoothInQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_DYNAMIC_QUANT_INPUT_QUEUE_NUM> expandRowIdxInQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_DYNAMIC_QUANT_OUTPUT_QUEUE_NUM> inputXOutQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_DYNAMIC_QUANT_OUTPUT_QUEUE_NUM> scaleOutQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> quantSmoothGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<float> quantTempGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

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
    int64_t colsAsInt8_;
    int64_t perLoopColsAsInt8_;
    int64_t colsTileLengthAsInt8_;

    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;

    TBuf<AscendC::TPosition::VECCALC> tempCalcBuf_;

    constexpr static MicroAPI::CastTrait castTraitF32ToF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
    constexpr static MicroAPI::CastTrait castTraitF16ToI8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF16ToI4 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::StoreInt4QuantOut(__ubuf__ uint8_t *outUbAddr,
                                                                              MicroAPI::RegTensor<float> &inReg,
                                                                              MicroAPI::MaskReg &maskRegInLoop,
                                                                              MicroAPI::MaskReg &maskRegHalf)
{
    MicroAPI::RegTensor<half> outRegF16;
    MicroAPI::RegTensor<uint8_t> outRegI4;
    MicroAPI::RegTensor<uint16_t> outRegPackedHalf;

    MicroAPI::Cast<half, float, castTraitF32ToF16>(outRegF16, inReg, maskRegInLoop);
    MicroAPI::Pack(outRegPackedHalf, (MicroAPI::RegTensor<uint32_t> &)outRegF16);
    MicroAPI::Cast<int4x2_t, half, castTraitF16ToI4>((MicroAPI::RegTensor<int4x2_t> &)outRegI4,
                                                     (MicroAPI::RegTensor<half> &)outRegPackedHalf, maskRegInLoop);
    MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr, outRegI4, maskRegHalf);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    if constexpr (!IsSameType<QuantT, int4b_t>::value) {
        DataCopyPad(indicesLocal[currentLoopRowsAlign_], expandedExpertIdxGm_[indicesOffset_], dataCopyParams,
                    dataCopyPadParams);
    }
    expandRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T, typename QuantT>
template <bool IS_INPUT_SCALE>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::Compute(LocalTensor<float> &smoothLocal)
{
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        if (cols_ <= FLOAT_REG_TENSOR_LENGTH) {
            ComputeInt4SinglePass<IS_INPUT_SCALE>(smoothLocal);
            return;
        }
    }
    ComputeMultiPass<IS_INPUT_SCALE>(smoothLocal);
}

template <typename T, typename QuantT>
template <bool IS_INPUT_SCALE>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::ComputeInt4SinglePass(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

    __ubuf__ float *inUbAddr = (__ubuf__ float *)inLocal.GetPhyAddr();
    __ubuf__ float *scaleUbAddr = (__ubuf__ float *)scaleLocal.GetPhyAddr();
    __ubuf__ int8_t *outUbAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
    __ubuf__ T *inUbAddrCastT;
    if constexpr (!IsSameType<T, float>::value) {
        inUbAddrCastT = (__ubuf__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + perLoopColsAlign_;
    }
    __ubuf__ float *smoothUbAddr;
    if constexpr (IS_INPUT_SCALE) {
        smoothUbAddr = (__ubuf__ float *)smoothLocal.GetPhyAddr();
    }

    uint32_t sreg = static_cast<uint32_t>(cols_);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, absReg, smoothReg, scaleValueReg, quantFactorReg, zeroReg;
        MicroAPI::Duplicate(scaleValueReg, 0.0f);
        MicroAPI::Duplicate(absReg, 0.0f);

        MicroAPI::MaskReg maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegHalf = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
        MicroAPI::MaskReg maskRegVL1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        MicroAPI::MaskReg maskRegScaleZero;

        if constexpr (!IsSameType<T, float>::value) {
            ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegInLoop, 0);
        } else {
            MicroAPI::LoadAlign(inReg, inUbAddr);
        }
        if constexpr (IS_INPUT_SCALE) {
            MicroAPI::LoadAlign(smoothReg, smoothUbAddr);
            MicroAPI::Mul(inReg, inReg, smoothReg, maskRegInLoop);
        }
        MicroAPI::Abs(absReg, inReg, maskRegInLoop);
        MicroAPI::Max(scaleValueReg, scaleValueReg, absReg, maskRegAll);
        Reg::Reduce<Reg::ReduceType::MAX>(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::Duplicate(quantFactorReg, DYNAMIC_QUANT_INT4_SYM_SCALE, maskRegVL1);
        MicroAPI::Div(quantFactorReg, quantFactorReg, scaleValueReg, maskRegVL1);
        MicroAPI::Duplicate(zeroReg, 0.0f, maskRegVL1);
        MicroAPI::Compares<float, CMPMODE::EQ>(maskRegScaleZero, scaleValueReg, 0.0f, maskRegVL1);
        MicroAPI::Select(quantFactorReg, zeroReg, quantFactorReg, maskRegScaleZero);
        MicroAPI::Duplicate(quantFactorReg, quantFactorReg, maskRegAll);

        MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / DYNAMIC_QUANT_INT4_SYM_SCALE, maskRegVL1);
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::StoreAlign(scaleUbAddr, scaleValueReg, maskRegVL1);

        MicroAPI::Mul(inReg, inReg, quantFactorReg, maskRegInLoop);
        StoreInt4QuantOut((__ubuf__ uint8_t *)outUbAddr, inReg, maskRegInLoop, maskRegHalf);
    }

    inputXOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(scaleLocal);
}

template <typename T, typename QuantT>
template <bool IS_INPUT_SCALE>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::ComputeMultiPass(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

    __ubuf__ float *inUbAddr = (__ubuf__ float *)inLocal.GetPhyAddr();
    __ubuf__ float *scaleUbAddr = (__ubuf__ float *)scaleLocal.GetPhyAddr();
    __ubuf__ int8_t *outUbAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
    __ubuf__ T *inUbAddrCastT;
    if constexpr (!IsSameType<T, float>::value) {
        inUbAddrCastT = (__ubuf__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + perLoopColsAlign_;
    }
    __ubuf__ float *smoothUbAddr;
    if constexpr (IS_INPUT_SCALE) {
        smoothUbAddr = (__ubuf__ float *)smoothLocal.GetPhyAddr();
    }

    uint16_t repeatTimes = Ceil(cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, smoothReg, scaleValueReg, quantFactorReg, zeroReg;
        MicroAPI::Duplicate(scaleValueReg, 0.0f);
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;

        MicroAPI::MaskReg maskRegInLoop;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegHalf = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
        MicroAPI::MaskReg maskRegVL1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        MicroAPI::MaskReg maskRegVL8 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL8>();
        MicroAPI::MaskReg maskRegScaleZero;

        sreg = static_cast<uint32_t>(cols_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            if constexpr (!IsSameType<T, float>::value) {
                ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH);
            } else {
                MicroAPI::LoadAlign(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            }
            if constexpr (IS_INPUT_SCALE) {
                MicroAPI::LoadAlign(smoothReg, smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
                MicroAPI::Mul(inReg, inReg, smoothReg, maskRegInLoop);
            }
            if constexpr (!IsSameType<T, float>::value || IS_INPUT_SCALE) {
                MicroAPI::StoreAlign(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegInLoop);
            }
            MicroAPI::Abs(inReg, inReg, maskRegInLoop);
            MicroAPI::Max(scaleValueReg, scaleValueReg, inReg, maskRegAll);
        }
        Reg::Reduce<Reg::ReduceType::MAX>(scaleValueReg, scaleValueReg, maskRegAll);
        if constexpr (!IsSameType<QuantT, int4b_t>::value) {
            MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / 127.0f, maskRegVL1);
        }
        if constexpr (IsSameType<QuantT, int4b_t>::value) {
            MicroAPI::Duplicate(quantFactorReg, DYNAMIC_QUANT_INT4_SYM_SCALE, maskRegVL1);
            MicroAPI::Div(quantFactorReg, quantFactorReg, scaleValueReg, maskRegVL1);
            MicroAPI::Duplicate(zeroReg, 0.0f, maskRegVL1);
            MicroAPI::Compares<float, CMPMODE::EQ>(maskRegScaleZero, scaleValueReg, 0.0f, maskRegVL1);
            MicroAPI::Select(quantFactorReg, zeroReg, quantFactorReg, maskRegScaleZero);
            MicroAPI::Duplicate(quantFactorReg, quantFactorReg, maskRegAll);
            MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / DYNAMIC_QUANT_INT4_SYM_SCALE, maskRegVL1);
        }
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);
        MicroAPI::StoreAlign(scaleUbAddr, scaleValueReg, maskRegVL8);

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        sreg = static_cast<uint32_t>(cols_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::LoadAlign(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            if constexpr (IsSameType<QuantT, int4b_t>::value) {
                MicroAPI::Mul(inReg, inReg, quantFactorReg, maskRegInLoop);
                StoreInt4QuantOut((__ubuf__ uint8_t *)outUbAddr + i * FLOAT_REG_TENSOR_LENGTH / 2, inReg, maskRegInLoop,
                                  maskRegHalf);
            } else {
                MicroAPI::Div(inReg, inReg, scaleValueReg, maskRegInLoop);
                MicroAPI::Cast<half, float, castTraitF32ToF16>(outRegF16, inReg, maskRegInLoop);
                MicroAPI::Cast<int8_t, half, castTraitF16ToI8>(outRegI8, outRegF16, maskRegInLoop);
                MicroAPI::StoreAlign<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outUbAddr + i * FLOAT_REG_TENSOR_LENGTH, outRegI8, maskRegInLoop);
            }
        }
    }

    inputXOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(scaleLocal);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::CopyOutXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(perLoopColsAsInt8_), 0, 0, 0};

    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    int32_t lastExpertIdx = -1;
    for (int64_t i = 0; i < currentLoopRows_; i++) {
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = 0;
        if constexpr (!IsSameType<QuantT, int4b_t>::value) {
            expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i) - expertStart_;
        }
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, inputXGm_[srcIdx / k_ * cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[perLoopColsAlign_], inputXGm_[srcIdx / k_ * cols_], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);

        if (isInputScale_) {
            if constexpr (IsSameType<QuantT, int4b_t>::value) {
                if (lastExpertIdx == -1) {
                    DataCopyPad(smoothLocal, quantSmoothGm_[0], smoothParams, {false, 0, 0, 0});
                    smoothInQueue_.EnQue(smoothLocal);
                    smoothLocal = smoothInQueue_.DeQue<float>();
                    lastExpertIdx = 0;
                }
            } else {
                if (expertIdx != lastExpertIdx) {
                    int64_t smoothOffset = expertIdx * cols_;
                    DataCopyPad(smoothLocal, quantSmoothGm_[smoothOffset], smoothParams, {false, 0, 0, 0});
                    smoothInQueue_.EnQue(smoothLocal);
                    smoothLocal = smoothInQueue_.DeQue<float>();
                    lastExpertIdx = expertIdx;
                }
            }
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
        DataCopyPad(expandedXGm_[(rowOffset + i) * colsAsInt8_], outLocal, copyOutParams);

        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }

    smoothInQueue_.FreeTensor(smoothLocal);
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T, typename QuantT>
template <bool IS_INPUT_SCALE>
__aicore__ inline float MoeGatherOutDynamicQuant<T, QuantT>::ComputeMax(LocalTensor<float> &inLocal,
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
        DataCopyPad(smoothLocal, quantSmoothGm_[j * perLoopCols_], intriParamsFp32, {false, 0, 0, 0});
        smoothInQueue_.EnQue(smoothLocal);
        smoothLocal = smoothInQueue_.DeQue<float>();
    }

    __ubuf__ float *inUbAddr = (__ubuf__ float *)inLocal.GetPhyAddr();
    __ubuf__ float *scaleUbAddr = (__ubuf__ float *)scaleLocal.GetPhyAddr();
    __ubuf__ T *inUbAddrCastT;
    if constexpr (!IsSameType<T, float>::value) {
        inUbAddrCastT = (__ubuf__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + perLoopColsAlign_;
    }
    __ubuf__ float *smoothUbAddr;
    if constexpr (IS_INPUT_SCALE) {
        smoothUbAddr = (__ubuf__ float *)smoothLocal.GetPhyAddr();
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
                MicroAPI::LoadAlign(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            }
            if constexpr (IS_INPUT_SCALE) {
                MicroAPI::LoadAlign(smoothReg, smoothUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
                MicroAPI::Mul(inReg, inReg, smoothReg, maskRegLoop);
            }
            MicroAPI::StoreAlign(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegLoop);
            MicroAPI::Abs(inReg, inReg, maskRegLoop);
            MicroAPI::Max(scaleReg, scaleReg, inReg, maskRegAll);
        }
        Reg::Reduce<Reg::ReduceType::MAX>(scaleReg, scaleReg, maskRegAll);
        MicroAPI::StoreAlign(scaleUbAddr + 8, scaleReg, maskRegVL2);
    }

    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyPad(quantTempGm_[j * perLoopCols_], inLocal, intriParamsFp32);

    smoothInQueue_.FreeTensor(smoothLocal);

    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    return scaleLocal.GetValue(8);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::QuantizeToInt4(LocalTensor<float> &inLocal,
                                                                           LocalTensor<int8_t> &outLocal,
                                                                           float quantFactor)
{
    __ubuf__ float *inUbAddr = (__ubuf__ float *)inLocal.GetPhyAddr();
    __ubuf__ uint8_t *outUbAddr = (__ubuf__ uint8_t *)outLocal.GetPhyAddr();

    uint16_t repeatTimes = Ceil(colsTileLength_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(colsTileLength_);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, tempReg;
        MicroAPI::MaskReg maskRegLoop;
        MicroAPI::MaskReg maskRegHalf = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();

        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::Duplicate(tempReg, quantFactor, maskRegLoop);
            MicroAPI::LoadAlign(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Mul(tempReg, inReg, tempReg, maskRegLoop);
            StoreInt4QuantOut(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH / 2, tempReg, maskRegLoop, maskRegHalf);
        }
    }
}

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::SetColTileParams(int64_t colLoopindex)
{
    colsTileLength_ = perLoopCols_;
    if (colLoopindex == colLoops_ - 1) {
        colsTileLength_ = lastLoopCols_;
    }
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        colsTileLengthAsInt8_ = Ceil(colsTileLength_, static_cast<int64_t>(2));
    } else {
        colsTileLengthAsInt8_ = colsTileLength_;
    }
}

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::QuantizeToInt8(LocalTensor<float> &inLocal,
                                                                           LocalTensor<int8_t> &outLocal,
                                                                           float scaleTemp)
{
    __ubuf__ float *inUbAddr = (__ubuf__ float *)inLocal.GetPhyAddr();
    __ubuf__ int8_t *outUbAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();

    uint16_t repeatTimes = Ceil(colsTileLength_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg = static_cast<uint32_t>(colsTileLength_);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, tempReg;
        MicroAPI::RegTensor<half> outRegF16;
        MicroAPI::RegTensor<int8_t> outRegI8;
        MicroAPI::MaskReg maskRegLoop;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::Duplicate(tempReg, scaleTemp, maskRegLoop);
            MicroAPI::LoadAlign(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Div(tempReg, inReg, tempReg, maskRegLoop);
            MicroAPI::Cast<half, float, castTraitF32ToF16>(outRegF16, tempReg, maskRegLoop);
            MicroAPI::Cast<int8_t, half, castTraitF16ToI8>(outRegI8, outRegF16, maskRegLoop);
            MicroAPI::StoreAlign<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                              outRegI8, maskRegLoop);
        }
    }
}

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::ComputeScale(LocalTensor<float> &inLocal, float scaleTemp,
                                                                         int64_t dstIndex, int64_t j)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsTileLengthAsInt8_), 0, 0, 0};

    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();

    DataCopyPad(inLocal, quantTempGm_[j * perLoopCols_], copyInParams, {false, 0, 0, 0});
    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();

    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        float quantFactor = (scaleTemp != 0.0f) ? (1.0f / scaleTemp) : 0.0f;
        QuantizeToInt4(inLocal, outLocal, quantFactor);
    } else {
        QuantizeToInt8(inLocal, outLocal, scaleTemp);
    }

    inputXOutQueue_.EnQue(outLocal);
    outLocal = inputXOutQueue_.DeQue<int8_t>();
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        DataCopyPad(expandedXGm_[dstIndex * colsAsInt8_ + j * perLoopColsAsInt8_], outLocal, copyOutParams);
    } else {
        DataCopyPad(expandedXGm_[dstIndex * cols_ + j * perLoopCols_], outLocal, copyOutParams);
    }

    inputXOutQueue_.FreeTensor(outLocal);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::CopyOutPartialXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    for (int64_t i = 0; i < currentLoopRows_; i++) {
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = 0;
        if constexpr (!IsSameType<QuantT, int4b_t>::value) {
            expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i) - expertStart_;
        }

        LocalTensor<float> inLocal = inputXInQueue_.AllocTensor<float>();
        LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

        uint32_t tmp = 0xFF7FFFFF;
        float reduceMax = *((float *)&tmp);
        for (int64_t j = 0; j < colLoops_; j++) {
            SetColTileParams(j);
            float tileMax;
            if (isInputScale_) {
                tileMax = ComputeMax<true>(inLocal, scaleLocal, srcIdx / k_, expertIdx, j);
            } else {
                tileMax = ComputeMax<false>(inLocal, scaleLocal, srcIdx / k_, expertIdx, j);
            }
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp;
        float scaleOutVal = 0.0f;
        if constexpr (IsSameType<QuantT, int4b_t>::value) {
            scaleOutVal = reduceMax * (1.0f / DYNAMIC_QUANT_INT4_SYM_SCALE);
            scaleTemp = scaleOutVal;
        } else {
            scaleTemp = reduceMax / 127.0f;
        }
        if constexpr (IsSameType<QuantT, int4b_t>::value) {
            Duplicate<float>(scaleLocal, scaleOutVal, 8);
        } else {
            Duplicate<float>(scaleLocal, scaleTemp, 8);
        }
        scaleOutQueue_.EnQue(scaleLocal);
        scaleLocal = scaleOutQueue_.DeQue<float>();

        DataCopyPad(expandedScaleGm_[(rowOffset + i)], scaleLocal, {1, 4, 0, 0, 0});

        for (int64_t j = 0; j < colLoops_; j++) {
            SetColTileParams(j);
            ComputeScale(inLocal, scaleTemp, rowOffset + i, j);
        }
        inputXInQueue_.FreeTensor(inLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::InitBaseData(
    GM_ADDR sortedExpertIdx, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
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

    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        colsAsInt8_ = Ceil(cols_, static_cast<int64_t>(2));
    } else {
        colsAsInt8_ = cols_;
    }

    // core split
    int64_t actualExpertNum = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum, sizeof(int32_t)),
                                        1);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        expertTotalCountGm_);

    int64_t expertTotalCount = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalCount, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount - (needCoreNum_ - 1) * perCoreRow_;

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
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        perLoopColsAsInt8_ = Ceil(perLoopCols_, static_cast<int64_t>(2));
    } else {
        perLoopColsAsInt8_ = perLoopCols_;
    }
}

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::Init(GM_ADDR inputX, GM_ADDR quantSmooth,
                                                                 GM_ADDR sortedExpertIdx, GM_ADDR expandedRowIdx,
                                                                 GM_ADDR expandedX, GM_ADDR expandedScale,
                                                                 const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                                 TPipe *tPipe)
{
    InitBaseData(sortedExpertIdx, tilingData, tPipe);
    inputXGm_.SetGlobalBuffer((__gm__ T *)inputX);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + blockIdx_ * perCoreRow_,
                                         Align(coreRows_, sizeof(int32_t)));

    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreRow_,
                                          Align(perCoreRow_, sizeof(int32_t)));
    } else {
        expandedRowIdxGm_.SetGlobalBuffer(
            (__gm__ int32_t *)sortedExpertIdx + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
            Align(perCoreRow_, sizeof(int32_t)));
    }

    if (isInputScale_) {
        quantSmoothGm_.SetGlobalBuffer((__gm__ float *)quantSmooth);
    }
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);

    int64_t actualExpertNum = tilingData->actualExpertNum;
    if (colLoops_ > 1) {
        // cols非全载 smooth*x结果临时存储
        quantTempGm_.SetGlobalBuffer((__gm__ float *)sortedExpertIdx + Align(totalLength_, sizeof(int32_t)) * 2 +
                                         Align(actualExpertNum, sizeof(int32_t)) + Align(1, sizeof(int32_t)) +
                                         blockIdx_ * cols_,
                                     cols_ * sizeof(float));
    }

    currentLoopRowsAlign_ = Align(perLoopRows_, sizeof(int32_t));

    int64_t perLoopColsAlignBytes = AlignBytes(perLoopCols_, sizeof(T));
    perLoopColsAlignBytes = Max(static_cast<int64_t>(perLoopColsAlignBytes * sizeof(float) / sizeof(T)),
                                static_cast<int64_t>(BLOCK_BYTES + BLOCK_BYTES));
    pipe_->InitBuffer(inputXInQueue_, GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM, perLoopColsAlignBytes);
    if constexpr (IsSameType<QuantT, int4b_t>::value) {
        pipe_->InitBuffer(expandRowIdxInQueue_, GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM,
                          AlignBytes(perLoopRows_, sizeof(int32_t)));
    } else {
        pipe_->InitBuffer(expandRowIdxInQueue_, GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM,
                          2 * AlignBytes(perLoopRows_, sizeof(int32_t)));
    }
    pipe_->InitBuffer(smoothInQueue_, GATHER_OUT_DYNAMIC_QUANT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(float)));
    pipe_->InitBuffer(inputXOutQueue_, GATHER_OUT_DYNAMIC_QUANT_OUTPUT_QUEUE_NUM,
                      AlignBytes(perLoopColsAsInt8_, sizeof(int8_t)));
    pipe_->InitBuffer(scaleOutQueue_, GATHER_OUT_DYNAMIC_QUANT_OUTPUT_QUEUE_NUM, BLOCK_BYTES + BLOCK_BYTES);
}

template <typename T, typename QuantT>
__aicore__ inline void MoeGatherOutDynamicQuant<T, QuantT>::Process()
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
#endif // MOE_V3_GATHER_DYNAMIC_QUANT_H_REGBASE_ARCH35
