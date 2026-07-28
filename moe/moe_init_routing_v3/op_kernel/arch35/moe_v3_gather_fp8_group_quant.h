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
 * \file moe_v3_gather_fp8_group_quant.h
 * \brief FP8 PerGroup dynamic quantization kernel. GroupSize=128, scale dtype=FP32.
 */

#ifndef MOE_V3_GATHER_FP8_GROUP_QUANT_H
#define MOE_V3_GATHER_FP8_GROUP_QUANT_H

#include "moe_v3_common.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr int64_t FP8_GROUP_QUANT_SIZE = 128;
constexpr float FP8_GROUP_E4M3_MAX_VALUE = 448.0f;
constexpr float FP8_GROUP_E5M2_MAX_VALUE = 57344.0f;
constexpr uint32_t FP32_MANTISSA_MASK = 0x007fffffU;
constexpr uint32_t FP32_EXPONENT_MASK = 0x000000ffU;
constexpr int16_t FP32_EXPONENT_SHIFT = 23;

constexpr CastTrait CAST_TRAIT_B16_TO_F32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};
constexpr CastTrait CAST_TRAIT_F32_TO_FP8 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                             RoundMode::CAST_RINT};

template <typename T>
__simd_callee__ inline void LoadFp8GroupInput(RegTensor<float> &dst, __local_mem__ T *src, MaskReg mask,
                                              uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst, src + offset);
    } else {
        RegTensor<T> tmp;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(tmp, src + offset);
        Cast<float, T, CAST_TRAIT_B16_TO_F32>(dst, tmp, mask);
    }
}

template <typename T, typename U, bool CLAMP_AMAX>
__simd_vf__ inline void ComputeFp8GroupQuantVF(__ubuf__ T *xAddr, __ubuf__ float *scaleAddr, __ubuf__ U *yAddr,
                                               uint32_t groupElemNum, float fp8MaxValue)
{
    const uint16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t inputElemNum = groupElemNum;
    uint32_t outputElemNum = groupElemNum;

    static constexpr DivSpecificMode divMode = {MaskMergeMode::ZEROING, false};

    __VEC_SCOPE__
    {
        RegTensor<float> xLeftReg;
        RegTensor<float> xRightReg;
        RegTensor<float> absLeftReg;
        RegTensor<float> absRightReg;
        RegTensor<float> maxReg;
        RegTensor<float> maxRightReg;
        RegTensor<float> fp8MaxReg;
        RegTensor<float> rawScaleReg;
        RegTensor<float> roundScaleReg;
        RegTensor<float> invScaleReg;
        RegTensor<float> quantLeftReg;
        RegTensor<float> quantRightReg;
        RegTensor<U> outLeftReg;
        RegTensor<U> outRightReg;

        RegTensor<uint32_t> expBitsReg;
        RegTensor<uint32_t> mantBitsReg;
        RegTensor<uint32_t> expMaskReg;
        RegTensor<uint32_t> mantMaskReg;
        RegTensor<uint32_t> oneIntReg;
        RegTensor<uint32_t> zeroIntReg;
        RegTensor<uint32_t> mantAddReg;
        RegTensor<uint32_t> roundedExpBitsReg;

        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskAllUint = CreateMask<uint32_t, MaskPattern::ALL>();
        MaskReg maskLoop;
        MaskReg maskRight;
        MaskReg maskScalar = CreateMask<float, MaskPattern::VL1>();
        MaskReg maskMantNonZero;
        MaskReg maskExpZero;

        Duplicate(maxReg, 0.0f, maskAll);
        Duplicate(fp8MaxReg, fp8MaxValue, maskAll);
        Duplicate(expMaskReg, FP32_EXPONENT_MASK, maskAllUint);
        Duplicate(mantMaskReg, FP32_MANTISSA_MASK, maskAllUint);
        Duplicate(oneIntReg, 1U, maskAllUint);
        Duplicate(zeroIntReg, 0U, maskAllUint);

        if (inputElemNum > vfLen) {
            uint32_t rightElemNum = inputElemNum - vfLen;
            maskRight = UpdateMask<float>(rightElemNum);
            LoadFp8GroupInput<T>(xLeftReg, xAddr, maskAll, 0);
            LoadFp8GroupInput<T>(xRightReg, xAddr, maskRight, vfLen);
            Abs(absLeftReg, xLeftReg, maskAll);
            Abs(absRightReg, xRightReg, maskRight);
            ReduceMax(maxReg, absLeftReg, maskAll);
            ReduceMax(maxRightReg, absRightReg, maskRight);
            Max(maxReg, maxReg, maxRightReg, maskScalar);
        } else if (inputElemNum > 0) {
            maskLoop = UpdateMask<float>(inputElemNum);
            LoadFp8GroupInput<T>(xLeftReg, xAddr, maskLoop, 0);
            Abs(absLeftReg, xLeftReg, maskLoop);
            ReduceMax(maxReg, absLeftReg, maskLoop);
        }
        Duplicate(maxReg, maxReg, maskAll);
        if constexpr (CLAMP_AMAX) {
            Maxs(maxReg, maxReg, 0.0001f, maskAll);
        }

        Div<float, &divMode>(rawScaleReg, maxReg, fp8MaxReg, maskAll);
        ShiftRights(expBitsReg, (RegTensor<uint32_t> &)rawScaleReg, FP32_EXPONENT_SHIFT, maskAllUint);
        And(expBitsReg, expBitsReg, expMaskReg, maskAllUint);
        And(mantBitsReg, (RegTensor<uint32_t> &)rawScaleReg, mantMaskReg, maskAllUint);
        Compare<uint32_t, CMPMODE::NE>(maskMantNonZero, mantBitsReg, zeroIntReg, maskAllUint);
        Select(mantAddReg, oneIntReg, zeroIntReg, maskMantNonZero);
        Compare<uint32_t, CMPMODE::EQ>(maskExpZero, expBitsReg, zeroIntReg, maskAllUint);
        Select(mantAddReg, oneIntReg, mantAddReg, maskExpZero);
        Add(roundedExpBitsReg, expBitsReg, mantAddReg, maskAllUint);
        ShiftLefts((RegTensor<uint32_t> &)roundScaleReg, roundedExpBitsReg, FP32_EXPONENT_SHIFT, maskAllUint);

        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr, roundScaleReg, maskAll);
        Duplicate(fp8MaxReg, 1.0f, maskAll);
        Div<float, &divMode>(invScaleReg, fp8MaxReg, roundScaleReg, maskAll);

        if (outputElemNum > vfLen) {
            Mul(quantLeftReg, xLeftReg, invScaleReg, maskAll);
            Mul(quantRightReg, xRightReg, invScaleReg, maskRight);
            Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outLeftReg, quantLeftReg, maskAll);
            Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outRightReg, quantRightReg, maskRight);
            DataCopy<U, StoreDist::DIST_PACK4_B32>(yAddr, outLeftReg, maskAll);
            DataCopy<U, StoreDist::DIST_PACK4_B32>(yAddr + vfLen, outRightReg, maskRight);
        } else if (outputElemNum > 0) {
            maskLoop = UpdateMask<float>(outputElemNum);
            Mul(quantLeftReg, xLeftReg, invScaleReg, maskLoop);
            Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outLeftReg, quantLeftReg, maskLoop);
            DataCopy<U, StoreDist::DIST_PACK4_B32>(yAddr, outLeftReg, maskLoop);
        }
    }
}

template <typename T, typename U, bool CLAMP_AMAX>
class MoeV3GatherFP8GroupQuant {
public:
    __aicore__ inline MoeV3GatherFP8GroupQuant(){};

    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR unusedScaleAddr, GM_ADDR sortedExpertIdxAddr,
                                GM_ADDR expandedRowIdxAddr, GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKernelTiling(GM_ADDR sortedExpertIdxAddr,
                                            const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyExpandedXAndQuant(int64_t progress);
    __aicore__ inline void CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);
    __aicore__ inline void Compute(int64_t loopCols, int64_t loopScaleCols);
    __aicore__ inline void CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols, int64_t loopScaleCols);

private:
    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> sortedRowIdxInQueue_;

    GlobalTensor<T> xInGm_;
    GlobalTensor<U> expandedXOutGm_;
    GlobalTensor<float> expandedScaleOutGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_{nullptr};

    int64_t needCoreNum_{0};
    int64_t blockIdx_{0};
    int64_t cols_{0};
    int64_t scaleCols_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t perCoreRow_{0};
    int64_t currentLoopRows_{0};
    int64_t coreRows_{0};
    int64_t perLoopRows_{0};
    int64_t lastLoopRows_{0};
    int64_t rowLoops_{0};
    int64_t perLoopCols_{0};
    int64_t lastLoopCols_{0};
    int64_t colLoops_{0};
    int64_t perLoopScaleCols_{0};
    int64_t lastLoopScaleCols_{0};
    int64_t indicesOffset_{0};
    int64_t rowIdxType_{0};

    float fp8MaxValue_{FP8_GROUP_E5M2_MAX_VALUE};
};

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::Init(
    GM_ADDR xAddr, GM_ADDR unusedScaleAddr, GM_ADDR sortedExpertIdxAddr, GM_ADDR expandedRowIdxAddr,
    GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    (void)unusedScaleAddr;
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(sortedExpertIdxAddr, tilingData);

    xInGm_.SetGlobalBuffer((__gm__ T *)xAddr);
    expandedXOutGm_.SetGlobalBuffer((__gm__ U *)expandedXAddr);
    expandedScaleOutGm_.SetGlobalBuffer((__gm__ float *)expandedScaleAddr);

    if (rowIdxType_ == SCATTER) {
        sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                        Align(perCoreRow_, sizeof(int32_t)));
    } else {
        sortedRowIdxGm_.SetGlobalBuffer(
            (__gm__ int32_t *)sortedExpertIdxAddr + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
            Align(perCoreRow_, sizeof(int32_t)));
    }

    InitBuffer();
    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        fp8MaxValue_ = FP8_GROUP_E4M3_MAX_VALUE;
    } else {
        fp8MaxValue_ = FP8_GROUP_E5M2_MAX_VALUE;
    }
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::InitKernelTiling(
    GM_ADDR sortedExpertIdxAddr, const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;

    scaleCols_ = Ceil(cols_, FP8_GROUP_QUANT_SIZE);

    int64_t actualExpertNum = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdxAddr + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum, sizeof(int32_t)),
                                        actualExpertNum);

    int64_t expertTotalCount = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalCount, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount - (needCoreNum_ - 1) * perCoreRow_;

    coreRows_ = perCoreRow_;
    int64_t originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    }

    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;
    perLoopScaleCols_ = perLoopCols_ / FP8_GROUP_QUANT_SIZE;
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::InitBuffer()
{
    pipe_->InitBuffer(inQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(outQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(U)));
    pipe_->InitBuffer(scaleQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopScaleCols_, sizeof(float)));
    pipe_->InitBuffer(sortedRowIdxInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopRows_, sizeof(int32_t)));
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::Process()
{
    if (blockIdx_ >= needCoreNum_) {
        return;
    }
    currentLoopRows_ = perLoopRows_;
    for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
        CopyInExpandedExpertIdx(loop);
        CopyExpandedXAndQuant(loop);
    }
    currentLoopRows_ = lastLoopRows_;
    CopyInExpandedExpertIdx(rowLoops_ - 1);
    CopyExpandedXAndQuant(rowLoops_ - 1);
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, padParams);
    sortedRowIdxInQueue_.EnQue(indicesLocal);
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::CopyExpandedXAndQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t index = 0; index < currentLoopRows_; index++) {
        int32_t srcIdx = indicesLocal.GetValue(index);
        int64_t dstIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress + index;
        for (int64_t j = 0; j < colLoops_; j++) {
            int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
            int64_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;
            CopyIn(srcIdx / k_, j, loopCols);
            Compute(loopCols, loopScaleCols);
            CopyOut(dstIdx, j, loopCols, loopScaleCols);
        }
    }
    sortedRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::CopyIn(int64_t srcIdx, int64_t colIdx,
                                                                          int64_t loopCols)
{
    LocalTensor<T> inLocal = inQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(loopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    DataCopyPad<T, PaddingMode::Compact>(inLocal, xInGm_[srcIdx * cols_ + colIdx * perLoopCols_], copyParams,
                                         padParams);
    inQueue_.EnQue(inLocal);
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::Compute(int64_t loopCols, int64_t loopScaleCols)
{
    LocalTensor<T> xLocal = inQueue_.DeQue<T>();
    __ubuf__ T *xAddr = reinterpret_cast<__ubuf__ T *>(xLocal.GetPhyAddr());

    LocalTensor<U> outLocal = outQueue_.AllocTensor<U>();
    __ubuf__ U *outAddr = reinterpret_cast<__ubuf__ U *>(outLocal.GetPhyAddr());

    LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
    __ubuf__ float *scaleAddr = reinterpret_cast<__ubuf__ float *>(scaleLocal.GetPhyAddr());

    for (int64_t groupIdx = 0; groupIdx < loopScaleCols; groupIdx++) {
        int64_t groupOffset = groupIdx * FP8_GROUP_QUANT_SIZE;
        int64_t groupElemNum = (groupIdx == loopScaleCols - 1) ? (loopCols - groupOffset) : FP8_GROUP_QUANT_SIZE;
        VF_CALL<ComputeFp8GroupQuantVF<T, U, CLAMP_AMAX>>(xAddr + groupOffset, scaleAddr + groupIdx,
                                                          outAddr + groupOffset, static_cast<uint32_t>(groupElemNum),
                                                          fp8MaxValue_);
    }

    inQueue_.FreeTensor(xLocal);
    outQueue_.EnQue(outLocal);
    scaleQueue_.EnQue(scaleLocal);
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::CopyOut(int64_t dstIdx, int64_t colIdx,
                                                                           int64_t loopCols, int64_t loopScaleCols)
{
    LocalTensor<U> outLocal = outQueue_.DeQue<U>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(loopCols * sizeof(U)), 0, 0, 0};
    DataCopyPad<U>(expandedXOutGm_[dstIdx * cols_ + colIdx * perLoopCols_], outLocal, copyOutParams);

    LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();
    DataCopyExtParams copyScaleParams{1, static_cast<uint32_t>(loopScaleCols * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(expandedScaleOutGm_[dstIdx * scaleCols_ + colIdx * perLoopScaleCols_], scaleLocal,
                       copyScaleParams);

    outQueue_.FreeTensor(outLocal);
    scaleQueue_.FreeTensor(scaleLocal);
}

} // namespace MoeInitRoutingV3

#endif // MOE_V3_GATHER_FP8_GROUP_QUANT_H
