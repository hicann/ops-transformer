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
constexpr uint32_t FP32_MANTISSA_MASK = 0x007fffffU;
constexpr uint32_t FP32_EXPONENT_MASK = 0x000000ffU;
constexpr int16_t FP32_EXPONENT_SHIFT = 23;
constexpr uint32_t FP32_INV_EXP_SUB = 2U * FP32_BIAS;
constexpr uint32_t FP8_GROUP_FP8_MAX_MANT = 0x00600000U;
constexpr uint32_t FP8_GROUP_E4M3_EMAX = 8U;
constexpr uint32_t FP8_GROUP_E5M2_EMAX = 15U;
constexpr uint32_t FP8_GROUP_FP32_INF_BITS = 0x7F800000U;
constexpr uint32_t FP8_GROUP_FP32_NAN_BITS = 0x7FC00000U;

constexpr CastTrait CAST_TRAIT_B16_TO_F32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};
constexpr CastTrait CAST_TRAIT_F32_TO_FP8 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                             RoundMode::CAST_RINT};

template <typename T>
__simd_callee__ inline void LoadFp8GroupInput(RegTensor<float> &dst, __ubuf__ T *src, MaskReg mask, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, src + offset);
    } else {
        RegTensor<T> tmp;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tmp, src + offset);
        Cast<float, T, CAST_TRAIT_B16_TO_F32>(dst, tmp, mask);
    }
}

template <typename T, bool CLAMP_AMAX>
__simd_vf__ inline void VfComputeAmax(__ubuf__ T *xAddr, __ubuf__ float *amaxOutAddr, uint32_t totalElemNum,
                                      uint16_t groupNum)
{
    const uint16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    __VEC_SCOPE__
    {
        RegTensor<float> xLeftReg;
        RegTensor<float> xRightReg;
        RegTensor<float> absLeftReg;
        RegTensor<float> absRightReg;
        RegTensor<float> maxReg;

        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskRight;
        MaskReg maskLoop;

        uint32_t tailElemNum = totalElemNum % FP8_GROUP_QUANT_SIZE;
        uint16_t fullGroupNum = (tailElemNum == 0) ? groupNum : static_cast<uint16_t>(groupNum - 1);
        for (uint16_t groupIdx = 0; groupIdx < fullGroupNum; groupIdx++) {
            uint32_t groupOffset = groupIdx * FP8_GROUP_QUANT_SIZE;

            LoadFp8GroupInput<T>(xLeftReg, xAddr + groupOffset, maskAll, 0);
            LoadFp8GroupInput<T>(xRightReg, xAddr + groupOffset, maskAll, vfLen);
            Abs(absLeftReg, xLeftReg, maskAll);
            Abs(absRightReg, xRightReg, maskAll);
            Max(absLeftReg, absLeftReg, absRightReg, maskAll);
            Reg::Reduce<Reg::ReduceType::MAX>(maxReg, absLeftReg, maskAll);
            Duplicate(maxReg, maxReg, maskAll);
            if constexpr (CLAMP_AMAX) {
                Maxs(maxReg, maxReg, 0.0001f, maskAll);
            }

            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(amaxOutAddr + groupIdx, maxReg, maskAll);
        }

        if (tailElemNum > 0) {
            uint32_t groupOffset = fullGroupNum * FP8_GROUP_QUANT_SIZE;
            Duplicate(maxReg, 0.0f, maskAll);

            if (tailElemNum > vfLen) {
                uint32_t rightElemNum = tailElemNum - vfLen;
                maskRight = UpdateMask<float>(rightElemNum);
                LoadFp8GroupInput<T>(xLeftReg, xAddr + groupOffset, maskAll, 0);
                LoadFp8GroupInput<T>(xRightReg, xAddr + groupOffset, maskRight, vfLen);
                Abs(absLeftReg, xLeftReg, maskAll);
                Abs(absRightReg, xRightReg, maskRight);
                Max<float, MaskMergeMode::MERGING>(absLeftReg, absLeftReg, absRightReg, maskRight);
                Reg::Reduce<Reg::ReduceType::MAX>(maxReg, absLeftReg, maskAll);
            } else {
                maskLoop = UpdateMask<float>(tailElemNum);
                LoadFp8GroupInput<T>(xLeftReg, xAddr + groupOffset, maskLoop, 0);
                Abs(absLeftReg, xLeftReg, maskLoop);
                Reg::Reduce<Reg::ReduceType::MAX>(maxReg, absLeftReg, maskLoop);
            }
            Duplicate(maxReg, maxReg, maskAll);
            if constexpr (CLAMP_AMAX) {
                Maxs(maxReg, maxReg, 0.0001f, maskAll);
            }

            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(amaxOutAddr + fullGroupNum, maxReg, maskAll);
        }
    }
}

template <bool CLAMP_AMAX>
__simd_vf__ inline void VfComputeRoundScale(__ubuf__ float *amaxInAddr, __ubuf__ float *scaleOutAddr,
                                            __ubuf__ float *invScaleOutAddr, uint16_t groupNum, uint32_t fp8EmaxValue)
{
    const uint16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vfLoopNum = (groupNum + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        RegTensor<float> maxReg;
        RegTensor<float> roundScaleReg;
        RegTensor<float> invScaleReg;
        RegTensor<uint32_t> expBitsReg;
        RegTensor<uint32_t> mantBitsReg;
        RegTensor<uint32_t> expMaskReg;
        RegTensor<uint32_t> mantMaskReg;
        RegTensor<uint32_t> oneIntReg;
        RegTensor<uint32_t> zeroIntReg;
        RegTensor<uint32_t> mantAddReg;
        RegTensor<uint32_t> roundedExpBitsReg;
        RegTensor<uint32_t> invExpBitsReg;
        RegTensor<uint32_t> invExpSubReg;
        RegTensor<uint32_t> fp8MantReg;
        RegTensor<uint32_t> fp8EmaxReg;
        RegTensor<uint32_t> threshold254Reg;
        RegTensor<uint32_t> infThresholdReg;
        RegTensor<uint32_t> nanIntReg;

        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskAllUint = CreateMask<uint32_t, MaskPattern::ALL>();
        MaskReg maskGroupNum;
        MaskReg maskMantHigh;
        MaskReg maskClamp;
        MaskReg maskInfNaN;

        Duplicate(expMaskReg, FP32_EXPONENT_MASK, maskAllUint);
        Duplicate(mantMaskReg, FP32_MANTISSA_MASK, maskAllUint);
        Duplicate(oneIntReg, 1U, maskAllUint);
        Duplicate(zeroIntReg, 0U, maskAllUint);
        Duplicate(invExpSubReg, FP32_INV_EXP_SUB, maskAllUint);
        Duplicate(fp8MantReg, FP8_GROUP_FP8_MAX_MANT, maskAllUint);
        Duplicate(fp8EmaxReg, fp8EmaxValue, maskAllUint);
        Duplicate(threshold254Reg, 254U, maskAllUint);
        Duplicate(infThresholdReg, FP8_GROUP_FP32_INF_BITS, maskAllUint);
        Duplicate(nanIntReg, FP8_GROUP_FP32_NAN_BITS, maskAllUint);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            uint32_t remaining = static_cast<uint32_t>(groupNum - i * vfLen);
            maskGroupNum = UpdateMask<float>(remaining);

            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(maxReg, amaxInAddr, vfLen);

            ShiftRights(expBitsReg, (RegTensor<uint32_t> &)maxReg, FP32_EXPONENT_SHIFT, maskAllUint);
            And(expBitsReg, expBitsReg, expMaskReg, maskAllUint);
            And(mantBitsReg, (RegTensor<uint32_t> &)maxReg, mantMaskReg, maskAllUint);
            Compare<uint32_t, CMPMODE::GE>(maskInfNaN, (RegTensor<uint32_t> &)maxReg, infThresholdReg, maskAllUint);
            Compare<uint32_t, CMPMODE::GT>(maskMantHigh, mantBitsReg, fp8MantReg, maskAllUint);
            Select(mantAddReg, oneIntReg, zeroIntReg, maskMantHigh);
            Sub<uint32_t>(roundedExpBitsReg, expBitsReg, fp8EmaxReg, maskAllUint);
            Add<uint32_t>(roundedExpBitsReg, roundedExpBitsReg, mantAddReg, maskAllUint);
            Compare<uint32_t, CMPMODE::EQ>(maskClamp, roundedExpBitsReg, zeroIntReg, maskAllUint);
            Select(roundedExpBitsReg, oneIntReg, roundedExpBitsReg, maskClamp);
            Compare<uint32_t, CMPMODE::GT>(maskClamp, roundedExpBitsReg, threshold254Reg, maskAllUint);
            Select(roundedExpBitsReg, oneIntReg, roundedExpBitsReg, maskClamp);
            ShiftLefts((RegTensor<uint32_t> &)roundScaleReg, roundedExpBitsReg, FP32_EXPONENT_SHIFT, maskAllUint);
            Select<uint32_t>((RegTensor<uint32_t> &)roundScaleReg, nanIntReg, (RegTensor<uint32_t> &)roundScaleReg,
                             maskInfNaN);

            StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(scaleOutAddr, roundScaleReg, vfLen, maskGroupNum);
            Sub<uint32_t>(invExpBitsReg, invExpSubReg, roundedExpBitsReg, maskAllUint);
            ShiftLefts((RegTensor<uint32_t> &)invScaleReg, invExpBitsReg, FP32_EXPONENT_SHIFT, maskAllUint);
            Select<uint32_t>((RegTensor<uint32_t> &)invScaleReg, nanIntReg, (RegTensor<uint32_t> &)invScaleReg,
                             maskInfNaN);
            StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(invScaleOutAddr, invScaleReg, vfLen, maskGroupNum);
        }
    }
}

template <typename T, typename U>
__simd_vf__ inline void VfComputeData(__ubuf__ T *xAddr, __ubuf__ float *invScaleInAddr, __ubuf__ U *yAddr,
                                      uint32_t totalElemNum, uint16_t groupNum)
{
    const uint16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    __VEC_SCOPE__
    {
        RegTensor<float> xLeftReg;
        RegTensor<float> xRightReg;
        RegTensor<float> invScaleReg;
        RegTensor<float> quantLeftReg;
        RegTensor<float> quantRightReg;
        RegTensor<U> outLeftReg;
        RegTensor<U> outRightReg;

        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskRight;
        MaskReg maskLoop;

        uint32_t tailElemNum = totalElemNum % FP8_GROUP_QUANT_SIZE;
        uint16_t fullGroupNum = (tailElemNum == 0) ? groupNum : static_cast<uint16_t>(groupNum - 1);
        for (uint16_t groupIdx = 0; groupIdx < fullGroupNum; groupIdx++) {
            uint32_t groupOffset = groupIdx * FP8_GROUP_QUANT_SIZE;

            LoadAlign<float, LoadDist::DIST_BRC_B32>(invScaleReg, invScaleInAddr + groupIdx);
            LoadFp8GroupInput<T>(xLeftReg, xAddr + groupOffset, maskAll, 0);
            LoadFp8GroupInput<T>(xRightReg, xAddr + groupOffset, maskAll, vfLen);
            Mul(quantLeftReg, xLeftReg, invScaleReg, maskAll);
            Mul(quantRightReg, xRightReg, invScaleReg, maskAll);
            Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outLeftReg, quantLeftReg, maskAll);
            Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outRightReg, quantRightReg, maskAll);
            StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + groupOffset, outLeftReg, maskAll);
            StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + groupOffset + vfLen, outRightReg, maskAll);
        }

        if (tailElemNum > 0) {
            uint32_t groupOffset = fullGroupNum * FP8_GROUP_QUANT_SIZE;

            LoadAlign<float, LoadDist::DIST_BRC_B32>(invScaleReg, invScaleInAddr + fullGroupNum);

            if (tailElemNum > vfLen) {
                uint32_t rightElemNum = tailElemNum - vfLen;
                maskRight = UpdateMask<float>(rightElemNum);
                LoadFp8GroupInput<T>(xLeftReg, xAddr + groupOffset, maskAll, 0);
                LoadFp8GroupInput<T>(xRightReg, xAddr + groupOffset, maskRight, vfLen);
                Mul(quantLeftReg, xLeftReg, invScaleReg, maskAll);
                Mul(quantRightReg, xRightReg, invScaleReg, maskRight);
                Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outLeftReg, quantLeftReg, maskAll);
                Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outRightReg, quantRightReg, maskRight);
                StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + groupOffset, outLeftReg, maskAll);
                StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + groupOffset + vfLen, outRightReg, maskRight);
            } else {
                maskLoop = UpdateMask<float>(tailElemNum);
                LoadFp8GroupInput<T>(xLeftReg, xAddr + groupOffset, maskLoop, 0);
                Mul(quantLeftReg, xLeftReg, invScaleReg, maskLoop);
                Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outLeftReg, quantLeftReg, maskLoop);
                StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + groupOffset, outLeftReg, maskLoop);
            }
        }
    }
}

template <typename T, typename U, bool CLAMP_AMAX>
class MoeV3GatherFP8GroupQuant {
public:
    __aicore__ inline MoeV3GatherFP8GroupQuant(){};

    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR unusedScaleAddr, GM_ADDR workspace, GM_ADDR expandedRowIdxAddr,
                                GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKernelTiling(GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void ScatterCopyExpandedXAndQuant(int64_t progress);
    __aicore__ inline void GatherCopyExpandedXAndQuant(int64_t progress);
    __aicore__ inline void CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);
    __aicore__ inline void Compute(int64_t loopCols, int64_t loopScaleCols);
    __aicore__ inline void CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols, int64_t loopScaleCols);

private:
    TPipe *pipe_{nullptr};
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_BUFFER_NUM> scaleQueue_;
    TBuf<QuePosition::VECCALC> amaxBuffer_;
    TBuf<QuePosition::VECCALC> invScaleBuffer_;
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
    int64_t useGatherCopy_{0};

    uint32_t fp8Emax_{FP8_GROUP_E5M2_EMAX};
};

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::Init(
    GM_ADDR xAddr, GM_ADDR unusedScaleAddr, GM_ADDR workspace, GM_ADDR expandedRowIdxAddr, GM_ADDR expandedXAddr,
    GM_ADDR expandedScaleAddr, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    (void)unusedScaleAddr;
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(workspace, tilingData);

    xInGm_.SetGlobalBuffer((__gm__ T *)xAddr);
    expandedXOutGm_.SetGlobalBuffer((__gm__ U *)expandedXAddr);
    expandedScaleOutGm_.SetGlobalBuffer((__gm__ float *)expandedScaleAddr);

    if (useGatherCopy_) {
        if (rowIdxType_ == SCATTER) {
            sortedRowIdxGm_.SetGlobalBuffer(
                (__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
                Align(perCoreRow_, sizeof(int32_t)));
        } else {
            sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                            Align(perCoreRow_, sizeof(int32_t)));
        }
    } else {
        if (rowIdxType_ == SCATTER) {
            sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                            Align(perCoreRow_, sizeof(int32_t)));
        } else {
            sortedRowIdxGm_.SetGlobalBuffer(
                (__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
                Align(perCoreRow_, sizeof(int32_t)));
        }
    }

    InitBuffer();
    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        fp8Emax_ = FP8_GROUP_E4M3_EMAX;
    } else {
        fp8Emax_ = FP8_GROUP_E5M2_EMAX;
    }
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::InitKernelTiling(
    GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;
    useGatherCopy_ = tilingData->useGatherCopy;

    scaleCols_ = Ceil(cols_, FP8_GROUP_QUANT_SIZE);

    int64_t actualExpertNum = tilingData->actualExpertNum;

    int64_t scanRowCount = n_ * k_;
    if (!useGatherCopy_) {
        expertTotalCountGm_.SetGlobalBuffer(
            (__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) * 2 + Align(actualExpertNum, sizeof(int32_t)),
            1);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertTotalCountGm_);
        scanRowCount = expertTotalCountGm_.GetValue(0);
    }
    perCoreRow_ = Ceil(scanRowCount, tilingData->coreNum);
    needCoreNum_ = Ceil(scanRowCount, perCoreRow_);
    int64_t lastCoreIndicesElements = scanRowCount - (needCoreNum_ - 1) * perCoreRow_;

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
    int64_t tmpBufferSize = Max(static_cast<int64_t>(AlignBytes(perLoopScaleCols_, sizeof(float))),
                                static_cast<int64_t>(AscendC::VECTOR_REG_WIDTH));
    pipe_->InitBuffer(amaxBuffer_, tmpBufferSize);
    pipe_->InitBuffer(invScaleBuffer_, tmpBufferSize);
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
        if (!useGatherCopy_) {
            ScatterCopyExpandedXAndQuant(loop);
        } else {
            GatherCopyExpandedXAndQuant(loop);
        }
    }
    currentLoopRows_ = lastLoopRows_;
    CopyInExpandedExpertIdx(rowLoops_ - 1);
    if (!useGatherCopy_) {
        ScatterCopyExpandedXAndQuant(rowLoops_ - 1);
    } else {
        GatherCopyExpandedXAndQuant(rowLoops_ - 1);
    }
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
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::ScatterCopyExpandedXAndQuant(int64_t progress)
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
__aicore__ inline void MoeV3GatherFP8GroupQuant<T, U, CLAMP_AMAX>::GatherCopyExpandedXAndQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    for (int64_t j = 0; j < colLoops_; j++) {
        int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
        int64_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;

        int64_t globalSortIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int64_t curLoopRow = 0;
        int64_t currentLoopStartRow = globalSortIdx / k_;
        int64_t currentLoopLastRow = (globalSortIdx + currentLoopRows_ - 1) / k_;

        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            bool hasValidOut = false;
            while (curLoopRow < currentLoopRows_ && globalSortIdx / k_ == row) {
                if (indicesLocal.GetValue(curLoopRow) >= 0) {
                    hasValidOut = true;
                    break;
                }
                curLoopRow++;
                globalSortIdx++;
            }
            if (!hasValidOut) {
                continue;
            }

            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyIn(row, j, loopCols);
            Compute(loopCols, loopScaleCols);

            LocalTensor<U> outLocal = outQueue_.DeQue<U>();
            LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();

            DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(U)), 0, 0, 0};
            DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(float)), 0, 0, 0};
            while (curLoopRow < currentLoopRows_ && globalSortIdx / k_ == row) {
                int32_t outIndex = indicesLocal.GetValue(curLoopRow);
                curLoopRow++;
                globalSortIdx++;
                if (outIndex < 0) {
                    continue;
                }
                int64_t outOffset = static_cast<int64_t>(outIndex) * cols_ + j * perLoopCols_;
                DataCopyPad<U>(expandedXOutGm_[outOffset], outLocal, copyOutParams);

                int64_t outScaleOffset = static_cast<int64_t>(outIndex) * scaleCols_ + j * perLoopScaleCols_;
                DataCopyPad<float>(expandedScaleOutGm_[outScaleOffset], scaleLocal, copyScaleParams);
            }

            outQueue_.FreeTensor(outLocal);
            scaleQueue_.FreeTensor(scaleLocal);
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

    LocalTensor<float> amaxLocal = amaxBuffer_.Get<float>();
    __ubuf__ float *amaxAddr = reinterpret_cast<__ubuf__ float *>(amaxLocal.GetPhyAddr());

    LocalTensor<float> invScaleLocal = invScaleBuffer_.Get<float>();
    __ubuf__ float *invScaleAddr = reinterpret_cast<__ubuf__ float *>(invScaleLocal.GetPhyAddr());

    __VEC_SCOPE__
    {
        VfComputeAmax<T, CLAMP_AMAX>(xAddr, amaxAddr, static_cast<uint32_t>(loopCols),
                                     static_cast<uint16_t>(loopScaleCols));
    }
    __VEC_SCOPE__
    {
        VfComputeRoundScale<CLAMP_AMAX>(amaxAddr, scaleAddr, invScaleAddr, static_cast<uint16_t>(loopScaleCols),
                                        fp8Emax_);
    }
    __VEC_SCOPE__
    {
        VfComputeData<T, U>(xAddr, invScaleAddr, outAddr, static_cast<uint32_t>(loopCols),
                            static_cast<uint16_t>(loopScaleCols));
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
