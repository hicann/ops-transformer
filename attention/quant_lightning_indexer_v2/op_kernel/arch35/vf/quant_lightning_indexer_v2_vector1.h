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
 * \file quant_lightning_indexer_v2_vector1.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_VECTOR1_H
#define QUANT_LIGHTNING_INDEXER_V2_VECTOR1_H

#include "kernel_operator.h"
#include "../../../../lightning_indexer_v2/op_kernel/arch35/common/vf/lightning_indexer_v2_vector1_base.h"

namespace vector1 {
__simd_vf__ void UIntToFloatReturnValueVF(__ubuf__ bfloat16_t *outBuf, __ubuf__ uint16_t *inBuf, uint16_t vfLoop)
{
    Reg::RegTensor<uint16_t> regIn;
    Reg::RegTensor<bfloat16_t> regOut;
    Reg::MaskReg maskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint16_t>(regIn, inBuf + i * 128);

        liV2Vector1::UIntSortConstCtx<bfloat16_t> uint16Ctx;
        liV2Vector1::InitUIntSortConstCtx(uint16Ctx, maskAllB16);

        liV2Vector1::UIntToSortableKey<bfloat16_t>(regOut, regIn, uint16Ctx, maskAllB16);

        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_NORM>(outBuf + i * 128, regOut, maskAllB16);
    }
}

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<bfloat16_t> &out_, const LocalTensor<uint16_t> &in,
                                              const uint32_t topK)
{
    __ubuf__ bfloat16_t *outBuf = (__ubuf__ bfloat16_t *)out_.GetPhyAddr();
    __ubuf__ uint16_t *inBuf = (__ubuf__ uint16_t *)in.GetPhyAddr();
    const uint16_t repeatSize16 = 128;
    uint16_t topkLoopNum = (topK + repeatSize16 - 1) / repeatSize16;
    UIntToFloatReturnValueVF(outBuf, inBuf, topkLoopNum);
}

// 可排序键还原为 bf16 返回值，并将无效位（score==0）刷为 -inf
__simd_vf__ void UIntToFloatReturnValueWithInfMaskVF(__ubuf__ bfloat16_t *valueOutBuf, __ubuf__ uint16_t *scoreOutBuf,
                                                     uint16_t vfLoop, uint16_t negInfBits)
{
    Reg::RegTensor<uint16_t> regIn;
    Reg::RegTensor<bfloat16_t> regOut;
    Reg::RegTensor<uint16_t> regNegInf;
    Reg::RegTensor<uint16_t> regZero;
    Reg::MaskReg maskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskInvalid;

    // 常量寄存器初始化：-inf(0xFF80) 与 0（用于识别无效位）
    Reg::Duplicate(regNegInf, negInfBits, maskAllB16);
    Reg::Duplicate(regZero, (uint16_t)0, maskAllB16);

    liV2Vector1::UIntSortConstCtx<bfloat16_t> uint16Ctx;
    liV2Vector1::InitUIntSortConstCtx(uint16Ctx, maskAllB16);

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint16_t>(regIn, scoreOutBuf + i * 128);
        // 比较得无效位掩码：score==0 即无效位（可排序键性质，真实值永不为0）
        Reg::Compare<uint16_t, CMPMODE::EQ>(maskInvalid, regIn, regZero, maskAllB16);
        // 逆变换：可排序键 → bf16 值
        liV2Vector1::UIntToSortableKey<bfloat16_t>(regOut, regIn, uint16Ctx, maskAllB16);
        // 无效位覆盖为 -inf，有效位保留还原值
        Reg::Select((Reg::RegTensor<uint16_t> &)regOut, regNegInf, (Reg::RegTensor<uint16_t> &)regOut, maskInvalid);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_NORM>(valueOutBuf + i * 128, regOut, maskAllB16);
    }
}

__aicore__ inline void UIntToFloatReturnValueWithInfMask(const LocalTensor<bfloat16_t> &valueOutLocal,
                                                         const LocalTensor<uint16_t> &scoreOutLocal,
                                                         const uint32_t topK, const uint16_t negInfBits)
{
    __ubuf__ bfloat16_t *valueOutBuf = (__ubuf__ bfloat16_t *)valueOutLocal.GetPhyAddr();
    __ubuf__ uint16_t *scoreOutBuf = (__ubuf__ uint16_t *)scoreOutLocal.GetPhyAddr();
    const uint16_t repeatSize16 = 128;
    uint16_t topkLoopNum = (topK + repeatSize16 - 1) / repeatSize16;
    UIntToFloatReturnValueWithInfMaskVF(valueOutBuf, scoreOutBuf, topkLoopNum, negInfBits);
}

__simd_callee__ inline void LoadKScaleFP16(AscendC::Reg::RegTensor<half> (&qliScaleRegKScaleFP16)[2],
                                           AscendC::Reg::RegTensor<float> (&qliScaleRegKScale)[2],
                                           AscendC::Reg::MaskReg &qliScaleMaskAllB16, __ubuf__ half *qliScaleKScale)
{
    constexpr static Reg::CastTrait qliScaleCastTraitFP16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                   Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(qliScaleRegKScaleFP16[0], qliScaleKScale);
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(qliScaleRegKScaleFP16[1],
                                                                           qliScaleKScale + 64);
    AscendC::Reg::Cast<float, half, qliScaleCastTraitFP16ToFP32>(qliScaleRegKScale[0], qliScaleRegKScaleFP16[0],
                                                                 qliScaleMaskAllB16);
    AscendC::Reg::Cast<float, half, qliScaleCastTraitFP16ToFP32>(qliScaleRegKScale[1], qliScaleRegKScaleFP16[1],
                                                                 qliScaleMaskAllB16);
}

__simd_callee__ inline void CastFP32ToFP16ToFP32(AscendC::Reg::RegTensor<float> (&qliCastRegQK)[2],
                                                 AscendC::Reg::RegTensor<half> (&qliCastRegQKHalf)[2],
                                                 AscendC::Reg::MaskReg &qliCastMaskAllB32)
{
    AscendC::Reg::MaskReg qliCastMaskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
    constexpr static Reg::CastTrait qliCastTraitFP32ToFP16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static Reg::CastTrait qliCastTraitFP16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    float qliCastMulsScalar = 1.0 / 1024;

    Reg::Muls(qliCastRegQK[0], qliCastRegQK[0], qliCastMulsScalar, qliCastMaskAllB32);
    Reg::Muls(qliCastRegQK[1], qliCastRegQK[1], qliCastMulsScalar, qliCastMaskAllB32);

    Reg::Cast<half, float, qliCastTraitFP32ToFP16>(qliCastRegQKHalf[0], qliCastRegQK[0], qliCastMaskAllB32);
    Reg::Cast<half, float, qliCastTraitFP32ToFP16>(qliCastRegQKHalf[1], qliCastRegQK[1], qliCastMaskAllB32);

    Reg::Cast<float, half, qliCastTraitFP16ToFP32>(qliCastRegQK[0], qliCastRegQKHalf[0], qliCastMaskAllB16);
    Reg::Cast<float, half, qliCastTraitFP16ToFP32>(qliCastRegQK[1], qliCastRegQKHalf[1], qliCastMaskAllB16);
}

// int32 in uint16 out
__simd_vf__ void MulWeightAndReduceSumInt32GSizeOddVF(__ubuf__ uint16_t *qliI32OddOut, __ubuf__ int32_t *qliI32OddQk,
                                                      uint32_t qliI32OddQkVLStride, __ubuf__ half *qliI32OddWeight,
                                                      __ubuf__ half *qliI32OddKScale, __ubuf__ half *qliI32OddQScale,
                                                      uint16_t qliI32OddGSize)
{
    Reg::RegTensor<float> qliI32OddRegwBrc;
    Reg::RegTensor<float> qliI32OddRegQK[2];
    Reg::RegTensor<half> qliI32OddRegQKHalf[2];
    Reg::RegTensor<int32_t> qliI32OddRegQKInt32[2];
    Reg::RegTensor<float> qliI32OddRegW;
    Reg::RegTensor<half> qliI32OddRegWFP16;
    Reg::RegTensor<half> qliI32OddRegWFP16Temp;
    Reg::RegTensor<float> qliI32OddRegQScale;
    Reg::RegTensor<half> qliI32OddRegQScaleFP16;
    Reg::RegTensor<float> qliI32OddRegKScale[2];
    Reg::RegTensor<half> qliI32OddRegKScaleFP16[2];
    Reg::RegTensor<float> qliI32OddRegSum0[2];
    Reg::RegTensor<float> qliI32OddRegSum1[2];
    Reg::MaskReg qliI32OddMaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliI32OddMaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliI32OddBf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliI32OddBf16Ctx, qliI32OddMaskAllB16);

    constexpr static Reg::CastTrait qliI32OddCastTraitF32ToF16_EVEN = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliI32OddCastTraitF32ToF16_ODD = {
        Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliI32OddCastTraitF16ToF32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    constexpr static Reg::CastTrait qliI32OddCastTraitInt32ToFP32 = {
        Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliI32OddCastTraitF32ToF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(qliI32OddRegWFP16, qliI32OddWeight);
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(qliI32OddRegQScaleFP16, qliI32OddQScale);
    Reg::Cast<float, half, qliI32OddCastTraitF16ToF32>(qliI32OddRegW, qliI32OddRegWFP16, qliI32OddMaskAllB16);
    Reg::Cast<float, half, qliI32OddCastTraitF16ToF32>(qliI32OddRegQScale, qliI32OddRegQScaleFP16, qliI32OddMaskAllB16);
    Reg::Mul(qliI32OddRegW, qliI32OddRegW, qliI32OddRegQScale, qliI32OddMaskAllB32);
    Reg::Cast<half, float, qliI32OddCastTraitF32ToF16>(qliI32OddRegWFP16Temp, qliI32OddRegW, qliI32OddMaskAllB32);
    Reg::Cast<float, half, qliI32OddCastTraitF16ToF32>(qliI32OddRegW, qliI32OddRegWFP16Temp, qliI32OddMaskAllB16);
    liV2Vector1::DuplicateZero(qliI32OddRegSum0, qliI32OddMaskAllB32);
    liV2Vector1::DuplicateZero(qliI32OddRegSum1, qliI32OddMaskAllB32);

    LoadKScaleFP16(qliI32OddRegKScaleFP16, qliI32OddRegKScale, qliI32OddMaskAllB16, qliI32OddKScale);
    // unroll2
    for (uint16_t qliI32OddI = (uint16_t)(0); (uint16_t)(qliI32OddI + 1) < qliI32OddGSize; qliI32OddI += 2) {
        Reg::LoadAlign<int32_t>(qliI32OddRegQKInt32[0], qliI32OddQk + 128 * qliI32OddI);
        Reg::LoadAlign<int32_t>(qliI32OddRegQKInt32[1], qliI32OddQk + 128 * qliI32OddI + qliI32OddQkVLStride);
        Reg::Cast<float, int32_t, qliI32OddCastTraitInt32ToFP32>(qliI32OddRegQK[0], qliI32OddRegQKInt32[0],
                                                                 qliI32OddMaskAllB32);
        Reg::Cast<float, int32_t, qliI32OddCastTraitInt32ToFP32>(qliI32OddRegQK[1], qliI32OddRegQKInt32[1],
                                                                 qliI32OddMaskAllB32);

        CastFP32ToFP16ToFP32(qliI32OddRegQK, qliI32OddRegQKHalf, qliI32OddMaskAllB32);

        liV2Vector1::BroadcastLane(qliI32OddRegwBrc, qliI32OddRegW, qliI32OddI);
        liV2Vector1::WeightedAccum(qliI32OddRegSum0, qliI32OddRegQK, qliI32OddRegwBrc, qliI32OddMaskAllB32);

        Reg::LoadAlign<int32_t>(qliI32OddRegQKInt32[0], qliI32OddQk + 128 * qliI32OddI + 128);
        Reg::LoadAlign<int32_t>(qliI32OddRegQKInt32[1], qliI32OddQk + 128 * qliI32OddI + 128 + qliI32OddQkVLStride);
        Reg::Cast<float, int32_t, qliI32OddCastTraitInt32ToFP32>(qliI32OddRegQK[0], qliI32OddRegQKInt32[0],
                                                                 qliI32OddMaskAllB32);
        Reg::Cast<float, int32_t, qliI32OddCastTraitInt32ToFP32>(qliI32OddRegQK[1], qliI32OddRegQKInt32[1],
                                                                 qliI32OddMaskAllB32);

        CastFP32ToFP16ToFP32(qliI32OddRegQK, qliI32OddRegQKHalf, qliI32OddMaskAllB32);

        liV2Vector1::BroadcastLane(qliI32OddRegwBrc, qliI32OddRegW, qliI32OddI + 1);
        liV2Vector1::WeightedAccum(qliI32OddRegSum1, qliI32OddRegQK, qliI32OddRegwBrc, qliI32OddMaskAllB32);
    }

    Reg::LoadAlign<int32_t>(qliI32OddRegQKInt32[0], qliI32OddQk + 128 * (qliI32OddGSize - 1));
    Reg::LoadAlign<int32_t>(qliI32OddRegQKInt32[1], qliI32OddQk + 128 * (qliI32OddGSize - 1) + qliI32OddQkVLStride);
    Reg::Cast<float, int32_t, qliI32OddCastTraitInt32ToFP32>(qliI32OddRegQK[0], qliI32OddRegQKInt32[0],
                                                             qliI32OddMaskAllB32);
    Reg::Cast<float, int32_t, qliI32OddCastTraitInt32ToFP32>(qliI32OddRegQK[1], qliI32OddRegQKInt32[1],
                                                             qliI32OddMaskAllB32);

    CastFP32ToFP16ToFP32(qliI32OddRegQK, qliI32OddRegQKHalf, qliI32OddMaskAllB32);

    liV2Vector1::BroadcastLane(qliI32OddRegwBrc, qliI32OddRegW, qliI32OddGSize - 1);
    liV2Vector1::WeightedAccum(qliI32OddRegSum0, qliI32OddRegQK, qliI32OddRegwBrc, qliI32OddMaskAllB32);

    Reg::Add(qliI32OddRegSum0[0], qliI32OddRegSum0[0], qliI32OddRegSum1[0], qliI32OddMaskAllB32);
    Reg::Add(qliI32OddRegSum0[1], qliI32OddRegSum0[1], qliI32OddRegSum1[1], qliI32OddMaskAllB32);

    Reg::Mul(qliI32OddRegSum0[0], qliI32OddRegSum0[0], qliI32OddRegKScale[0], qliI32OddMaskAllB32);
    Reg::Mul(qliI32OddRegSum0[1], qliI32OddRegSum0[1], qliI32OddRegKScale[1], qliI32OddMaskAllB32);

    Reg::RegTensor<bfloat16_t> qliI32OddRegSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    Reg::DeInterleave(qliI32OddRegSum0[0], qliI32OddRegSum0[1], qliI32OddRegSum0[0], qliI32OddRegSum0[1]);
    Reg::Cast<bfloat16_t, float, qliI32OddCastTraitF32ToF16_ODD>(qliI32OddRegSumBF16, qliI32OddRegSum0[1],
                                                                 qliI32OddMaskAllB32);
    Reg::Cast<bfloat16_t, float, qliI32OddCastTraitF32ToF16_EVEN>(qliI32OddRegSumBF16, qliI32OddRegSum0[0],
                                                                  qliI32OddMaskAllB32);

    Reg::RegTensor<uint16_t> qliI32OddRegOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(qliI32OddRegOut, qliI32OddRegSumBF16, qliI32OddBf16Ctx,
                                                qliI32OddMaskAllB16);
    // normal store
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliI32OddOut, qliI32OddRegOut, qliI32OddMaskAllB16);
}

// float in uint16 out
__simd_vf__ void MulWeightAndReduceSumF32GSizeOddVF(__ubuf__ uint16_t *qliF32OddOut, __ubuf__ float *qliF32OddQk,
                                                    uint32_t qliF32OddQkVLStride, __ubuf__ float *qliF32OddWeight,
                                                    __ubuf__ float *qliF32OddKScale, __ubuf__ float *qliF32OddQScale,
                                                    uint16_t qliF32OddGSize)
{
    Reg::RegTensor<float> qliF32OddRegwBrc;
    Reg::RegTensor<float> qliF32OddRegQK[2];
    Reg::RegTensor<float> qliF32OddRegW;

    Reg::RegTensor<float> qliF32OddRegQScale;
    Reg::RegTensor<float> qliF32OddRegKScale[2];
    Reg::RegTensor<float> qliF32OddRegSum0[2];
    Reg::RegTensor<float> qliF32OddRegSum1[2];
    Reg::MaskReg qliF32OddMaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliF32OddMaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliF32OddBf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliF32OddBf16Ctx, qliF32OddMaskAllB16);

    constexpr static Reg::CastTrait qliF32OddCastTraitF32ToF16_EVEN = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliF32OddCastTraitF32ToF16_ODD = {
        Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    Reg::LoadAlign<float>(qliF32OddRegW, qliF32OddWeight);
    Reg::LoadAlign<float>(qliF32OddRegQScale, qliF32OddQScale);
    Reg::Mul(qliF32OddRegW, qliF32OddRegW, qliF32OddRegQScale, qliF32OddMaskAllB32);

    liV2Vector1::DuplicateZero(qliF32OddRegSum0, qliF32OddMaskAllB32);
    liV2Vector1::DuplicateZero(qliF32OddRegSum1, qliF32OddMaskAllB32);

    Reg::LoadAlign<float>(qliF32OddRegKScale[0], qliF32OddKScale);
    Reg::LoadAlign<float>(qliF32OddRegKScale[1], qliF32OddKScale + 64);

    // unroll2
    for (uint16_t qliF32OddI = (uint16_t)(0); (uint16_t)(qliF32OddI + 1) < qliF32OddGSize; qliF32OddI += 2) {
        Reg::LoadAlign<float>(qliF32OddRegQK[0], qliF32OddQk + 128 * qliF32OddI); // RowStride是128, 行都落在一个bank上
        Reg::LoadAlign<float>(qliF32OddRegQK[1], qliF32OddQk + 128 * qliF32OddI + qliF32OddQkVLStride);
        liV2Vector1::BroadcastLane(qliF32OddRegwBrc, qliF32OddRegW, qliF32OddI);
        liV2Vector1::WeightedAccum(qliF32OddRegSum0, qliF32OddRegQK, qliF32OddRegwBrc, qliF32OddMaskAllB32);

        Reg::LoadAlign<float>(qliF32OddRegQK[0], qliF32OddQk + 128 * qliF32OddI + 128);
        Reg::LoadAlign<float>(qliF32OddRegQK[1], qliF32OddQk + 128 * qliF32OddI + 128 + qliF32OddQkVLStride);
        liV2Vector1::BroadcastLane(qliF32OddRegwBrc, qliF32OddRegW, qliF32OddI + 1);
        liV2Vector1::WeightedAccum(qliF32OddRegSum1, qliF32OddRegQK, qliF32OddRegwBrc, qliF32OddMaskAllB32);
    }

    Reg::LoadAlign<float>(qliF32OddRegQK[0],
                          qliF32OddQk + 128 * (qliF32OddGSize - 1)); // RowStride是128, 行都落在一个bank上
    Reg::LoadAlign<float>(qliF32OddRegQK[1], qliF32OddQk + 128 * (qliF32OddGSize - 1) + qliF32OddQkVLStride);
    liV2Vector1::BroadcastLane(qliF32OddRegwBrc, qliF32OddRegW, (qliF32OddGSize - 1));
    liV2Vector1::WeightedAccum(qliF32OddRegSum0, qliF32OddRegQK, qliF32OddRegwBrc, qliF32OddMaskAllB32);

    Reg::Add(qliF32OddRegSum0[0], qliF32OddRegSum0[0], qliF32OddRegSum1[0], qliF32OddMaskAllB32);
    Reg::Add(qliF32OddRegSum0[1], qliF32OddRegSum0[1], qliF32OddRegSum1[1], qliF32OddMaskAllB32);

    Reg::Mul(qliF32OddRegSum0[0], qliF32OddRegSum0[0], qliF32OddRegKScale[0], qliF32OddMaskAllB32);
    Reg::Mul(qliF32OddRegSum0[1], qliF32OddRegSum0[1], qliF32OddRegKScale[1], qliF32OddMaskAllB32);

    Reg::RegTensor<bfloat16_t> qliF32OddRegSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    Reg::DeInterleave(qliF32OddRegSum0[0], qliF32OddRegSum0[1], qliF32OddRegSum0[0], qliF32OddRegSum0[1]);
    Reg::Cast<bfloat16_t, float, qliF32OddCastTraitF32ToF16_ODD>(qliF32OddRegSumBF16, qliF32OddRegSum0[1],
                                                                 qliF32OddMaskAllB32);
    Reg::Cast<bfloat16_t, float, qliF32OddCastTraitF32ToF16_EVEN>(qliF32OddRegSumBF16, qliF32OddRegSum0[0],
                                                                  qliF32OddMaskAllB32);

    Reg::RegTensor<uint16_t> qliF32OddRegOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(qliF32OddRegOut, qliF32OddRegSumBF16, qliF32OddBf16Ctx,
                                                qliF32OddMaskAllB16);
    // normal store
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliF32OddOut, qliF32OddRegOut, qliF32OddMaskAllB16);
}

// int32 in uint16 out
__simd_vf__ void MulWeightAndReduceSumInt32GSizeEvenVF(__ubuf__ uint16_t *qliI32EvenOut, __ubuf__ int32_t *qliI32EvenQk,
                                                       uint32_t qliI32EvenQkVLStride, __ubuf__ half *qliI32EvenWeight,
                                                       __ubuf__ half *qliI32EvenKScale, __ubuf__ half *qliI32EvenQScale,
                                                       uint16_t qliI32EvenGSize)
{
    Reg::RegTensor<float> qliI32EvenRegwBrc;
    Reg::RegTensor<float> qliI32EvenRegQK[2];
    Reg::RegTensor<half> qliI32EvenRegQKHalf[2];
    Reg::RegTensor<int32_t> qliI32EvenRegQKInt32[2];
    Reg::RegTensor<float> qliI32EvenRegW;
    Reg::RegTensor<half> qliI32EvenRegWFP16;
    Reg::RegTensor<half> qliI32EvenRegWFP16Temp;
    Reg::RegTensor<float> qliI32EvenRegQScale;
    Reg::RegTensor<half> qliI32EvenRegQScaleFP16;
    Reg::RegTensor<float> qliI32EvenRegKScale[2];
    Reg::RegTensor<half> qliI32EvenRegKScaleFP16[2];
    Reg::RegTensor<float> qliI32EvenRegSum0[2];
    Reg::RegTensor<float> qliI32EvenRegSum1[2];
    Reg::MaskReg qliI32EvenMaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliI32EvenMaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliI32EvenBf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliI32EvenBf16Ctx, qliI32EvenMaskAllB16);

    constexpr static Reg::CastTrait qliI32EvenCastTraitF32ToF16_EVEN = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliI32EvenCastTraitF32ToF16_ODD = {
        Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliI32EvenCastTraitF16ToF32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                   Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    constexpr static Reg::CastTrait qliI32EvenCastTraitInt32ToFP32 = {
        Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliI32EvenCastTraitF32ToF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                   Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(qliI32EvenRegWFP16, qliI32EvenWeight);
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(qliI32EvenRegQScaleFP16, qliI32EvenQScale);
    Reg::Cast<float, half, qliI32EvenCastTraitF16ToF32>(qliI32EvenRegW, qliI32EvenRegWFP16, qliI32EvenMaskAllB16);
    Reg::Cast<float, half, qliI32EvenCastTraitF16ToF32>(qliI32EvenRegQScale, qliI32EvenRegQScaleFP16,
                                                        qliI32EvenMaskAllB16);
    Reg::Mul(qliI32EvenRegW, qliI32EvenRegW, qliI32EvenRegQScale, qliI32EvenMaskAllB32);
    Reg::Cast<half, float, qliI32EvenCastTraitF32ToF16>(qliI32EvenRegWFP16Temp, qliI32EvenRegW, qliI32EvenMaskAllB32);
    Reg::Cast<float, half, qliI32EvenCastTraitF16ToF32>(qliI32EvenRegW, qliI32EvenRegWFP16Temp, qliI32EvenMaskAllB16);
    liV2Vector1::DuplicateZero(qliI32EvenRegSum0, qliI32EvenMaskAllB32);
    liV2Vector1::DuplicateZero(qliI32EvenRegSum1, qliI32EvenMaskAllB32);

    LoadKScaleFP16(qliI32EvenRegKScaleFP16, qliI32EvenRegKScale, qliI32EvenMaskAllB16, qliI32EvenKScale);
    // unroll2
    for (uint16_t qliI32EvenI = (uint16_t)(0); qliI32EvenI < qliI32EvenGSize; qliI32EvenI += 2) {
        Reg::LoadAlign<int32_t>(qliI32EvenRegQKInt32[0], qliI32EvenQk + 128 * qliI32EvenI);
        Reg::LoadAlign<int32_t>(qliI32EvenRegQKInt32[1], qliI32EvenQk + 128 * qliI32EvenI + qliI32EvenQkVLStride);
        Reg::Cast<float, int32_t, qliI32EvenCastTraitInt32ToFP32>(qliI32EvenRegQK[0], qliI32EvenRegQKInt32[0],
                                                                  qliI32EvenMaskAllB32);
        Reg::Cast<float, int32_t, qliI32EvenCastTraitInt32ToFP32>(qliI32EvenRegQK[1], qliI32EvenRegQKInt32[1],
                                                                  qliI32EvenMaskAllB32);

        CastFP32ToFP16ToFP32(qliI32EvenRegQK, qliI32EvenRegQKHalf, qliI32EvenMaskAllB32);

        liV2Vector1::BroadcastLane(qliI32EvenRegwBrc, qliI32EvenRegW, qliI32EvenI);
        liV2Vector1::WeightedAccum(qliI32EvenRegSum0, qliI32EvenRegQK, qliI32EvenRegwBrc, qliI32EvenMaskAllB32);

        Reg::LoadAlign<int32_t>(qliI32EvenRegQKInt32[0], qliI32EvenQk + 128 * qliI32EvenI + 128);
        Reg::LoadAlign<int32_t>(qliI32EvenRegQKInt32[1], qliI32EvenQk + 128 * qliI32EvenI + 128 + qliI32EvenQkVLStride);
        Reg::Cast<float, int32_t, qliI32EvenCastTraitInt32ToFP32>(qliI32EvenRegQK[0], qliI32EvenRegQKInt32[0],
                                                                  qliI32EvenMaskAllB32);
        Reg::Cast<float, int32_t, qliI32EvenCastTraitInt32ToFP32>(qliI32EvenRegQK[1], qliI32EvenRegQKInt32[1],
                                                                  qliI32EvenMaskAllB32);

        CastFP32ToFP16ToFP32(qliI32EvenRegQK, qliI32EvenRegQKHalf, qliI32EvenMaskAllB32);

        liV2Vector1::BroadcastLane(qliI32EvenRegwBrc, qliI32EvenRegW, qliI32EvenI + 1);
        liV2Vector1::WeightedAccum(qliI32EvenRegSum1, qliI32EvenRegQK, qliI32EvenRegwBrc, qliI32EvenMaskAllB32);
    }

    Reg::Add(qliI32EvenRegSum0[0], qliI32EvenRegSum0[0], qliI32EvenRegSum1[0], qliI32EvenMaskAllB32);
    Reg::Add(qliI32EvenRegSum0[1], qliI32EvenRegSum0[1], qliI32EvenRegSum1[1], qliI32EvenMaskAllB32);

    Reg::Mul(qliI32EvenRegSum0[0], qliI32EvenRegSum0[0], qliI32EvenRegKScale[0], qliI32EvenMaskAllB32);
    Reg::Mul(qliI32EvenRegSum0[1], qliI32EvenRegSum0[1], qliI32EvenRegKScale[1], qliI32EvenMaskAllB32);

    Reg::RegTensor<bfloat16_t> qliI32EvenRegSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    Reg::DeInterleave(qliI32EvenRegSum0[0], qliI32EvenRegSum0[1], qliI32EvenRegSum0[0], qliI32EvenRegSum0[1]);
    Reg::Cast<bfloat16_t, float, qliI32EvenCastTraitF32ToF16_ODD>(qliI32EvenRegSumBF16, qliI32EvenRegSum0[1],
                                                                  qliI32EvenMaskAllB32);
    Reg::Cast<bfloat16_t, float, qliI32EvenCastTraitF32ToF16_EVEN>(qliI32EvenRegSumBF16, qliI32EvenRegSum0[0],
                                                                   qliI32EvenMaskAllB32);

    Reg::RegTensor<uint16_t> qliI32EvenRegOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(qliI32EvenRegOut, qliI32EvenRegSumBF16, qliI32EvenBf16Ctx,
                                                qliI32EvenMaskAllB16);
    // normal store
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliI32EvenOut, qliI32EvenRegOut, qliI32EvenMaskAllB16);
}

__simd_vf__ void MulWeightAndReduceSumF32GSizeEvenVF(__ubuf__ uint16_t *qliF32EvenOut, __ubuf__ float *qliF32EvenQk,
                                                     uint32_t qliF32EvenQkVLStride, __ubuf__ float *qliF32EvenWeight,
                                                     __ubuf__ float *qliF32EvenKScale, __ubuf__ float *qliF32EvenQScale,
                                                     uint16_t qliF32EvenGSize)
{
    Reg::RegTensor<float> qliF32EvenRegwBrc;
    Reg::RegTensor<float> qliF32EvenRegQK[2];
    Reg::RegTensor<float> qliF32EvenRegW;

    Reg::RegTensor<float> qliF32EvenRegQScale;
    Reg::RegTensor<float> qliF32EvenRegKScale[2];
    Reg::RegTensor<float> qliF32EvenRegSum0[2];
    Reg::RegTensor<float> qliF32EvenRegSum1[2];
    Reg::MaskReg qliF32EvenMaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliF32EvenMaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliF32EvenBf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliF32EvenBf16Ctx, qliF32EvenMaskAllB16);

    constexpr static Reg::CastTrait qliF32EvenCastTraitF32ToF16_EVEN = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliF32EvenCastTraitF32ToF16_ODD = {
        Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    Reg::LoadAlign<float>(qliF32EvenRegW, qliF32EvenWeight);
    Reg::LoadAlign<float>(qliF32EvenRegQScale, qliF32EvenQScale);
    Reg::Mul(qliF32EvenRegW, qliF32EvenRegW, qliF32EvenRegQScale, qliF32EvenMaskAllB32);

    liV2Vector1::DuplicateZero(qliF32EvenRegSum0, qliF32EvenMaskAllB32);
    liV2Vector1::DuplicateZero(qliF32EvenRegSum1, qliF32EvenMaskAllB32);

    Reg::LoadAlign<float>(qliF32EvenRegKScale[0], qliF32EvenKScale);
    Reg::LoadAlign<float>(qliF32EvenRegKScale[1], qliF32EvenKScale + 64);

    // unroll2
    for (uint16_t qliF32EvenI = (uint16_t)(0); qliF32EvenI < qliF32EvenGSize; qliF32EvenI += 2) {
        Reg::LoadAlign<float>(qliF32EvenRegQK[0],
                              qliF32EvenQk + 128 * qliF32EvenI); // RowStride是128, 行都落在一个bank上
        Reg::LoadAlign<float>(qliF32EvenRegQK[1], qliF32EvenQk + 128 * qliF32EvenI + qliF32EvenQkVLStride);
        liV2Vector1::BroadcastLane(qliF32EvenRegwBrc, qliF32EvenRegW, qliF32EvenI);
        liV2Vector1::WeightedAccum(qliF32EvenRegSum0, qliF32EvenRegQK, qliF32EvenRegwBrc, qliF32EvenMaskAllB32);

        Reg::LoadAlign<float>(qliF32EvenRegQK[0], qliF32EvenQk + 128 * qliF32EvenI + 128);
        Reg::LoadAlign<float>(qliF32EvenRegQK[1], qliF32EvenQk + 128 * qliF32EvenI + 128 + qliF32EvenQkVLStride);
        liV2Vector1::BroadcastLane(qliF32EvenRegwBrc, qliF32EvenRegW, qliF32EvenI + 1);
        liV2Vector1::WeightedAccum(qliF32EvenRegSum1, qliF32EvenRegQK, qliF32EvenRegwBrc, qliF32EvenMaskAllB32);
    }

    Reg::Add(qliF32EvenRegSum0[0], qliF32EvenRegSum0[0], qliF32EvenRegSum1[0], qliF32EvenMaskAllB32);
    Reg::Add(qliF32EvenRegSum0[1], qliF32EvenRegSum0[1], qliF32EvenRegSum1[1], qliF32EvenMaskAllB32);

    Reg::Mul(qliF32EvenRegSum0[0], qliF32EvenRegSum0[0], qliF32EvenRegKScale[0], qliF32EvenMaskAllB32);
    Reg::Mul(qliF32EvenRegSum0[1], qliF32EvenRegSum0[1], qliF32EvenRegKScale[1], qliF32EvenMaskAllB32);

    Reg::RegTensor<bfloat16_t> qliF32EvenRegSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    Reg::DeInterleave(qliF32EvenRegSum0[0], qliF32EvenRegSum0[1], qliF32EvenRegSum0[0], qliF32EvenRegSum0[1]);
    Reg::Cast<bfloat16_t, float, qliF32EvenCastTraitF32ToF16_ODD>(qliF32EvenRegSumBF16, qliF32EvenRegSum0[1],
                                                                  qliF32EvenMaskAllB32);
    Reg::Cast<bfloat16_t, float, qliF32EvenCastTraitF32ToF16_EVEN>(qliF32EvenRegSumBF16, qliF32EvenRegSum0[0],
                                                                   qliF32EvenMaskAllB32);

    Reg::RegTensor<uint16_t> qliF32EvenRegOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(qliF32EvenRegOut, qliF32EvenRegSumBF16, qliF32EvenBf16Ctx,
                                                qliF32EvenMaskAllB16);
    // normal store
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliF32EvenOut, qliF32EvenRegOut, qliF32EvenMaskAllB16);
}

__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
                                             const LocalTensor<float> &qk_,     // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,
                                             const LocalTensor<float> &weight_, // w      [G]          [64    ]
                                             const LocalTensor<float> &kScale_, // kScale [S2Base]     [128   ]
                                             const LocalTensor<float> &qScale_, // qScale [G]          [64    ]
                                             const int gSize)                   // G 64
{
    __ubuf__ uint16_t *out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ float *weight = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *qScale = (__ubuf__ float *)qScale_.GetPhyAddr();
    __ubuf__ float *kScale = (__ubuf__ float *)kScale_.GetPhyAddr();
    __ubuf__ float *qk = (__ubuf__ float *)qk_.GetPhyAddr();
    if (gSize % 2 == 0) { // 2：判断奇偶性
        MulWeightAndReduceSumF32GSizeEvenVF(out, qk, qkVLStride, weight, kScale, qScale, (uint16_t)gSize);
    } else {
        MulWeightAndReduceSumF32GSizeOddVF(out, qk, qkVLStride, weight, kScale, qScale, (uint16_t)gSize);
    }
}

__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
                                             const LocalTensor<int32_t> &qk_,   // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,
                                             const LocalTensor<half> &weight_, // w      [G]          [64    ]
                                             const LocalTensor<half> &kScale_, // kScale [S2Base]     [128   ]
                                             const LocalTensor<half> &qScale_, // qScale [G]          [64    ]
                                             const int gSize)                  // G 64
{
    __ubuf__ uint16_t *out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ half *weight = (__ubuf__ half *)weight_.GetPhyAddr();
    __ubuf__ half *qScale = (__ubuf__ half *)qScale_.GetPhyAddr();
    __ubuf__ half *kScale = (__ubuf__ half *)kScale_.GetPhyAddr();
    __ubuf__ int32_t *qk = (__ubuf__ int32_t *)qk_.GetPhyAddr();
    if (gSize % 2 != 0) { // 2：判断奇偶性
        MulWeightAndReduceSumInt32GSizeOddVF(out, qk, qkVLStride, weight, kScale, qScale, (uint16_t)gSize);
    } else {
        MulWeightAndReduceSumInt32GSizeEvenVF(out, qk, qkVLStride, weight, kScale, qScale, (uint16_t)gSize);
    }
}

// bfloat16_t in uint16 out
__simd_vf__ void MulWeightAndReduceSumB16VF(__ubuf__ uint16_t *qliB16Out, __ubuf__ bfloat16_t *qliB16Qk,
                                            __ubuf__ float *qliB16Weight, __ubuf__ float *qliB16KScale,
                                            __ubuf__ float *qliB16QScale, uint16_t qliB16GSize)
{
    Reg::RegTensor<float> qliB16RegQK[4];
    Reg::RegTensor<bfloat16_t> qliB16RegQKB16[2];
    Reg::RegTensor<float> qliB16RegW;
    Reg::RegTensor<float> qliB16RegwBrc[2];
    Reg::RegTensor<float> qliB16RegQScale;
    Reg::RegTensor<float> qliB16RegKScale[2];
    Reg::RegTensor<float> qliB16RegSum[2];

    Reg::MaskReg qliB16MaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliB16MaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<bfloat16_t> qliB16RegSumBF16;

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliB16Bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliB16Bf16Ctx, qliB16MaskAllB16);

    using CastTrait = Reg::CastTrait;
    static constexpr CastTrait castTraitB162B32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait castTraitB162B32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static CastTrait castTraitF32ToF16_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static CastTrait castTraitF32ToF16_ODD = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    Reg::LoadAlign<float>(qliB16RegW, qliB16Weight);
    Reg::LoadAlign<float>(qliB16RegQScale, qliB16QScale);
    Reg::Mul(qliB16RegW, qliB16RegW, qliB16RegQScale, qliB16MaskAllB32);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(qliB16Weight, qliB16RegW, qliB16MaskAllB32);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

    liV2Vector1::DuplicateZero(qliB16RegSum, qliB16MaskAllB32);

    // interleave load
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(qliB16RegKScale[0], qliB16RegKScale[1], qliB16KScale);

    // Duplicate + Gather方法劣化
    // Relu在cube随路做
    for (uint16_t qliB16I = (uint16_t)(0); qliB16I < qliB16GSize; qliB16I++) {
        // RowStride是256, 行都落在一个bank上
        Reg::LoadAlign<bfloat16_t>(qliB16RegQKB16[0], qliB16Qk + 256 * qliB16I);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(qliB16RegwBrc[0], qliB16Weight + qliB16I);
        // interleave cast
        Reg::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(qliB16RegQK[0], qliB16RegQKB16[0], qliB16MaskAllB16);
        Reg::Cast<float, bfloat16_t, castTraitB162B32_ODD>(qliB16RegQK[1], qliB16RegQKB16[0], qliB16MaskAllB16);
        Reg::MulAddDst(qliB16RegSum[0], qliB16RegQK[0], qliB16RegwBrc[0], qliB16MaskAllB32);
        Reg::MulAddDst(qliB16RegSum[1], qliB16RegQK[1], qliB16RegwBrc[0], qliB16MaskAllB32);
    }

    Reg::Mul(qliB16RegSum[0], qliB16RegSum[0], qliB16RegKScale[0], qliB16MaskAllB32);
    Reg::Mul(qliB16RegSum[1], qliB16RegSum[1], qliB16RegKScale[1], qliB16MaskAllB32);
    // interleave cast back
    Reg::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(qliB16RegSumBF16, qliB16RegSum[1], qliB16MaskAllB32);
    Reg::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(qliB16RegSumBF16, qliB16RegSum[0], qliB16MaskAllB32);

    Reg::RegTensor<uint16_t> qliB16RegOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(qliB16RegOut, qliB16RegSumBF16, qliB16Bf16Ctx, qliB16MaskAllB16);
    // norm load
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliB16Out, qliB16RegOut, qliB16MaskAllB16);
}

__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_,  // out    [S2Base]     [128   ]
                                             const LocalTensor<bfloat16_t> &qk_, // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,          // unused for bfloat16
                                             const LocalTensor<float> &weight_,  // w      [G]          [64    ]
                                             const LocalTensor<float> &kScale_,  // kScale [S2Base]     [128   ]
                                             const LocalTensor<float> &qScale_,  // qScale [G]          [64    ]
                                             const int gSize)                    // G 64
{
    __ubuf__ uint16_t *out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ float *weight = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *qScale = (__ubuf__ float *)qScale_.GetPhyAddr();
    __ubuf__ bfloat16_t *qk = (__ubuf__ bfloat16_t *)qk_.GetPhyAddr();
    __ubuf__ float *kScale = (__ubuf__ float *)kScale_.GetPhyAddr();
    MulWeightAndReduceSumB16VF(out, qk, weight, kScale, qScale, (uint16_t)gSize);
}

// 计算S1=2
// float in uint16 out
__simd_vf__ void MulWeightAndReduceSum2F32VF(__ubuf__ uint16_t *qliTwoF32Out0, __ubuf__ uint16_t *qliTwoF32Out1,
                                             __ubuf__ float *qliTwoF32Qk0, __ubuf__ float *qliTwoF32Qk1,
                                             uint32_t qliTwoF32QkVLStride, __ubuf__ float *qliTwoF32Weight0,
                                             __ubuf__ float *qliTwoF32Weight1, __ubuf__ float *weightTemp,
                                             __ubuf__ float *qliTwoF32QScale0, __ubuf__ float *qliTwoF32QScale1,
                                             __ubuf__ float *qliTwoF32KScale0, uint16_t qliTwoF32GSize)
{
    Reg::RegTensor<float> qliTwoF32RegwBrc[2];
    Reg::RegTensor<float> qliTwoF32RegQK0[2];
    Reg::RegTensor<float> qliTwoF32RegQK1[2];
    Reg::RegTensor<float> qliTwoF32RegW[2];

    Reg::RegTensor<float> qliTwoF32RegQScale[2];
    Reg::RegTensor<float> qliTwoF32RegKScale[2];
    Reg::RegTensor<float> qliTwoF32RegSum0[2];
    Reg::RegTensor<float> qliTwoF32RegSum1[2];
    Reg::MaskReg qliTwoF32MaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliTwoF32MaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliTwoF32Bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliTwoF32Bf16Ctx, qliTwoF32MaskAllB16);

    constexpr static Reg::CastTrait qliTwoF32CastTraitF32ToF16_EVEN = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliTwoF32CastTraitF32ToF16_ODD = {
        Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    // regW[0]与weight1混合使用
    Reg::LoadAlign<float>(qliTwoF32RegW[0], qliTwoF32Weight0);
    Reg::LoadAlign<float>(qliTwoF32RegW[1], qliTwoF32Weight1);
    Reg::LoadAlign<float>(qliTwoF32RegQScale[0], qliTwoF32QScale0);
    Reg::LoadAlign<float>(qliTwoF32RegQScale[1], qliTwoF32QScale1);
    Reg::Mul(qliTwoF32RegW[0], qliTwoF32RegW[0], qliTwoF32RegQScale[0], qliTwoF32MaskAllB32);
    Reg::Mul(qliTwoF32RegW[1], qliTwoF32RegW[1], qliTwoF32RegQScale[1], qliTwoF32MaskAllB32);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(weightTemp, qliTwoF32RegW[1], qliTwoF32MaskAllB32);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    liV2Vector1::DuplicateZero(qliTwoF32RegSum0, qliTwoF32MaskAllB32);
    liV2Vector1::DuplicateZero(qliTwoF32RegSum1, qliTwoF32MaskAllB32);

    Reg::LoadAlign<float>(qliTwoF32RegKScale[0], qliTwoF32KScale0);
    Reg::LoadAlign<float>(qliTwoF32RegKScale[1], qliTwoF32KScale0 + 64);

    for (uint16_t qliTwoF32I = (uint16_t)(0); qliTwoF32I < qliTwoF32GSize; qliTwoF32I++) {
        Reg::LoadAlign<float>(qliTwoF32RegQK0[0], qliTwoF32Qk0 + 128 * qliTwoF32I);
        Reg::LoadAlign<float>(qliTwoF32RegQK0[1], qliTwoF32Qk0 + 128 * qliTwoF32I + qliTwoF32QkVLStride);
        Reg::LoadAlign<float>(qliTwoF32RegQK1[0], qliTwoF32Qk1 + 128 * qliTwoF32I);
        Reg::LoadAlign<float>(qliTwoF32RegQK1[1], qliTwoF32Qk1 + 128 * qliTwoF32I + qliTwoF32QkVLStride);
        // 混合使用对整体性能更好
        liV2Vector1::BroadcastLane(qliTwoF32RegwBrc[0], qliTwoF32RegW[0], qliTwoF32I);
        // Weight无bank冲突，用LoadAlign来提取weight标量
        // 地址空间处理：原 BroadcastLane(ptr) 内联为 LoadAlign BRC，避免 __ubuf__ 传给 __local_mem__ 参数
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(qliTwoF32RegwBrc[1], weightTemp + qliTwoF32I);
        Reg::Relu(qliTwoF32RegQK0[0], qliTwoF32RegQK0[0], qliTwoF32MaskAllB32);
        Reg::Relu(qliTwoF32RegQK0[1], qliTwoF32RegQK0[1], qliTwoF32MaskAllB32);
        Reg::Relu(qliTwoF32RegQK1[0], qliTwoF32RegQK1[0], qliTwoF32MaskAllB32);
        Reg::Relu(qliTwoF32RegQK1[1], qliTwoF32RegQK1[1], qliTwoF32MaskAllB32);
        Reg::MulAddDst(qliTwoF32RegSum0[0], qliTwoF32RegQK0[0], qliTwoF32RegwBrc[0], qliTwoF32MaskAllB32);
        Reg::MulAddDst(qliTwoF32RegSum0[1], qliTwoF32RegQK0[1], qliTwoF32RegwBrc[0], qliTwoF32MaskAllB32);
        Reg::MulAddDst(qliTwoF32RegSum1[0], qliTwoF32RegQK1[0], qliTwoF32RegwBrc[1], qliTwoF32MaskAllB32);
        Reg::MulAddDst(qliTwoF32RegSum1[1], qliTwoF32RegQK1[1], qliTwoF32RegwBrc[1], qliTwoF32MaskAllB32);
    }

    // Apply kScale scaling
    Reg::Mul(qliTwoF32RegSum0[0], qliTwoF32RegSum0[0], qliTwoF32RegKScale[0], qliTwoF32MaskAllB32);
    Reg::Mul(qliTwoF32RegSum0[1], qliTwoF32RegSum0[1], qliTwoF32RegKScale[1], qliTwoF32MaskAllB32);
    Reg::Mul(qliTwoF32RegSum1[0], qliTwoF32RegSum1[0], qliTwoF32RegKScale[0], qliTwoF32MaskAllB32);
    Reg::Mul(qliTwoF32RegSum1[1], qliTwoF32RegSum1[1], qliTwoF32RegKScale[1], qliTwoF32MaskAllB32);

    // Convert to bfloat16 and store output channel
    Reg::RegTensor<bfloat16_t> qliTwoF32RegSumBF16[2];
    Reg::RegTensor<uint16_t> qliTwoF32RegOut[2];
    Reg::DeInterleave(qliTwoF32RegSum0[0], qliTwoF32RegSum0[1], qliTwoF32RegSum0[0], qliTwoF32RegSum0[1]);
    Reg::DeInterleave(qliTwoF32RegSum1[0], qliTwoF32RegSum1[1], qliTwoF32RegSum1[0], qliTwoF32RegSum1[1]);
    Reg::Cast<bfloat16_t, float, qliTwoF32CastTraitF32ToF16_ODD>(qliTwoF32RegSumBF16[0], qliTwoF32RegSum0[1],
                                                                 qliTwoF32MaskAllB32);
    Reg::Cast<bfloat16_t, float, qliTwoF32CastTraitF32ToF16_ODD>(qliTwoF32RegSumBF16[1], qliTwoF32RegSum1[1],
                                                                 qliTwoF32MaskAllB32);
    Reg::Cast<bfloat16_t, float, qliTwoF32CastTraitF32ToF16_EVEN>(qliTwoF32RegSumBF16[0], qliTwoF32RegSum0[0],
                                                                  qliTwoF32MaskAllB32);
    Reg::Cast<bfloat16_t, float, qliTwoF32CastTraitF32ToF16_EVEN>(qliTwoF32RegSumBF16[1], qliTwoF32RegSum1[0],
                                                                  qliTwoF32MaskAllB32);

    liV2Vector1::FloatX2ToSortableKey<bfloat16_t>(qliTwoF32RegOut[0], qliTwoF32RegOut[1], qliTwoF32RegSumBF16[0],
                                                  qliTwoF32RegSumBF16[1], qliTwoF32Bf16Ctx, qliTwoF32MaskAllB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliTwoF32Out0, qliTwoF32RegOut[0], qliTwoF32MaskAllB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliTwoF32Out1, qliTwoF32RegOut[1], qliTwoF32MaskAllB16);
}

// 计算S1=2
// int32 in uint16 out
__simd_vf__ void MulWeightAndReduceSum2Int32VF(__ubuf__ uint16_t *qliTwoI32Out0, __ubuf__ uint16_t *qliTwoI32Out1,
                                               __ubuf__ int32_t *qliTwoI32Qk0, __ubuf__ int32_t *qliTwoI32Qk1,
                                               uint32_t qliTwoI32QkVLStride, __ubuf__ half *qliTwoI32Weight0,
                                               __ubuf__ half *qliTwoI32Weight1, __ubuf__ float *weightTemp,
                                               __ubuf__ half *qliTwoI32QScale0, __ubuf__ half *qliTwoI32QScale1,
                                               __ubuf__ half *qliTwoI32KScale0, uint16_t qliTwoI32GSize)
{
    Reg::RegTensor<float> qliTwoI32RegwBrc[2];
    Reg::RegTensor<float> qliTwoI32RegQK0[2];
    Reg::RegTensor<float> qliTwoI32RegQK1[2];
    Reg::RegTensor<half> qliTwoI32RegQK0Half[2];
    Reg::RegTensor<half> qliTwoI32RegQK1Half[2];
    Reg::RegTensor<int32_t> qliTwoI32RegQK0Int32[2];
    Reg::RegTensor<int32_t> qliTwoI32RegQK1Int32[2];
    Reg::RegTensor<float> qliTwoI32RegW[2];
    Reg::RegTensor<half> qliTwoI32RegWFP16[2];
    Reg::RegTensor<half> qliTwoI32RegWFP16Temp[2];
    Reg::RegTensor<float> qliTwoI32RegQScale[2];
    Reg::RegTensor<half> qliTwoI32RegQScaleFP16[2];
    Reg::RegTensor<float> qliTwoI32RegKScale[2];
    Reg::RegTensor<half> qliTwoI32RegKScaleFP16[2];
    Reg::RegTensor<float> qliTwoI32RegSum0[2];
    Reg::RegTensor<float> qliTwoI32RegSum1[2];
    Reg::MaskReg qliTwoI32MaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliTwoI32MaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliTwoI32Bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliTwoI32Bf16Ctx, qliTwoI32MaskAllB16);

    constexpr static Reg::CastTrait qliTwoI32CastTraitF32ToF16_EVEN = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliTwoI32CastTraitF32ToF16_ODD = {
        Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliTwoI32CastTraitF16ToF32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    constexpr static Reg::CastTrait qliTwoI32CastTraitF32ToF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static Reg::CastTrait qliTwoI32CastTraitInt32ToFP32 = {
        Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(qliTwoI32RegWFP16[0], qliTwoI32Weight0);
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(qliTwoI32RegWFP16[1], qliTwoI32Weight1);
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(qliTwoI32RegQScaleFP16[0], qliTwoI32QScale0);
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(qliTwoI32RegQScaleFP16[1], qliTwoI32QScale1);
    Reg::Cast<float, half, qliTwoI32CastTraitF16ToF32>(qliTwoI32RegW[0], qliTwoI32RegWFP16[0], qliTwoI32MaskAllB16);
    Reg::Cast<float, half, qliTwoI32CastTraitF16ToF32>(qliTwoI32RegW[1], qliTwoI32RegWFP16[1], qliTwoI32MaskAllB16);
    Reg::Cast<float, half, qliTwoI32CastTraitF16ToF32>(qliTwoI32RegQScale[0], qliTwoI32RegQScaleFP16[0],
                                                       qliTwoI32MaskAllB16);
    Reg::Cast<float, half, qliTwoI32CastTraitF16ToF32>(qliTwoI32RegQScale[1], qliTwoI32RegQScaleFP16[1],
                                                       qliTwoI32MaskAllB16);

    Reg::Mul(qliTwoI32RegW[0], qliTwoI32RegW[0], qliTwoI32RegQScale[0], qliTwoI32MaskAllB32);
    Reg::Mul(qliTwoI32RegW[1], qliTwoI32RegW[1], qliTwoI32RegQScale[1], qliTwoI32MaskAllB32);

    // regW[0]与weight1混合使用
    Reg::Cast<half, float, qliTwoI32CastTraitF32ToF16>(qliTwoI32RegWFP16Temp[0], qliTwoI32RegW[0], qliTwoI32MaskAllB32);
    Reg::Cast<float, half, qliTwoI32CastTraitF16ToF32>(qliTwoI32RegW[0], qliTwoI32RegWFP16Temp[0], qliTwoI32MaskAllB16);
    Reg::Cast<half, float, qliTwoI32CastTraitF32ToF16>(qliTwoI32RegWFP16Temp[1], qliTwoI32RegW[1], qliTwoI32MaskAllB32);
    Reg::Cast<float, half, qliTwoI32CastTraitF16ToF32>(qliTwoI32RegW[1], qliTwoI32RegWFP16Temp[1], qliTwoI32MaskAllB16);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(weightTemp, qliTwoI32RegW[1], qliTwoI32MaskAllB32);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    liV2Vector1::DuplicateZero(qliTwoI32RegSum0, qliTwoI32MaskAllB32);
    liV2Vector1::DuplicateZero(qliTwoI32RegSum1, qliTwoI32MaskAllB32);

    LoadKScaleFP16(qliTwoI32RegKScaleFP16, qliTwoI32RegKScale, qliTwoI32MaskAllB16, qliTwoI32KScale0);

    for (uint16_t qliTwoI32I = (uint16_t)(0); qliTwoI32I < qliTwoI32GSize; qliTwoI32I++) {
        Reg::LoadAlign<int32_t>(qliTwoI32RegQK0Int32[0], qliTwoI32Qk0 + 128 * qliTwoI32I);
        Reg::Cast<float, int32_t, qliTwoI32CastTraitInt32ToFP32>(qliTwoI32RegQK0[0], qliTwoI32RegQK0Int32[0],
                                                                 qliTwoI32MaskAllB32);
        Reg::LoadAlign<int32_t>(qliTwoI32RegQK0Int32[1], qliTwoI32Qk0 + 128 * qliTwoI32I + qliTwoI32QkVLStride);
        Reg::Cast<float, int32_t, qliTwoI32CastTraitInt32ToFP32>(qliTwoI32RegQK0[1], qliTwoI32RegQK0Int32[1],
                                                                 qliTwoI32MaskAllB32);
        Reg::LoadAlign<int32_t>(qliTwoI32RegQK1Int32[0], qliTwoI32Qk1 + 128 * qliTwoI32I);
        Reg::Cast<float, int32_t, qliTwoI32CastTraitInt32ToFP32>(qliTwoI32RegQK1[0], qliTwoI32RegQK1Int32[0],
                                                                 qliTwoI32MaskAllB32);
        Reg::LoadAlign<int32_t>(qliTwoI32RegQK1Int32[1], qliTwoI32Qk1 + 128 * qliTwoI32I + qliTwoI32QkVLStride);
        Reg::Cast<float, int32_t, qliTwoI32CastTraitInt32ToFP32>(qliTwoI32RegQK1[1], qliTwoI32RegQK1Int32[1],
                                                                 qliTwoI32MaskAllB32);
        // 混合使用对整体性能更好
        liV2Vector1::BroadcastLane(qliTwoI32RegwBrc[0], qliTwoI32RegW[0], qliTwoI32I);
        // Weight无bank冲突，用LoadAlign来提取weight标量
        // 地址空间处理：原 BroadcastLane(ptr) 内联为 LoadAlign BRC，避免 __ubuf__ 传给 __local_mem__ 参数
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(qliTwoI32RegwBrc[1], weightTemp + qliTwoI32I);
        Reg::Relu(qliTwoI32RegQK0[0], qliTwoI32RegQK0[0], qliTwoI32MaskAllB32);
        Reg::Relu(qliTwoI32RegQK0[1], qliTwoI32RegQK0[1], qliTwoI32MaskAllB32);
        Reg::Relu(qliTwoI32RegQK1[0], qliTwoI32RegQK1[0], qliTwoI32MaskAllB32);
        Reg::Relu(qliTwoI32RegQK1[1], qliTwoI32RegQK1[1], qliTwoI32MaskAllB32);

        CastFP32ToFP16ToFP32(qliTwoI32RegQK0, qliTwoI32RegQK0Half, qliTwoI32MaskAllB32);
        CastFP32ToFP16ToFP32(qliTwoI32RegQK1, qliTwoI32RegQK1Half, qliTwoI32MaskAllB32);

        Reg::MulAddDst(qliTwoI32RegSum0[0], qliTwoI32RegQK0[0], qliTwoI32RegwBrc[0], qliTwoI32MaskAllB32);
        Reg::MulAddDst(qliTwoI32RegSum0[1], qliTwoI32RegQK0[1], qliTwoI32RegwBrc[0], qliTwoI32MaskAllB32);
        Reg::MulAddDst(qliTwoI32RegSum1[0], qliTwoI32RegQK1[0], qliTwoI32RegwBrc[1], qliTwoI32MaskAllB32);
        Reg::MulAddDst(qliTwoI32RegSum1[1], qliTwoI32RegQK1[1], qliTwoI32RegwBrc[1], qliTwoI32MaskAllB32);
    }

    // Apply kScale scaling
    Reg::Mul(qliTwoI32RegSum0[0], qliTwoI32RegSum0[0], qliTwoI32RegKScale[0], qliTwoI32MaskAllB32);
    Reg::Mul(qliTwoI32RegSum0[1], qliTwoI32RegSum0[1], qliTwoI32RegKScale[1], qliTwoI32MaskAllB32);
    Reg::Mul(qliTwoI32RegSum1[0], qliTwoI32RegSum1[0], qliTwoI32RegKScale[0], qliTwoI32MaskAllB32);
    Reg::Mul(qliTwoI32RegSum1[1], qliTwoI32RegSum1[1], qliTwoI32RegKScale[1], qliTwoI32MaskAllB32);

    // Convert to bfloat16 and store output channel
    Reg::RegTensor<bfloat16_t> qliTwoI32RegSumBF16[2];
    Reg::RegTensor<uint16_t> qliTwoI32RegOut[2];
    Reg::DeInterleave(qliTwoI32RegSum0[0], qliTwoI32RegSum0[1], qliTwoI32RegSum0[0], qliTwoI32RegSum0[1]);
    Reg::DeInterleave(qliTwoI32RegSum1[0], qliTwoI32RegSum1[1], qliTwoI32RegSum1[0], qliTwoI32RegSum1[1]);
    Reg::Cast<bfloat16_t, float, qliTwoI32CastTraitF32ToF16_ODD>(qliTwoI32RegSumBF16[0], qliTwoI32RegSum0[1],
                                                                 qliTwoI32MaskAllB32);
    Reg::Cast<bfloat16_t, float, qliTwoI32CastTraitF32ToF16_ODD>(qliTwoI32RegSumBF16[1], qliTwoI32RegSum1[1],
                                                                 qliTwoI32MaskAllB32);
    Reg::Cast<bfloat16_t, float, qliTwoI32CastTraitF32ToF16_EVEN>(qliTwoI32RegSumBF16[0], qliTwoI32RegSum0[0],
                                                                  qliTwoI32MaskAllB32);
    Reg::Cast<bfloat16_t, float, qliTwoI32CastTraitF32ToF16_EVEN>(qliTwoI32RegSumBF16[1], qliTwoI32RegSum1[0],
                                                                  qliTwoI32MaskAllB32);

    liV2Vector1::FloatX2ToSortableKey<bfloat16_t>(qliTwoI32RegOut[0], qliTwoI32RegOut[1], qliTwoI32RegSumBF16[0],
                                                  qliTwoI32RegSumBF16[1], qliTwoI32Bf16Ctx, qliTwoI32MaskAllB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliTwoI32Out0, qliTwoI32RegOut[0], qliTwoI32MaskAllB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliTwoI32Out1, qliTwoI32RegOut[1], qliTwoI32MaskAllB16);
}

__aicore__ inline void MulWeightAndReduceSum2(const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
                                              uint32_t outStride,
                                              const LocalTensor<float> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
                                              uint32_t qkVLStride, uint32_t qkStride,
                                              const LocalTensor<float> &weight_, // w      [2, G]          [64    ]
                                              uint32_t weightStride, const LocalTensor<float> &weightTemp_,
                                              const LocalTensor<float> &kScale_, // kScale [S2Base]        [128   ]
                                              uint32_t kScaleStride,
                                              const LocalTensor<float> &qScale_, // qScale [2, G]          [64    ]
                                              uint32_t qScaleStride,
                                              const int gSize) // G 64
{
    __ubuf__ float *weight0 = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *weightTemp = (__ubuf__ float *)weightTemp_.GetPhyAddr();
    __ubuf__ float *qScale0 = (__ubuf__ float *)qScale_.GetPhyAddr();
    __ubuf__ float *kScale0 = (__ubuf__ float *)kScale_.GetPhyAddr();
    __ubuf__ float *qk0 = (__ubuf__ float *)qk_.GetPhyAddr();
    __ubuf__ uint16_t *out0 = (__ubuf__ uint16_t *)out_.GetPhyAddr();

    __ubuf__ float *weight1 = weight0 + weightStride;
    __ubuf__ float *qScale1 = qScale0 + qScaleStride;
    __ubuf__ float *qk1 = qk0 + qkStride;
    // kScaleStride is zero
    __ubuf__ uint16_t *out1 = out0 + outStride;

    MulWeightAndReduceSum2F32VF(out0, out1, qk0, qk1, qkVLStride, weight0, weight1, weightTemp, qScale0, qScale1,
                                kScale0, (uint16_t)gSize);
}

__aicore__ inline void MulWeightAndReduceSum2(const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
                                              uint32_t outStride,
                                              const LocalTensor<int32_t> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
                                              uint32_t qkVLStride, uint32_t qkStride,
                                              const LocalTensor<half> &weight_, // w      [2, G]          [64    ]
                                              uint32_t weightStride, const LocalTensor<float> &weightTemp_,
                                              const LocalTensor<half> &kScale_, // kScale [S2Base]        [128   ]
                                              uint32_t kScaleStride,
                                              const LocalTensor<half> &qScale_, // qScale [2, G]          [64    ]
                                              uint32_t qScaleStride,
                                              const int gSize) // G 64
{
    __ubuf__ half *weight0 = (__ubuf__ half *)weight_.GetPhyAddr();
    __ubuf__ float *weightTemp = (__ubuf__ float *)weightTemp_.GetPhyAddr();
    __ubuf__ half *qScale0 = (__ubuf__ half *)qScale_.GetPhyAddr();
    __ubuf__ half *kScale0 = (__ubuf__ half *)kScale_.GetPhyAddr();
    __ubuf__ int32_t *qk0 = (__ubuf__ int32_t *)qk_.GetPhyAddr();
    __ubuf__ uint16_t *out0 = (__ubuf__ uint16_t *)out_.GetPhyAddr();

    __ubuf__ half *weight1 = weight0 + weightStride;
    __ubuf__ half *qScale1 = qScale0 + qScaleStride;
    __ubuf__ int32_t *qk1 = qk0 + qkStride;
    // kScaleStride is zero
    __ubuf__ uint16_t *out1 = out0 + outStride;

    MulWeightAndReduceSum2Int32VF(out0, out1, qk0, qk1, qkVLStride, weight0, weight1, weightTemp, qScale0, qScale1,
                                  kScale0, (uint16_t)gSize);
}

// 计算S1=2
// bfloat16 in uint16 out
__simd_vf__ void MulWeightAndReduceSum2B16VF(__ubuf__ uint16_t *qliTwoB16Out0, __ubuf__ uint16_t *qliTwoB16Out1,
                                             __ubuf__ bfloat16_t *qliTwoB16Qk0, __ubuf__ bfloat16_t *qliTwoB16Qk1,
                                             __ubuf__ float *qliTwoB16Weight0, __ubuf__ float *qliTwoB16Weight1,
                                             __ubuf__ float *weightTemp0, __ubuf__ float *weightTemp1,
                                             __ubuf__ float *qliTwoB16QScale0, __ubuf__ float *qliTwoB16QScale1,
                                             __ubuf__ float *qliTwoB16KScale0, uint16_t qliTwoB16GSize)
{
    Reg::RegTensor<float> qliTwoB16RegwBrc[2];
    Reg::RegTensor<float> qliTwoB16RegQK0[2];
    Reg::RegTensor<float> qliTwoB16RegQK1[2];
    Reg::RegTensor<float> qliTwoB16RegW[2];
    Reg::RegTensor<bfloat16_t> qliTwoB16RegQKB16[2];

    Reg::RegTensor<float> qliTwoB16RegQScale[2];
    Reg::RegTensor<float> qliTwoB16RegKScale[2];
    Reg::RegTensor<float> qliTwoB16RegSum0[2];
    Reg::RegTensor<float> qliTwoB16RegSum1[2];
    Reg::MaskReg qliTwoB16MaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliTwoB16MaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliTwoB16Bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliTwoB16Bf16Ctx, qliTwoB16MaskAllB16);

    using CastTrait = Reg::CastTrait;
    static constexpr CastTrait castTraitB162B32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait castTraitB162B32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static Reg::CastTrait qliTwoB16CastTraitF32ToF16_EVEN = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliTwoB16CastTraitF32ToF16_ODD = {
        Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    Reg::LoadAlign<float>(qliTwoB16RegW[0], qliTwoB16Weight0);
    Reg::LoadAlign<float>(qliTwoB16RegW[1], qliTwoB16Weight1);
    Reg::LoadAlign<float>(qliTwoB16RegQScale[0], qliTwoB16QScale0);
    Reg::LoadAlign<float>(qliTwoB16RegQScale[1], qliTwoB16QScale1);
    Reg::Mul(qliTwoB16RegW[0], qliTwoB16RegW[0], qliTwoB16RegQScale[0], qliTwoB16MaskAllB32);
    Reg::Mul(qliTwoB16RegW[1], qliTwoB16RegW[1], qliTwoB16RegQScale[1], qliTwoB16MaskAllB32);
    // 读写依赖，寄存器可以保序
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(weightTemp0, qliTwoB16RegW[0], qliTwoB16MaskAllB32);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(weightTemp1, qliTwoB16RegW[1], qliTwoB16MaskAllB32);
    liV2Vector1::DuplicateZero(qliTwoB16RegSum0, qliTwoB16MaskAllB32);
    liV2Vector1::DuplicateZero(qliTwoB16RegSum1, qliTwoB16MaskAllB32);

    // interleave load
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(qliTwoB16RegKScale[0], qliTwoB16RegKScale[1],
                                                          qliTwoB16KScale0);

    for (uint16_t qliTwoB16I = (uint16_t)(0); qliTwoB16I < qliTwoB16GSize; qliTwoB16I++) {
        // RowStride是256, 行都落在一个bank上
        Reg::LoadAlign<bfloat16_t>(qliTwoB16RegQKB16[0], qliTwoB16Qk0 + 256 * qliTwoB16I);
        // RowStride是256, 行都落在一个bank上
        Reg::LoadAlign<bfloat16_t>(qliTwoB16RegQKB16[1], qliTwoB16Qk1 + 256 * qliTwoB16I);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(qliTwoB16RegwBrc[0], weightTemp0 + qliTwoB16I);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(qliTwoB16RegwBrc[1], weightTemp1 + qliTwoB16I);
        // interleave cast
        Reg::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(qliTwoB16RegQK0[0], qliTwoB16RegQKB16[0],
                                                            qliTwoB16MaskAllB32);
        Reg::Cast<float, bfloat16_t, castTraitB162B32_ODD>(qliTwoB16RegQK0[1], qliTwoB16RegQKB16[0],
                                                           qliTwoB16MaskAllB32);
        Reg::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(qliTwoB16RegQK1[0], qliTwoB16RegQKB16[1],
                                                            qliTwoB16MaskAllB32);
        Reg::Cast<float, bfloat16_t, castTraitB162B32_ODD>(qliTwoB16RegQK1[1], qliTwoB16RegQKB16[1],
                                                           qliTwoB16MaskAllB32);
        Reg::MulAddDst(qliTwoB16RegSum0[0], qliTwoB16RegQK0[0], qliTwoB16RegwBrc[0], qliTwoB16MaskAllB32);
        Reg::MulAddDst(qliTwoB16RegSum0[1], qliTwoB16RegQK0[1], qliTwoB16RegwBrc[0], qliTwoB16MaskAllB32);
        Reg::MulAddDst(qliTwoB16RegSum1[0], qliTwoB16RegQK1[0], qliTwoB16RegwBrc[1], qliTwoB16MaskAllB32);
        Reg::MulAddDst(qliTwoB16RegSum1[1], qliTwoB16RegQK1[1], qliTwoB16RegwBrc[1], qliTwoB16MaskAllB32);
    }

    // Apply kScale scaling
    Reg::Mul(qliTwoB16RegSum0[0], qliTwoB16RegSum0[0], qliTwoB16RegKScale[0], qliTwoB16MaskAllB32);
    Reg::Mul(qliTwoB16RegSum0[1], qliTwoB16RegSum0[1], qliTwoB16RegKScale[1], qliTwoB16MaskAllB32);
    Reg::Mul(qliTwoB16RegSum1[0], qliTwoB16RegSum1[0], qliTwoB16RegKScale[0], qliTwoB16MaskAllB32);
    Reg::Mul(qliTwoB16RegSum1[1], qliTwoB16RegSum1[1], qliTwoB16RegKScale[1], qliTwoB16MaskAllB32);

    // Convert to bfloat16 and store output channel
    Reg::RegTensor<bfloat16_t> qliTwoB16RegSumBF16[2];
    Reg::RegTensor<uint16_t> qliTwoB16RegOut[2];
    Reg::Cast<bfloat16_t, float, qliTwoB16CastTraitF32ToF16_ODD>(qliTwoB16RegSumBF16[0], qliTwoB16RegSum0[1],
                                                                 qliTwoB16MaskAllB32);
    Reg::Cast<bfloat16_t, float, qliTwoB16CastTraitF32ToF16_ODD>(qliTwoB16RegSumBF16[1], qliTwoB16RegSum1[1],
                                                                 qliTwoB16MaskAllB32);
    Reg::Cast<bfloat16_t, float, qliTwoB16CastTraitF32ToF16_EVEN>(qliTwoB16RegSumBF16[0], qliTwoB16RegSum0[0],
                                                                  qliTwoB16MaskAllB32);
    Reg::Cast<bfloat16_t, float, qliTwoB16CastTraitF32ToF16_EVEN>(qliTwoB16RegSumBF16[1], qliTwoB16RegSum1[0],
                                                                  qliTwoB16MaskAllB32);

    liV2Vector1::FloatX2ToSortableKey<bfloat16_t>(qliTwoB16RegOut[0], qliTwoB16RegOut[1], qliTwoB16RegSumBF16[0],
                                                  qliTwoB16RegSumBF16[1], qliTwoB16Bf16Ctx, qliTwoB16MaskAllB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliTwoB16Out0, qliTwoB16RegOut[0], qliTwoB16MaskAllB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliTwoB16Out1, qliTwoB16RegOut[1], qliTwoB16MaskAllB16);
}

__aicore__ inline void MulWeightAndReduceSum2(const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
                                              uint32_t outStride,
                                              const LocalTensor<bfloat16_t> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
                                              uint32_t qkVLStride,
                                              uint32_t qkStride,                 // gSize * 256
                                              const LocalTensor<float> &weight_, // w      [2, G]          [64    ]
                                              uint32_t weightStride, const LocalTensor<float> &weightTemp_,
                                              const LocalTensor<float> &kScale_, // kScale [S2Base]        [128   ]
                                              uint32_t kScaleStride,
                                              const LocalTensor<float> &qScale_, // qScale [2, G]          [64    ]
                                              uint32_t qScaleStride,
                                              const int gSize) // G 64
{
    __ubuf__ float *weight0 = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *weightTemp0 = (__ubuf__ float *)weightTemp_.GetPhyAddr();
    __ubuf__ float *qScale0 = (__ubuf__ float *)qScale_.GetPhyAddr();
    __ubuf__ float *kScale0 = (__ubuf__ float *)kScale_.GetPhyAddr();
    __ubuf__ bfloat16_t *qk0 = (__ubuf__ bfloat16_t *)qk_.GetPhyAddr();
    __ubuf__ uint16_t *out0 = (__ubuf__ uint16_t *)out_.GetPhyAddr();

    __ubuf__ float *weightTemp1 = weightTemp0 + weightStride;
    __ubuf__ float *weight1 = weight0 + weightStride;
    __ubuf__ float *qScale1 = qScale0 + qScaleStride;
    __ubuf__ bfloat16_t *qk1 = qk0 + qkStride;
    // kScaleStride is zero
    __ubuf__ uint16_t *out1 = out0 + outStride;

    MulWeightAndReduceSum2B16VF(out0, out1, qk0, qk1, weight0, weight1, weightTemp0, weightTemp1, qScale0, qScale1,
                                kScale0, (uint16_t)gSize);
}

template <typename QK_T, typename SCORE_T, typename WEIGHT_T, typename SCALE_T>
__aicore__ inline void BatchMulWeightAndReduceSum(
    const LocalTensor<SCORE_T> &qliBatchOut, // out    [S2Base]     [128   ]
    uint32_t qliBatchOutStride,
    const LocalTensor<QK_T> &qliBatchQk, // q*k^t  [G, S2Base]  [64 128]
    uint32_t qliBatchQkVLStride, uint32_t qliBatchQkStride,
    const LocalTensor<WEIGHT_T> &qliBatchWeight, // w      [G]      [64    ]
    uint32_t qliBatchWeightStride, const LocalTensor<float> &qliBatchWeightTemp,
    const LocalTensor<SCALE_T> &qliBatchKScale, // kScale [S2Base]    [128   ]
    uint32_t qliBatchKScaleStride,
    const LocalTensor<SCALE_T> &qliBatchQScale, // qScale [G]         [64    ]
    uint32_t qliBatchQScaleStride,
    const int qliBatchGSize, // G 64
    const int qliBatchSize)
{
    // 暂只支持这两种情况, 后续改成循环
    if (qliBatchSize != 2 && qliBatchSize != 1) {
        return;
    }
    if (qliBatchSize == 2) {
        MulWeightAndReduceSum2(qliBatchOut, qliBatchOutStride, qliBatchQk, qliBatchQkVLStride, qliBatchQkStride,
                               qliBatchWeight, qliBatchWeightStride, qliBatchWeightTemp, qliBatchKScale,
                               qliBatchKScaleStride, qliBatchQScale, qliBatchQScaleStride, qliBatchGSize);
    } else {
        MulWeightAndReduceSum(qliBatchOut, qliBatchQk, qliBatchQkVLStride, qliBatchWeight, qliBatchKScale,
                              qliBatchQScale, qliBatchGSize);
    }
}

// per_tensor与MX共用的weight加权归约实现，WITH_SCALE控制是否额外应用scalar scale
// float in uint16 out
template <bool WITH_SCALE>
__simd_vf__ void MulWeightAndReduceSumOptionalScaleGSizeEvenVF(__ubuf__ uint16_t *qliOptEvenOut,
                                                               __ubuf__ float *qliOptEvenQk,
                                                               uint32_t qliOptEvenQkVLStride,
                                                               __ubuf__ float *qliOptEvenWeight,
                                                               float qliOptEvenKScaleValue, float qliOptEvenQScaleValue,
                                                               uint16_t qliOptEvenGSize)
{
    if constexpr (!WITH_SCALE) {
        (void)qliOptEvenKScaleValue;
        (void)qliOptEvenQScaleValue;
    }

    Reg::RegTensor<float> qliOptEvenRegwBrc;
    Reg::RegTensor<float> qliOptEvenRegQK[2];
    Reg::RegTensor<float> qliOptEvenRegW;
    Reg::RegTensor<float> qliOptEvenRegSum0[2];
    Reg::RegTensor<float> qliOptEvenRegSum1[2];
    Reg::MaskReg qliOptEvenMaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliOptEvenMaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliOptEvenBf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliOptEvenBf16Ctx, qliOptEvenMaskAllB16);

    constexpr static Reg::CastTrait qliOptEvenCastTraitF32ToF16_EVEN = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliOptEvenCastTraitF32ToF16_ODD = {
        Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    Reg::LoadAlign<float>(qliOptEvenRegW, qliOptEvenWeight);
    if constexpr (WITH_SCALE) {
        Reg::Muls(qliOptEvenRegW, qliOptEvenRegW, qliOptEvenQScaleValue, qliOptEvenMaskAllB32);
    }

    liV2Vector1::DuplicateZero(qliOptEvenRegSum0, qliOptEvenMaskAllB32);
    liV2Vector1::DuplicateZero(qliOptEvenRegSum1, qliOptEvenMaskAllB32);

    // unroll2
    for (uint16_t qliOptEvenI = (uint16_t)(0); qliOptEvenI < qliOptEvenGSize; qliOptEvenI += 2) {
        Reg::LoadAlign<float>(qliOptEvenRegQK[0],
                              qliOptEvenQk + 128 * qliOptEvenI); // RowStride是128, 行都落在一个bank上
        Reg::LoadAlign<float>(qliOptEvenRegQK[1], qliOptEvenQk + 128 * qliOptEvenI + qliOptEvenQkVLStride);
        liV2Vector1::BroadcastLane(qliOptEvenRegwBrc, qliOptEvenRegW, qliOptEvenI);
        liV2Vector1::WeightedAccum(qliOptEvenRegSum0, qliOptEvenRegQK, qliOptEvenRegwBrc, qliOptEvenMaskAllB32);

        Reg::LoadAlign<float>(qliOptEvenRegQK[0], qliOptEvenQk + 128 * qliOptEvenI + 128);
        Reg::LoadAlign<float>(qliOptEvenRegQK[1], qliOptEvenQk + 128 * qliOptEvenI + 128 + qliOptEvenQkVLStride);
        liV2Vector1::BroadcastLane(qliOptEvenRegwBrc, qliOptEvenRegW, qliOptEvenI + 1);
        liV2Vector1::WeightedAccum(qliOptEvenRegSum1, qliOptEvenRegQK, qliOptEvenRegwBrc, qliOptEvenMaskAllB32);
    }

    Reg::Add(qliOptEvenRegSum0[0], qliOptEvenRegSum0[0], qliOptEvenRegSum1[0], qliOptEvenMaskAllB32);
    Reg::Add(qliOptEvenRegSum0[1], qliOptEvenRegSum0[1], qliOptEvenRegSum1[1], qliOptEvenMaskAllB32);

    if constexpr (WITH_SCALE) {
        Reg::Muls(qliOptEvenRegSum0[0], qliOptEvenRegSum0[0], qliOptEvenKScaleValue, qliOptEvenMaskAllB32);
        Reg::Muls(qliOptEvenRegSum0[1], qliOptEvenRegSum0[1], qliOptEvenKScaleValue, qliOptEvenMaskAllB32);
    }

    Reg::RegTensor<bfloat16_t> qliOptEvenRegSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    Reg::DeInterleave(qliOptEvenRegSum0[0], qliOptEvenRegSum0[1], qliOptEvenRegSum0[0], qliOptEvenRegSum0[1]);
    Reg::Cast<bfloat16_t, float, qliOptEvenCastTraitF32ToF16_ODD>(qliOptEvenRegSumBF16, qliOptEvenRegSum0[1],
                                                                  qliOptEvenMaskAllB32);
    Reg::Cast<bfloat16_t, float, qliOptEvenCastTraitF32ToF16_EVEN>(qliOptEvenRegSumBF16, qliOptEvenRegSum0[0],
                                                                   qliOptEvenMaskAllB32);

    Reg::RegTensor<uint16_t> qliOptEvenRegOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(qliOptEvenRegOut, qliOptEvenRegSumBF16, qliOptEvenBf16Ctx,
                                                qliOptEvenMaskAllB16);
    // normal store
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliOptEvenOut, qliOptEvenRegOut, qliOptEvenMaskAllB16);
}

template <bool WITH_SCALE>
__simd_vf__ void MulWeightAndReduceSumOptionalScaleGSizeOddVF(__ubuf__ uint16_t *qliOptOddOut,
                                                              __ubuf__ float *qliOptOddQk, uint32_t qliOptOddQkVLStride,
                                                              __ubuf__ float *qliOptOddWeight,
                                                              float qliOptOddKScaleValue, float qliOptOddQScaleValue,
                                                              uint16_t qliOptOddGSize)
{
    if constexpr (!WITH_SCALE) {
        (void)qliOptOddKScaleValue;
        (void)qliOptOddQScaleValue;
    }

    Reg::RegTensor<float> qliOptOddRegwBrc;
    Reg::RegTensor<float> qliOptOddRegQK[2];
    Reg::RegTensor<float> qliOptOddRegW;
    Reg::RegTensor<float> qliOptOddRegSum0[2];
    Reg::RegTensor<float> qliOptOddRegSum1[2];
    Reg::MaskReg qliOptOddMaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliOptOddMaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliOptOddBf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliOptOddBf16Ctx, qliOptOddMaskAllB16);

    constexpr static Reg::CastTrait qliOptOddCastTraitF32ToF16_EVEN = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliOptOddCastTraitF32ToF16_ODD = {
        Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    Reg::LoadAlign<float>(qliOptOddRegW, qliOptOddWeight);
    if constexpr (WITH_SCALE) {
        Reg::Muls(qliOptOddRegW, qliOptOddRegW, qliOptOddQScaleValue, qliOptOddMaskAllB32);
    }

    liV2Vector1::DuplicateZero(qliOptOddRegSum0, qliOptOddMaskAllB32);
    liV2Vector1::DuplicateZero(qliOptOddRegSum1, qliOptOddMaskAllB32);

    // unroll2
    for (uint16_t qliOptOddI = (uint16_t)(0); (uint16_t)(qliOptOddI + 1) < qliOptOddGSize; qliOptOddI += 2) {
        Reg::LoadAlign<float>(qliOptOddRegQK[0], qliOptOddQk + 128 * qliOptOddI); // RowStride是128, 行都落在一个bank上
        Reg::LoadAlign<float>(qliOptOddRegQK[1], qliOptOddQk + 128 * qliOptOddI + qliOptOddQkVLStride);
        liV2Vector1::BroadcastLane(qliOptOddRegwBrc, qliOptOddRegW, qliOptOddI);
        liV2Vector1::WeightedAccum(qliOptOddRegSum0, qliOptOddRegQK, qliOptOddRegwBrc, qliOptOddMaskAllB32);

        Reg::LoadAlign<float>(qliOptOddRegQK[0], qliOptOddQk + 128 * qliOptOddI + 128);
        Reg::LoadAlign<float>(qliOptOddRegQK[1], qliOptOddQk + 128 * qliOptOddI + 128 + qliOptOddQkVLStride);
        liV2Vector1::BroadcastLane(qliOptOddRegwBrc, qliOptOddRegW, qliOptOddI + 1);
        liV2Vector1::WeightedAccum(qliOptOddRegSum1, qliOptOddRegQK, qliOptOddRegwBrc, qliOptOddMaskAllB32);
    }

    Reg::LoadAlign<float>(qliOptOddRegQK[0],
                          qliOptOddQk + 128 * (qliOptOddGSize - 1)); // RowStride是128, 行都落在一个bank上
    Reg::LoadAlign<float>(qliOptOddRegQK[1], qliOptOddQk + 128 * (qliOptOddGSize - 1) + qliOptOddQkVLStride);
    liV2Vector1::BroadcastLane(qliOptOddRegwBrc, qliOptOddRegW, qliOptOddGSize - 1);
    liV2Vector1::WeightedAccum(qliOptOddRegSum0, qliOptOddRegQK, qliOptOddRegwBrc, qliOptOddMaskAllB32);

    Reg::Add(qliOptOddRegSum0[0], qliOptOddRegSum0[0], qliOptOddRegSum1[0], qliOptOddMaskAllB32);
    Reg::Add(qliOptOddRegSum0[1], qliOptOddRegSum0[1], qliOptOddRegSum1[1], qliOptOddMaskAllB32);

    if constexpr (WITH_SCALE) {
        Reg::Muls(qliOptOddRegSum0[0], qliOptOddRegSum0[0], qliOptOddKScaleValue, qliOptOddMaskAllB32);
        Reg::Muls(qliOptOddRegSum0[1], qliOptOddRegSum0[1], qliOptOddKScaleValue, qliOptOddMaskAllB32);
    }

    Reg::RegTensor<bfloat16_t> qliOptOddRegSumBF16;
    // interleave cast ==> regSum[1] high regSum[0] low
    Reg::DeInterleave(qliOptOddRegSum0[0], qliOptOddRegSum0[1], qliOptOddRegSum0[0], qliOptOddRegSum0[1]);
    Reg::Cast<bfloat16_t, float, qliOptOddCastTraitF32ToF16_ODD>(qliOptOddRegSumBF16, qliOptOddRegSum0[1],
                                                                 qliOptOddMaskAllB32);
    Reg::Cast<bfloat16_t, float, qliOptOddCastTraitF32ToF16_EVEN>(qliOptOddRegSumBF16, qliOptOddRegSum0[0],
                                                                  qliOptOddMaskAllB32);

    Reg::RegTensor<uint16_t> qliOptOddRegOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(qliOptOddRegOut, qliOptOddRegSumBF16, qliOptOddBf16Ctx,
                                                qliOptOddMaskAllB16);
    // normal store
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliOptOddOut, qliOptOddRegOut, qliOptOddMaskAllB16);
}

template <bool WITH_SCALE>
__aicore__ inline void MulWeightAndReduceSumOptionalScaleImpl(
    const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
    const LocalTensor<float> &qk_,     // q*k^t  [G, S2Base]  [64 128]
    const uint32_t qkVLStride,
    const LocalTensor<float> &weight_, // w      [G]          [64    ]
    const float kScaleValue,           // kScale scalar
    const float qScaleValue,           // qScale scalar
    const int gSize)                   // G 64
{
    __ubuf__ uint16_t *out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ float *weight = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *qk = (__ubuf__ float *)qk_.GetPhyAddr();
    if (gSize % 2 == 0) {
        MulWeightAndReduceSumOptionalScaleGSizeEvenVF<WITH_SCALE>(out, qk, qkVLStride, weight, kScaleValue, qScaleValue,
                                                                  (uint16_t)gSize);
    } else {
        MulWeightAndReduceSumOptionalScaleGSizeOddVF<WITH_SCALE>(out, qk, qkVLStride, weight, kScaleValue, qScaleValue,
                                                                 (uint16_t)gSize);
    }
}

__aicore__ inline void MulWeightAndReduceSumPerTensor(const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
                                                      const LocalTensor<float> &qk_,     // q*k^t  [G, S2Base]  [64 128]
                                                      const uint32_t qkVLStride,
                                                      const LocalTensor<float> &weight_, // w      [G]          [64    ]
                                                      const float kScaleValue,           // kScale scalar
                                                      const float qScaleValue,           // qScale scalar
                                                      const int gSize)                   // G 64
{
    MulWeightAndReduceSumOptionalScaleImpl<true>(out_, qk_, qkVLStride, weight_, kScaleValue, qScaleValue, gSize);
}

__aicore__ inline void MulWeightAndReduceSumMX(const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
                                               const LocalTensor<float> &qk_,     // q*k^t  [G, S2Base]  [64 128]
                                               const uint32_t qkVLStride,
                                               const LocalTensor<float> &weight_, // w      [G]          [64    ]
                                               const int gSize)                   // G 64
{
    MulWeightAndReduceSumOptionalScaleImpl<false>(out_, qk_, qkVLStride, weight_, 1.0f, 1.0f, gSize);
}

// 计算S1=2
// float in uint16 out
template <bool WITH_SCALE>
__simd_vf__ void MulWeightAndReduceSumOptionalScale2VF(__ubuf__ uint16_t *qliOptTwoOut0,
                                                       __ubuf__ uint16_t *qliOptTwoOut1, __ubuf__ float *qliOptTwoQk0,
                                                       __ubuf__ float *qliOptTwoQk1, uint32_t qliOptTwoQkVLStride,
                                                       __ubuf__ float *qliOptTwoWeight0,
                                                       __ubuf__ float *qliOptTwoWeight1,
                                                       __ubuf__ float *qliOptTwoWeightTemp, float qliOptTwoKScaleValue,
                                                       float qliOptTwoQScaleValue, uint16_t qliOptTwoGSize)
{
    if constexpr (!WITH_SCALE) {
        (void)qliOptTwoKScaleValue;
        (void)qliOptTwoQScaleValue;
    }

    Reg::RegTensor<float> qliOptTwoRegwBrc[2];
    Reg::RegTensor<float> qliOptTwoRegQK0[2];
    Reg::RegTensor<float> qliOptTwoRegQK1[2];
    Reg::RegTensor<float> qliOptTwoRegW[2];

    Reg::RegTensor<float> qliOptTwoRegSum0[2];
    Reg::RegTensor<float> qliOptTwoRegSum1[2];
    Reg::MaskReg qliOptTwoMaskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliOptTwoMaskAllB16 = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> qliOptTwoBf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(qliOptTwoBf16Ctx, qliOptTwoMaskAllB16);

    constexpr static Reg::CastTrait qliOptTwoCastTraitF32ToF16_EVEN = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait qliOptTwoCastTraitF32ToF16_ODD = {
        Reg::RegLayout::ONE, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    Reg::LoadAlign<float>(qliOptTwoRegW[0], qliOptTwoWeight0);
    Reg::LoadAlign<float>(qliOptTwoRegW[1], qliOptTwoWeight1);
    if constexpr (WITH_SCALE) {
        Reg::Muls(qliOptTwoRegW[0], qliOptTwoRegW[0], qliOptTwoQScaleValue, qliOptTwoMaskAllB32);
        Reg::Muls(qliOptTwoRegW[1], qliOptTwoRegW[1], qliOptTwoQScaleValue, qliOptTwoMaskAllB32);
    }
    // regW[0]与weight1混合使用
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(qliOptTwoWeightTemp, qliOptTwoRegW[1], qliOptTwoMaskAllB32);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    liV2Vector1::DuplicateZero(qliOptTwoRegSum0, qliOptTwoMaskAllB32);
    liV2Vector1::DuplicateZero(qliOptTwoRegSum1, qliOptTwoMaskAllB32);

    for (uint16_t qliOptTwoI = (uint16_t)(0); qliOptTwoI < qliOptTwoGSize; qliOptTwoI++) {
        Reg::LoadAlign<float>(qliOptTwoRegQK0[0], qliOptTwoQk0 + 128 * qliOptTwoI);
        Reg::LoadAlign<float>(qliOptTwoRegQK0[1], qliOptTwoQk0 + 128 * qliOptTwoI + qliOptTwoQkVLStride);
        Reg::LoadAlign<float>(qliOptTwoRegQK1[0], qliOptTwoQk1 + 128 * qliOptTwoI);
        Reg::LoadAlign<float>(qliOptTwoRegQK1[1], qliOptTwoQk1 + 128 * qliOptTwoI + qliOptTwoQkVLStride);
        // 混合使用对整体性能更好
        liV2Vector1::BroadcastLane(qliOptTwoRegwBrc[0], qliOptTwoRegW[0], qliOptTwoI);
        // Weight无bank冲突，用LoadAlign来提取weight标量
        // 地址空间处理：原 BroadcastLane(ptr) 内联为 LoadAlign BRC，避免 __ubuf__ 传给 __local_mem__ 参数
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(qliOptTwoRegwBrc[1], qliOptTwoWeightTemp + qliOptTwoI);
        Reg::Relu(qliOptTwoRegQK0[0], qliOptTwoRegQK0[0], qliOptTwoMaskAllB32);
        Reg::Relu(qliOptTwoRegQK0[1], qliOptTwoRegQK0[1], qliOptTwoMaskAllB32);
        Reg::Relu(qliOptTwoRegQK1[0], qliOptTwoRegQK1[0], qliOptTwoMaskAllB32);
        Reg::Relu(qliOptTwoRegQK1[1], qliOptTwoRegQK1[1], qliOptTwoMaskAllB32);
        Reg::MulAddDst(qliOptTwoRegSum0[0], qliOptTwoRegQK0[0], qliOptTwoRegwBrc[0], qliOptTwoMaskAllB32);
        Reg::MulAddDst(qliOptTwoRegSum0[1], qliOptTwoRegQK0[1], qliOptTwoRegwBrc[0], qliOptTwoMaskAllB32);
        Reg::MulAddDst(qliOptTwoRegSum1[0], qliOptTwoRegQK1[0], qliOptTwoRegwBrc[1], qliOptTwoMaskAllB32);
        Reg::MulAddDst(qliOptTwoRegSum1[1], qliOptTwoRegQK1[1], qliOptTwoRegwBrc[1], qliOptTwoMaskAllB32);
    }

    if constexpr (WITH_SCALE) {
        Reg::Muls(qliOptTwoRegSum0[0], qliOptTwoRegSum0[0], qliOptTwoKScaleValue, qliOptTwoMaskAllB32);
        Reg::Muls(qliOptTwoRegSum0[1], qliOptTwoRegSum0[1], qliOptTwoKScaleValue, qliOptTwoMaskAllB32);
        Reg::Muls(qliOptTwoRegSum1[0], qliOptTwoRegSum1[0], qliOptTwoKScaleValue, qliOptTwoMaskAllB32);
        Reg::Muls(qliOptTwoRegSum1[1], qliOptTwoRegSum1[1], qliOptTwoKScaleValue, qliOptTwoMaskAllB32);
    }

    // Convert to bfloat16 and store output channel
    Reg::RegTensor<bfloat16_t> qliOptTwoRegSumBF16[2];
    Reg::RegTensor<uint16_t> qliOptTwoRegOut[2];
    Reg::DeInterleave(qliOptTwoRegSum0[0], qliOptTwoRegSum0[1], qliOptTwoRegSum0[0], qliOptTwoRegSum0[1]);
    Reg::DeInterleave(qliOptTwoRegSum1[0], qliOptTwoRegSum1[1], qliOptTwoRegSum1[0], qliOptTwoRegSum1[1]);
    Reg::Cast<bfloat16_t, float, qliOptTwoCastTraitF32ToF16_ODD>(qliOptTwoRegSumBF16[0], qliOptTwoRegSum0[1],
                                                                 qliOptTwoMaskAllB32);
    Reg::Cast<bfloat16_t, float, qliOptTwoCastTraitF32ToF16_ODD>(qliOptTwoRegSumBF16[1], qliOptTwoRegSum1[1],
                                                                 qliOptTwoMaskAllB32);
    Reg::Cast<bfloat16_t, float, qliOptTwoCastTraitF32ToF16_EVEN>(qliOptTwoRegSumBF16[0], qliOptTwoRegSum0[0],
                                                                  qliOptTwoMaskAllB32);
    Reg::Cast<bfloat16_t, float, qliOptTwoCastTraitF32ToF16_EVEN>(qliOptTwoRegSumBF16[1], qliOptTwoRegSum1[0],
                                                                  qliOptTwoMaskAllB32);

    liV2Vector1::FloatX2ToSortableKey<bfloat16_t>(qliOptTwoRegOut[0], qliOptTwoRegOut[1], qliOptTwoRegSumBF16[0],
                                                  qliOptTwoRegSumBF16[1], qliOptTwoBf16Ctx, qliOptTwoMaskAllB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliOptTwoOut0, qliOptTwoRegOut[0], qliOptTwoMaskAllB16);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(qliOptTwoOut1, qliOptTwoRegOut[1], qliOptTwoMaskAllB16);
}

template <bool WITH_SCALE>
__aicore__ inline void MulWeightAndReduceSumOptionalScale2Impl(
    const LocalTensor<uint16_t> &qliOptImplOut, // out    [2, S2Base]     [128   ]
    uint32_t qliOptImplOutStride,
    const LocalTensor<float> &qliOptImplQk, // q*k^t  [2, G, S2Base]  [64 128]
    uint32_t qliOptImplQkVLStride, uint32_t qliOptImplQkStride,
    const LocalTensor<float> &qliOptImplWeight, // w      [2, G]          [64    ]
    uint32_t qliOptImplWeightStride, const LocalTensor<float> &qliOptImplWeightTemp,
    const float qliOptImplKScaleValue, // kScale scalar
    const float qliOptImplQScaleValue, // qScale scalar for batch 0和1
    const int qliOptImplGSize)         // G 64
{
    __ubuf__ float *qliOptImplWeight0 = (__ubuf__ float *)qliOptImplWeight.GetPhyAddr();
    __ubuf__ float *qliOptImplWeightTempAddr = (__ubuf__ float *)qliOptImplWeightTemp.GetPhyAddr();
    __ubuf__ float *qliOptImplQk0 = (__ubuf__ float *)qliOptImplQk.GetPhyAddr();
    __ubuf__ uint16_t *qliOptImplOut0 = (__ubuf__ uint16_t *)qliOptImplOut.GetPhyAddr();

    __ubuf__ float *qliOptImplWeight1 = qliOptImplWeight0 + qliOptImplWeightStride;
    __ubuf__ float *qliOptImplQk1 = qliOptImplQk0 + qliOptImplQkStride;
    __ubuf__ uint16_t *qliOptImplOut1 = qliOptImplOut0 + qliOptImplOutStride;

    MulWeightAndReduceSumOptionalScale2VF<WITH_SCALE>(qliOptImplOut0, qliOptImplOut1, qliOptImplQk0, qliOptImplQk1,
                                                      qliOptImplQkVLStride, qliOptImplWeight0, qliOptImplWeight1,
                                                      qliOptImplWeightTempAddr, qliOptImplKScaleValue,
                                                      qliOptImplQScaleValue, (uint16_t)qliOptImplGSize);
}

__aicore__ inline void MulWeightAndReduceSumPerTensor2(
    const LocalTensor<uint16_t> &qliPerTensorOut, // out    [2, S2Base]     [128   ]
    uint32_t qliPerTensorOutStride,
    const LocalTensor<float> &qliPerTensorQk, // q*k^t  [2, G, S2Base]  [64 128]
    uint32_t qliPerTensorQkVLStride, uint32_t qliPerTensorQkStride,
    const LocalTensor<float> &qliPerTensorWeight, // w      [2, G]          [64    ]
    uint32_t qliPerTensorWeightStride, const LocalTensor<float> &qliPerTensorWeightTemp,
    const float qliPerTensorKScaleValue, // kScale scalar
    const float qliPerTensorQScaleValue, // qScale scalar for batch 0和1
    const int qliPerTensorGSize)         // G 64
{
    MulWeightAndReduceSumOptionalScale2Impl<true>(qliPerTensorOut, qliPerTensorOutStride, qliPerTensorQk,
                                                  qliPerTensorQkVLStride, qliPerTensorQkStride, qliPerTensorWeight,
                                                  qliPerTensorWeightStride, qliPerTensorWeightTemp,
                                                  qliPerTensorKScaleValue, qliPerTensorQScaleValue, qliPerTensorGSize);
}

__aicore__ inline void MulWeightAndReduceSumMX2(const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
                                                uint32_t outStride,
                                                const LocalTensor<float> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
                                                uint32_t qkVLStride, uint32_t qkStride,
                                                const LocalTensor<float> &weight_, // w      [2, G]          [64    ]
                                                uint32_t weightStride, const LocalTensor<float> &weightTemp_,
                                                const int gSize) // G 64
{
    MulWeightAndReduceSumOptionalScale2Impl<false>(out_, outStride, qk_, qkVLStride, qkStride, weight_, weightStride,
                                                   weightTemp_, 1.0f, 1.0f, gSize);
}

__simd_callee__ inline void CastWeightToBf16(AscendC::Reg::RegTensor<bfloat16_t> &dst, __ubuf__ float *src,
                                             AscendC::Reg::MaskReg &maskAllB32)
{
    using CastTrait = AscendC::Reg::CastTrait;
    static constexpr CastTrait castTraitF32ToBf16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    AscendC::Reg::RegTensor<float> regWeightF32;
    AscendC::Reg::LoadAlign<float>(regWeightF32, src);
    AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToBf16>(dst, regWeightF32, maskAllB32);
    // Cast结果按B32 lane落位，在目标寄存器内压紧为连续BF16，供BroadcastLane按元素索引。
    AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>((AscendC::Reg::RegTensor<uint16_t> &)dst,
                                                                              (AscendC::Reg::RegTensor<uint32_t> &)dst);
}

__simd_vf__ void MulWeightAndReduceSumMXFP4VF(__ubuf__ uint16_t *out, __ubuf__ bfloat16_t *qk, __ubuf__ float *weight,
                                              uint16_t gSize)
{
    constexpr uint32_t BF16_QK_ROW_STRIDE = UB_BANK_DEPTH_STRIDE / sizeof(bfloat16_t);
    AscendC::Reg::RegTensor<bfloat16_t> regQK;
    AscendC::Reg::RegTensor<bfloat16_t> regWeight;
    AscendC::Reg::RegTensor<bfloat16_t> regWeightBrc;
    AscendC::Reg::RegTensor<bfloat16_t> regSum;
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    CastWeightToBf16(regWeight, weight, maskAllB32);
    AscendC::Reg::Duplicate(regSum, bfloat16_t(0.0f), maskAllB16);

    for (uint16_t i = 0; i < gSize; i++) {
        AscendC::Reg::LoadAlign<bfloat16_t>(regQK, qk + BF16_QK_ROW_STRIDE * i);
        liV2Vector1::BroadcastLane(regWeightBrc, regWeight, i);
        AscendC::Reg::MulAddDst(regSum, regQK, regWeightBrc, maskAllB16);
    }

    AscendC::Reg::RegTensor<uint16_t> regOut;
    liV2Vector1::FloatToSortableKey<bfloat16_t>(regOut, regSum, bf16Ctx, maskAllB16);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
}

__aicore__ inline void MulWeightAndReduceSumMXFP4(const LocalTensor<uint16_t> &out_, const LocalTensor<bfloat16_t> &qk_,
                                                  const uint32_t qkVLStride, const LocalTensor<float> &weight_,
                                                  const int gSize)
{
    (void)qkVLStride;
    __ubuf__ uint16_t *out = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ bfloat16_t *qk = (__ubuf__ bfloat16_t *)qk_.GetPhyAddr();
    __ubuf__ float *weight = (__ubuf__ float *)weight_.GetPhyAddr();
    MulWeightAndReduceSumMXFP4VF(out, qk, weight, static_cast<uint16_t>(gSize));
}

__simd_vf__ void MulWeightAndReduceSumMXFP4TwoRowsVF(__ubuf__ uint16_t *out0, __ubuf__ uint16_t *out1,
                                                     __ubuf__ bfloat16_t *qk0, __ubuf__ bfloat16_t *qk1,
                                                     __ubuf__ float *weight0, __ubuf__ float *weight1, uint16_t gSize)
{
    constexpr uint32_t BF16_QK_ROW_STRIDE = UB_BANK_DEPTH_STRIDE / sizeof(bfloat16_t);
    AscendC::Reg::RegTensor<bfloat16_t> regQK0;
    AscendC::Reg::RegTensor<bfloat16_t> regQK1;
    AscendC::Reg::RegTensor<bfloat16_t> regWeight[2];
    AscendC::Reg::RegTensor<bfloat16_t> regWeightBrc[2];
    AscendC::Reg::RegTensor<bfloat16_t> regSum[2];
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    liV2Vector1::FloatSortConstCtx<bfloat16_t> bf16Ctx;
    liV2Vector1::InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    CastWeightToBf16(regWeight[0], weight0, maskAllB32);
    CastWeightToBf16(regWeight[1], weight1, maskAllB32);
    AscendC::Reg::Duplicate(regSum[0], bfloat16_t(0.0f), maskAllB16);
    AscendC::Reg::Duplicate(regSum[1], bfloat16_t(0.0f), maskAllB16);

    for (uint16_t i = 0; i < gSize; i++) {
        AscendC::Reg::LoadAlign<bfloat16_t>(regQK0, qk0 + BF16_QK_ROW_STRIDE * i);
        AscendC::Reg::LoadAlign<bfloat16_t>(regQK1, qk1 + BF16_QK_ROW_STRIDE * i);
        liV2Vector1::BroadcastLane(regWeightBrc[0], regWeight[0], i);
        liV2Vector1::BroadcastLane(regWeightBrc[1], regWeight[1], i);
        AscendC::Reg::MulAddDst(regSum[0], regQK0, regWeightBrc[0], maskAllB16);
        AscendC::Reg::MulAddDst(regSum[1], regQK1, regWeightBrc[1], maskAllB16);
    }

    AscendC::Reg::RegTensor<uint16_t> regOut[2];
    liV2Vector1::FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSum[0], regSum[1], bf16Ctx, maskAllB16);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regOut[0], maskAllB16);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out1, regOut[1], maskAllB16);
}

__aicore__ inline void MulWeightAndReduceSumMXFP4TwoRows(const LocalTensor<uint16_t> &out_, uint32_t outStride,
                                                         const LocalTensor<bfloat16_t> &qk_, uint32_t qkVLStride,
                                                         uint32_t qkStride, const LocalTensor<float> &weight_,
                                                         uint32_t weightStride, const int gSize)
{
    (void)qkVLStride;
    __ubuf__ uint16_t *out0 = (__ubuf__ uint16_t *)out_.GetPhyAddr();
    __ubuf__ uint16_t *out1 = out0 + outStride;
    __ubuf__ bfloat16_t *qk0 = (__ubuf__ bfloat16_t *)qk_.GetPhyAddr();
    __ubuf__ bfloat16_t *qk1 = qk0 + qkStride;
    __ubuf__ float *weight0 = (__ubuf__ float *)weight_.GetPhyAddr();
    __ubuf__ float *weight1 = weight0 + weightStride;
    MulWeightAndReduceSumMXFP4TwoRowsVF(out0, out1, qk0, qk1, weight0, weight1, static_cast<uint16_t>(gSize));
}

template <typename QK_T, typename SCORE_T, typename WEIGHT_T>
__aicore__ inline void BatchMulWeightAndReduceSumMXFP4(const LocalTensor<SCORE_T> &out_, uint32_t outStride,
                                                       const LocalTensor<QK_T> &qk_, uint32_t qkVLStride,
                                                       uint32_t qkStride, const LocalTensor<WEIGHT_T> &weight_,
                                                       uint32_t weightStride, const int gSize, const int batch)
{
    static_assert(std::is_same_v<QK_T, bfloat16_t>);
    static_assert(std::is_same_v<SCORE_T, uint16_t>);
    static_assert(std::is_same_v<WEIGHT_T, float>);
    if (batch == 2) {
        MulWeightAndReduceSumMXFP4TwoRows(out_, outStride, qk_, qkVLStride, qkStride, weight_, weightStride, gSize);
    } else if (batch == 1) {
        MulWeightAndReduceSumMXFP4(out_, qk_, qkVLStride, weight_, gSize);
    }
}
template <typename QK_T, typename SCORE_T, typename WEIGHT_T>
__aicore__ inline void BatchMulWeightAndReduceSumMX(const LocalTensor<SCORE_T> &out_,
                                                    uint32_t outStride,           // out    [S2Base]     [128   ]
                                                    const LocalTensor<QK_T> &qk_, // q*k^t  [G, S2Base]  [64 128]
                                                    uint32_t qkVLStride, uint32_t qkStride,
                                                    const LocalTensor<WEIGHT_T> &weight_, // w      [G]      [64    ]
                                                    uint32_t weightStride, const LocalTensor<float> &weightTemp_,
                                                    const int gSize, const int batch)
{
    // 暂只支持这两种情况, 后续改成循环
    if (batch != 2 && batch != 1) {
        return;
    }
    if (batch == 2) {
        MulWeightAndReduceSumMX2(out_, outStride, qk_, qkVLStride, qkStride, weight_, weightStride, weightTemp_, gSize);
    } else {
        MulWeightAndReduceSumMX(out_, qk_, qkVLStride, weight_, gSize);
    }
}

template <typename QK_T, typename SCORE_T, typename WEIGHT_T>
__aicore__ inline void BatchMulWeightAndReduceSumPerTensor(
    const LocalTensor<SCORE_T> &out_,
    uint32_t outStride,           // out    [S2Base]     [128   ]
    const LocalTensor<QK_T> &qk_, // q*k^t  [G, S2Base]  [64 128]
    uint32_t qkVLStride, uint32_t qkStride,
    const LocalTensor<WEIGHT_T> &weight_, // w  [G]          [64    ]
    uint32_t weightStride, const LocalTensor<float> &weightTemp_, const float kScaleValue, const float qScaleValue,
    const int gSize, const int batch)
{
    // 暂只支持这两种情况, 后续改成循环
    if (batch != 2 && batch != 1) {
        return;
    }
    if (batch == 2) {
        MulWeightAndReduceSumPerTensor2(out_, outStride, qk_, qkVLStride, qkStride, weight_, weightStride, weightTemp_,
                                        kScaleValue, qScaleValue, gSize);
    } else {
        MulWeightAndReduceSumPerTensor(out_, qk_, qkVLStride, weight_, kScaleValue, qScaleValue, gSize);
    }
}

} // namespace vector1

#endif // QUANT_LIGHTNING_INDEXER_V2_VECTOR1_H
