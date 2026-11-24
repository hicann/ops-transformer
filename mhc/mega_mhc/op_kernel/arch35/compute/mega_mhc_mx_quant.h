/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_MX_QUANT_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_MX_QUANT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

// OCP shared-exponent scales and MXFP8 conversion for FP16/BF16 inputs.
// Preserve dtype-specific scaling, saturating cast layouts and rint rounding.
namespace MegaMhc::Compute::MxQuant {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint16_t kVectorBytes = static_cast<int64_t>(Ops::Base::GetVRegSize());
constexpr uint32_t kFp32VectorElements = kVectorBytes / sizeof(float);
constexpr uint32_t kB16VectorElements = kVectorBytes / sizeof(half);
constexpr int64_t kUbBlockBytes = static_cast<int64_t>(Ops::Base::GetUbBlockSize());
constexpr uint16_t kGroupsPerStep = kVectorBytes / kUbBlockBytes;
constexpr uint32_t kInputRegistersPerStep = 2U;
constexpr uint32_t kInputElementsPerStep = kB16VectorElements * kInputRegistersPerStep;
constexpr uint32_t kPackedScaleElementsPerStep = kB16VectorElements * sizeof(uint8_t) / sizeof(uint16_t);
constexpr int64_t kFp8OutputBytesPerStep = kVectorBytes;

// BF16 exponent bits occupy [14:7]; OCP exponent arithmetic keeps this bit alignment.
constexpr int16_t kBf16MantissaBits = 7;
constexpr uint16_t kBf16ExponentMask = 0x7f80;
constexpr uint16_t kFp16ExponentMask = 0x7c00;
constexpr uint16_t kBf16InverseScaleBias = 0x7f00; // 2 * BF16 exponent bias (127), shifted by 7.
constexpr uint16_t kBf16NanBits = 0x7f81;
constexpr uint16_t kE8m0NanBits = 0x00ff;
// The reciprocal of a scale at shared exponent 254 uses the BF16 subnormal 2^-127.
constexpr uint16_t kBf16MinScaleReciprocalBits = 0x0040;
constexpr uint16_t kFp8E4m3MaxExponentBits = 8U << kBf16MantissaBits;
constexpr uint16_t kFp8E5m2MaxExponentBits = 15U << kBf16MantissaBits;
constexpr uint16_t kFp4E2m1MaxExponentBits = 2U << kBf16MantissaBits;
constexpr uint16_t kFp4E1m2MaxExponentBits = 0U;

constexpr AscendC::Reg::CastTrait castTraitHalf2Bf16 = {AscendC::Reg::RegLayout::UNKNOWN,
                                                        AscendC::Reg::SatMode::UNKNOWN,
                                                        AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

constexpr AscendC::Reg::CastTrait castTraitZero = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                   AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait castTraitOne = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                  AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait castTrait32to80 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait castTrait32to81 = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait castTrait32to82 = {AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait castTrait32to83 = {AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode, typename T1, typename T2>
__aicore__ inline void ComputeData(__ubuf__ T1* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                   __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    uint16_t elementAfterReduce = kGroupsPerStep;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg dataMask1;
        AscendC::Reg::MaskReg dataMask2;
        AscendC::Reg::MaskReg dataMask3;
        AscendC::Reg::MaskReg dataMask4;
        AscendC::Reg::MaskReg dataMask5;
        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
        AscendC::Reg::RegTensor<float> floatScaleForMul;
        AscendC::Reg::RegTensor<T1> vdExp0;
        AscendC::Reg::RegTensor<T1> vdExp1;
        AscendC::Reg::RegTensor<float> vdExp0FP32Zero;
        AscendC::Reg::RegTensor<float> vdExp0FP32One;
        AscendC::Reg::RegTensor<float> vdExp1FP32Zero;
        AscendC::Reg::RegTensor<float> vdExp1FP32One;
        AscendC::Reg::RegTensor<T2> vdExp0FP8Zero;
        AscendC::Reg::RegTensor<T2> vdExp0FP8One;
        AscendC::Reg::RegTensor<T2> vdExp1FP8Zero;
        AscendC::Reg::RegTensor<T2> vdExp1FP8One;
        dataMask1 = AscendC::Reg::CreateMask<T1>();
        dataMask2 = AscendC::Reg::CreateMask<T1>();
        dataMask3 = AscendC::Reg::CreateMask<T1>();
        dataMask4 = AscendC::Reg::CreateMask<T1>();
        dataMask5 = AscendC::Reg::CreateMask<T2>();
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::Reg::LoadAlign<T1, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                             kInputElementsPerStep);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                          elementAfterReduce);
            if constexpr (IsSameType<T1, half>::value) {
                AscendC::Reg::Cast<float, T1, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                AscendC::Reg::Cast<float, T1, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                AscendC::Reg::Cast<float, bfloat16_t, castTraitZero>(
                    floatScaleForMul, (AscendC::Reg::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                AscendC::Reg::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, dataMask3);
                AscendC::Reg::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, dataMask4);

                AscendC::Reg::Cast<float, T1, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask1);
                AscendC::Reg::Cast<float, T1, castTraitOne>(vdExp1FP32One, vdExp1, dataMask1);
                AscendC::Reg::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, dataMask3);
                AscendC::Reg::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, dataMask4);

            } else {
                AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T1>&)halfScaleForMul, dataMask1);
                AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T1>&)halfScaleForMul, dataMask1);

                AscendC::Reg::Cast<float, T1, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                AscendC::Reg::Cast<float, T1, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                AscendC::Reg::Cast<float, T1, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                AscendC::Reg::Cast<float, T1, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
            }
            AscendC::Reg::Cast<T2, float, castTrait32to80>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            AscendC::Reg::Cast<T2, float, castTrait32to82>(vdExp0FP8One, vdExp0FP32One, dataMask3);
            AscendC::Reg::Cast<T2, float, castTrait32to81>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            AscendC::Reg::Cast<T2, float, castTrait32to83>(vdExp1FP8One, vdExp1FP32One, dataMask4);

            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8One, dataMask5);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp1FP8Zero, dataMask5);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp1FP8One, dataMask5);

            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP8Zero, kFp8OutputBytesPerStep, dataMask5);
        }
    }
    return;
}

template <typename T>

__aicore__ inline void ComputeMaxExpOCP(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB,
                                        uint16_t loopNum)
{
    uint16_t elementAfterReduce = kGroupsPerStep;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vdExp0;
        AscendC::Reg::RegTensor<T> vdExp1;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1BF16;
        AscendC::Reg::RegTensor<uint16_t> vdExpSelect0;
        AscendC::Reg::RegTensor<uint16_t> vdExpSelect1;
        AscendC::Reg::RegTensor<uint16_t> vdExpExtract0;
        AscendC::Reg::RegTensor<uint16_t> vdExpExtract1;

        AscendC::Reg::RegTensor<uint16_t> expMaskBF16;
        AscendC::Reg::Duplicate(expMaskBF16, kBf16ExponentMask);

        AscendC::Reg::RegTensor<uint16_t> invalidMaskFP16;
        AscendC::Reg::Duplicate(invalidMaskFP16, kFp16ExponentMask);
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::MaskReg scaleMask1;
        AscendC::Reg::MaskReg scaleMask2;
        AscendC::Reg::MaskReg invalidDataMask0;
        AscendC::Reg::MaskReg invalidDataMask1;
        AscendC::Reg::UnalignRegForStore u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            scaleMask2 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                             kInputElementsPerStep);
            if constexpr (IsSameType<T, half>::value) {
                AscendC::Reg::And(vdExpSelect0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0, invalidMaskFP16,
                                  scaleMask1);
                AscendC::Reg::And(vdExpSelect1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1, invalidMaskFP16,
                                  scaleMask1);
                AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, vdExpSelect0, invalidMaskFP16,
                                                             scaleMask1);
                AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, vdExpSelect1, invalidMaskFP16,
                                                             scaleMask1);
                AscendC::Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                AscendC::Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16,
                                  scaleMask1);
                AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16,
                                  scaleMask1);
                AscendC::Reg::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                AscendC::Reg::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
                AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);
            }

            AscendC::Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(vdMaxExp, vdMaxExp, scaleMask1);

            AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, vdMaxExp, u1,
                                                                                              elementAfterReduce);
        }
        AscendC::Reg::StoreUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

template <typename T1, typename T2>
__aicore__ inline void ComputeScaleOCP(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                       __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
                                       uint16_t loopNumScale)
{
    uint16_t dtypeEmax = 0;
    if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
        dtypeEmax = kFp8E4m3MaxExponentBits;
    } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
        dtypeEmax = kFp8E5m2MaxExponentBits;
    } else if constexpr (IsSameType<T2, fp4x2_e2m1_t>::value) {
        dtypeEmax = kFp4E2m1MaxExponentBits;
    } else {
        dtypeEmax = kFp4E1m2MaxExponentBits;
    }
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::MaskReg cmpResult;
        AscendC::Reg::RegTensor<uint16_t> expMask;
        AscendC::Reg::MaskReg zeroMask;
        AscendC::Reg::MaskReg preMaskScale;
        AscendC::Reg::Duplicate(expMask, kBf16ExponentMask);
        AscendC::Reg::RegTensor<uint16_t> maxExpValue;
        AscendC::Reg::Duplicate(maxExpValue, dtypeEmax);
        AscendC::Reg::RegTensor<uint16_t> sharedExp;
        AscendC::Reg::RegTensor<uint16_t> scaleValue;
        AscendC::Reg::RegTensor<uint16_t> scaleBias;
        AscendC::Reg::Duplicate(scaleBias, kBf16InverseScaleBias);
        AscendC::Reg::RegTensor<uint16_t> halfScale;
        AscendC::Reg::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::Reg::Duplicate(fp8NanRegTensor, kE8m0NanBits);
        AscendC::Reg::RegTensor<uint16_t> zeroRegTensor;
        AscendC::Reg::Duplicate(zeroRegTensor, 0);
        AscendC::Reg::RegTensor<uint16_t> nanRegTensor;
        AscendC::Reg::Duplicate(nanRegTensor, kBf16NanBits);
        AscendC::Reg::MaskReg invalidDataMask;
        AscendC::Reg::MaskReg specialDataMask;
        AscendC::Reg::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::Reg::Duplicate(specialExpRegTensor, kBf16MinScaleReciprocalBits);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr,
                                                                                           kB16VectorElements);
            AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
            AscendC::Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

            AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

            AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, kBf16MantissaBits, preMaskScale);

            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);

            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                             kPackedScaleElementsPerStep, preMaskScale);

            AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroRegTensor, preMaskScale);
            AscendC::Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, kB16VectorElements, preMaskScale);
        }
    }
    return;
}

} // namespace MegaMhc::Compute::MxQuant

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_MX_QUANT_H
