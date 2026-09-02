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
 * \file lightning_indexer_vector1.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_VECTOR1_H
#define LIGHTNING_INDEXER_VECTOR1_H

#include "kernel_operator.h"
#include "common/lightning_indexer_vector1_base.h"

namespace vector1 {

template <typename T>
struct UIntSortTraits;

template <>
struct UIntSortTraits<float> {
    using UInt = uint32_t;
    static constexpr UInt ZERO = 0x00000000;
    static constexpr UInt SIGN_MASK = 0x80000000;
    static constexpr UInt NAN_MASK = 0xFFC00000;
    static constexpr UInt ALL_ONE = 0xFFFFFFFF;
};

template <typename FloatT>
struct UIntSortConstCtx {
    using Traits = UIntSortTraits<FloatT>;
    using UInt = typename Traits::UInt;
    AscendC::MicroAPI::RegTensor<UInt> zeros;
    AscendC::MicroAPI::RegTensor<UInt> allOne;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};

template <typename FloatT>
__simd_callee__ inline void InitUIntSortConstCtx(UIntSortConstCtx<FloatT> &ctx, AscendC::MicroAPI::MaskReg &maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros, Traits::ZERO, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.allOne, Traits::ALL_ONE, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.nan, Traits::NAN_MASK, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void UIntToSortableKey(
    AscendC::MicroAPI::RegTensor<FloatT> &outKey,
    AscendC::MicroAPI::RegTensor<typename UIntSortConstCtx<FloatT>::UInt> &inVal, UIntSortConstCtx<FloatT> &ctx,
    AscendC::MicroAPI::MaskReg &maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp;
    AscendC::MicroAPI::RegTensor<UInt> regMask;
    AscendC::MicroAPI::MaskReg regSelectZero;
    AscendC::MicroAPI::MaskReg regSelectSign;

    auto &inBits = inVal;

    // 1. 0 check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectZero, inBits, ctx.zeros, maskAll);

    // 2. 0 -> -NAN
    AscendC::MicroAPI::Select((AscendC::MicroAPI::RegTensor<UInt> &)outKey, ctx.nan, inBits, regSelectZero);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, (AscendC::MicroAPI::RegTensor<UInt> &)outKey, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask, ctx.signMask, ctx.allOne, regSelectSign);
    AscendC::MicroAPI::Xor((AscendC::MicroAPI::RegTensor<UInt> &)outKey, (AscendC::MicroAPI::RegTensor<UInt> &)outKey,
                           regMask, maskAll);
}

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<bfloat16_t> &out_, const LocalTensor<uint32_t> &in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ bfloat16_t *)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t *)in.GetPhyAddr();

    const uint16_t repeatSize32 = 128;
    uint16_t topkLoopNum = (topK + repeatSize32 - 1) / repeatSize32;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> regIn[2];
        AscendC::MicroAPI::RegTensor<float> regOut[2];
        AscendC::MicroAPI::RegTensor<bfloat16_t> regOutBF16[2];
        AscendC::MicroAPI::RegTensor<bfloat16_t> regOutValue;
        AscendC::MicroAPI::RegTensor<bfloat16_t> regInvalid;
        AscendC::MicroAPI::MaskReg maskAllB32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB16 =
            AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        constexpr static MicroAPI::CastTrait castTraitFP32ToBF16 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
            RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::MicroAPI::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::MicroAPI::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);
            UIntSortConstCtx<float> uint32Ctx;
            InitUIntSortConstCtx(uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFP32ToBF16>(regOutBF16[0], regOut[0], maskAllB32);
            AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFP32ToBF16>(regOutBF16[1], regOut[1], maskAllB32);

            AscendC::MicroAPI::DeInterleave(regOutValue, regInvalid, regOutBF16[0], regOutBF16[1]);

            AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                outBuf + i * repeatSize32, regOutValue, maskAllB16);
        }
    }
}

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<half> &out_, const LocalTensor<uint32_t> &in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ half *)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t *)in.GetPhyAddr();

    const uint16_t repeatSize32 = 128;
    uint16_t topkLoopNum = (topK + repeatSize32 - 1) / repeatSize32;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> regIn[2];
        AscendC::MicroAPI::RegTensor<float> regOut[2];
        AscendC::MicroAPI::RegTensor<half> regOutFP16[2];
        AscendC::MicroAPI::RegTensor<half> regOutValue;
        AscendC::MicroAPI::RegTensor<half> regInvalid;
        AscendC::MicroAPI::MaskReg maskAllB32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB16 =
            AscendC::MicroAPI::CreateMask<half, AscendC::MicroAPI::MaskPattern::ALL>();
        constexpr static MicroAPI::CastTrait castTraitFP32ToFP16 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
            RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::MicroAPI::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::MicroAPI::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);
            UIntSortConstCtx<float> uint32Ctx;
            InitUIntSortConstCtx(uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            AscendC::MicroAPI::Cast<half, float, castTraitFP32ToFP16>(regOutFP16[0], regOut[0], maskAllB32);
            AscendC::MicroAPI::Cast<half, float, castTraitFP32ToFP16>(regOutFP16[1], regOut[1], maskAllB32);

            AscendC::MicroAPI::DeInterleave(regOutValue, regInvalid, regOutFP16[0], regOutFP16[1]);

            AscendC::MicroAPI::StoreAlign<half, AscendC::MicroAPI::StoreDist::DIST_NORM>(outBuf + i * repeatSize32,
                                                                                         regOutValue, maskAllB16);
        }
    }
}

__simd_callee__ inline void BroadcastLane(AscendC::MicroAPI::RegTensor<float> &dst, __local_mem__ float *src,
                                          uint16_t laneIdx)
{
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + laneIdx);
}

template <typename W_T>
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint32_t> &out, // out    [S2Base]     [128   ] 2
                                             const LocalTensor<float> &qk,     // q*k^t  [G, S2Base]  [64 128] 2
                                             const LocalTensor<W_T> &weight,   // w      [G]          [64    ] 1
                                             const int gSize)                  // G 64
{
    __local_mem__ W_T *weight_ = (__local_mem__ W_T *)weight.GetPhyAddr();

    constexpr uint32_t VL = 64; // vector length

    auto qk0 = (__local_mem__ float *)qk.GetPhyAddr();
    auto qk1 = qk0 + VL;
    auto out0 = (__local_mem__ uint32_t *)out.GetPhyAddr();
    auto out1 = out0 + VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> brcGatherIndex;
        AscendC::MicroAPI::RegTensor<float> regQK[2];
        AscendC::MicroAPI::RegTensor<float> regW;
        AscendC::MicroAPI::RegTensor<float> regwBrc;
        AscendC::MicroAPI::RegTensor<float> regQScale;
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum[2];
        AscendC::MicroAPI::RegTensor<W_T> regWWT;

        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAll16 =
            AscendC::MicroAPI::CreateMask<W_T, AscendC::MicroAPI::MaskPattern::ALL>();

        FloatSortConstCtx<float> fp32Ctx;
        InitFloatSortConstCtx(fp32Ctx, maskAll);

        constexpr static MicroAPI::CastTrait castTraitWTToFP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        AscendC::MicroAPI::LoadAlign<W_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regWWT, weight_);
        AscendC::MicroAPI::Cast<float, W_T, castTraitWTToFP32>(regW, regWWT, maskAll16);

        AscendC::MicroAPI::Duplicate(regSum[0], 0.0f, maskAll);
        AscendC::MicroAPI::Duplicate(regSum[1], 0.0f, maskAll);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); ++i) {
            AscendC::MicroAPI::Duplicate(brcGatherIndex, i);
            AscendC::MicroAPI::LoadAlign<float>(regQK[0], qk0 + 128 * i);
            AscendC::MicroAPI::LoadAlign<float>(regQK[1], qk1 + 128 * i);
            AscendC::MicroAPI::Gather(regwBrc, regW, brcGatherIndex);

            AscendC::MicroAPI::Relu(regQK[0], regQK[0], maskAll);
            AscendC::MicroAPI::Relu(regQK[1], regQK[1], maskAll);

            AscendC::MicroAPI::MulAddDst(regSum[0], regQK[0], regwBrc, maskAll);
            AscendC::MicroAPI::MulAddDst(regSum[1], regQK[1], regwBrc, maskAll);
        }

        AscendC::MicroAPI::RegTensor<uint32_t> regOut[2];
        FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
    }
}
} // namespace vector1

#endif
