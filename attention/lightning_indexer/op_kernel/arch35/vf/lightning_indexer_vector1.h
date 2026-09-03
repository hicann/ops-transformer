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
    AscendC::Reg::RegTensor<UInt> zeros;
    AscendC::Reg::RegTensor<UInt> allOne;
    AscendC::Reg::RegTensor<UInt> signMask;
    AscendC::Reg::RegTensor<UInt> nan;
};

template <typename FloatT>
__simd_callee__ inline void InitUIntSortConstCtx(UIntSortConstCtx<FloatT> &ctx, AscendC::Reg::MaskReg &maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    AscendC::Reg::Duplicate(ctx.zeros, Traits::ZERO, maskAll);
    AscendC::Reg::Duplicate(ctx.allOne, Traits::ALL_ONE, maskAll);
    AscendC::Reg::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::Reg::Duplicate(ctx.nan, Traits::NAN_MASK, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void UIntToSortableKey(AscendC::Reg::RegTensor<FloatT> &outKey,
                                              AscendC::Reg::RegTensor<typename UIntSortConstCtx<FloatT>::UInt> &inVal,
                                              UIntSortConstCtx<FloatT> &ctx, AscendC::Reg::MaskReg &maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::Reg::RegTensor<UInt> regTemp;
    AscendC::Reg::RegTensor<UInt> regMask;
    AscendC::Reg::MaskReg regSelectZero;
    AscendC::Reg::MaskReg regSelectSign;

    auto &inBits = inVal;

    // 1. 0 check
    AscendC::Reg::Compare<UInt, CMPMODE::EQ>(regSelectZero, inBits, ctx.zeros, maskAll);

    // 2. 0 -> -NAN
    AscendC::Reg::Select((AscendC::Reg::RegTensor<UInt> &)outKey, ctx.nan, inBits, regSelectZero);

    // 3. sign bit
    AscendC::Reg::And(regTemp, (AscendC::Reg::RegTensor<UInt> &)outKey, ctx.signMask, maskAll);

    AscendC::Reg::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::Reg::Select(regMask, ctx.signMask, ctx.allOne, regSelectSign);
    AscendC::Reg::Xor((AscendC::Reg::RegTensor<UInt> &)outKey, (AscendC::Reg::RegTensor<UInt> &)outKey, regMask,
                      maskAll);
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
        AscendC::Reg::RegTensor<uint32_t> regIn[2];
        AscendC::Reg::RegTensor<float> regOut[2];
        AscendC::Reg::RegTensor<bfloat16_t> regOutBF16[2];
        AscendC::Reg::RegTensor<bfloat16_t> regOutValue;
        AscendC::Reg::RegTensor<bfloat16_t> regInvalid;
        AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
        constexpr static Reg::CastTrait castTraitFP32ToBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::Reg::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::Reg::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);
            UIntSortConstCtx<float> uint32Ctx;
            InitUIntSortConstCtx(uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            AscendC::Reg::Cast<bfloat16_t, float, castTraitFP32ToBF16>(regOutBF16[0], regOut[0], maskAllB32);
            AscendC::Reg::Cast<bfloat16_t, float, castTraitFP32ToBF16>(regOutBF16[1], regOut[1], maskAllB32);

            AscendC::Reg::DeInterleave(regOutValue, regInvalid, regOutBF16[0], regOutBF16[1]);

            AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize32,
                                                                                     regOutValue, maskAllB16);
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
        AscendC::Reg::RegTensor<uint32_t> regIn[2];
        AscendC::Reg::RegTensor<float> regOut[2];
        AscendC::Reg::RegTensor<half> regOutFP16[2];
        AscendC::Reg::RegTensor<half> regOutValue;
        AscendC::Reg::RegTensor<half> regInvalid;
        AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();
        constexpr static Reg::CastTrait castTraitFP32ToFP16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::Reg::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::Reg::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);
            UIntSortConstCtx<float> uint32Ctx;
            InitUIntSortConstCtx(uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            AscendC::Reg::Cast<half, float, castTraitFP32ToFP16>(regOutFP16[0], regOut[0], maskAllB32);
            AscendC::Reg::Cast<half, float, castTraitFP32ToFP16>(regOutFP16[1], regOut[1], maskAllB32);

            AscendC::Reg::DeInterleave(regOutValue, regInvalid, regOutFP16[0], regOutFP16[1]);

            AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize32, regOutValue,
                                                                               maskAllB16);
        }
    }
}

__simd_callee__ inline void BroadcastLane(AscendC::Reg::RegTensor<float> &dst, __local_mem__ float *src,
                                          uint16_t laneIdx)
{
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dst, src + laneIdx);
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
        AscendC::Reg::RegTensor<uint32_t> brcGatherIndex;
        AscendC::Reg::RegTensor<float> regQK[2];
        AscendC::Reg::RegTensor<float> regW;
        AscendC::Reg::RegTensor<float> regwBrc;
        AscendC::Reg::RegTensor<float> regQScale;
        AscendC::Reg::RegTensor<float> regKScale[2];
        AscendC::Reg::RegTensor<float> regSum[2];
        AscendC::Reg::RegTensor<W_T> regWWT;

        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAll16 = AscendC::Reg::CreateMask<W_T, AscendC::Reg::MaskPattern::ALL>();

        FloatSortConstCtx<float> fp32Ctx;
        InitFloatSortConstCtx(fp32Ctx, maskAll);

        constexpr static Reg::CastTrait castTraitWTToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        AscendC::Reg::LoadAlign<W_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWWT, weight_);
        AscendC::Reg::Cast<float, W_T, castTraitWTToFP32>(regW, regWWT, maskAll16);

        AscendC::Reg::Duplicate(regSum[0], 0.0f, maskAll);
        AscendC::Reg::Duplicate(regSum[1], 0.0f, maskAll);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); ++i) {
            AscendC::Reg::Duplicate(brcGatherIndex, i);
            AscendC::Reg::LoadAlign<float>(regQK[0], qk0 + 128 * i);
            AscendC::Reg::LoadAlign<float>(regQK[1], qk1 + 128 * i);
            AscendC::Reg::Gather(regwBrc, regW, brcGatherIndex);

            AscendC::Reg::Relu(regQK[0], regQK[0], maskAll);
            AscendC::Reg::Relu(regQK[1], regQK[1], maskAll);

            AscendC::Reg::MulAddDst(regSum[0], regQK[0], regwBrc, maskAll);
            AscendC::Reg::MulAddDst(regSum[1], regQK[1], regwBrc, maskAll);
        }

        AscendC::Reg::RegTensor<uint32_t> regOut[2];
        FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
        AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
    }
}
} // namespace vector1

#endif
