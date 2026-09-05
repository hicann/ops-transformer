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
 * \file quant_lightning_indexer_vector1.h
 * \brief
 */
#ifndef quant_lightning_indexer_VECTOR1_H
#define quant_lightning_indexer_VECTOR1_H

#include "kernel_operator.h"

namespace vector1 {

template <typename T>
struct FloatSortTraits;

// fp32
template <>
struct FloatSortTraits<float> {
    using UInt = uint32_t;
    static constexpr UInt ZERO = 0x00000000;
    static constexpr UInt SIGN_MASK = 0x80000000;
    static constexpr UInt NAN_MASK = 0x7FC00000;
    static constexpr UInt ALL_ONE = 0xFFFFFFFF;
};

// bf16
template <>
struct FloatSortTraits<bfloat16_t> {
    using UInt = uint16_t;
    static constexpr UInt ZERO = 0x0000;
    static constexpr UInt SIGN_MASK = 0x8000;
    static constexpr UInt NAN_MASK = 0x7FC0;
    static constexpr UInt ALL_ONE = 0xFFFF;
};

template <typename FloatT>
struct FloatSortConstCtx {
    using Traits = FloatSortTraits<FloatT>;
    using UInt = typename Traits::UInt;
    AscendC::Reg::RegTensor<UInt> zeros;
    AscendC::Reg::RegTensor<UInt> allOne;
    AscendC::Reg::RegTensor<UInt> signMask;
    AscendC::Reg::RegTensor<UInt> nan;
};

template <typename FloatT>
__simd_callee__ inline void InitFloatSortConstCtx(FloatSortConstCtx<FloatT> &ctx, AscendC::Reg::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    AscendC::Reg::Duplicate(ctx.zeros, Traits::ZERO, maskAll);
    AscendC::Reg::Duplicate(ctx.allOne, Traits::ALL_ONE, maskAll);
    AscendC::Reg::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::Reg::Duplicate(ctx.nan, Traits::NAN_MASK, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatToSortableKey(AscendC::Reg::RegTensor<typename FloatSortTraits<FloatT>::UInt> &outKey,
                                               AscendC::Reg::RegTensor<FloatT> &inVal, FloatSortConstCtx<FloatT> &ctx,
                                               AscendC::Reg::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::Reg::RegTensor<UInt> regTemp;
    AscendC::Reg::RegTensor<UInt> regMask;
    AscendC::Reg::MaskReg regSelectNan;
    AscendC::Reg::MaskReg regSelectSign;

    auto &inBits = (AscendC::Reg::RegTensor<UInt> &)inVal;

    // 1. NaN check
    AscendC::Reg::Compare<UInt, CMPMODE::EQ>(regSelectNan, inBits, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::Reg::Select(outKey, ctx.allOne, inBits, regSelectNan);

    // 3. sign bit
    AscendC::Reg::And(regTemp, outKey, ctx.signMask, maskAll);

    AscendC::Reg::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::Reg::Select(regMask, ctx.allOne, ctx.signMask, regSelectSign);
    AscendC::Reg::Xor(outKey, outKey, regMask, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatX2ToSortableKey(
    AscendC::Reg::RegTensor<typename FloatSortTraits<FloatT>::UInt> &outKey0,
    AscendC::Reg::RegTensor<typename FloatSortTraits<FloatT>::UInt> &outKey1, AscendC::Reg::RegTensor<FloatT> &inVal0,
    AscendC::Reg::RegTensor<FloatT> &inVal1, FloatSortConstCtx<FloatT> &ctx, AscendC::Reg::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::Reg::RegTensor<UInt> regTemp[2];
    AscendC::Reg::RegTensor<UInt> regMask[2];
    AscendC::Reg::MaskReg regSelectNan[2];
    AscendC::Reg::MaskReg regSelectSign[2];

    auto &inBits0 = (AscendC::Reg::RegTensor<UInt> &)inVal0;
    auto &inBits1 = (AscendC::Reg::RegTensor<UInt> &)inVal1;

    // 1. NaN check
    AscendC::Reg::Compare<UInt, CMPMODE::EQ>(regSelectNan[0], inBits0, ctx.nan, maskAll);
    AscendC::Reg::Compare<UInt, CMPMODE::EQ>(regSelectNan[1], inBits1, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::Reg::Select(outKey0, ctx.allOne, inBits0, regSelectNan[0]);
    AscendC::Reg::Select(outKey1, ctx.allOne, inBits1, regSelectNan[1]);

    // 3. sign bit
    AscendC::Reg::And(regTemp[0], outKey0, ctx.signMask, maskAll);
    AscendC::Reg::And(regTemp[1], outKey1, ctx.signMask, maskAll);

    AscendC::Reg::Compare<UInt, CMPMODE::GT>(regSelectSign[0], regTemp[0], ctx.zeros, maskAll);
    AscendC::Reg::Compare<UInt, CMPMODE::GT>(regSelectSign[1], regTemp[1], ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::Reg::Select(regMask[0], ctx.allOne, ctx.signMask, regSelectSign[0]);
    AscendC::Reg::Select(regMask[1], ctx.allOne, ctx.signMask, regSelectSign[1]);
    AscendC::Reg::Xor(outKey0, outKey0, regMask[0], maskAll);
    AscendC::Reg::Xor(outKey1, outKey1, regMask[1], maskAll);
}

template <typename T, size_t N>
__simd_callee__ inline void DuplicateZero(AscendC::Reg::RegTensor<T> (&regArray)[N], AscendC::Reg::MaskReg &mask)
{
    static_assert(N <= 4, "N must be <= 4");
    // 不能用循环, 会导致fatal error: error in backend: Unsupported Inst must be hoisted.
    if constexpr (N >= 1) {
        AscendC::Reg::Duplicate(regArray[0], static_cast<T>(0), mask);
    }
    if constexpr (N >= 2) {
        AscendC::Reg::Duplicate(regArray[1], static_cast<T>(0), mask);
    }
    if constexpr (N >= 3) {
        AscendC::Reg::Duplicate(regArray[2], static_cast<T>(0), mask);
    }
    if constexpr (N >= 4) {
        AscendC::Reg::Duplicate(regArray[3], static_cast<T>(0), mask);
    }
}

template <typename T, size_t N, bool ApplyRelu = true>
__simd_callee__ inline void WeightedAccum(AscendC::Reg::RegTensor<T> (&accum)[N],
                                          AscendC::Reg::RegTensor<T> (&input)[N], AscendC::Reg::RegTensor<T> &weight,
                                          AscendC::Reg::MaskReg &mask)
{
    static_assert(N <= 2, "N must be <= 2");
    // ---- Relu block ----
    if constexpr (ApplyRelu) {
        if constexpr (N >= 1) {
            AscendC::Reg::Relu(input[0], input[0], mask);
        }
        if constexpr (N >= 2) {
            AscendC::Reg::Relu(input[1], input[1], mask);
        }
    }
    // ---- MulAdd block ----
    if constexpr (N >= 1) {
        AscendC::Reg::MulAddDst(accum[0], input[0], weight, mask);
    }
    if constexpr (N >= 2) {
        AscendC::Reg::MulAddDst(accum[1], input[1], weight, mask);
    }
}

__simd_callee__ inline void BroadcastLane(AscendC::Reg::RegTensor<float> &dst, AscendC::Reg::RegTensor<float> &src,
                                          uint16_t laneIdx)
{
    AscendC::Reg::RegTensor<uint32_t> brcGatherIndex;
    AscendC::Reg::Duplicate(brcGatherIndex, laneIdx);
    AscendC::Reg::Gather(dst, src, brcGatherIndex);
}

__simd_callee__ inline void BroadcastLane(AscendC::Reg::RegTensor<float> &dst, __local_mem__ float *src,
                                          uint16_t laneIdx)
{
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dst, src + laneIdx);
}

// float in uint16 out
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_, // out    [S2Base]     [128   ]
                                             const LocalTensor<float> &qk_,     // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,
                                             const LocalTensor<float> &weight_, // w      [G]          [64    ]
                                             const LocalTensor<float> &kScale_, // kScale [S2Base]     [128   ]
                                             const LocalTensor<float> &qScale_, // qScale [G]          [64    ]
                                             const int gSize)                   // G 64
{
    auto weight = (__local_mem__ float *)weight_.GetPhyAddr();
    auto qScale = (__local_mem__ float *)qScale_.GetPhyAddr();
    auto kScale = (__local_mem__ float *)kScale_.GetPhyAddr();
    auto qk = (__local_mem__ float *)qk_.GetPhyAddr();
    auto out = (__local_mem__ uint16_t *)out_.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> regwBrc;
        AscendC::Reg::RegTensor<float> regQK[2];
        AscendC::Reg::RegTensor<float> regW;

        AscendC::Reg::RegTensor<float> regQScale;
        AscendC::Reg::RegTensor<float> regKScale[2];
        AscendC::Reg::RegTensor<float> regSum0[2];
        AscendC::Reg::RegTensor<float> regSum1[2];
        AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);

        constexpr static Reg::CastTrait castTraitF32ToF16_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                  Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
        constexpr static Reg::CastTrait castTraitF32ToF16_ODD = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

        AscendC::Reg::LoadAlign<float>(regW, weight);
        AscendC::Reg::LoadAlign<float>(regQScale, qScale);
        AscendC::Reg::Mul(regW, regW, regQScale, maskAllB32);

        DuplicateZero(regSum0, maskAllB32);
        DuplicateZero(regSum1, maskAllB32);

        Reg::LoadAlign<float>(regKScale[0], kScale);
        Reg::LoadAlign<float>(regKScale[1], kScale + 64);

        // unroll2
        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i += 2) {
            Reg::LoadAlign<float>(regQK[0], qk + 128 * i); // RowStride是128, 行都落在一个bank上
            Reg::LoadAlign<float>(regQK[1], qk + 128 * i + qkVLStride);
            BroadcastLane(regwBrc, regW, i);
            WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

            Reg::LoadAlign<float>(regQK[0], qk + 128 * i + 128);
            Reg::LoadAlign<float>(regQK[1], qk + 128 * i + 128 + qkVLStride);
            BroadcastLane(regwBrc, regW, i + 1);
            WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
        }

        AscendC::Reg::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
        AscendC::Reg::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

        AscendC::Reg::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
        AscendC::Reg::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);

        AscendC::Reg::RegTensor<bfloat16_t> regSumBF16;
        // interleave cast ==> regSum[1] high regSum[0] low
        AscendC::Reg::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

        AscendC::Reg::RegTensor<uint16_t> regOut;
        FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
        // normal store
        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
    }
}

// bfloat16_t in uint16 out
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint16_t> &out_,  // out    [S2Base]     [128   ]
                                             const LocalTensor<bfloat16_t> &qk_, // q*k^t  [G, S2Base]  [64 128]
                                             const uint32_t qkVLStride,          // unused for bfloat16
                                             const LocalTensor<float> &weight_,  // w      [G]          [64    ]
                                             const LocalTensor<float> &kScale_,  // kScale [S2Base]     [128   ]
                                             const LocalTensor<float> &qScale_,  // qScale [G]          [64    ]
                                             const int gSize)                    // G 64
{
    auto weight = (__local_mem__ float *)weight_.GetPhyAddr();
    auto qScale = (__local_mem__ float *)qScale_.GetPhyAddr();
    auto qk = (__local_mem__ bfloat16_t *)qk_.GetPhyAddr();
    auto kScale = (__local_mem__ float *)kScale_.GetPhyAddr();
    auto out = (__local_mem__ uint16_t *)out_.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> regQK[4];
        AscendC::Reg::RegTensor<bfloat16_t> regQKB16[2];
        AscendC::Reg::RegTensor<float> regW;
        AscendC::Reg::RegTensor<float> regwBrc[2];
        AscendC::Reg::RegTensor<float> regQScale;
        AscendC::Reg::RegTensor<float> regKScale[2];
        AscendC::Reg::RegTensor<float> regSum[2];

        AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::RegTensor<bfloat16_t> regSumBF16;

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);

        using CastTrait = AscendC::Reg::CastTrait;
        static constexpr CastTrait castTraitB162B32_EVEN = {AscendC::Reg::RegLayout::ZERO,
                                                            AscendC::Reg::SatMode::UNKNOWN,
                                                            AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr CastTrait castTraitB162B32_ODD = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                           AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        constexpr static CastTrait castTraitF32ToF16_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                             Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
        constexpr static CastTrait castTraitF32ToF16_ODD = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

        AscendC::Reg::LoadAlign<float>(regW, weight);
        AscendC::Reg::LoadAlign<float>(regQScale, qScale);
        AscendC::Reg::Mul(regW, regW, regQScale, maskAllB32);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(weight, regW, maskAllB32);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

        DuplicateZero(regSum, maskAllB32);

        // interleave load
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(regKScale[0], regKScale[1], kScale);

        // Duplicate + Gather方法劣化
        // Relu在cube随路做
        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
            AscendC::Reg::LoadAlign<bfloat16_t>(regQKB16[0], qk + 256 * i); // RowStride是256, 行都落在一个bank上
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(regwBrc[0], weight + i);
            // interleave cast
            AscendC::Reg::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(regQK[0], regQKB16[0], maskAllB16);
            AscendC::Reg::Cast<float, bfloat16_t, castTraitB162B32_ODD>(regQK[1], regQKB16[0], maskAllB16);
            AscendC::Reg::MulAddDst(regSum[0], regQK[0], regwBrc[0], maskAllB32);
            AscendC::Reg::MulAddDst(regSum[1], regQK[1], regwBrc[0], maskAllB32);
        }

        AscendC::Reg::Mul(regSum[0], regSum[0], regKScale[0], maskAllB32);
        AscendC::Reg::Mul(regSum[1], regSum[1], regKScale[1], maskAllB32);
        // interleave cast back
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum[1], maskAllB32);
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum[0], maskAllB32);

        AscendC::Reg::RegTensor<uint16_t> regOut;
        FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
        // norm load
        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out, regOut, maskAllB16);
    }
}

// 计算S1=2
// float in uint16 out
__aicore__ inline void MulWeightAndReduceSum2(const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
                                              uint32_t outStride,
                                              const LocalTensor<float> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
                                              uint32_t qkVLStride, uint32_t qkStride,
                                              const LocalTensor<float> &weight_, // w      [2, G]          [64    ]
                                              uint32_t weightStride,
                                              const LocalTensor<float> &kScale_, // kScale [S2Base]        [128   ]
                                              uint32_t kScaleStride,
                                              const LocalTensor<float> &qScale_, // qScale [2, G]          [64    ]
                                              uint32_t qScaleStride,
                                              const int gSize) // G 64
{
    auto weight0 = (__local_mem__ float *)weight_.GetPhyAddr();
    auto qScale0 = (__local_mem__ float *)qScale_.GetPhyAddr();
    auto kScale0 = (__local_mem__ float *)kScale_.GetPhyAddr();
    auto qk0 = (__local_mem__ float *)qk_.GetPhyAddr();
    auto out0 = (__local_mem__ uint16_t *)out_.GetPhyAddr();

    auto weight1 = weight0 + weightStride;
    auto qScale1 = qScale0 + qScaleStride;
    auto qk1 = qk0 + qkStride;
    // kScaleStride is zero
    auto out1 = out0 + outStride;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> regwBrc[2];
        AscendC::Reg::RegTensor<float> regQK0[2];
        AscendC::Reg::RegTensor<float> regQK1[2];
        AscendC::Reg::RegTensor<float> regW[2];

        AscendC::Reg::RegTensor<float> regQScale[2];
        AscendC::Reg::RegTensor<float> regKScale[2];
        AscendC::Reg::RegTensor<float> regSum0[2];
        AscendC::Reg::RegTensor<float> regSum1[2];
        AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);

        constexpr static Reg::CastTrait castTraitF32ToF16_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                  Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
        constexpr static Reg::CastTrait castTraitF32ToF16_ODD = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

        AscendC::Reg::LoadAlign<float>(regW[0], weight0);
        AscendC::Reg::LoadAlign<float>(regW[1], weight1);
        AscendC::Reg::LoadAlign<float>(regQScale[0], qScale0);
        AscendC::Reg::LoadAlign<float>(regQScale[1], qScale1);
        AscendC::Reg::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
        AscendC::Reg::Mul(regW[1], regW[1], regQScale[1], maskAllB32);
        // regW[0]与weight1混合使用
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(weight1, regW[1], maskAllB32);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        DuplicateZero(regSum0, maskAllB32);
        DuplicateZero(regSum1, maskAllB32);

        Reg::LoadAlign<float>(regKScale[0], kScale0);
        Reg::LoadAlign<float>(regKScale[1], kScale0 + 64);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
            Reg::LoadAlign<float>(regQK0[0], qk0 + 128 * i);
            Reg::LoadAlign<float>(regQK0[1], qk0 + 128 * i + qkVLStride);
            Reg::LoadAlign<float>(regQK1[0], qk1 + 128 * i);
            Reg::LoadAlign<float>(regQK1[1], qk1 + 128 * i + qkVLStride);
            // 混合使用对整体性能更好
            BroadcastLane(regwBrc[0], regW[0], i);
            // Weight无bank冲突，用LoadAlign来提取weight标量
            BroadcastLane(regwBrc[1], weight1, i);
            AscendC::Reg::Relu(regQK0[0], regQK0[0], maskAllB32);
            AscendC::Reg::Relu(regQK0[1], regQK0[1], maskAllB32);
            AscendC::Reg::Relu(regQK1[0], regQK1[0], maskAllB32);
            AscendC::Reg::Relu(regQK1[1], regQK1[1], maskAllB32);
            AscendC::Reg::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
            AscendC::Reg::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
            AscendC::Reg::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
            AscendC::Reg::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
        }

        // Apply kScale scaling
        AscendC::Reg::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
        AscendC::Reg::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);
        AscendC::Reg::Mul(regSum1[0], regSum1[0], regKScale[0], maskAllB32);
        AscendC::Reg::Mul(regSum1[1], regSum1[1], regKScale[1], maskAllB32);

        // Convert to bfloat16 and store output channel
        AscendC::Reg::RegTensor<bfloat16_t> regSumBF16[2];
        AscendC::Reg::RegTensor<uint16_t> regOut[2];
        AscendC::Reg::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
        AscendC::Reg::DeInterleave(regSum1[0], regSum1[1], regSum1[0], regSum1[1]);
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

        FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx, maskAllB16);
        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regOut[0], maskAllB16);
        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out1, regOut[1], maskAllB16);
    }
}

// 计算S1=2
// bfloat16 in uint16 out
__aicore__ inline void MulWeightAndReduceSum2(const LocalTensor<uint16_t> &out_, // out    [2, S2Base]     [128   ]
                                              uint32_t outStride,
                                              const LocalTensor<bfloat16_t> &qk_, // q*k^t  [2, G, S2Base]  [64 128]
                                              uint32_t qkVLStride,
                                              uint32_t qkStride,                 // gSize * 256
                                              const LocalTensor<float> &weight_, // w      [2, G]          [64    ]
                                              uint32_t weightStride,
                                              const LocalTensor<float> &kScale_, // kScale [S2Base]        [128   ]
                                              uint32_t kScaleStride,
                                              const LocalTensor<float> &qScale_, // qScale [2, G]          [64    ]
                                              uint32_t qScaleStride,
                                              const int gSize) // G 64
{
    auto weight0 = (__local_mem__ float *)weight_.GetPhyAddr();
    auto qScale0 = (__local_mem__ float *)qScale_.GetPhyAddr();
    auto kScale0 = (__local_mem__ float *)kScale_.GetPhyAddr();
    auto qk0 = (__local_mem__ bfloat16_t *)qk_.GetPhyAddr();
    auto out0 = (__local_mem__ uint16_t *)out_.GetPhyAddr();

    auto weight1 = weight0 + weightStride;
    auto qScale1 = qScale0 + qScaleStride;
    auto qk1 = qk0 + qkStride;
    // kScaleStride is zero
    auto out1 = out0 + outStride;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> regwBrc[2];
        AscendC::Reg::RegTensor<float> regQK0[2];
        AscendC::Reg::RegTensor<float> regQK1[2];
        AscendC::Reg::RegTensor<float> regW[2];
        AscendC::Reg::RegTensor<bfloat16_t> regQKB16[2];

        AscendC::Reg::RegTensor<float> regQScale[2];
        AscendC::Reg::RegTensor<float> regKScale[2];
        AscendC::Reg::RegTensor<float> regSum0[2];
        AscendC::Reg::RegTensor<float> regSum1[2];
        AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

        FloatSortConstCtx<bfloat16_t> bf16Ctx;
        InitFloatSortConstCtx(bf16Ctx, maskAllB16);

        using CastTrait = AscendC::Reg::CastTrait;
        static constexpr CastTrait castTraitB162B32_EVEN = {AscendC::Reg::RegLayout::ZERO,
                                                            AscendC::Reg::SatMode::UNKNOWN,
                                                            AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr CastTrait castTraitB162B32_ODD = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                           AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        constexpr static Reg::CastTrait castTraitF32ToF16_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                  Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
        constexpr static Reg::CastTrait castTraitF32ToF16_ODD = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

        AscendC::Reg::LoadAlign<float>(regW[0], weight0);
        AscendC::Reg::LoadAlign<float>(regW[1], weight1);
        AscendC::Reg::LoadAlign<float>(regQScale[0], qScale0);
        AscendC::Reg::LoadAlign<float>(regQScale[1], qScale1);
        AscendC::Reg::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
        AscendC::Reg::Mul(regW[1], regW[1], regQScale[1], maskAllB32);
        // 读写依赖，寄存器可以保序
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(weight0, regW[0], maskAllB32);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(weight1, regW[1], maskAllB32);
        DuplicateZero(regSum0, maskAllB32);
        DuplicateZero(regSum1, maskAllB32);

        // interleave load
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(regKScale[0], regKScale[1], kScale0);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
            AscendC::Reg::LoadAlign<bfloat16_t>(regQKB16[0], qk0 + 256 * i); // RowStride是256, 行都落在一个bank上
            AscendC::Reg::LoadAlign<bfloat16_t>(regQKB16[1], qk1 + 256 * i); // RowStride是256, 行都落在一个bank上
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(regwBrc[0], weight0 + i);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(regwBrc[1], weight1 + i);
            // interleave cast
            AscendC::Reg::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(regQK0[0], regQKB16[0], maskAllB32);
            AscendC::Reg::Cast<float, bfloat16_t, castTraitB162B32_ODD>(regQK0[1], regQKB16[0], maskAllB32);
            AscendC::Reg::Cast<float, bfloat16_t, castTraitB162B32_EVEN>(regQK1[0], regQKB16[1], maskAllB32);
            AscendC::Reg::Cast<float, bfloat16_t, castTraitB162B32_ODD>(regQK1[1], regQKB16[1], maskAllB32);
            AscendC::Reg::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
            AscendC::Reg::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
            AscendC::Reg::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
            AscendC::Reg::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
        }

        // Apply kScale scaling
        AscendC::Reg::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
        AscendC::Reg::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);
        AscendC::Reg::Mul(regSum1[0], regSum1[0], regKScale[0], maskAllB32);
        AscendC::Reg::Mul(regSum1[1], regSum1[1], regKScale[1], maskAllB32);

        // Convert to bfloat16 and store output channel
        AscendC::Reg::RegTensor<bfloat16_t> regSumBF16[2];
        AscendC::Reg::RegTensor<uint16_t> regOut[2];
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
        AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

        FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx, maskAllB16);
        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regOut[0], maskAllB16);
        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out1, regOut[1], maskAllB16);
    }
}

template <typename QK_T, typename SCORE_T>
__aicore__ inline void BatchMulWeightAndReduceSum(const LocalTensor<SCORE_T> &out_, // out    [S2Base]     [128   ]
                                                  uint32_t outStride,
                                                  const LocalTensor<QK_T> &qk_, // q*k^t  [G, S2Base]  [64 128]
                                                  uint32_t qkVLStride, uint32_t qkStride,
                                                  const LocalTensor<float> &weight_, // w      [G]          [64    ]
                                                  uint32_t weightStride,
                                                  const LocalTensor<float> &kScale_, // kScale [S2Base]     [128   ]
                                                  uint32_t kScaleStride,
                                                  const LocalTensor<float> &qScale_, // qScale [G]          [64    ]
                                                  uint32_t qScaleStride,
                                                  const int gSize, // G 64
                                                  const int batch)
{
    // 暂只支持这两种情况, 后续改成循环
    if (batch != 2 && batch != 1) {
        return;
    }
    if (batch == 2) {
        MulWeightAndReduceSum2(out_, outStride, qk_, qkVLStride, qkStride, weight_, weightStride, kScale_, kScaleStride,
                               qScale_, qScaleStride, gSize);
    } else {
        MulWeightAndReduceSum(out_, qk_, qkVLStride, weight_, kScale_, qScale_, gSize);
    }
}

} // namespace vector1

#endif
