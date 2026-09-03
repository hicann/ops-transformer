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
#ifndef QUANT_LIGHTNING_INDEXER_VECTOR1_H
#define QUANT_LIGHTNING_INDEXER_VECTOR1_H

#include "kernel_operator.h"
#if __has_include("../../../lightning_indexer/arch35/vf/common/lightning_indexer_vector1_base.h")
#include "../../../lightning_indexer/arch35/vf/common/lightning_indexer_vector1_base.h"
#else
#include "../../../../lightning_indexer/op_kernel/arch35/vf/common/lightning_indexer_vector1_base.h"
#endif

namespace vector1 {

__simd_callee__ inline void BroadcastLane(AscendC::Reg::RegTensor<float> &dst, __ubuf__ float *src, uint16_t laneIdx)
{
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dst, src + laneIdx);
}

__simd_callee__ inline void CastFP32ToFP16ToFP32(AscendC::Reg::RegTensor<float> (&regQK0)[2],
                                                 AscendC::Reg::RegTensor<half> (&regQK0Half)[2],
                                                 AscendC::Reg::MaskReg &maskAllB32)
{
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
    constexpr static Reg::CastTrait castTraitFP32ToFP16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static Reg::CastTrait castTraitFP16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    float mulsScalar = 1.0 / 1024;

    Reg::Muls(regQK0[0], regQK0[0], mulsScalar, maskAllB32);
    Reg::Muls(regQK0[1], regQK0[1], mulsScalar, maskAllB32);

    Reg::Cast<half, float, castTraitFP32ToFP16>(regQK0Half[0], regQK0[0], maskAllB32);
    Reg::Cast<half, float, castTraitFP32ToFP16>(regQK0Half[1], regQK0[1], maskAllB32);

    Reg::Cast<float, half, castTraitFP16ToFP32>(regQK0[0], regQK0Half[0], maskAllB16);
    Reg::Cast<float, half, castTraitFP16ToFP32>(regQK0[1], regQK0Half[1], maskAllB16);
}

template <typename QK_T>
__simd_callee__ inline void ReduceSumLoopBody(
    AscendC::Reg::RegTensor<float> (&regQK)[2], AscendC::Reg::RegTensor<half> (&regQKHalf)[2],
    AscendC::Reg::RegTensor<int32_t> (&regQKInt32)[2], AscendC::Reg::RegTensor<float> &regwBrc,
    AscendC::Reg::RegTensor<float> &regW, AscendC::Reg::RegTensor<float> (&regSum0)[2],
    AscendC::Reg::RegTensor<float> (&regSum1)[2], AscendC::Reg::MaskReg &maskAllB32, __ubuf__ QK_T *qk_,
    const uint32_t qkVLStride, const int gSize)
{
    constexpr static Reg::CastTrait castTraitInt32ToFP32 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                                            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i += 2) {
        if constexpr (std::is_same<QK_T, int32_t>::value) {
            Reg::LoadAlign<int32_t>(regQKInt32[0], qk_ + 128 * i);
            Reg::LoadAlign<int32_t>(regQKInt32[1], qk_ + 128 * i + qkVLStride);
            Reg::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[0], regQKInt32[0], maskAllB32);
            Reg::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[1], regQKInt32[1], maskAllB32);

            CastFP32ToFP16ToFP32(regQK, regQKHalf, maskAllB32);
        } else {
            Reg::LoadAlign<float>(regQK[0], qk_ + 128 * i);
            Reg::LoadAlign<float>(regQK[1], qk_ + 128 * i + qkVLStride);
        }
        BroadcastLane(regwBrc, regW, i);
        WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);
        if constexpr (std::is_same<QK_T, int32_t>::value) {
            Reg::LoadAlign<int32_t>(regQKInt32[0], qk_ + 128 * i + 128);
            Reg::LoadAlign<int32_t>(regQKInt32[1], qk_ + 128 * i + 128 + qkVLStride);
            Reg::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[0], regQKInt32[0], maskAllB32);
            Reg::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[1], regQKInt32[1], maskAllB32);

            CastFP32ToFP16ToFP32(regQK, regQKHalf, maskAllB32);
        } else {
            Reg::LoadAlign<float>(regQK[0], qk_ + 128 * i + 128);
            Reg::LoadAlign<float>(regQK[1], qk_ + 128 * i + 128 + qkVLStride);
        }
        BroadcastLane(regwBrc, regW, i + 1);
        WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }
}

__simd_callee__ inline void ReduceSumFinalize(AscendC::Reg::RegTensor<float> (&regSum0)[2],
                                              AscendC::Reg::RegTensor<float> (&regSum1)[2],
                                              AscendC::Reg::RegTensor<float> (&regKScale)[2],
                                              AscendC::Reg::MaskReg &maskAllB32, FloatSortConstCtx<bfloat16_t> &bf16Ctx,
                                              AscendC::Reg::MaskReg &maskAllB16, __ubuf__ uint16_t *out_)
{
    AscendC::Reg::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    AscendC::Reg::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    AscendC::Reg::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    AscendC::Reg::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);

    constexpr static Reg::CastTrait castTraitF32ToF16_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                              Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait castTraitF32ToF16_ODD = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    AscendC::Reg::RegTensor<bfloat16_t> regSumBF16;
    AscendC::Reg::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16, regSum0[1], maskAllB32);
    AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16, regSum0[0], maskAllB32);

    AscendC::Reg::RegTensor<uint16_t> regOut;
    FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out_, regOut, maskAllB16);
}

// uint32 out
__simd_callee__ inline void ReduceSumFinalize(AscendC::Reg::RegTensor<float> (&regSum0)[2],
                                              AscendC::Reg::RegTensor<float> (&regSum1)[2],
                                              AscendC::Reg::RegTensor<float> (&regKScale)[2],
                                              AscendC::Reg::MaskReg &maskAllB32, FloatSortConstCtx<float> &fp32Ctx,
                                              __ubuf__ uint32_t *out_)
{
    AscendC::Reg::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    AscendC::Reg::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    AscendC::Reg::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    AscendC::Reg::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);

    AscendC::Reg::RegTensor<uint32_t> regOut[2];
    FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum0[0], regSum0[1], fp32Ctx, maskAllB32);

    AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out_, regOut[0], maskAllB32);
    AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out_ + 64, regOut[1], maskAllB32);
}

template <typename QK_T>
__simd_callee__ inline void ReduceSum2LoopBody(
    AscendC::Reg::RegTensor<float> (&regQK0)[2], AscendC::Reg::RegTensor<float> (&regQK1)[2],
    AscendC::Reg::RegTensor<half> (&regQK0Half)[2], AscendC::Reg::RegTensor<half> (&regQK1Half)[2],
    AscendC::Reg::RegTensor<int32_t> (&regQK0Int32)[2], AscendC::Reg::RegTensor<int32_t> (&regQK1Int32)[2],
    AscendC::Reg::RegTensor<float> (&regwBrc)[2], AscendC::Reg::RegTensor<float> (&regW)[2],
    AscendC::Reg::RegTensor<float> (&regSum0)[2], AscendC::Reg::RegTensor<float> (&regSum1)[2],
    AscendC::Reg::MaskReg &maskAllB32, __ubuf__ QK_T *qk0_, __ubuf__ QK_T *qk1_, const uint32_t qkVLStride,
    __ubuf__ float *brcWeight_, const int gSize)
{
    constexpr static Reg::CastTrait castTraitInt32ToFP32 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                                            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
        if constexpr (std::is_same<QK_T, int32_t>::value) {
            Reg::LoadAlign<int32_t>(regQK0Int32[0], qk0_ + 128 * i);
            Reg::Cast<float, int32_t, castTraitInt32ToFP32>(regQK0[0], regQK0Int32[0], maskAllB32);
            Reg::LoadAlign<int32_t>(regQK0Int32[1], qk0_ + 128 * i + qkVLStride);
            Reg::Cast<float, int32_t, castTraitInt32ToFP32>(regQK0[1], regQK0Int32[1], maskAllB32);
            Reg::LoadAlign<int32_t>(regQK1Int32[0], qk1_ + 128 * i);
            Reg::Cast<float, int32_t, castTraitInt32ToFP32>(regQK1[0], regQK1Int32[0], maskAllB32);
            Reg::LoadAlign<int32_t>(regQK1Int32[1], qk1_ + 128 * i + qkVLStride);
            Reg::Cast<float, int32_t, castTraitInt32ToFP32>(regQK1[1], regQK1Int32[1], maskAllB32);
        } else {
            Reg::LoadAlign<float>(regQK0[0], qk0_ + 128 * i);
            Reg::LoadAlign<float>(regQK0[1], qk0_ + 128 * i + qkVLStride);
            Reg::LoadAlign<float>(regQK1[0], qk1_ + 128 * i);
            Reg::LoadAlign<float>(regQK1[1], qk1_ + 128 * i + qkVLStride);
        }
        BroadcastLane(regwBrc[0], regW[0], i);
        BroadcastLane(regwBrc[1], brcWeight_, i);
        AscendC::Reg::Relu(regQK0[0], regQK0[0], maskAllB32);
        AscendC::Reg::Relu(regQK0[1], regQK0[1], maskAllB32);
        AscendC::Reg::Relu(regQK1[0], regQK1[0], maskAllB32);
        AscendC::Reg::Relu(regQK1[1], regQK1[1], maskAllB32);

        if constexpr (std::is_same<QK_T, int32_t>::value) {
            CastFP32ToFP16ToFP32(regQK0, regQK0Half, maskAllB32);
            CastFP32ToFP16ToFP32(regQK1, regQK1Half, maskAllB32);
        }
        AscendC::Reg::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
        AscendC::Reg::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
        AscendC::Reg::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
        AscendC::Reg::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
    }
}

__simd_callee__ inline void ReduceSum2Finalize(AscendC::Reg::RegTensor<float> (&regSum0)[2],
                                               AscendC::Reg::RegTensor<float> (&regSum1)[2],
                                               AscendC::Reg::RegTensor<float> (&regKScale)[2],
                                               AscendC::Reg::MaskReg &maskAllB32,
                                               FloatSortConstCtx<bfloat16_t> &bf16Ctx,
                                               AscendC::Reg::MaskReg &maskAllB16, __ubuf__ uint16_t *out0_,
                                               __ubuf__ uint16_t *out1_)
{
    AscendC::Reg::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    AscendC::Reg::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);
    AscendC::Reg::Mul(regSum1[0], regSum1[0], regKScale[0], maskAllB32);
    AscendC::Reg::Mul(regSum1[1], regSum1[1], regKScale[1], maskAllB32);

    constexpr static Reg::CastTrait castTraitF32ToF16_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                              Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static Reg::CastTrait castTraitF32ToF16_ODD = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    AscendC::Reg::RegTensor<bfloat16_t> regSumBF16[2];
    AscendC::Reg::RegTensor<uint16_t> regOut[2];
    AscendC::Reg::DeInterleave(regSum0[0], regSum0[1], regSum0[0], regSum0[1]);
    AscendC::Reg::DeInterleave(regSum1[0], regSum1[1], regSum1[0], regSum1[1]);
    AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[0], regSum0[1], maskAllB32);
    AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regSumBF16[1], regSum1[1], maskAllB32);
    AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[0], regSum0[0], maskAllB32);
    AscendC::Reg::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regSumBF16[1], regSum1[0], maskAllB32);

    FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx, maskAllB16);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out0_, regOut[0], maskAllB16);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out1_, regOut[1], maskAllB16);
}

__simd_callee__ inline void ReduceSum2Finalize(AscendC::Reg::RegTensor<float> (&regSum0)[2],
                                               AscendC::Reg::RegTensor<float> (&regSum1)[2],
                                               AscendC::Reg::RegTensor<float> (&regKScale)[2],
                                               AscendC::Reg::MaskReg &maskAllB32, FloatSortConstCtx<float> &fp32Ctx,
                                               __ubuf__ uint32_t *out0_, __ubuf__ uint32_t *out1_)
{
    AscendC::Reg::Mul(regSum0[0], regSum0[0], regKScale[0], maskAllB32);
    AscendC::Reg::Mul(regSum0[1], regSum0[1], regKScale[1], maskAllB32);
    AscendC::Reg::Mul(regSum1[0], regSum1[0], regKScale[0], maskAllB32);
    AscendC::Reg::Mul(regSum1[1], regSum1[1], regKScale[1], maskAllB32);

    AscendC::Reg::RegTensor<uint32_t> regOut0[2];
    AscendC::Reg::RegTensor<uint32_t> regOut1[2];

    FloatX2ToSortableKey<float>(regOut0[0], regOut0[1], regSum0[0], regSum0[1], fp32Ctx, maskAllB32);
    FloatX2ToSortableKey<float>(regOut1[0], regOut1[1], regSum1[0], regSum1[1], fp32Ctx, maskAllB32);
    AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out0_, regOut0[0], maskAllB32);
    AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out0_ + 64, regOut0[1], maskAllB32);
    AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out1_, regOut1[0], maskAllB32);
    AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out1_ + 64, regOut1[1], maskAllB32);
}

__simd_callee__ inline void LoadKScaleFP16(AscendC::Reg::RegTensor<half> (&regKScaleFP16)[2],
                                           AscendC::Reg::RegTensor<float> (&regKScale)[2],
                                           AscendC::Reg::MaskReg &maskAllB16, __ubuf__ half *kScale_)
{
    constexpr static Reg::CastTrait castTraitFP16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regKScaleFP16[0], kScale_);
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regKScaleFP16[1], kScale_ + 64);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regKScale[0], regKScaleFP16[0], maskAllB16);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regKScale[1], regKScaleFP16[1], maskAllB16);
}

// float in uint16 out
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum(__ubuf__ uint16_t *out_, __ubuf__ QK_T *qk_, const uint32_t qkVLStride,
                                              __ubuf__ float *weight_, __ubuf__ float *kScale_, __ubuf__ float *qScale_,
                                              const int gSize)
{
    AscendC::Reg::RegTensor<float> regwBrc;
    AscendC::Reg::RegTensor<float> regQK[2];
    AscendC::Reg::RegTensor<half> regQKHalf[2];
    AscendC::Reg::RegTensor<float> regW;
    AscendC::Reg::RegTensor<int32_t> regQKInt32[2];
    AscendC::Reg::RegTensor<float> regQScale;
    AscendC::Reg::RegTensor<float> regKScale[2];
    AscendC::Reg::RegTensor<float> regSum0[2];
    AscendC::Reg::RegTensor<float> regSum1[2];
    AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    AscendC::Reg::LoadAlign<float>(regW, weight_);
    AscendC::Reg::LoadAlign<float>(regQScale, qScale_);
    AscendC::Reg::Mul(regW, regW, regQScale, maskAllB32);

    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    Reg::LoadAlign<float>(regKScale[0], kScale_);
    Reg::LoadAlign<float>(regKScale[1], kScale_ + 64);

    ReduceSumLoopBody<QK_T>(regQK, regQKHalf, regQKInt32, regwBrc, regW, regSum0, regSum1, maskAllB32, qk_, qkVLStride,
                            gSize);

    ReduceSumFinalize(regSum0, regSum1, regKScale, maskAllB32, bf16Ctx, maskAllB16, out_);
}

// float in uint16 out
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum(__ubuf__ uint16_t *out_, __ubuf__ QK_T *qk_, const uint32_t qkVLStride,
                                              __ubuf__ half *weight_, __ubuf__ half *kScale_, __ubuf__ half *qScale_,
                                              const int gSize)
{
    AscendC::Reg::RegTensor<float> regwBrc;
    AscendC::Reg::RegTensor<float> regQK[2];
    AscendC::Reg::RegTensor<half> regQKHalf[2];
    AscendC::Reg::RegTensor<int32_t> regQKInt32[2];
    AscendC::Reg::RegTensor<float> regW;
    AscendC::Reg::RegTensor<half> regWFP16;
    AscendC::Reg::RegTensor<float> regQScale;
    AscendC::Reg::RegTensor<half> regQScaleFP16;
    AscendC::Reg::RegTensor<float> regKScale[2];
    AscendC::Reg::RegTensor<half> regKScaleFP16[2];
    AscendC::Reg::RegTensor<float> regSum0[2];
    AscendC::Reg::RegTensor<float> regSum1[2];
    AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static Reg::CastTrait castTraitFP16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWFP16, weight_);
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regQScaleFP16, qScale_);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regW, regWFP16, maskAllB16);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regQScale, regQScaleFP16, maskAllB16);
    AscendC::Reg::Mul(regW, regW, regQScale, maskAllB32);

    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    LoadKScaleFP16(regKScaleFP16, regKScale, maskAllB16, kScale_);

    ReduceSumLoopBody<QK_T>(regQK, regQKHalf, regQKInt32, regwBrc, regW, regSum0, regSum1, maskAllB32, qk_, qkVLStride,
                            gSize);

    ReduceSumFinalize(regSum0, regSum1, regKScale, maskAllB32, bf16Ctx, maskAllB16, out_);
}

// float in uint16 out
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum(__ubuf__ uint16_t *out_, __ubuf__ QK_T *qk_, const uint32_t qkVLStride,
                                              __ubuf__ bfloat16_t *weight_, __ubuf__ float *kScale_,
                                              __ubuf__ float *qScale_, const int gSize)
{
    AscendC::Reg::RegTensor<float> regwBrc;
    AscendC::Reg::RegTensor<float> regQK[2];
    AscendC::Reg::RegTensor<half> regQKHalf[2];
    AscendC::Reg::RegTensor<bfloat16_t> regWBF16;
    AscendC::Reg::RegTensor<float> regW;
    AscendC::Reg::RegTensor<float> regQScale;
    AscendC::Reg::RegTensor<float> regKScale[2];
    AscendC::Reg::RegTensor<int32_t> regQKInt32[2];
    AscendC::Reg::RegTensor<float> regSum0[2];
    AscendC::Reg::RegTensor<float> regSum1[2];
    AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static Reg::CastTrait castTraitBF16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWBF16, weight_);
    AscendC::Reg::Cast<float, bfloat16_t, castTraitBF16ToFP32>(regW, regWBF16, maskAllB16);
    AscendC::Reg::LoadAlign<float>(regQScale, qScale_);
    AscendC::Reg::Mul(regW, regW, regQScale, maskAllB32);

    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    Reg::LoadAlign<float>(regKScale[0], kScale_);
    Reg::LoadAlign<float>(regKScale[1], kScale_ + 64);

    ReduceSumLoopBody<QK_T>(regQK, regQKHalf, regQKInt32, regwBrc, regW, regSum0, regSum1, maskAllB32, qk_, qkVLStride,
                            gSize);

    ReduceSumFinalize(regSum0, regSum1, regKScale, maskAllB32, bf16Ctx, maskAllB16, out_);
}

// float in uint32 out
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum(__ubuf__ uint32_t *out_, __ubuf__ QK_T *qk_, const uint32_t qkVLStride,
                                              __ubuf__ half *weight_, __ubuf__ half *kScale_, __ubuf__ half *qScale_,
                                              const int gSize)
{
    AscendC::Reg::RegTensor<float> regwBrc;
    AscendC::Reg::RegTensor<float> regQK[2];
    AscendC::Reg::RegTensor<half> regQKHalf[2];
    AscendC::Reg::RegTensor<int32_t> regQKInt32[2];
    AscendC::Reg::RegTensor<float> regW;
    AscendC::Reg::RegTensor<half> regWFP16;
    AscendC::Reg::RegTensor<half> regWFP16Temp;
    AscendC::Reg::RegTensor<float> regQScale;
    AscendC::Reg::RegTensor<half> regQScaleFP16;
    AscendC::Reg::RegTensor<float> regKScale[2];
    AscendC::Reg::RegTensor<half> regKScaleFP16[2];
    AscendC::Reg::RegTensor<float> regSum0[2];
    AscendC::Reg::RegTensor<float> regSum1[2];
    AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

    FloatSortConstCtx<float> fp32Ctx;
    InitFloatSortConstCtx(fp32Ctx, maskAllB32);

    constexpr static Reg::CastTrait castTraitFP16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWFP16, weight_);
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regQScaleFP16, qScale_);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regW, regWFP16, maskAllB16);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regQScale, regQScaleFP16, maskAllB16);
    AscendC::Reg::Mul(regW, regW, regQScale, maskAllB32);

    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);
    constexpr static Reg::CastTrait castTraitFP32ToFP16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    Reg::Cast<half, float, castTraitFP32ToFP16>(regWFP16Temp, regW, maskAllB32);
    Reg::Cast<float, half, castTraitFP16ToFP32>(regW, regWFP16Temp, maskAllB16);

    LoadKScaleFP16(regKScaleFP16, regKScale, maskAllB16, kScale_);

    ReduceSumLoopBody<QK_T>(regQK, regQKHalf, regQKInt32, regwBrc, regW, regSum0, regSum1, maskAllB32, qk_, qkVLStride,
                            gSize);

    ReduceSumFinalize(regSum0, regSum1, regKScale, maskAllB32, fp32Ctx, out_);
}

// 计算S1=2
// float weight in uint16 out
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum2(__ubuf__ uint16_t *out0_, __ubuf__ uint16_t *out1_, uint32_t outStride,
                                               __ubuf__ QK_T *qk0_, __ubuf__ QK_T *qk1_, uint32_t qkVLStride,
                                               uint32_t qkStride, __ubuf__ float *weight0_, __ubuf__ float *weight1_,
                                               uint32_t weightStride, __ubuf__ float *weightFloat_,
                                               __ubuf__ float *kScale_, uint32_t kScaleStride, __ubuf__ float *qScale0_,
                                               __ubuf__ float *qScale1_, uint32_t qScaleStride, const int gSize)
{
    AscendC::Reg::RegTensor<float> regwBrc[2];
    AscendC::Reg::RegTensor<float> regQK0[2], regQK1[2];
    AscendC::Reg::RegTensor<half> regQK0Half[2], regQK1Half[2];
    AscendC::Reg::RegTensor<float> regW[2];
    AscendC::Reg::RegTensor<int32_t> regQK0Int32[2], regQK1Int32[2];
    AscendC::Reg::RegTensor<float> regQScale[2], regKScale[2];
    AscendC::Reg::RegTensor<float> regSum0[2], regSum1[2];
    AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    AscendC::Reg::LoadAlign<float>(regW[0], weight0_);
    AscendC::Reg::LoadAlign<float>(regW[1], weight1_);
    AscendC::Reg::LoadAlign<float>(regQScale[0], qScale0_);
    AscendC::Reg::LoadAlign<float>(regQScale[1], qScale1_);
    AscendC::Reg::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
    AscendC::Reg::Mul(regW[1], regW[1], regQScale[1], maskAllB32);
    // regW[0]与weight1混合使用
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(weight1_, regW[1], maskAllB32);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    Reg::LoadAlign<float>(regKScale[0], kScale_);
    Reg::LoadAlign<float>(regKScale[1], kScale_ + 64);

    ReduceSum2LoopBody<QK_T>(regQK0, regQK1, regQK0Half, regQK1Half, regQK0Int32, regQK1Int32, regwBrc, regW, regSum0,
                             regSum1, maskAllB32, qk0_, qk1_, qkVLStride, weight1_, gSize);

    ReduceSum2Finalize(regSum0, regSum1, regKScale, maskAllB32, bf16Ctx, maskAllB16, out0_, out1_);
}

// 计算S1=2
// bfloat16_t weight in uint16 out
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum2(__ubuf__ uint16_t *out0_, __ubuf__ uint16_t *out1_, uint32_t outStride,
                                               __ubuf__ QK_T *qk0_, __ubuf__ QK_T *qk1_, uint32_t qkVLStride,
                                               uint32_t qkStride, __ubuf__ bfloat16_t *weight0_,
                                               __ubuf__ bfloat16_t *weight1_, uint32_t weightStride,
                                               __ubuf__ float *weightFloat_, __ubuf__ float *kScale_,
                                               uint32_t kScaleStride, __ubuf__ float *qScale0_,
                                               __ubuf__ float *qScale1_, uint32_t qScaleStride, const int gSize)
{
    AscendC::Reg::RegTensor<float> regW[2];
    AscendC::Reg::RegTensor<float> regwBrc[2];
    AscendC::Reg::RegTensor<float> regQK0[2], regQK1[2];
    AscendC::Reg::RegTensor<half> regQK0Half[2], regQK1Half[2];
    AscendC::Reg::RegTensor<int32_t> regQK0Int32[2], regQK1Int32[2];
    AscendC::Reg::RegTensor<bfloat16_t> regWBF16[2];

    AscendC::Reg::RegTensor<float> regQScale[2], regKScale[2];
    AscendC::Reg::RegTensor<float> regSum0[2], regSum1[2];
    AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static Reg::CastTrait castTraitBF16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWBF16[0], weight0_);
    AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWBF16[1], weight1_);
    AscendC::Reg::Cast<float, bfloat16_t, castTraitBF16ToFP32>(regW[0], regWBF16[0], maskAllB16);
    AscendC::Reg::Cast<float, bfloat16_t, castTraitBF16ToFP32>(regW[1], regWBF16[1], maskAllB16);

    AscendC::Reg::LoadAlign<float>(regQScale[0], qScale0_);
    AscendC::Reg::LoadAlign<float>(regQScale[1], qScale1_);
    AscendC::Reg::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
    AscendC::Reg::Mul(regW[1], regW[1], regQScale[1], maskAllB32);
    // regW[0]与weight1混合使用
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(weightFloat_, regW[1], maskAllB32);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    Reg::LoadAlign<float>(regKScale[0], kScale_);
    Reg::LoadAlign<float>(regKScale[1], kScale_ + 64);

    ReduceSum2LoopBody<QK_T>(regQK0, regQK1, regQK0Half, regQK1Half, regQK0Int32, regQK1Int32, regwBrc, regW, regSum0,
                             regSum1, maskAllB32, qk0_, qk1_, qkVLStride, weightFloat_, gSize);

    ReduceSum2Finalize(regSum0, regSum1, regKScale, maskAllB32, bf16Ctx, maskAllB16, out0_, out1_);
}

// 计算S1=2
// half weight in uint16 out
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum2(__ubuf__ uint16_t *out0_, __ubuf__ uint16_t *out1_, uint32_t outStride,
                                               __ubuf__ QK_T *qk0_, __ubuf__ QK_T *qk1_, uint32_t qkVLStride,
                                               uint32_t qkStride, __ubuf__ half *weight0_, __ubuf__ half *weight1_,
                                               uint32_t weightStride, __ubuf__ float *weightFloat_,
                                               __ubuf__ half *kScale_, uint32_t kScaleStride, __ubuf__ half *qScale0_,
                                               __ubuf__ half *qScale1_, uint32_t qScaleStride, const int gSize)
{
    AscendC::Reg::RegTensor<float> regwBrc[2];
    AscendC::Reg::RegTensor<float> regQK0[2], regQK1[2];
    AscendC::Reg::RegTensor<half> regQK0Half[2], regQK1Half[2];
    AscendC::Reg::RegTensor<int32_t> regQK0Int32[2], regQK1Int32[2];
    AscendC::Reg::RegTensor<float> regW[2];
    AscendC::Reg::RegTensor<half> regWFP16[2];

    AscendC::Reg::RegTensor<half> regQScaleFP16[2], regKScaleFP16[2];
    AscendC::Reg::RegTensor<float> regQScale[2], regKScale[2], regSum0[2], regSum1[2];
    AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static Reg::CastTrait castTraitFP16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWFP16[0], weight0_);
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWFP16[1], weight1_);
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regQScaleFP16[0], qScale0_);
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regQScaleFP16[1], qScale1_);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regW[0], regWFP16[0], maskAllB16);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regW[1], regWFP16[1], maskAllB16);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regQScale[0], regQScaleFP16[0], maskAllB16);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regQScale[1], regQScaleFP16[1], maskAllB16);

    AscendC::Reg::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
    AscendC::Reg::Mul(regW[1], regW[1], regQScale[1], maskAllB32);
    // regW[0]与weight1混合使用
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(weightFloat_, regW[1], maskAllB32);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    LoadKScaleFP16(regKScaleFP16, regKScale, maskAllB16, kScale_);

    ReduceSum2LoopBody<QK_T>(regQK0, regQK1, regQK0Half, regQK1Half, regQK0Int32, regQK1Int32, regwBrc, regW, regSum0,
                             regSum1, maskAllB32, qk0_, qk1_, qkVLStride, weightFloat_, gSize);

    ReduceSum2Finalize(regSum0, regSum1, regKScale, maskAllB32, bf16Ctx, maskAllB16, out0_, out1_);
}

// 计算S1=2
// float in uint32 out
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum2(__ubuf__ uint32_t *out0_, __ubuf__ uint32_t *out1_, uint32_t outStride,
                                               __ubuf__ QK_T *qk0_, __ubuf__ QK_T *qk1_, uint32_t qkVLStride,
                                               uint32_t qkStride, __ubuf__ half *weight0_, __ubuf__ half *weight1_,
                                               uint32_t weightStride, __ubuf__ float *weightFloat_,
                                               __ubuf__ half *kScale_, uint32_t kScaleStride, __ubuf__ half *qScale0_,
                                               __ubuf__ half *qScale1_, uint32_t qScaleStride, const int gSize)
{
    AscendC::Reg::RegTensor<float> regwBrc[2];
    AscendC::Reg::RegTensor<float> regQK0[2], regQK1[2];
    AscendC::Reg::RegTensor<half> regQK0Half[2], regQK1Half[2];
    AscendC::Reg::RegTensor<int32_t> regQK0Int32[2], regQK1Int32[2];
    AscendC::Reg::RegTensor<float> regW[2];
    AscendC::Reg::RegTensor<half> regWFP16[2], regWFP16Temp[2];

    AscendC::Reg::RegTensor<half> regQScaleFP16[2], regKScaleFP16[2];
    AscendC::Reg::RegTensor<float> regQScale[2], regKScale[2], regSum0[2], regSum1[2];
    AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();

    FloatSortConstCtx<float> fp32Ctx;
    InitFloatSortConstCtx(fp32Ctx, maskAllB32);

    constexpr static Reg::CastTrait castTraitFP16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWFP16[0], weight0_);
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regWFP16[1], weight1_);
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regQScaleFP16[0], qScale0_);
    AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regQScaleFP16[1], qScale1_);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regW[0], regWFP16[0], maskAllB16);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regW[1], regWFP16[1], maskAllB16);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regQScale[0], regQScaleFP16[0], maskAllB16);
    AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regQScale[1], regQScaleFP16[1], maskAllB16);

    AscendC::Reg::Mul(regW[0], regW[0], regQScale[0], maskAllB32);
    AscendC::Reg::Mul(regW[1], regW[1], regQScale[1], maskAllB32);

    constexpr static Reg::CastTrait castTraitFP32ToFP16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    Reg::Cast<half, float, castTraitFP32ToFP16>(regWFP16Temp[0], regW[0], maskAllB32);
    Reg::Cast<half, float, castTraitFP32ToFP16>(regWFP16Temp[1], regW[1], maskAllB32);

    Reg::Cast<float, half, castTraitFP16ToFP32>(regW[0], regWFP16Temp[0], maskAllB16);
    Reg::Cast<float, half, castTraitFP16ToFP32>(regW[1], regWFP16Temp[1], maskAllB16);

    // regW[0]与weight1混合使用
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(weightFloat_, regW[1], maskAllB32);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    LoadKScaleFP16(regKScaleFP16, regKScale, maskAllB16, kScale_);

    ReduceSum2LoopBody<QK_T>(regQK0, regQK1, regQK0Half, regQK1Half, regQK0Int32, regQK1Int32, regwBrc, regW, regSum0,
                             regSum1, maskAllB32, qk0_, qk1_, qkVLStride, weightFloat_, gSize);

    ReduceSum2Finalize(regSum0, regSum1, regKScale, maskAllB32, fp32Ctx, out0_, out1_);
}

template <typename QK_T, typename W_T, typename SCALE_T, typename SCORE_T>
__aicore__ inline void BatchMulWeightAndReduceSum(const LocalTensor<SCORE_T> &out_, // out    [S2Base]     [128   ]
                                                  uint32_t outStride,
                                                  const LocalTensor<QK_T> &qk_, // q*k^t  [G, S2Base]  [64 128]
                                                  uint32_t qkVLStride, uint32_t qkStride,
                                                  const LocalTensor<W_T> &weight_, // w      [G]          [64    ]
                                                  uint32_t weightStride, const LocalTensor<float> &weightFloat_,
                                                  const LocalTensor<SCALE_T> &kScale_, // kScale [S2Base]     [128   ]
                                                  uint32_t kScaleStride,
                                                  const LocalTensor<SCALE_T> &qScale_, // qScale [G]          [64    ]
                                                  uint32_t qScaleStride,
                                                  const int gSize, // G 64
                                                  const int batch)
{
    // 暂只支持这两种情况, 后续改成循环
    if (batch != 2 && batch != 1) {
        return;
    }
    auto weight = (__ubuf__ W_T *)weight_.GetPhyAddr();
    auto weightFloat = (__ubuf__ float *)weightFloat_.GetPhyAddr();
    auto qScale = (__ubuf__ SCALE_T *)qScale_.GetPhyAddr();
    auto kScale = (__ubuf__ SCALE_T *)kScale_.GetPhyAddr();
    auto qk = (__ubuf__ QK_T *)qk_.GetPhyAddr();
    auto out = (__ubuf__ SCORE_T *)out_.GetPhyAddr();

    if (batch == 2) {
        auto weight1 = weight + weightStride;
        auto qScale1 = qScale + qScaleStride;
        auto qk1 = qk + qkStride;
        auto out1 = out + outStride;

        MulWeightAndReduceSum2(out, out1, outStride, qk, qk1, qkVLStride, qkStride, weight, weight1, weightStride,
                               weightFloat, kScale, kScaleStride, qScale, qScale1, qScaleStride, gSize);
    } else {
        MulWeightAndReduceSum(out, qk, qkVLStride, weight, kScale, qScale, gSize);
    }
}
} // namespace vector1

#endif
