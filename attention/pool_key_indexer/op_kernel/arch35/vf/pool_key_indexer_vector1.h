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
 * \file pool_key_indexer_vector1.h
 * \brief
 */
#ifndef POOL_KEY_INDEXER_VECTOR1_H
#define POOL_KEY_INDEXER_VECTOR1_H

#include "kernel_operator.h"
#include "common/pool_key_indexer_vector1_base.h"

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

template <>
struct UIntSortTraits<bfloat16_t> {
    using UInt = uint16_t;
    static constexpr UInt ZERO = 0x0000;
    static constexpr UInt SIGN_MASK = 0x8000;
    static constexpr UInt NAN_MASK = 0xFFC0;
    static constexpr UInt ALL_ONE = 0xFFFF;
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
__simd_callee__ inline void InitUIntSortConstCtx(UIntSortConstCtx<FloatT>& ctx, AscendC::Reg::MaskReg& maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    AscendC::Reg::Duplicate(ctx.zeros, Traits::ZERO, maskAll);
    AscendC::Reg::Duplicate(ctx.allOne, Traits::ALL_ONE, maskAll);
    AscendC::Reg::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::Reg::Duplicate(ctx.nan, Traits::NAN_MASK, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void UIntToSortableKey(AscendC::Reg::RegTensor<FloatT>& outKey,
                                              AscendC::Reg::RegTensor<typename UIntSortConstCtx<FloatT>::UInt>& inVal,
                                              UIntSortConstCtx<FloatT>& ctx, AscendC::Reg::MaskReg& maskAll)
{
    using Traits = UIntSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::Reg::RegTensor<UInt> regTemp;
    AscendC::Reg::RegTensor<UInt> regMask;
    AscendC::Reg::MaskReg regSelectZero;
    AscendC::Reg::MaskReg regSelectSign;

    auto& inBits = inVal;

    // 1. 0 check
    AscendC::Reg::Compare<UInt, CMPMODE::EQ>(regSelectZero, inBits, ctx.zeros, maskAll);

    // 2. 0 -> -NAN
    AscendC::Reg::Select((AscendC::Reg::RegTensor<UInt>&)outKey, ctx.nan, inBits, regSelectZero);

    // 3. sign bit
    AscendC::Reg::And(regTemp, (AscendC::Reg::RegTensor<UInt>&)outKey, ctx.signMask, maskAll);

    AscendC::Reg::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::Reg::Select(regMask, ctx.signMask, ctx.allOne, regSelectSign);
    AscendC::Reg::Xor((AscendC::Reg::RegTensor<UInt>&)outKey, (AscendC::Reg::RegTensor<UInt>&)outKey, regMask, maskAll);
}

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<bfloat16_t>& out_, const LocalTensor<uint32_t>& in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ bfloat16_t*)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t*)in.GetPhyAddr();

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

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<half>& out_, const LocalTensor<uint32_t>& in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ half*)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t*)in.GetPhyAddr();

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

// PKI: float output version (no Cast needed, values output is FLOAT)
__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<float>& out_, const LocalTensor<uint32_t>& in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ float*)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t*)in.GetPhyAddr();

    const uint16_t repeatSize32 = 128;
    uint16_t topkLoopNum = (topK + repeatSize32 - 1) / repeatSize32;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint32_t> regIn[2];
        AscendC::Reg::RegTensor<float> regOut[2];
        AscendC::Reg::MaskReg maskAllB32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB32f = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::Reg::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::Reg::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);
            UIntSortConstCtx<float> uint32Ctx;
            InitUIntSortConstCtx(uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            // float 入/出同宽(4B->4B), 不能沿用 bf16/fp16 的 DeInterleave
            // 半宽合并(同宽对上是 2:1 抽取丢元素); 两个 64 lane 寄存器分别落回原位
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize32, regOut[0],
                                                                                maskAllB32f);
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize32 + 64,
                                                                                regOut[1], maskAllB32f);
        }
    }
}

// uint16(bf16 sortable key) -> fp32 values: 反变换到 bf16 值域后经 values 缓冲
// 原地 bf16 视图中转, 半宽对两段做 bf16->fp32 展宽落回输出(同域复用, float store
// 完全覆盖 scratch 字节范围, 原位安全)
__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<float>& out_, const LocalTensor<uint16_t>& in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ float*)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint16_t*)in.GetPhyAddr();
    auto bfView = (__local_mem__ bfloat16_t*)out_.GetPhyAddr();

    const uint16_t repeatSize = 128;
    uint16_t topkLoopNum = (topK + repeatSize - 1) / repeatSize;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> regIn;
        AscendC::Reg::RegTensor<bfloat16_t> regBf;
        AscendC::Reg::RegTensor<bfloat16_t> regLo;
        AscendC::Reg::RegTensor<bfloat16_t> regHi;
        AscendC::Reg::RegTensor<float> regF[2];
        AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskHalfB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::H>();
        AscendC::Reg::MaskReg maskAllF = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        constexpr static Reg::CastTrait castTraitBf16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        UIntSortConstCtx<bfloat16_t> bf16KeyCtx;
        InitUIntSortConstCtx(bf16KeyCtx, maskAllB16);
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::Reg::LoadAlign<uint16_t>(regIn, inBuf + i * repeatSize);
            UIntToSortableKey<bfloat16_t>(regBf, regIn, bf16KeyCtx, maskAllB16);
            AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_NORM>(bfView + i * repeatSize, regBf,
                                                                                     maskAllB16);
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
            AscendC::Reg::LoadAlign<bfloat16_t>(regLo, bfView + i * repeatSize);
            AscendC::Reg::LoadAlign<bfloat16_t>(regHi, bfView + i * repeatSize + 64);
            AscendC::Reg::Cast<float, bfloat16_t, castTraitBf16ToFP32>(regF[0], regLo, maskHalfB16);
            AscendC::Reg::Cast<float, bfloat16_t, castTraitBf16ToFP32>(regF[1], regHi, maskHalfB16);
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize, regF[0],
                                                                                maskAllF);
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize + 64, regF[1],
                                                                                maskAllF);
        }
    }
}

__simd_callee__ inline void BroadcastLane(AscendC::Reg::RegTensor<float>& dst, __local_mem__ float* src,
                                          uint16_t laneIdx)
{
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dst, src + laneIdx);
}

template <typename KEY_T, typename W_T>
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<KEY_T>& out,  // out    [S2Base]     [128   ] 2
                                             const LocalTensor<float>& qk,   // q*k^t  [G, S2Base]  [64 128] 2
                                             const LocalTensor<W_T>& weight, // w      [G]          [64    ] 1
                                             const int gSize,                // G 64
                                             const float scale)              // 1/sqrt(headDim)
{
    __local_mem__ W_T* weight_ = (__local_mem__ W_T*)weight.GetPhyAddr();

    constexpr uint32_t VL = 64; // vector length

    auto qk0 = (__local_mem__ float*)qk.GetPhyAddr();
    auto qk1 = qk0 + VL;
    auto out0 = (__local_mem__ KEY_T*)out.GetPhyAddr();
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
        AscendC::Reg::RegTensor<float> regScale;
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

        // 池级分数乘 1/sqrt(headDim)(文档公式 S = Q@K^T/sqrt(headDim));
        // 缩放为正数, 与 ReLU/加权求和可交换, 在聚合后统一应用
        AscendC::Reg::Duplicate(regScale, scale, maskAll);
        AscendC::Reg::Mul(regSum[0], regSum[0], regScale, maskAll);
        AscendC::Reg::Mul(regSum[1], regSum[1], regScale, maskAll);

        if constexpr (sizeof(KEY_T) == 4) {
            AscendC::Reg::RegTensor<uint32_t> regOut[2];
            FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

            AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
            AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
        } else {
            // uint16 键: DeInterleave 预排 + 奇/偶 lane 压缩 cast -> bf16(RNE) 全宽可排序键
            // (quant_lightning_indexer_v2 同款配方: ODD 先写 ZEROING, EVEN 后写 MERGING 保留奇槽)
            (void)out1;
            constexpr static Reg::CastTrait traitF32ToBf16Even = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                  Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
            constexpr static Reg::CastTrait traitF32ToBf16Odd = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
            AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::DeInterleave(regSum[0], regSum[1], regSum[0], regSum[1]);
            AscendC::Reg::RegTensor<bfloat16_t> regBf;
            AscendC::Reg::Cast<bfloat16_t, float, traitF32ToBf16Odd>(regBf, regSum[1], maskAll);
            AscendC::Reg::Cast<bfloat16_t, float, traitF32ToBf16Even>(regBf, regSum[0], maskAll);
            FloatSortConstCtx<bfloat16_t> bf16Ctx;
            InitFloatSortConstCtx(bf16Ctx, maskAllB16);
            AscendC::Reg::RegTensor<uint16_t> regKey;
            FloatToSortableKey<bfloat16_t>(regKey, regBf, bf16Ctx, maskAllB16);
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regKey, maskAllB16);
        }
    }
}

// mode=0 (FP8 per-token-head) scale 融合版: scale 为正标量, 可从 D 维点积中
// 完全提出, out = (Σ_g relu(Σ_d q·k)·w_g·dq)·dk; weight 为已预乘 qScale 的 fp32 UB
template <typename KEY_T>
__aicore__ inline void MulWeightAndReduceSumWithScale(
    const LocalTensor<KEY_T>& out,    // out    [S2Base]     [128   ] 2
    const LocalTensor<float>& qk,     // q*k^t  [G, S2Base]  [64 128] 2
    const LocalTensor<float>& weight, // w*dq   [G]          [64    ] 1
    const LocalTensor<float>& kScale, // kScale [S2Base]     [128   ] 2
    const int gSize,                  // G 64
    const float scale)                // 1/sqrt(headDim)
{
    __local_mem__ float* weight_ = (__local_mem__ float*)weight.GetPhyAddr();
    __local_mem__ float* kScale_ = (__local_mem__ float*)kScale.GetPhyAddr();

    constexpr uint32_t VL = 64; // vector length

    auto qk0 = (__local_mem__ float*)qk.GetPhyAddr();
    auto qk1 = qk0 + VL;
    auto out0 = (__local_mem__ KEY_T*)out.GetPhyAddr();
    auto out1 = out0 + VL;
    auto kScale0 = kScale_;
    auto kScale1 = kScale_ + VL;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> regQK[2];
        AscendC::Reg::RegTensor<float> regwBrc;
        AscendC::Reg::RegTensor<float> regKScale[2];
        AscendC::Reg::RegTensor<float> regSum[2];
        AscendC::Reg::RegTensor<float> regScale;

        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

        FloatSortConstCtx<float> fp32Ctx;
        InitFloatSortConstCtx(fp32Ctx, maskAll);

        AscendC::Reg::Duplicate(regSum[0], 0.0f, maskAll);
        AscendC::Reg::Duplicate(regSum[1], 0.0f, maskAll);

        AscendC::Reg::LoadAlign<float>(regKScale[0], kScale0);
        AscendC::Reg::LoadAlign<float>(regKScale[1], kScale1);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); ++i) {
            // BroadcastLane: 从 UB 直接广播第 i 个 weight
            // (Gather 与 LoadAlign fp32 产物的 lane 布局不兼容)
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(regwBrc, weight_ + i);
            AscendC::Reg::LoadAlign<float>(regQK[0], qk0 + 128 * i);
            AscendC::Reg::LoadAlign<float>(regQK[1], qk1 + 128 * i);

            AscendC::Reg::Relu(regQK[0], regQK[0], maskAll);
            AscendC::Reg::Relu(regQK[1], regQK[1], maskAll);

            AscendC::Reg::MulAddDst(regSum[0], regQK[0], regwBrc, maskAll);
            AscendC::Reg::MulAddDst(regSum[1], regQK[1], regwBrc, maskAll);
        }

        // kScale 乘入聚合和(逐 s2 元素), 随后统一乘 1/sqrt(headDim)
        AscendC::Reg::Mul(regSum[0], regSum[0], regKScale[0], maskAll);
        AscendC::Reg::Mul(regSum[1], regSum[1], regKScale[1], maskAll);

        // 池级分数乘 1/sqrt(headDim)(文档公式 S = Q@K^T/sqrt(headDim));
        // 缩放为正数, 与 ReLU/加权求和可交换, 在聚合后统一应用
        AscendC::Reg::Duplicate(regScale, scale, maskAll);
        AscendC::Reg::Mul(regSum[0], regSum[0], regScale, maskAll);
        AscendC::Reg::Mul(regSum[1], regSum[1], regScale, maskAll);

        if constexpr (sizeof(KEY_T) == 4) {
            AscendC::Reg::RegTensor<uint32_t> regOut[2];
            FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

            AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
            AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
        } else {
            // uint16 键: DeInterleave 预排 + 奇/偶 lane 压缩 cast -> bf16(RNE) 全宽可排序键
            // (quant_lightning_indexer_v2 同款配方: ODD 先写 ZEROING, EVEN 后写 MERGING 保留奇槽)
            (void)out1;
            constexpr static Reg::CastTrait traitF32ToBf16Even = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                  Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
            constexpr static Reg::CastTrait traitF32ToBf16Odd = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
            AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::DeInterleave(regSum[0], regSum[1], regSum[0], regSum[1]);
            AscendC::Reg::RegTensor<bfloat16_t> regBf;
            AscendC::Reg::Cast<bfloat16_t, float, traitF32ToBf16Odd>(regBf, regSum[1], maskAll);
            AscendC::Reg::Cast<bfloat16_t, float, traitF32ToBf16Even>(regBf, regSum[0], maskAll);
            FloatSortConstCtx<bfloat16_t> bf16Ctx;
            InitFloatSortConstCtx(bf16Ctx, maskAllB16);
            AscendC::Reg::RegTensor<uint16_t> regKey;
            FloatToSortableKey<bfloat16_t>(regKey, regBf, bf16Ctx, maskAllB16);
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regKey, maskAllB16);
        }
    }
}
// mode=0: weight×qScale SIMD 预乘(单 s1 行, 64 lanes): 半精度 weight -> fp32
// cast -> 乘 qScale -> 落 fp32 UB; 独立 __simd_vf__ 保证 bf16 Cast 硬件展开
template <typename W_T>
__simd_vf__ void MulWeightQScaleSIMD(__local_mem__ W_T* weight_, __local_mem__ float* qScale_,
                                     __local_mem__ float* dst_)
{
    Reg::MaskReg maskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskAllB16 = Reg::CreateMask<W_T, Reg::MaskPattern::ALL>();

    Reg::RegTensor<W_T> regWHalf;
    Reg::RegTensor<float> regW;
    Reg::RegTensor<float> regQScale;

    constexpr static Reg::CastTrait castTraitWTToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    Reg::LoadAlign<W_T, Reg::LoadDist::DIST_UNPACK_B16>(regWHalf, weight_);
    Reg::Cast<float, W_T, castTraitWTToFP32>(regW, regWHalf, maskAllB16);
    Reg::LoadAlign<float>(regQScale, qScale_);
    Reg::Mul(regW, regW, regQScale, maskAllB32);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(dst_, regW, maskAllB32);
}

// int8: fp32 -> ×2^-10 -> fp16 round-trip -> fp32; ×2^-10 为纯指数运算无舍入,
// fp16 cast(CAST_RINT=RNE)单次舍入, 结果与 fixpipe DEQF16 反量化(fp16 cast)同值
__simd_callee__ inline void CastFP32ToFP16ToFP32(AscendC::Reg::RegTensor<float> (&regQK)[2],
                                                 AscendC::Reg::RegTensor<half> (&regQKHalf)[2],
                                                 AscendC::Reg::MaskReg& maskAllB32)
{
    AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();
    constexpr static Reg::CastTrait castTraitFP32ToFP16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr static Reg::CastTrait castTraitFP16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    float mulsScalar = 1.0 / 1024;

    Reg::Muls(regQK[0], regQK[0], mulsScalar, maskAllB32);
    Reg::Muls(regQK[1], regQK[1], mulsScalar, maskAllB32);

    Reg::Cast<half, float, castTraitFP32ToFP16>(regQKHalf[0], regQK[0], maskAllB32);
    Reg::Cast<half, float, castTraitFP32ToFP16>(regQKHalf[1], regQK[1], maskAllB32);

    Reg::Cast<float, half, castTraitFP16ToFP32>(regQK[0], regQKHalf[0], maskAllB16);
    Reg::Cast<float, half, castTraitFP16ToFP32>(regQK[1], regQKHalf[1], maskAllB16);
}

// int8: weight(half)×q_descale(half) 单 s1 行 SIMD 预乘: 双 half->fp32->Mul->fp16 RNE round-trip->fp32;
// 写满 128 lane([gSize,64) 掩码保留积, 其余置 0); fp32 域积+单次 RNE=half 硬件乘的正确舍入, N1≤64 由 PKI_N1_LIMIT 保证
__simd_vf__ void MulWeightQScaleSIMDHalf(__local_mem__ half* weight_, __local_mem__ half* qScale_,
                                         __local_mem__ float* dst_, int64_t gSize)
{
    Reg::MaskReg maskAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();

    Reg::RegTensor<half> regWHalf;
    Reg::RegTensor<half> regQSHalf;
    Reg::RegTensor<float> regW;
    Reg::RegTensor<float> regQS;
    Reg::RegTensor<half> regWTmpHalf;

    constexpr static Reg::CastTrait castTraitHToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    constexpr static Reg::CastTrait castTraitFP32ToH = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(regWHalf, weight_);
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(regQSHalf, qScale_);
    Reg::Cast<float, half, castTraitHToFP32>(regW, regWHalf, maskAllB16);
    Reg::Cast<float, half, castTraitHToFP32>(regQS, regQSHalf, maskAllB16);
    Reg::Mul(regW, regW, regQS, maskAllB32);
    Reg::Cast<half, float, castTraitFP32ToH>(regWTmpHalf, regW, maskAllB32);
    Reg::Cast<float, half, castTraitHToFP32>(regW, regWTmpHalf, maskAllB16);

    // [gSize,64) 掩码清 0(lane 序号 < gSize 保留乘积, 其余置 0)
    Reg::RegTensor<int32_t> regIdx;
    Reg::RegTensor<int32_t> regGSize;
    Reg::RegTensor<float> regZero;
    Reg::Arange(regIdx, 0);
    Reg::Duplicate(regGSize, static_cast<int32_t>(gSize), maskAllB32);
    Reg::Duplicate(regZero, 0.0f, maskAllB32);
    Reg::MaskReg pregLT;
    Reg::Compare<int32_t, CMPMODE::LT>(pregLT, regIdx, regGSize, maskAllB32);
    Reg::Select(regW, regW, regZero, pregLT);

    // 写满 128 lane: [0,64) 乘积(掩码后) + [64,128) 恒 0
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(dst_, regW, maskAllB32);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(dst_ + 64, regZero, maskAllB32);
}

// int8 聚合版: qk 为 fixpipe 直达 UB 的 int32 精确点积, int32 -> fp32 Cast ->
// CastFP32ToFP16ToFP32(×2^-10+round-trip) -> Relu(与 arch22 DEQF16 舍入后
// reluPre 的硬件序一致) -> wq 加权累加 -> dk 后乘 -> qkScale'(=1024/sqrt(d), D4
// 补偿折入 host) -> uint32 sortable key
template <typename KEY_T>
__aicore__ inline void MulWeightAndReduceSumWithScaleInt8(
    const LocalTensor<KEY_T>& out,    // out    [S2Base]     [128   ]
    const LocalTensor<int32_t>& qk,   // q*k^t  [G, S2Base]  [64 128] int32
    const LocalTensor<float>& weight, // w*dq   [G]          [64    ] fp32 预乘
    const LocalTensor<half>& kScale,  // kScale [S2Base]     [128   ] half
    const int gSize,                  // G 64
    const float scale)                // 1024/sqrt(headDim)
{
    __local_mem__ float* weight_ = (__local_mem__ float*)weight.GetPhyAddr();
    __local_mem__ half* kScale_ = (__local_mem__ half*)kScale.GetPhyAddr();

    constexpr uint32_t VL = 64; // vector length

    auto qk0 = (__local_mem__ int32_t*)qk.GetPhyAddr();
    auto qk1 = qk0 + VL;
    auto out0 = (__local_mem__ KEY_T*)out.GetPhyAddr();
    auto out1 = out0 + VL;
    auto kScale0 = kScale_;
    auto kScale1 = kScale_ + VL;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> regQK[2];
        AscendC::Reg::RegTensor<half> regQKHalf[2];
        AscendC::Reg::RegTensor<int32_t> regQKInt[2];
        AscendC::Reg::RegTensor<float> regwBrc;
        AscendC::Reg::RegTensor<half> regKScaleHalf[2];
        AscendC::Reg::RegTensor<float> regKScale[2];
        AscendC::Reg::RegTensor<float> regSum[2];

        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();

        FloatSortConstCtx<float> fp32Ctx;
        InitFloatSortConstCtx(fp32Ctx, maskAll);
        (void)scale; // 排序键域不含 ×(1024/√d), 参数保留兼容调用点

        constexpr static Reg::CastTrait castTraitInt32ToFP32 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                                                Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        constexpr static Reg::CastTrait castTraitFP16ToFP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        AscendC::Reg::Duplicate(regSum[0], 0.0f, maskAll);
        AscendC::Reg::Duplicate(regSum[1], 0.0f, maskAll);

        // k_descale(half) -> fp32(v2 LoadKScaleFP16 同款)
        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regKScaleHalf[0], kScale0);
        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regKScaleHalf[1], kScale1);
        AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regKScale[0], regKScaleHalf[0], maskAllB16);
        AscendC::Reg::Cast<float, half, castTraitFP16ToFP32>(regKScale[1], regKScaleHalf[1], maskAllB16);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); ++i) {
            AscendC::Reg::LoadAlign<int32_t>(regQKInt[0], qk0 + 128 * i);
            AscendC::Reg::LoadAlign<int32_t>(regQKInt[1], qk1 + 128 * i);
            AscendC::Reg::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[0], regQKInt[0], maskAll);
            AscendC::Reg::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[1], regQKInt[1], maskAll);

            CastFP32ToFP16ToFP32(regQK, regQKHalf, maskAll);
            AscendC::Reg::Relu(regQK[0], regQK[0], maskAll);
            AscendC::Reg::Relu(regQK[1], regQK[1], maskAll);

            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(regwBrc, weight_ + i);
            AscendC::Reg::MulAddDst(regSum[0], regQK[0], regwBrc, maskAll);
            AscendC::Reg::MulAddDst(regSum[1], regQK[1], regwBrc, maskAll);
        }

        // kScale(dk) 乘入聚合和(逐 s2 元素)。
        // [键宽契约] 量化模式排序键域 = Σ_g wq·relu(dot/1024)·dk, 不含 ×qkScale(1024/√d)
        // 物理缩放; 物理分数由 values 输出侧统一补乘, 排序域与输出域分离。
        AscendC::Reg::Mul(regSum[0], regSum[0], regKScale[0], maskAll);
        AscendC::Reg::Mul(regSum[1], regSum[1], regKScale[1], maskAll);

        // 非有限分数消毒(仅 NaN, PR#15559 意见2): dot 溢出 ±inf 再乘 0 可成 NaN
        // (键位型 0xFFFF 抢占 TopK), 须置 -inf 沉底; 判据收紧为幅值位型 > +inf 位型
        // (即指数域全 1 且尾数≠0, 符号位已掩除) —— ±inf 是契约正确值(如
        // wq=256×256=65536 fp16 NO_SAT 上溢的有效池分数)须保留自然排序位,
        // 垃圾 lane 治理走 lane 有效性而非值判别。
        {
            auto& s0u = (AscendC::Reg::RegTensor<uint32_t>&)regSum[0];
            auto& s1u = (AscendC::Reg::RegTensor<uint32_t>&)regSum[1];
            AscendC::Reg::RegTensor<uint32_t> regMagMask, regMag0, regMag1, regInfBits, regNegInf;
            AscendC::Reg::MaskReg regSel0, regSel1;
            AscendC::Reg::MaskReg maskAll32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::Duplicate(regMagMask, 0x7FFFFFFFu, maskAll32); // 幅值位(去符号)
            AscendC::Reg::Duplicate(regInfBits, 0x7F800000u, maskAll32); // +inf 位型
            AscendC::Reg::Duplicate(regNegInf, 0xFF800000u, maskAll32);  // -inf
            AscendC::Reg::And(regMag0, s0u, regMagMask, maskAll32);
            AscendC::Reg::And(regMag1, s1u, regMagMask, maskAll32);
            // 幅值 > +inf 位型 <=> 指数全 1 且尾数≠0 <=> NaN; ±inf(尾数 0)不命中
            AscendC::Reg::Compare<uint32_t, CMPMODE::GT>(regSel0, regMag0, regInfBits, maskAll32);
            AscendC::Reg::Compare<uint32_t, CMPMODE::GT>(regSel1, regMag1, regInfBits, maskAll32);
            AscendC::Reg::Select(s0u, regNegInf, s0u, regSel0);
            AscendC::Reg::Select(s1u, regNegInf, s1u, regSel1);
        }

        // [键宽契约] 排序键域不含 ×qkScale(1024/√d) 物理缩放(见上注), scale 不参与排序键
        if constexpr (sizeof(KEY_T) == 4) {
            AscendC::Reg::RegTensor<uint32_t> regOut[2];
            FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

            AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
            AscendC::Reg::StoreAlign<uint32_t, AscendC::Reg::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
        } else {
            // uint16 键: DeInterleave 预排 + 奇/偶 lane 压缩 cast -> bf16(RNE) 全宽可排序键
            // (quant_lightning_indexer_v2 同款配方: ODD 先写 ZEROING, EVEN 后写 MERGING 保留奇槽)
            (void)out1;
            constexpr static Reg::CastTrait traitF32ToBf16Even = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                  Reg::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
            constexpr static Reg::CastTrait traitF32ToBf16Odd = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
            AscendC::Reg::MaskReg maskAllB16 = AscendC::Reg::CreateMask<bfloat16_t, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::DeInterleave(regSum[0], regSum[1], regSum[0], regSum[1]);
            AscendC::Reg::RegTensor<bfloat16_t> regBf;
            AscendC::Reg::Cast<bfloat16_t, float, traitF32ToBf16Odd>(regBf, regSum[1], maskAll);
            AscendC::Reg::Cast<bfloat16_t, float, traitF32ToBf16Even>(regBf, regSum[0], maskAll);
            FloatSortConstCtx<bfloat16_t> bf16Ctx;
            InitFloatSortConstCtx(bf16Ctx, maskAllB16);
            AscendC::Reg::RegTensor<uint16_t> regKey;
            FloatToSortableKey<bfloat16_t>(regKey, regBf, bf16Ctx, maskAllB16);
            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(out0, regKey, maskAllB16);
        }
    }
}
} // namespace vector1

#endif
