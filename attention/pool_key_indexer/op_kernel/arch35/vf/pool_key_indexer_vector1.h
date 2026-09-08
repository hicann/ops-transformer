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

// PKI: float output version (no Cast needed, values output is FLOAT)
__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<float> &out_, const LocalTensor<uint32_t> &in,
                                              const uint32_t topK)
{
    auto outBuf = (__local_mem__ float *)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t *)in.GetPhyAddr();

    const uint16_t repeatSize32 = 128;
    uint16_t topkLoopNum = (topK + repeatSize32 - 1) / repeatSize32;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> regIn[2];
        AscendC::MicroAPI::RegTensor<float> regOut[2];
        AscendC::MicroAPI::MaskReg maskAllB32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB32f =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::MicroAPI::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::MicroAPI::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);
            UIntSortConstCtx<float> uint32Ctx;
            InitUIntSortConstCtx(uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            // float 版本入/出同宽(4B->4B), 不能沿用 bf16/fp16 版本的
            // DeInterleave 半宽合并模式: 同宽寄存器对上 DeInterleave 是
            // 2:1 元素抽取(丢弃奇数位元素), 且单次 StoreAlign 只写 64 lane,
            // 会造成 out[k]=in[2k] 且 [128i+64,128i+128) 残留脏数据。
            // 正确做法: 两个 64 lane 寄存器分别落回各自位置。
            AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(outBuf + i * repeatSize32,
                                                                                          regOut[0], maskAllB32f);
            AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                outBuf + i * repeatSize32 + 64, regOut[1], maskAllB32f);
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
                                             const int gSize,                  // G 64
                                             const float scale)                // 1/sqrt(headDim)
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
        AscendC::MicroAPI::RegTensor<float> regScale;
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

        // 池级分数乘 1/sqrt(headDim)(文档公式 S = Q@K^T/sqrt(headDim));
        // 缩放为正数, 与 ReLU/加权求和可交换, 在聚合后统一应用
        AscendC::MicroAPI::Duplicate(regScale, scale, maskAll);
        AscendC::MicroAPI::Mul(regSum[0], regSum[0], regScale, maskAll);
        AscendC::MicroAPI::Mul(regSum[1], regSum[1], regScale, maskAll);

        AscendC::MicroAPI::RegTensor<uint32_t> regOut[2];
        FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
    }
}

// mode=0 (FP8 per-token-head) scale 融合版: 数学依据 —— per-token-head scale
// 为正标量, 可从 D 维点积中完全提出:
//   out = (Σ_g relu(Σ_d q·k) · w_g · dq) · dk
// weight 参数为已预乘 qScale 的 fp32 UB(Gather 对寄存器 Mul 产物的 lane
// 布局不兼容, 参考 QLIv2: weight/qScale 均为 float UB, Mul 后走
// LoadAlign<float>+BroadcastLane/Gather 路径)。
//   qk:      [G, S2Base] float(Mmad fp32 累加结果)
//   weight:  [G]         float(已含 dq: w_g * dq_g)
//   kScale:  [S2Base]    float(每池 1 个 scale, 逐 s2 元素)
__aicore__ inline void MulWeightAndReduceSumWithScale(
    const LocalTensor<uint32_t> &out, // out    [S2Base]     [128   ] 2
    const LocalTensor<float> &qk,     // q*k^t  [G, S2Base]  [64 128] 2
    const LocalTensor<float> &weight, // w*dq   [G]          [64    ] 1
    const LocalTensor<float> &kScale, // kScale [S2Base]     [128   ] 2
    const int gSize,                  // G 64
    const float scale)                // 1/sqrt(headDim)
{
    __local_mem__ float *weight_ = (__local_mem__ float *)weight.GetPhyAddr();
    __local_mem__ float *kScale_ = (__local_mem__ float *)kScale.GetPhyAddr();

    constexpr uint32_t VL = 64; // vector length

    auto qk0 = (__local_mem__ float *)qk.GetPhyAddr();
    auto qk1 = qk0 + VL;
    auto out0 = (__local_mem__ uint32_t *)out.GetPhyAddr();
    auto out1 = out0 + VL;
    auto kScale0 = kScale_;
    auto kScale1 = kScale_ + VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> regQK[2];
        AscendC::MicroAPI::RegTensor<float> regwBrc;
        AscendC::MicroAPI::RegTensor<float> regKScale[2];
        AscendC::MicroAPI::RegTensor<float> regSum[2];
        AscendC::MicroAPI::RegTensor<float> regScale;

        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

        FloatSortConstCtx<float> fp32Ctx;
        InitFloatSortConstCtx(fp32Ctx, maskAll);

        AscendC::MicroAPI::Duplicate(regSum[0], 0.0f, maskAll);
        AscendC::MicroAPI::Duplicate(regSum[1], 0.0f, maskAll);

        AscendC::MicroAPI::LoadAlign<float>(regKScale[0], kScale0);
        AscendC::MicroAPI::LoadAlign<float>(regKScale[1], kScale1);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); ++i) {
            // BroadcastLane: 从 UB 直接广播第 i 个 weight(Gather 对 LoadAlign
            // fp32 产物的 lane 布局不兼容, 参考 QLIv2 的 BroadcastLane 模式)
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(regwBrc, weight_ + i);
            AscendC::MicroAPI::LoadAlign<float>(regQK[0], qk0 + 128 * i);
            AscendC::MicroAPI::LoadAlign<float>(regQK[1], qk1 + 128 * i);

            AscendC::MicroAPI::Relu(regQK[0], regQK[0], maskAll);
            AscendC::MicroAPI::Relu(regQK[1], regQK[1], maskAll);

            AscendC::MicroAPI::MulAddDst(regSum[0], regQK[0], regwBrc, maskAll);
            AscendC::MicroAPI::MulAddDst(regSum[1], regQK[1], regwBrc, maskAll);
        }

        // kScale 乘入聚合和(逐 s2 元素), 随后统一乘 1/sqrt(headDim)
        AscendC::MicroAPI::Mul(regSum[0], regSum[0], regKScale[0], maskAll);
        AscendC::MicroAPI::Mul(regSum[1], regSum[1], regKScale[1], maskAll);

        // 池级分数乘 1/sqrt(headDim)(文档公式 S = Q@K^T/sqrt(headDim));
        // 缩放为正数, 与 ReLU/加权求和可交换, 在聚合后统一应用
        AscendC::MicroAPI::Duplicate(regScale, scale, maskAll);
        AscendC::MicroAPI::Mul(regSum[0], regSum[0], regScale, maskAll);
        AscendC::MicroAPI::Mul(regSum[1], regSum[1], regScale, maskAll);

        AscendC::MicroAPI::RegTensor<uint32_t> regOut[2];
        FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
    }
}
// mode=0 (FP8 per-token-head) weight×qScale SIMD 预乘(单 s1 行, 64 lanes)。
// 半精度 weight(half/bfloat16_t) -> fp32 cast -> 乘 qScale -> 落 fp32 UB,
// 参考 QLIv2 weightTemp 模式(LoadAlign+Cast+Mul+StoreAlign, 4 条向量指令),
// 替代逐 g 标量预乘循环(gSize 次标量/行; W_T=bfloat16_t 时还触发
// bisheng bf16 标量语义限制)。独立 __simd_vf__ 函数保证 bf16 Cast
// 的硬件展开(内联到普通函数会报 "Do not know how to split this operator")。
//   weight_:   [G]  半精度(行内 gSizeAlign16 布局, [gSize,64) 已补 0)
//   qScale_:   [G]  fp32(同布局)
//   dst_:      [G]  fp32 输出(行距 128 元素 bank 对齐)
template <typename W_T>
__simd_vf__ void MulWeightQScaleSIMD(__local_mem__ W_T *weight_, __local_mem__ float *qScale_,
                                     __local_mem__ float *dst_)
{
    MicroAPI::MaskReg maskAllB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = MicroAPI::CreateMask<W_T, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<W_T> regWHalf;
    MicroAPI::RegTensor<float> regW;
    MicroAPI::RegTensor<float> regQScale;

    constexpr static MicroAPI::CastTrait castTraitWTToFP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    MicroAPI::LoadAlign<W_T, MicroAPI::LoadDist::DIST_UNPACK_B16>(regWHalf, weight_);
    MicroAPI::Cast<float, W_T, castTraitWTToFP32>(regW, regWHalf, maskAllB16);
    MicroAPI::LoadAlign<float>(regQScale, qScale_);
    MicroAPI::Mul(regW, regW, regQScale, maskAllB32);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(dst_, regW, maskAllB32);
}
} // namespace vector1

#endif
