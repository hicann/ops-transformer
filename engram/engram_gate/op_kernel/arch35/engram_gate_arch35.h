/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ENGRAM_GATE_ARCH35_H
#define ENGRAM_GATE_ARCH35_H
#include "kernel_operator.h"
using namespace AscendC;
namespace EngramGateOps {
using namespace AscendC::Reg;
constexpr int32_t VEC_LANES = 64;
constexpr int32_t ACCUMULATOR_COUNT = 3;
constexpr uint32_t STATS_BLOCK_FLOATS = 8;
constexpr int32_t DOT_STATS_INDEX = 0;
constexpr int32_t GATE_STATS_INDEX = 1;
constexpr int32_t RSTDX_STATS_INDEX = 2;
constexpr int32_t RSTDK_STATS_INDEX = 3;
constexpr int64_t ROWS_PER_GROUP = 16;
constexpr AscendC::Reg::CastTrait castTraitB162B32Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

// FP32 -> BF16: ZERO 布局 (偶半 lane), RNE 舍入 (CAST_RINT), 无饱和 (f322bf16 vcvt 要求 RS_DISABLE/RS_ENABLE, NO_SAT ->
// RS_DISABLE), 配合 DIST_PACK_B32 打包存储。支持矩阵见 Cast 文档表 7 (float -> bfloat16_t: ZERO/ONE, NO_SAT/SAT,
// ZEROING/MERGING, RINT 等)。
constexpr AscendC::Reg::CastTrait castTraitF322B16Rint = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

/*!
 * \brief 第一遍扫描一个分块 (直接 __simd_vf__ 调用): 累加 sum_x2 / sum_k2 / raw_dot。
 *
 * x/k 为 BF16 (解包升精度), w 为 FP32。D 和 Host 下发的 chunk 均按 64 元素对齐，
 * 因此每次 VF 迭代都使用完整 64 lane。
 * 累加器跨块保存在 UB acc 区 (3 x 64 float, lane0 有效, 高位为确定性 0),
 * firstChunk 时直接以本块和初始化。
 */
template <typename TIn, bool FIRST_CHUNK>
__simd_vf__ inline void EngramGatePass1ChunkVF(__ubuf__ TIn* xChunk, __ubuf__ TIn* kChunk, __ubuf__ float* wChunk,
                                               __ubuf__ float* accUb, int32_t chunkElems)
{
    const int32_t numFull = (chunkElems + VEC_LANES - 1) >> 6;
    __VEC_SCOPE__
    {
        MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
        MaskReg lane0Mask = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> xAcc;
        RegTensor<float> kAcc;
        RegTensor<float> dAcc;
        RegTensor<float> dError;
        Duplicate(xAcc, 0.0F, fullMask);
        Duplicate(kAcc, 0.0F, fullMask);
        Duplicate(dAcc, 0.0F, fullMask);
        Duplicate(dError, 0.0F, fullMask);

        RegTensor<TIn> xBf;
        RegTensor<TIn> kBf;
        RegTensor<float> w;
        RegTensor<float> x;
        RegTensor<float> k;
        RegTensor<float> t;
        for (uint16_t seg = 0; seg < static_cast<uint16_t>(numFull); ++seg) {
            const uint32_t off = static_cast<uint32_t>(seg) * VEC_LANES;
            LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xChunk + off);
            LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(kBf, kChunk + off);
            LoadAlign<float, LoadDist::DIST_NORM>(w, wChunk + off);
            Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
            Cast<float, TIn, castTraitB162B32Even>(k, kBf, fullMask);
            MulAddDst(xAcc, x, x, fullMask); // FMA: xAcc += x*x
            MulAddDst(kAcc, k, k, fullMask); // FMA: kAcc += k*k
            Mul(t, x, w, fullMask);          // t = x*w
            // Preserve the sign of a cancellation-heavy dot before signed sqrt/clamp.
            // FMA residuals recover both product roundings; TwoSum carries lane errors.
            RegTensor<float> productError, product, productLow, next, delta, aLow, bLow;
            Muls(productError, t, -1.0F, fullMask);
            MulAddDst(productError, x, w, fullMask);
            Mul(product, t, k, fullMask);
            Muls(productLow, product, -1.0F, fullMask);
            MulAddDst(productLow, t, k, fullMask);
            MulAddDst(productLow, productError, k, fullMask);
            Add(next, dAcc, product, fullMask);
            Sub(delta, next, dAcc, fullMask);
            Sub(aLow, next, delta, fullMask);
            Sub(aLow, dAcc, aLow, fullMask);
            Sub(bLow, product, delta, fullMask);
            Add(aLow, aLow, bLow, fullMask);
            Add(aLow, aLow, productLow, fullMask);
            Add(dError, dError, aLow, fullMask);
            dAcc = next;
        }
        // 规约到 lane0 并累加进 UB acc 区。
        MaskReg sumMask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> sumX;
        RegTensor<float> sumK;
        RegTensor<float> sumD;
        RegTensor<float> acc;
        ReduceSum(sumX, xAcc, sumMask);
        ReduceSum(sumK, kAcc, sumMask);
        ReduceSum(sumD, dAcc, sumMask);
        // Split mantissas before the tree reduction to reduce cancellation error.
        RegTensor<uint32_t> mantissaMask, highBits;
        Duplicate(mantissaMask, 0xFFFFF000U, sumMask);
        And(highBits, (RegTensor<uint32_t>&)dAcc, mantissaMask, sumMask);
        RegTensor<float> high = (RegTensor<float>&)highBits;
        RegTensor<float> low, highSum, lowSum, corrected, correctedAbs;
        Sub(low, dAcc, high, sumMask);
        Add(low, low, dError, sumMask);
        ReduceSum(highSum, high, sumMask);
        ReduceSum(lowSum, low, sumMask);
        Add(corrected, highSum, lowSum, lane0Mask);
        Abs(correctedAbs, corrected, lane0Mask);
        MaskReg correctionFinite;
        CompareScalar<float, CMPMODE::LT>(correctionFinite, correctedAbs, 3.402823466e38F, lane0Mask);
        RegTensor<float> zero;
        Duplicate(zero, 0.0F, lane0Mask);
        Select(highSum, highSum, sumD, correctionFinite);
        Select(lowSum, lowSum, zero, correctionFinite);
        if constexpr (!FIRST_CHUNK) {
            LoadAlign<float, LoadDist::DIST_NORM>(acc, accUb);
            Add(sumX, sumX, acc, lane0Mask);
            LoadAlign<float, LoadDist::DIST_NORM>(acc, accUb + VEC_LANES);
            Add(sumK, sumK, acc, lane0Mask);
            // Keep the dot high/low pair across chunks, without changing the UB budget.
            // The extra scalar occupies an aligned unused lane in the existing dot block.
            RegTensor<float> previousLow, nextHigh, delta, aLow, bLow;
            LoadAlign<float, LoadDist::DIST_NORM>(acc, accUb + (ACCUMULATOR_COUNT - 1) * VEC_LANES);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(previousLow,
                                                     accUb + (ACCUMULATOR_COUNT - 1) * VEC_LANES + STATS_BLOCK_FLOATS);
            Add(nextHigh, acc, highSum, lane0Mask);
            Sub(delta, nextHigh, acc, lane0Mask);
            Sub(aLow, nextHigh, delta, lane0Mask);
            Sub(aLow, acc, aLow, lane0Mask);
            Sub(bLow, highSum, delta, lane0Mask);
            Add(aLow, aLow, bLow, lane0Mask);
            Add(lowSum, lowSum, previousLow, lane0Mask);
            Add(lowSum, lowSum, aLow, lane0Mask);
            highSum = nextHigh;
        }
        // On nonfinite high values, residual arithmetic may be NaN; retain IEEE high.
        RegTensor<float> highAbs;
        Abs(highAbs, highSum, lane0Mask);
        MaskReg highFinite;
        CompareScalar<float, CMPMODE::LT>(highFinite, highAbs, 3.402823466e38F, lane0Mask);
        Select(lowSum, lowSum, zero, highFinite);
        StoreAlign<float, StoreDist::DIST_NORM>(accUb, sumX, fullMask);
        StoreAlign<float, StoreDist::DIST_NORM>(accUb + VEC_LANES, sumK, fullMask);
        StoreAlign<float, StoreDist::DIST_NORM>(accUb + (ACCUMULATOR_COUNT - 1) * VEC_LANES, highSum, fullMask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            accUb + (ACCUMULATOR_COUNT - 1) * VEC_LANES + STATS_BLOCK_FLOATS, lowSum, lane0Mask);
    }
}

/*!
 * \brief gate 数学 (直接 __simd_vf__ 调用, lane0): 由 3 个和计算统计量并写 UB 低区。
 *
 * stats = (raw_dot, gate, rstd_x, rstd_k), 每类占一个 32B 对齐块:
 *   rstd = 1/sqrt(sum*invDim + eps); raw_dot = sum(x*w*k);
 *   normalized = raw_dot * rstd_x * rstd_k * invSqrtDim;
 *   s = copysign(sqrt(max(|normalized|, clamp)), normalized); gate = 1/(1+exp(-s))。
 */
__simd_vf__ inline void EngramGateFinalizeVF(__ubuf__ float* accUb, __ubuf__ float* statsUb, float eps,
                                             float clampValue, float invDim, float invSqrtDim)
{
    __VEC_SCOPE__
    {
        MaskReg lane0Mask = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> sumX;
        RegTensor<float> sumK;
        RegTensor<float> sumD;
        LoadAlign<float, LoadDist::DIST_NORM>(sumX, accUb);
        LoadAlign<float, LoadDist::DIST_NORM>(sumK, accUb + VEC_LANES);
        LoadAlign<float, LoadDist::DIST_NORM>(sumD, accUb + (ACCUMULATOR_COUNT - 1) * VEC_LANES);
        RegTensor<float> dotLow;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dotLow,
                                                 accUb + (ACCUMULATOR_COUNT - 1) * VEC_LANES + STATS_BLOCK_FLOATS);
        Add(sumD, sumD, dotLow, lane0Mask);

        RegTensor<float> one;
        Duplicate(one, 1.0F, lane0Mask);
        RegTensor<float> rmsX;
        RegTensor<float> rmsK;
        Muls(rmsX, sumX, invDim, lane0Mask);
        Adds(rmsX, rmsX, eps, lane0Mask);
        Sqrt(rmsX, rmsX, lane0Mask);
        Muls(rmsK, sumK, invDim, lane0Mask);
        Adds(rmsK, rmsK, eps, lane0Mask);
        Sqrt(rmsK, rmsK, lane0Mask);
        RegTensor<float> rstdX;
        RegTensor<float> rstdK;
        Div(rstdX, one, rmsX, lane0Mask);
        Div(rstdK, one, rmsK, lane0Mask);

        RegTensor<float> dot;
        Mul(dot, sumD, rstdX, lane0Mask);
        Mul(dot, dot, rstdK, lane0Mask);
        Muls(dot, dot, invSqrtDim, lane0Mask);

        RegTensor<float> magnitude;
        Abs(magnitude, dot, lane0Mask);
        RegTensor<float> floor;
        Duplicate(floor, clampValue, lane0Mask);
        MaskReg belowFloor;
        CompareScalar<float, CMPMODE::LT>(belowFloor, magnitude, clampValue, lane0Mask);
        Select(magnitude, floor, magnitude, belowFloor);
        Sqrt(magnitude, magnitude, lane0Mask);

        // copysign: retain the sign bit, including negative zero and signed NaNs.
        RegTensor<uint32_t> signMask, signBits, rootBits;
        Duplicate(signMask, 0x80000000U, lane0Mask);
        And(signBits, (RegTensor<uint32_t>&)dot, signMask, lane0Mask);
        Or(rootBits, (RegTensor<uint32_t>&)magnitude, signBits, lane0Mask);
        RegTensor<float> signedRoot = (RegTensor<float>&)rootBits;

        // sigmoid 直接公式 (与 cannbot-dsl 一致): gate = 1/(1+exp(-s))。
        RegTensor<float> negRootExp;
        Muls(negRootExp, signedRoot, -1.0F, lane0Mask);
        RegTensor<float> expVal;
        Exp<float>(expVal, negRootExp, lane0Mask);
        RegTensor<float> denom;
        Add(denom, one, expVal, lane0Mask);
        RegTensor<float> gate;
        Div(gate, one, denom, lane0Mask);

        // 四个 lane0 标量写入 UB 低区；输出 dot 保存原始点积，gate 使用归一化点积。
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statsUb + DOT_STATS_INDEX * STATS_BLOCK_FLOATS, sumD,
                                                             lane0Mask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statsUb + GATE_STATS_INDEX * STATS_BLOCK_FLOATS, gate,
                                                             lane0Mask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statsUb + RSTDX_STATS_INDEX * STATS_BLOCK_FLOATS, rstdX,
                                                             lane0Mask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statsUb + RSTDK_STATS_INDEX * STATS_BLOCK_FLOATS, rstdK,
                                                             lane0Mask);
    }
}

template <typename T>
__simd_vf__ inline void EmitVF(__ubuf__ T* out, __ubuf__ T* xPtr, __ubuf__ T* vPtr, __ubuf__ float* stats,
                               uint32_t count)
{
    const uint32_t numFull = (count + VEC_LANES - 1) >> 6;
    __VEC_SCOPE__
    {
        MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> gate;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(gate, stats + STATS_BLOCK_FLOATS);
        for (uint16_t i = 0; i < static_cast<uint16_t>(numFull); ++i) {
            RegTensor<T> xb, vb, yb;
            RegTensor<float> x, v, y;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(xb, xPtr + i * VEC_LANES);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(vb, vPtr + i * VEC_LANES);
            Cast<float, T, castTraitB162B32Even>(x, xb, fullMask);
            Cast<float, T, castTraitB162B32Even>(v, vb, fullMask);
            Mul(y, v, gate, fullMask);
            Add(y, x, y, fullMask);
            Cast<T, float, castTraitF322B16Rint>(yb, y, fullMask);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + i * VEC_LANES, yb, fullMask);
        }
    }
}

} // namespace EngramGateOps
#endif
