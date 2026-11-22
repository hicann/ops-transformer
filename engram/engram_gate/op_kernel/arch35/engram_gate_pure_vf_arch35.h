/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Derived from reference PR 13974 EngramGatePureFullRowVF.
// Keep fused register reduction/gate/emit; save raw dot for backward; IEEE copysign follows the final requirement.
// Both save modes use the same positive/negative dot reduction. SAVE controls
// only whether the four backward statistics are stored.
#ifndef ENGRAM_GATE_PURE_VF_ARCH35_H
#define ENGRAM_GATE_PURE_VF_ARCH35_H
#include "engram_gate_arch35.h"
namespace EngramGateOps {
// Preserve both multiplication roundings and the addition rounding for one
// vector lane. The accumulator and its low-part channel are updated together;
// callers provide independent channels to hide the loop-carried dependency.
__simd_callee__ inline void AccumulateCompensatedDot(RegTensor<float>& acc, RegTensor<float>& error,
                                                     RegTensor<float>& x, RegTensor<float>& w, RegTensor<float>& k,
                                                     RegTensor<float>& t, RegTensor<float>& productError,
                                                     RegTensor<float>& product, RegTensor<float>& productLow,
                                                     RegTensor<float>& next, RegTensor<float>& delta,
                                                     RegTensor<float>& aLow, MaskReg& mask)
{
    // Residual of t = x * w.
    Muls(productError, t, -1.0F, mask);
    MulAddDst(productError, x, w, mask);
    // Residual of product = t * k, including the error propagated from t.
    Mul(product, t, k, mask);
    Muls(productLow, product, -1.0F, mask);
    MulAddDst(productLow, t, k, mask);
    MulAddDst(productLow, productError, k, mask);

    // TwoSum(acc, product): the accumulator carries the rounded sum while the
    // low-part channel carries both the addition residual and product residual.
    Add(next, acc, product, mask);
    Sub(delta, next, acc, mask);
    Sub(aLow, next, delta, mask);
    Sub(aLow, acc, aLow, mask);
    // Reuse product as bLow after the rounded product has been consumed.
    Sub(product, product, delta, mask);
    Add(aLow, aLow, product, mask);
    Add(aLow, aLow, productLow, mask);
    Add(error, error, aLow, mask);
    acc = next;
}

// TwoSum merge of two compensated channels: acc/error absorbs otherAcc/
// otherError along with the rounding residual of the high-part addition, so
// cross-channel cancellation cannot drop low-order terms before the final
// split reduction.
__simd_callee__ inline void MergeCompensatedChannel(RegTensor<float>& acc, RegTensor<float>& error,
                                                    RegTensor<float>& otherAcc, RegTensor<float>& otherError,
                                                    RegTensor<float>& next, RegTensor<float>& delta,
                                                    RegTensor<float>& low, MaskReg& mask)
{
    Add(next, acc, otherAcc, mask);
    Sub(delta, next, acc, mask);
    Sub(low, next, delta, mask);
    Sub(low, acc, low, mask);
    Sub(otherAcc, otherAcc, delta, mask);
    Add(low, low, otherAcc, mask);
    Add(error, error, otherError, mask);
    Add(error, error, low, mask);
    acc = next;
}

template <typename TIn, typename TOut, int32_t HEADS, bool SAVE>
__simd_vf__ inline void EngramGatePureFullRowVF(__ubuf__ TOut* outUb, __ubuf__ TIn* xUb, __ubuf__ TIn* kUb,
                                                __ubuf__ TIn* vUb, __ubuf__ float* wUb, __ubuf__ float* dotUb,
                                                __ubuf__ float* gateUb, __ubuf__ float* rstdXUb,
                                                __ubuf__ float* rstdKUb, int32_t dim, int32_t /*numFullHint*/,
                                                float eps, float clampValue, float invDim, float invSqrtDim)
{
    constexpr uint16_t heads = static_cast<uint16_t>(HEADS);
    const uint16_t numFull = static_cast<uint16_t>(dim / VEC_LANES);
    uint32_t VL = static_cast<uint32_t>(VEC_LANES);
    uint32_t stride = static_cast<uint32_t>(dim);

    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg lane0Mask = CreateMask<float, MaskPattern::VL1>();

    RegTensor<float> xAccA;
    RegTensor<float> xAccB;
    RegTensor<float> kAccA;
    RegTensor<float> kAccB;
    RegTensor<float> dAccA;
    RegTensor<float> dAccB;
    RegTensor<float> dAccC;
    RegTensor<float> dAccD;
    RegTensor<float> dErrA;
    RegTensor<float> dErrB;
    RegTensor<float> dErrC;
    RegTensor<float> dErrD;
    RegTensor<float> x;
    RegTensor<float> w;
    RegTensor<float> k;
    RegTensor<float> t;
    RegTensor<float> dotProductError;
    RegTensor<float> dotProduct;
    RegTensor<float> dotProductLow;
    RegTensor<float> dotNext;
    RegTensor<float> dotDelta;
    RegTensor<float> dotLow;
    RegTensor<float> zero;
    RegTensor<float> sumX;
    RegTensor<float> sumK;
    RegTensor<float> sumD;
    RegTensor<float> dot;
    RegTensor<float> unitValue;
    RegTensor<float> floor;
    RegTensor<float> magnitude;
    RegTensor<float> signedRoot;
    RegTensor<float> gate;
    RegTensor<float> gateB;
    RegTensor<float> v;
    RegTensor<float> out;

    RegTensor<TIn> xBf;
    RegTensor<TIn> kBf;
    RegTensor<TIn> vBf;
    RegTensor<TOut> outBf;

    RegTensor<uint32_t> signMask;

    Duplicate(unitValue, 1.0F, lane0Mask);
    Duplicate(floor, clampValue, lane0Mask);
    Duplicate(zero, 0.0F, fullMask);

    __VEC_SCOPE__
    {
        for (uint16_t h = 0; h < heads; ++h) {
            Duplicate(xAccA, 0.0F, fullMask);
            Duplicate(xAccB, 0.0F, fullMask);
            Duplicate(kAccA, 0.0F, fullMask);
            Duplicate(kAccB, 0.0F, fullMask);
            Duplicate(dAccA, 0.0F, fullMask);
            Duplicate(dAccB, 0.0F, fullMask);
            Duplicate(dAccC, 0.0F, fullMask);
            Duplicate(dAccD, 0.0F, fullMask);
            Duplicate(dErrA, 0.0F, fullMask);
            Duplicate(dErrB, 0.0F, fullMask);
            Duplicate(dErrC, 0.0F, fullMask);
            Duplicate(dErrD, 0.0F, fullMask);
            Duplicate(signMask, 0x80000000U, lane0Mask);
            const uint16_t numVec8 = static_cast<uint16_t>(numFull & 0xFFF8U);
            const uint32_t headBase = h * stride;
            for (uint16_t seg = 0; seg < numVec8; seg += 8U) {
                const uint32_t off0 = headBase + static_cast<uint32_t>(seg) * VL;
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xUb + off0);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(kBf, kUb + off0);
                LoadAlign<float, LoadDist::DIST_NORM>(w, wUb + off0);
                Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
                Cast<float, TIn, castTraitB162B32Even>(k, kBf, fullMask);
                MulAddDst(xAccA, x, x, fullMask);
                MulAddDst(kAccA, k, k, fullMask);
                Mul(t, x, w, fullMask);
                AccumulateCompensatedDot(dAccA, dErrA, x, w, k, t, dotProductError, dotProduct, dotProductLow, dotNext,
                                         dotDelta, dotLow, fullMask);

                const uint32_t off1 = headBase + static_cast<uint32_t>(seg + 1U) * VL;
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xUb + off1);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(kBf, kUb + off1);
                LoadAlign<float, LoadDist::DIST_NORM>(w, wUb + off1);
                Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
                Cast<float, TIn, castTraitB162B32Even>(k, kBf, fullMask);
                MulAddDst(xAccB, x, x, fullMask);
                MulAddDst(kAccB, k, k, fullMask);
                Mul(t, x, w, fullMask);
                AccumulateCompensatedDot(dAccB, dErrB, x, w, k, t, dotProductError, dotProduct, dotProductLow, dotNext,
                                         dotDelta, dotLow, fullMask);

                const uint32_t off2 = headBase + static_cast<uint32_t>(seg + 2U) * VL;
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xUb + off2);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(kBf, kUb + off2);
                LoadAlign<float, LoadDist::DIST_NORM>(w, wUb + off2);
                Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
                Cast<float, TIn, castTraitB162B32Even>(k, kBf, fullMask);
                MulAddDst(xAccA, x, x, fullMask);
                MulAddDst(kAccA, k, k, fullMask);
                Mul(t, x, w, fullMask);
                AccumulateCompensatedDot(dAccC, dErrC, x, w, k, t, dotProductError, dotProduct, dotProductLow, dotNext,
                                         dotDelta, dotLow, fullMask);

                const uint32_t off3 = headBase + static_cast<uint32_t>(seg + 3U) * VL;
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xUb + off3);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(kBf, kUb + off3);
                LoadAlign<float, LoadDist::DIST_NORM>(w, wUb + off3);
                Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
                Cast<float, TIn, castTraitB162B32Even>(k, kBf, fullMask);
                MulAddDst(xAccB, x, x, fullMask);
                MulAddDst(kAccB, k, k, fullMask);
                Mul(t, x, w, fullMask);
                AccumulateCompensatedDot(dAccD, dErrD, x, w, k, t, dotProductError, dotProduct, dotProductLow, dotNext,
                                         dotDelta, dotLow, fullMask);

                const uint32_t off4 = headBase + static_cast<uint32_t>(seg + 4U) * VL;
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xUb + off4);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(kBf, kUb + off4);
                LoadAlign<float, LoadDist::DIST_NORM>(w, wUb + off4);
                Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
                Cast<float, TIn, castTraitB162B32Even>(k, kBf, fullMask);
                MulAddDst(xAccA, x, x, fullMask);
                MulAddDst(kAccA, k, k, fullMask);
                Mul(t, x, w, fullMask);
                AccumulateCompensatedDot(dAccA, dErrA, x, w, k, t, dotProductError, dotProduct, dotProductLow, dotNext,
                                         dotDelta, dotLow, fullMask);

                const uint32_t off5 = headBase + static_cast<uint32_t>(seg + 5U) * VL;
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xUb + off5);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(kBf, kUb + off5);
                LoadAlign<float, LoadDist::DIST_NORM>(w, wUb + off5);
                Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
                Cast<float, TIn, castTraitB162B32Even>(k, kBf, fullMask);
                MulAddDst(xAccB, x, x, fullMask);
                MulAddDst(kAccB, k, k, fullMask);
                Mul(t, x, w, fullMask);
                AccumulateCompensatedDot(dAccB, dErrB, x, w, k, t, dotProductError, dotProduct, dotProductLow, dotNext,
                                         dotDelta, dotLow, fullMask);

                const uint32_t off6 = headBase + static_cast<uint32_t>(seg + 6U) * VL;
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xUb + off6);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(kBf, kUb + off6);
                LoadAlign<float, LoadDist::DIST_NORM>(w, wUb + off6);
                Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
                Cast<float, TIn, castTraitB162B32Even>(k, kBf, fullMask);
                MulAddDst(xAccA, x, x, fullMask);
                MulAddDst(kAccA, k, k, fullMask);
                Mul(t, x, w, fullMask);
                AccumulateCompensatedDot(dAccC, dErrC, x, w, k, t, dotProductError, dotProduct, dotProductLow, dotNext,
                                         dotDelta, dotLow, fullMask);

                const uint32_t off7 = headBase + static_cast<uint32_t>(seg + 7U) * VL;
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xUb + off7);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(kBf, kUb + off7);
                LoadAlign<float, LoadDist::DIST_NORM>(w, wUb + off7);
                Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
                Cast<float, TIn, castTraitB162B32Even>(k, kBf, fullMask);
                MulAddDst(xAccB, x, x, fullMask);
                MulAddDst(kAccB, k, k, fullMask);
                Mul(t, x, w, fullMask);
                AccumulateCompensatedDot(dAccD, dErrD, x, w, k, t, dotProductError, dotProduct, dotProductLow, dotNext,
                                         dotDelta, dotLow, fullMask);
            }

            for (uint16_t seg = numVec8; seg < numFull; ++seg) {
                const uint32_t off = headBase + static_cast<uint32_t>(seg) * VL;
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xUb + off);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(kBf, kUb + off);
                LoadAlign<float, LoadDist::DIST_NORM>(w, wUb + off);
                Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
                Cast<float, TIn, castTraitB162B32Even>(k, kBf, fullMask);
                MulAddDst(xAccA, x, x, fullMask);
                MulAddDst(kAccA, k, k, fullMask);
                Mul(t, x, w, fullMask);
                AccumulateCompensatedDot(dAccA, dErrA, x, w, k, t, dotProductError, dotProduct, dotProductLow, dotNext,
                                         dotDelta, dotLow, fullMask);
            }

            Add(xAccA, xAccA, xAccB, fullMask);
            Add(kAccA, kAccA, kAccB, fullMask);
            ReduceSum(sumX, xAccA, fullMask);
            ReduceSum(sumK, kAccA, fullMask);
            RegTensor<uint32_t> mantissaMask;
            RegTensor<uint32_t> dotHighBits;
            Duplicate(mantissaMask, 0xFFFFF000U, fullMask);

            // Merge the four compensated accumulators and their low parts as
            // binary trees. Each high-part merge keeps its rounding residual in
            // the error channel, so cross-channel cancellation cannot drop
            // low-order terms before the final split reduction.
            MergeCompensatedChannel(dAccA, dErrA, dAccB, dErrB, dotNext, dotDelta, dotLow, fullMask);
            MergeCompensatedChannel(dAccC, dErrC, dAccD, dErrD, dotNext, dotDelta, dotLow, fullMask);
            MergeCompensatedChannel(dAccA, dErrA, dAccC, dErrC, dotNext, dotDelta, dotLow, fullMask);

            RegTensor<float> low = dAccA;
            And(dotHighBits, (RegTensor<uint32_t>&)low, mantissaMask, fullMask);
            dAccA = (RegTensor<float>&)dotHighBits;
            Sub(low, low, dAccA, fullMask);
            Add(low, low, dErrA, fullMask);
            // Inf - Inf has no finite low part. Keep the high Inf (or NaN)
            // and discard only the NaN introduced by the low-part subtraction.
            MaskReg validLow;
            Compare<float, CMPMODE::EQ>(validLow, low, low, fullMask);
            Select(low, low, zero, validLow);
            RegTensor<float> highSum;
            RegTensor<float> lowSum;
            ReduceSum(highSum, dAccA, fullMask);
            ReduceSum(lowSum, low, fullMask);
            Add(sumD, highSum, lowSum, lane0Mask);

            // ---- finalize (lane0, 与 EngramGateFinalizeVF 同构, 寄存器直通) ----
            Muls(sumX, sumX, invDim, lane0Mask);
            Muls(sumK, sumK, invDim, lane0Mask);
            Adds(sumX, sumX, eps, lane0Mask);
            Adds(sumK, sumK, eps, lane0Mask);
            Sqrt(sumX, sumX, lane0Mask);
            Sqrt(sumK, sumK, lane0Mask);
            Div(sumX, unitValue, sumX, lane0Mask);
            Div(sumK, unitValue, sumK, lane0Mask);

            Mul(dot, sumD, sumX, lane0Mask);
            Mul(dot, dot, sumK, lane0Mask);
            Muls(dot, dot, invSqrtDim, lane0Mask);

            Abs(magnitude, dot, lane0Mask);
            Max(magnitude, magnitude, floor, lane0Mask);
            Sqrt(magnitude, magnitude, lane0Mask);

            // copysign: retain the sign bit, including negative zero and signed NaNs.
            And(signMask, (RegTensor<uint32_t>&)dot, signMask, lane0Mask);
            Or(signMask, (RegTensor<uint32_t>&)magnitude, signMask, lane0Mask);
            signedRoot = (RegTensor<float>&)signMask;

            Muls(signedRoot, signedRoot, -1.0F, lane0Mask);
            Exp<float>(signedRoot, signedRoot, lane0Mask);
            Add(signedRoot, unitValue, signedRoot, lane0Mask);
            Div(gate, unitValue, signedRoot, lane0Mask);

            // ---- stats store (每类统计量独立窗口) ----
            if constexpr (SAVE) {
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dotUb + h, sumD, lane0Mask);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(gateUb + h, gate, lane0Mask);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdXUb + h, sumX, lane0Mask);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdKUb + h, sumK, lane0Mask);
            }

            // ---- gate 寄存器广播 (lane0 -> 全 lane) ----
            Duplicate(gateB, gate, fullMask);

            // ---- emit: out = bf16_rne(x + gate * v) ----
            for (uint16_t seg = 0; seg < numFull; ++seg) {
                AscendC::Reg::AddrReg offAddr = AscendC::Reg::CreateAddrReg<TIn>(h, stride, seg, VL);
                AscendC::Reg::AddrReg vOffAddr = AscendC::Reg::CreateAddrReg<TIn>(seg, VL);
                AscendC::Reg::AddrReg outAddr = AscendC::Reg::CreateAddrReg<TOut>(h, stride, seg, VL);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(xBf, xUb, offAddr);
                LoadAlign<TIn, LoadDist::DIST_UNPACK_B16>(vBf, vUb, vOffAddr);
                Cast<float, TIn, castTraitB162B32Even>(x, xBf, fullMask);
                Cast<float, TIn, castTraitB162B32Even>(v, vBf, fullMask);
                Mul(v, gateB, v, fullMask);
                Add(out, x, v, fullMask);
                Cast<TOut, float, castTraitF322B16Rint>(outBf, out, fullMask);
                StoreAlign<TOut, StoreDist::DIST_PACK_B32>(outUb, outBf, outAddr, fullMask);
            }
        }
    }
}

} // namespace EngramGateOps
#endif
