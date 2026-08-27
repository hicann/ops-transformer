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
 * \file indexer_quant_cache_base.h
 * \brief
 */

#ifndef INDEXER_QUANT_CACHE_BASE_H
#define INDEXER_QUANT_CACHE_BASE_H

#include "kernel_operator.h"

namespace IndexerQuantCache {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UnalignRegForStore;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t VL_FP32 = 64;
constexpr int32_t PER_BLOCK_FP16 = 128;
constexpr uint32_t REPEAT_SIZE = 256;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float FP8_E5M2_MIN_VALUE = -57344.0f;
constexpr float FP8_E4M3FN_MIN_VALUE = -448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (((uint32_t)1 << (uint32_t)23) - (uint32_t)1);
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;
constexpr uint32_t INV_HIFP8_MAX_VALUE = 0x38000000;
constexpr int64_t HIFLOAT_QUANT_MODE = 2;

#define FLOAT_OVERFLOW_MODE_CTRL 60
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float POS_INFINITY = INFINITY;
constexpr float NEG_INFINITY = -INFINITY;

__aicore__ inline int32_t CeilDiv(int32_t a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t PagedSlotOffset(int64_t slot, int64_t blockSize, int64_t blockStride, int64_t rowStride)
{
    int64_t block = slot / blockSize;
    int64_t pos = slot % blockSize;
    return block * blockStride + pos * rowStride;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num, int32_t elemNum)
{
    return CeilAlign(num, elemNum);
}

constexpr AscendC::Reg::CastTrait castTraitB162B32Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitB322B16Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::Reg::CastTrait castTraitF32toFp8Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::Reg::CastTrait castTraitU32toU8Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

constexpr AscendC::Reg::CastTrait traitB16ToB32Layout1 = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::Reg::CastTrait castTraitF32toh8 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                             AscendC::Reg::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::CAST_ROUND};

template <typename T>
__simd_callee__ inline void LoadInputData(RegTensor<float> &dst, __ubuf__ T *src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__simd_callee__ inline void StoreOutputData(__ubuf__ T *dst, RegTensor<float> &src, MaskReg pregLoop,
                                            uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign(dst + dstOffset, src, pregLoop);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, pregLoop);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitF32toFp8Even>(tmp, src, pregLoop);
        StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK4_B32>(dst + dstOffset, tmp, pregLoop);
    } else if constexpr (IsSameType<T, hifloat8_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitF32toh8>(tmp, src, pregLoop);
        StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK4_B32>(dst + dstOffset, tmp, pregLoop);
    }
}

template <typename T>
__simd_callee__ inline void StoreMxFp8Scale(__ubuf__ T *dst, RegTensor<float> &src, MaskReg pregLoop,
                                            uint32_t dstOffset)
{
    RegTensor<uint32_t> tmp;
    RegTensor<uint8_t> tmp1;
    ShiftRights(tmp, (RegTensor<uint32_t> &)src, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregLoop);
    Cast<uint8_t, uint32_t, castTraitU32toU8Even>(tmp1, tmp, pregLoop);
    StoreAlign<T, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B8>(dst + dstOffset, (RegTensor<T> &)tmp1, pregLoop);
}

template <typename TX, typename TS>
__simd_vf__ inline void VFProcessMxFp8InvScaleVF(__ubuf__ float *yLocalAddr, __ubuf__ uint8_t *scaleOriginAddr,
                                                 __ubuf__ float *invScaleLocalAddr, __ubuf__ TX *xLocalAddr,
                                                 float coeff, const uint16_t curRowNum, const uint32_t curColNum,
                                                 const uint32_t MXB, const uint16_t vlLen, const uint16_t loopCount,
                                                 const uint32_t xRowAlign, const uint32_t curColNumAlignFloat,
                                                 const uint32_t scaleColNumAlign, const uint32_t invScaleColNumAlign)
{
    {
        RegTensor<TX> raw;
        RegTensor<float> x0Layout0;
        RegTensor<float> x0Layout1;
        RegTensor<float> y0;
        RegTensor<float> y1;
        RegTensor<float> yMax0;
        RegTensor<float> yMax1;
        RegTensor<float> yMax1Layout0;
        RegTensor<float> yMax1Layout1;
        RegTensor<float> scale;
        RegTensor<float> clampScale;
        RegTensor<float> invScale;
        RegTensor<float> invScale0;
        RegTensor<float> invScale1;
        RegTensor<uint32_t> scale2;
        RegTensor<uint32_t> scale3;
        RegTensor<uint32_t> scaleSel;
        RegTensor<int32_t> scale4;
        RegTensor<int32_t> scale5;
        RegTensor<uint8_t> scale6;
        RegTensor<uint16_t> scale7;
        RegTensor<uint32_t> zeroUint32;
        RegTensor<uint32_t> oneUint32;
        RegTensor<uint32_t> cAndExp;
        RegTensor<uint32_t> cAndMan;
        RegTensor<uint32_t> c127;
        RegTensor<int32_t> c127Int;
        RegTensor<int32_t> c255Int;
        RegTensor<int32_t> fp32Nan;
        RegTensor<uint8_t> tmp5;
        UnalignRegForStore uReg;
        MaskReg pregMain0 = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregMain1 = CreateMask<TX, AscendC::Reg::MaskPattern::ALL>();
        MaskReg compareMask0;
        MaskReg specialDataMask;
        uint32_t sregMerge = static_cast<uint32_t>(vlLen) / MXB;         // 4
        uint32_t sregMerge1 = 4U * (static_cast<uint32_t>(vlLen) / MXB); // 16
        MaskReg pregMerge = UpdateMask<float>(sregMerge);
        MaskReg pregMerge1 = UpdateMask<float>(sregMerge1);
        Duplicate(zeroUint32, static_cast<uint32_t>(0), pregMain0);
        Duplicate(oneUint32, static_cast<uint32_t>(1), pregMain0);
        Duplicate(cAndExp, FAST_LOG_AND_VALUE1, pregMain0);
        Duplicate(cAndMan, FAST_LOG_AND_VALUE2, pregMain0);
        Duplicate(c127, static_cast<uint32_t>(127), pregMain0);
        Duplicate(c127Int, static_cast<int32_t>(127), pregMain0);
        Duplicate(c255Int, static_cast<int32_t>(255), pregMain0);
        Duplicate(fp32Nan, static_cast<int32_t>(0x7FC00000), pregMain0);
        for (uint16_t i = 0; i < curRowNum; i++) {
            __ubuf__ uint8_t *scaleRowAddr = scaleOriginAddr + i * scaleColNumAlign; // 本行 32B 对齐基址
            uint32_t eCnt = curColNum; // y 存储尾块的 running counter (UpdateMask 饱和到 0)
            for (uint16_t j = 0; j < loopCount; j++) {
                // 载入 128 个 B16, 拆偶/拆奇 -> 两路 fp32, Interleave 还原自然顺序写 fp32 中间 buffer
                LoadAlign(raw, xLocalAddr + i * xRowAlign + j * vlLen);
                Cast<float, TX, castTraitB162B32Even>(x0Layout0, raw, pregMain1); // even
                Cast<float, TX, traitB16ToB32Layout1>(x0Layout1, raw, pregMain1); // odd
                Interleave(y0, y1, x0Layout0, x0Layout1);
                MaskReg my0 = UpdateMask<float>(eCnt); // [0,64) 尾块自适应
                MaskReg my1 = UpdateMask<float>(eCnt); // [64,128) 尾块自适应 (尾块自动 0 mask, 免 if)
                StoreOutputData<float>(yLocalAddr, y0, my0, 2 * j * VL_FP32 + i * curColNumAlignFloat);
                StoreOutputData<float>(yLocalAddr, y1, my1, (2 * j + 1) * VL_FP32 + i * curColNumAlignFloat);
                // per-32-block amax
                Abs(x0Layout0, x0Layout0, pregMain0);
                Abs(x0Layout1, x0Layout1, pregMain0);
                Max(yMax0, x0Layout0, x0Layout1, pregMain0);
                ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(yMax1, yMax0, pregMain0);
                DeInterleave(yMax1Layout0, yMax1Layout1, yMax1, yMax1);
                Max(scale, yMax1Layout0, yMax1Layout1, pregMerge);
                Maxs(clampScale, scale, static_cast<float>(1e-4), pregMerge); // amax
                Muls(scale, clampScale, coeff, pregMerge);                    // sf = amax / fp8max
                // round 成 e8m0 = exp + (mantissa != 0); 存储 scale = 2^(exp_scale); invScale = 2^(-exp_scale)
                ShiftRights(scale2, (RegTensor<uint32_t> &)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                And(scale2, scale2, cAndExp, pregMerge);                                              // exp
                And(scale3, (RegTensor<uint32_t> &)scale, cAndMan, pregMerge);                        // man_bits
                Compare<uint32_t, AscendC::CMPMODE::EQ>(specialDataMask, scale2, cAndExp, pregMerge); // INF/NAN
                Compare<uint32_t, AscendC::CMPMODE::NE>(compareMask0, scale3, zeroUint32, pregMerge);
                Select(scaleSel, oneUint32, zeroUint32, compareMask0);                                // man_bits != 0
                Sub(scale3, scale2, c127, pregMerge);                                                 // exp - 127
                Add((RegTensor<uint32_t> &)scale4, scale3, scaleSel, pregMerge);                      // exp_scale
                Adds(scale5, scale4, 127, pregMerge);                                                 // exp_scale + 127
                Select<int32_t>(scale5, c255Int, scale5, specialDataMask);                            // E8M0 NAN
                ShiftLefts((RegTensor<int32_t> &)scale, scale5, static_cast<int16_t>(23), pregMerge); // 存储 scale
                Sub(scale5, c127Int, scale4, pregMerge);                                              // 127 - exp_scale
                // 2^(-exp_scale)
                ShiftLefts((RegTensor<int32_t> &)invScale, scale5, static_cast<int16_t>(23), pregMerge);
                Select<int32_t>((RegTensor<int32_t> &)invScale, fp32Nan, (RegTensor<int32_t> &)invScale,
                                specialDataMask);
                // e8m0 字节存储: 每 128-chunk 固定 4 字节 (越界块入本行 32B scratch padding, CopyOut 只取 scaleCol)
                ShiftRights(scale2, (RegTensor<uint32_t> &)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                Cast<uint8_t, int32_t, castTraitU32toU8Even>(tmp5, (RegTensor<int32_t> &)scale2, pregMerge);
                Pack(scale7, (RegTensor<uint32_t> &)tmp5);
                Pack(scale6, scale7);
                StoreUnAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(scaleRowAddr, scale6, uReg,
                                                                                   MXB / 8); // 4 bytes
                // invScale 广播 (每 scale x4): 固定 16 lane (越界块被 Quant 掩码)
                Interleave(invScale0, invScale1, invScale, invScale);
                Interleave(invScale1, invScale, invScale0, invScale0);
                StoreOutputData<float>(invScaleLocalAddr, invScale1, pregMerge1, j * 16 + i * invScaleColNumAlign);
            }
            StoreUnAlignPost(scaleRowAddr, uReg, 0);
        }
    }
}

template <typename TX, typename TS>
__aicore__ inline void VFProcessMxFp8InvScale(const LocalTensor<float> &yLocal, const LocalTensor<TS> &scaleLocal,
                                              const LocalTensor<float> &invScaleLocal, const LocalTensor<TX> &xLocal,
                                              float coeff, const uint16_t curRowNum, const uint32_t curColNum)
{
    __ubuf__ float *yLocalAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    __ubuf__ uint8_t *scaleOriginAddr = (__ubuf__ uint8_t *)scaleLocal.GetPhyAddr();
    __ubuf__ float *invScaleLocalAddr = (__ubuf__ float *)invScaleLocal.GetPhyAddr();
    __ubuf__ TX *xLocalAddr = (__ubuf__ TX *)xLocal.GetPhyAddr();
    constexpr uint32_t MXB = 32;
    uint16_t vlLen = REPEAT_SIZE / sizeof(TX); // 128 elements (B16) per chunk
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint32_t xRowAlign = RoundUp<TX>(curColNum, vlLen);       // x 输入 buffer 按 128 对齐
    uint32_t curColNumAlignFloat = RoundUp<float>(curColNum); // fp32 中间 buffer 每行跨步
    uint32_t scaleCol = CeilDiv(curColNum, MXB);
    uint32_t scaleColNumAlign = RoundUp<TS>(scaleCol);                    // e8m0 scale 每行 32B 对齐跨步
    uint32_t invScaleColNumAlign = static_cast<uint32_t>(loopCount) * 16; // 每 128-chunk 固定 16 个 invScale
    VFProcessMxFp8InvScaleVF<TX, TS>(yLocalAddr, scaleOriginAddr, invScaleLocalAddr, xLocalAddr, coeff, curRowNum,
                                     curColNum, MXB, vlLen, loopCount, xRowAlign, curColNumAlignFloat, scaleColNumAlign,
                                     invScaleColNumAlign);
}

template <typename TY>
__simd_vf__ inline void VFProcessMxFp8QuantVF(__ubuf__ TY *yQuantLocalAddr, __ubuf__ float *yLocalAddr,
                                              __ubuf__ float *scaleLocalAddr, const uint16_t curRowNum,
                                              const uint32_t curColNum, const uint16_t loopCount,
                                              const uint32_t curColNumAlign, const uint32_t dstCurColNumAlign,
                                              const uint32_t invScaleColNumAlign)
{
    {
        RegTensor<float> y;
        RegTensor<float> invScale;
        MaskReg pregMain;
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregMain = UpdateMask<float>(sreg); // 尾块按剩余元素数掩码
                LoadInputData<float>(y, yLocalAddr, pregMain, i * curColNumAlign + j * VL_FP32);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_E2B_B32>(
                    invScale, scaleLocalAddr + j * 8 + i * invScaleColNumAlign);
                Mul(y, y, invScale, pregMain);
                StoreOutputData<TY>(yQuantLocalAddr, y, pregMain, j * VL_FP32 + i * dstCurColNumAlign);
            }
        }
    }
}

template <typename TY>
__aicore__ inline void VFProcessMxFp8Quant(const LocalTensor<TY> &yQuantLocal, const LocalTensor<float> &yLocal,
                                           const LocalTensor<float> &invScaleLocal, const uint16_t curRowNum,
                                           const uint32_t curColNum)
{
    __ubuf__ TY *yQuantLocalAddr = (__ubuf__ TY *)yQuantLocal.GetPhyAddr();
    __ubuf__ float *yLocalAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    __ubuf__ float *scaleLocalAddr = (__ubuf__ float *)invScaleLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<TY>(curColNum);
    // 与 InvScale 一致: 每 128-chunk 固定 16 个 invScale -> 每行跨步 ceil(d/128)*16
    uint32_t invScaleColNumAlign = CeilDiv(curColNum, PER_BLOCK_FP16) * 16;
    VFProcessMxFp8QuantVF<TY>(yQuantLocalAddr, yLocalAddr, scaleLocalAddr, curRowNum, curColNum, loopCount,
                              curColNumAlign, dstCurColNumAlign, invScaleColNumAlign);
}

// Normal 动态量化, 整行一个 float32 scale (scale = rowmax / fp8max)。两遍处理:
// 第一遍跨整行求 amax, 第二遍按该 scale 量化。NaN 安全 (排除 NaN 求 max, 结果 NaN 时保留原值)。
// roundScale=true 时把 scale 向上取到最近的 2 的幂 (与 kv_compress_epilog 的 roundScale 一致)。
template <typename T0, typename T1, bool roundScale = true>
__simd_vf__ inline void VFProcessDynamicBlockQuantVF(__ubuf__ T0 *yLocalAddr, __ubuf__ float *scaleLocalAddr,
                                                     __ubuf__ T1 *xLocalAddr, const uint32_t maxValueInt,
                                                     const uint16_t curRowNum, const uint32_t curColNum,
                                                     const uint16_t loopCount, const uint32_t curColNumAlign,
                                                     const uint32_t dstCurColNumAlign, const uint32_t scaleRowAlign)
{
    static constexpr AscendC::Reg::DivSpecificMode mode = {AscendC::Reg::MaskMergeMode::ZEROING, false};
    {
        RegTensor<float> x;
        RegTensor<float> xAbs;
        RegTensor<float> partMax;
        RegTensor<float> rowMax;
        RegTensor<float> x0;
        RegTensor<float> x1;
        RegTensor<float> scale;
        RegTensor<float> one;
        RegTensor<float> invScale;
        RegTensor<float> dupInvScale;
        RegTensor<float> inf;
        RegTensor<float> zero;
        RegTensor<uint32_t> coeffReg;
        RegTensor<uint32_t> rsExp;
        RegTensor<uint32_t> rsInvExp;
        RegTensor<uint32_t> rsMant;
        RegTensor<uint32_t> rsSel;
        RegTensor<uint32_t> rsZeroU;
        RegTensor<uint32_t> rsOneU;
        RegTensor<uint32_t> rs254;
        RegTensor<uint32_t> rsMantMask;
        MaskReg pregMain = CreateMask<float>();
        MaskReg preg1 = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        MaskReg pm;
        MaskReg cmpAbs;
        MaskReg cmpScalar;
        MaskReg cmpQ;
        MaskReg rsCmp;
        Duplicate(coeffReg, maxValueInt, pregMain);
        Duplicate(zero, 0.0f, pregMain);
        Duplicate(one, 1.0f, pregMain);
        Duplicate(inf, 1.0f, pregMain);
        Div<float, &mode>(inf, inf, zero, pregMain);
        if constexpr (roundScale) {
            Duplicate(rsZeroU, static_cast<uint32_t>(0), preg1);
            Duplicate(rsOneU, static_cast<uint32_t>(1), preg1);
            Duplicate(rs254, static_cast<uint32_t>(254), preg1);
            Duplicate(rsMantMask, FAST_LOG_AND_VALUE2, preg1);
        }
        for (uint16_t i = 0; i < curRowNum; i++) {
            // pass 1: 整行 amax (排除 NaN)
            Duplicate(rowMax, 0.0f, preg1);
            uint32_t sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pm = UpdateMask<float>(sreg);
                LoadInputData<T1>(x, xLocalAddr, pm, j * VL_FP32 + i * curColNumAlign);
                Muls(xAbs, x, 0.0f, pm);
                Compare<float, CMPMODE::NE>(cmpAbs, xAbs, xAbs, pm);
                Not(cmpAbs, cmpAbs, pm);
                Abs(xAbs, x, cmpAbs);
                Reduce<AscendC::Reg::ReduceType::MAX>(partMax, xAbs, cmpAbs);
                Max(rowMax, rowMax, partMax, preg1);
            }
            Compares<float, CMPMODE::NE>(cmpScalar, rowMax, (float)0.0, preg1);
            Mul(scale, rowMax, (RegTensor<float> &)coeffReg, cmpScalar);
            Min(scale, scale, inf, preg1);
            if constexpr (!roundScale) {
                Div<float, &mode>(invScale, one, scale, preg1);
            } else {
                ShiftRights(rsExp, (RegTensor<uint32_t> &)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), preg1);
                And(rsMant, (RegTensor<uint32_t> &)scale, rsMantMask, preg1);
                Compare<uint32_t, AscendC::CMPMODE::NE>(rsCmp, rsMant, rsZeroU, preg1);
                Select(rsSel, rsOneU, rsZeroU, rsCmp);
                Add(rsExp, rsExp, rsSel, preg1);
                ShiftLefts((RegTensor<int32_t> &)scale, (RegTensor<int32_t> &)rsExp, static_cast<int16_t>(23), preg1);
                Sub((RegTensor<int32_t> &)rsInvExp, (RegTensor<int32_t> &)rs254, (RegTensor<int32_t> &)rsExp, preg1);
                ShiftLefts((RegTensor<int32_t> &)invScale, (RegTensor<int32_t> &)rsInvExp, static_cast<int16_t>(23),
                           preg1);
            }
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr + i * scaleRowAlign,
                                                                               scale, preg1);
            Duplicate(dupInvScale, invScale, pregMain);
            sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pm = UpdateMask<float>(sreg);
                LoadInputData<T1>(x, xLocalAddr, pm, j * VL_FP32 + i * curColNumAlign);
                Mul(x0, x, dupInvScale, pm);
                Muls(x1, x0, 0.0f, pm);
                Compare<float, CMPMODE::NE>(cmpQ, x1, x1, pm);
                Select(x, x, x0, cmpQ);
                StoreOutputData<T0>(yLocalAddr, x, pm, j * VL_FP32 + i * dstCurColNumAlign);
            }
        }
    }
}

template <typename T0, typename T1, bool roundScale = true>
__aicore__ inline void VFProcessDynamicBlockQuant(const LocalTensor<T0> &yLocal, const LocalTensor<float> &scaleLocal,
                                                  const LocalTensor<T1> &xLocal, float coeff, const uint16_t curRowNum,
                                                  const uint32_t curColNum)
{
    (void)coeff;
    __ubuf__ T0 *yLocalAddr = (__ubuf__ T0 *)yLocal.GetPhyAddr();
    __ubuf__ float *scaleLocalAddr = (__ubuf__ float *)scaleLocal.GetPhyAddr();
    __ubuf__ T1 *xLocalAddr = (__ubuf__ T1 *)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    uint32_t scaleRowAlign = RoundUp<float>(1);
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<T0, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<T0, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    } else if constexpr (IsSameType<T0, hifloat8_t>::value) {
        maxValueInt = INV_HIFP8_MAX_VALUE;
    }

    VFProcessDynamicBlockQuantVF<T0, T1, roundScale>(yLocalAddr, scaleLocalAddr, xLocalAddr, maxValueInt, curRowNum,
                                                     curColNum, loopCount, curColNumAlign, dstCurColNumAlign,
                                                     scaleRowAlign);
}

template <typename T0, typename T1>
__simd_vf__ inline void VFProcessHifp8QuantVF(__ubuf__ hifloat8_t *yLocalAddr, __ubuf__ T1 *xLocalAddr,
                                              const uint16_t curRowNum, const uint32_t curColNum, const float scales,
                                              const uint16_t loopCount, const uint32_t curColNumAlign,
                                              const uint32_t dstCurColNumAlign)
{
    {
        RegTensor<float> xReg;
        RegTensor<hifloat8_t> tmp;
        MaskReg pregLoop = CreateMask<float>();
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T1>(xReg, xLocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
                Muls(xReg, xReg, scales, pregLoop);
                Cast<hifloat8_t, float, castTraitF32toh8>(tmp, xReg, pregLoop);
                StoreAlign<hifloat8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    yLocalAddr + j * VL_FP32 + i * dstCurColNumAlign, tmp, pregLoop);
            }
        }
    }
}

template <typename T0, typename T1>
__aicore__ inline void VFProcessHifp8Quant(const LocalTensor<T0> &yLocal, const LocalTensor<float> &scaleLocal,
                                           const LocalTensor<T1> &xLocal, float coeff, const uint16_t curRowNum,
                                           const uint32_t curColNum, const float scales)
{
    LocalTensor<hifloat8_t> outLocal = yLocal.template ReinterpretCast<hifloat8_t>();
    __ubuf__ hifloat8_t *yLocalAddr = (__ubuf__ hifloat8_t *)outLocal.GetPhyAddr();
    __ubuf__ float *scaleLocalAddr = (__ubuf__ float *)scaleLocal.GetPhyAddr();
    __ubuf__ T1 *xLocalAddr = (__ubuf__ T1 *)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    VFProcessHifp8QuantVF<T0, T1>(yLocalAddr, xLocalAddr, curRowNum, curColNum, scales, loopCount, curColNumAlign,
                                  dstCurColNumAlign);
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T> &inputGm, const LocalTensor<T> &inputTensor, const uint16_t nBurst,
                              const uint32_t copyLen, uint32_t srcStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = nBurst;
    dataCopyExtParams.blockLen = copyLen * sizeof(T);
    dataCopyExtParams.srcStride = srcStride * sizeof(T);
    dataCopyExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCopyExtParams, dataCopyPadExtParams);
}

template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
__aicore__ inline void CopyOut(const LocalTensor<T> &outputTensor, const GlobalTensor<T> &outputGm,
                               const uint16_t nBurst, const uint32_t copyLen, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad<T, mode>(outputGm, outputTensor, dataCopyParams);
}

} // namespace IndexerQuantCache

#endif
