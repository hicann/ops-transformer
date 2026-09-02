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
 * \file kv_compress_epilog_common.h
 * \brief
 */

#ifndef KV_COMPRESS_EPILOG_COMMON_H
#define KV_COMPRESS_EPILOG_COMMON_H

#include "kernel_operator.h"

namespace KvCompressEpilogOps {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UnalignRegForLoad;
using AscendC::Reg::UnalignRegForStore;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t VL_FP32 = 64;
constexpr int32_t PER_BLOCK_FP16 = 128;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float FP8_E5M2_MIN_VALUE = -57344.0f;
constexpr float FP8_E4M3FN_MIN_VALUE = -448.0f;
constexpr int64_t QUANT_MODE_GROUP_QUANT_BF16 = 0;
constexpr int64_t QUANT_MODE_GROUP_QUANT_E8M0 = 1;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (((uint32_t)1 << (uint32_t)23) - (uint32_t)1);
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;

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

constexpr static AscendC::Reg::CastTrait castTraitU32toU16Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

constexpr static AscendC::Reg::CastTrait castTraitB322B8Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

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
    }
}

template <typename T0, typename T1, bool roundScale = true, bool castBf16 = false>
__simd_vf__ inline void VFProcessFP8PerGroupQuantVF(
    __ubuf__ T1 *yLocalAddr, __ubuf__ T0 *xLocalAddr, __ubuf__ T1 *scaleLocalAddr, __ubuf__ T0 *scaleLocalBf16Addr,
    __ubuf__ T0 *ropeYLocalAddr, float coeff, float fp8Min, float fp8Max, const uint16_t curRowNum,
    const uint32_t quantColNum, const uint16_t ropeNum, const uint16_t loopCount, const uint32_t curColNumAlign,
    const uint32_t dstCurColNumAlign, const uint32_t concatColNum, const uint32_t padColNum, const uint32_t sregNum)
{
    {
        RegTensor<float> x0;
        RegTensor<float> x0Abs;
        RegTensor<float> x1;
        RegTensor<float> x1Abs;
        RegTensor<float> max0;
        RegTensor<float> max1;
        RegTensor<float> max2;
        RegTensor<uint32_t> tmp1;
        RegTensor<uint32_t> vreg0;
        RegTensor<uint32_t> vreg1;
        RegTensor<uint32_t> vreg2;
        RegTensor<uint32_t> vreg3;
        RegTensor<uint32_t> vreg4;
        RegTensor<int32_t> vreg5;
        RegTensor<uint32_t> zero;
        RegTensor<uint32_t> one;
        RegTensor<float> dupScale;
        RegTensor<T0> ropeReg;
        RegTensor<uint8_t> scaleTmp1;
        RegTensor<bfloat16_t> scaleTmp1Bf16;
        UnalignRegForLoad ureg0;
        UnalignRegForStore ureg1;
        UnalignRegForStore ureg2;
        MaskReg pregLoop;
        MaskReg preg1 = CreateMask<T0, AscendC::Reg::MaskPattern::VL1>();
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        MaskReg pregMain = CreateMask<float>();
        MaskReg pregRope = CreateMask<T0, AscendC::Reg::MaskPattern::VL64>();
        MaskReg cmpMask;
        Duplicate(tmp1, FAST_LOG_AND_VALUE2, pregMerge);
        Duplicate(zero, static_cast<uint32_t>(0), pregMerge);
        Duplicate(one, static_cast<uint32_t>(1), pregMerge);
        for (uint16_t i = 0; i < curRowNum; i++) {
            // cat rope
            __ubuf__ T0 *rowRopeXLocalAddr = xLocalAddr + i * curColNumAlign + quantColNum;

            LoadUnAlignPre(ureg0, rowRopeXLocalAddr);
            LoadUnAlign(ropeReg, ureg0, rowRopeXLocalAddr, 64);
            StoreAlign(ropeYLocalAddr + i * dstCurColNumAlign / 2, ropeReg, pregRope);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T0>(x0, xLocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
                Abs(x0Abs, x0, pregLoop);
                Reduce<AscendC::Reg::ReduceType::MAX>(max0, x0Abs, pregLoop);
                Maxs(max2, max0, static_cast<float>(1e-4), pregMerge);
                Muls(max2, max2, coeff, pregMerge);
                if constexpr (roundScale) {
                    ShiftRights(vreg1, (RegTensor<uint32_t> &)max2, static_cast<int16_t>(FAST_LOG_SHIFT_BITS),
                                pregMerge);
                    And(vreg2, (RegTensor<uint32_t> &)max2, tmp1, pregMerge);
                    Compare<uint32_t, AscendC::CMPMODE::NE>(cmpMask, vreg2, zero, pregMerge);
                    Select(vreg4, one, zero, cmpMask);
                    Add(vreg1, vreg1, vreg4, pregMerge);
                    ShiftLefts((RegTensor<int32_t> &)max2, (RegTensor<int32_t> &)vreg1, static_cast<int16_t>(23),
                               pregMerge);
                }
                Duplicate(dupScale, max2, pregMain);
                Div(x0, x0, dupScale, pregLoop);
                Maxs(x0, x0, fp8Min, pregLoop);
                Mins(x0, x0, fp8Max, pregLoop);
                RegTensor<fp8_e4m3fn_t> fp8Out;
                Cast<fp8_e4m3fn_t, float, castTraitF32toFp8Even>(fp8Out, x0, pregLoop);
                StoreAlign<T1, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    yLocalAddr + j * VL_FP32 + i * dstCurColNumAlign + ropeNum, (RegTensor<T1> &)fp8Out, pregLoop);

                // cat scale
                if constexpr (castBf16) {
                    Cast<bfloat16_t, float, castTraitB322B16Even>(scaleTmp1Bf16, max2, preg1);
                    StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(
                        scaleLocalBf16Addr + (((quantColNum + ropeNum + i * dstCurColNumAlign) >> 1) + j),
                        (RegTensor<bfloat16_t> &)scaleTmp1Bf16, preg1);
                } else {
                    RegTensor<uint32_t> scaleTmp0;
                    ShiftRights(scaleTmp0, (RegTensor<uint32_t> &)max2, static_cast<int16_t>(FAST_LOG_SHIFT_BITS),
                                preg1);
                    Cast<uint8_t, uint32_t, castTraitB322B8Even>(scaleTmp1, scaleTmp0, preg1);
                    StoreAlign<T1, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B8>(
                        scaleLocalAddr + quantColNum + ropeNum + j + i * dstCurColNumAlign, (RegTensor<T1> &)scaleTmp1,
                        preg1);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            // pad zero
            __ubuf__ T1 *rowPadLocalAddr = yLocalAddr + i * dstCurColNumAlign + concatColNum;
            StoreUnAlign(rowPadLocalAddr, (RegTensor<T1> &)zero, ureg1, padColNum);
            StoreUnAlignPost(rowPadLocalAddr, ureg1, 0);
        }
    }
}

template <typename T0, typename T1, bool roundScale = true, bool castBf16 = false>
__aicore__ inline void VFProcessFP8PerGroupQuant(const LocalTensor<T1> &yLocal, const LocalTensor<T0> &xLocal,
                                                 float coeff, float fp8Min, float fp8Max, const uint16_t curRowNum,
                                                 const uint32_t curColNum, const uint32_t concatColNum,
                                                 const uint32_t padColNum)
{
    __ubuf__ T1 *yLocalAddr = (__ubuf__ T1 *)yLocal.GetPhyAddr();
    __ubuf__ T0 *xLocalAddr = (__ubuf__ T0 *)xLocal.GetPhyAddr();
    __ubuf__ T1 *scaleLocalAddr;
    __ubuf__ T0 *scaleLocalBf16Addr;
    if constexpr (castBf16) {
        LocalTensor<T0> scaleBf16Tensor = yLocal.template ReinterpretCast<T0>();
        scaleLocalBf16Addr = (__ubuf__ T0 *)scaleBf16Tensor.GetPhyAddr();
    } else {
        scaleLocalAddr = (__ubuf__ T1 *)yLocal.GetPhyAddr();
    }
    __ubuf__ T0 *ropeYLocalAddr = (__ubuf__ T0 *)yLocal.GetPhyAddr();

    uint32_t quantColNum = curColNum - 64;
    uint16_t scaleColNum = CeilDiv(quantColNum, 128);
    uint16_t ropeNum = 128;
    uint16_t loopCount = CeilDiv(quantColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T0>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T1>(concatColNum + padColNum);
    uint16_t loopCountFoldTwo = loopCount / 2;
    uint16_t loopCountReminder = loopCount % 2;
    uint32_t tailReminder = quantColNum - (loopCount - 1) * VL_FP32;
    uint32_t scaleColNumAlign = RoundUp<T0>(scaleColNum);
    uint32_t sregNum = quantColNum;
    VFProcessFP8PerGroupQuantVF<T0, T1, roundScale, castBf16>(
        yLocalAddr, xLocalAddr, scaleLocalAddr, scaleLocalBf16Addr, ropeYLocalAddr, coeff, fp8Min, fp8Max, curRowNum,
        quantColNum, ropeNum, loopCount, curColNumAlign, dstCurColNumAlign, concatColNum, padColNum, sregNum);
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

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
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

// quant_mode=2 (rope hifloat8 静态 + nope FLOAT4_E2M1 动态) 量化 VF 及相关常量/cast trait/descriptor
constexpr int32_t FOUR_UNFOLD_FP4 = 4; // 拷出按 4 行展开
constexpr int64_t QUANT_MODE_HIF8_FP4 = 2;
constexpr int64_t ROPE_HIF8_COLS = 64;         // rope 列数 (= 输出 hifloat8 字节数)
constexpr float FP4_E2M1_MAX_VALUE = 6.0f;     // FLOAT4_E2M1 的最大可表示幅值
constexpr uint16_t BF16_ABS_MASK = 0x7FFF;     // 清 bf16 符号位 (求 |x| 的 bf16 bit 模式)
constexpr uint32_t SCRATCH_BLK_PER_CHUNK = 16; // 每 64 元素块在 scratch 中占 16 个 bf16(32B 对齐)槽位
constexpr uint32_t FP4_CHUNK_ELEMS = 64;       // 一个 fp4 对齐块 = 64 元素 (= 32B fp4)
constexpr uint32_t E2B_DATABLOCK_B16 = 16;     // DIST_E2B_B16 每个 datablock 的 b16 lane 数

// fp32 (除完 scale 的 xq, 已转 bf16) -> FLOAT4_E2M1: ZERO 布局, 不饱和(已 clamp), ZEROING, 就近舍入
constexpr static AscendC::Reg::CastTrait castTraitBf16toFp4 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// fp32 -> hifloat8: ZERO 布局, 饱和, ZEROING, CAST_ROUND (与 indexer_quant_cache rope 静态量化一致)
constexpr static AscendC::Reg::CastTrait castTraitF32toHif8 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

struct Hif8Fp4RowDesc {
    uint32_t nopeElems;     // d-64, 必为 64 的整数倍
    uint32_t numChunks;     // nopeElems / 64
    uint32_t xRowAlign;     // RoundUp<T0>(d)
    uint32_t dstRowStride;  // cacheColAlign (字节)
    uint32_t fp4ByteOff;    // = 64
    uint32_t scaleByteOff;  // = 64 + nopeElems/2
    uint32_t scratchRowB16; // 每行 scratch bf16 槽位数 = numChunks * 16
    uint32_t concatCol;
    uint32_t padCol;
    float scalesAttr;
};

template <typename T0, typename T1>
__simd_callee__ inline void Hif8Fp4Rope(__ubuf__ hifloat8_t *yHif8Addr, __ubuf__ T0 *xAddr, uint32_t rowByteBase,
                                        uint32_t rowElemBase, uint32_t nopeElems, float scalesAttr, MaskReg pregRope)
{
    RegTensor<float> ropeF32;
    RegTensor<hifloat8_t> ropeHif8;
    LoadInputData<T0>(ropeF32, xAddr, pregRope, rowElemBase + nopeElems);
    Muls(ropeF32, ropeF32, scalesAttr, pregRope);
    Cast<hifloat8_t, float, castTraitF32toHif8>(ropeHif8, ropeF32, pregRope);
    StoreAlign<hifloat8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(yHif8Addr + rowByteBase, ropeHif8, pregRope);
}

template <typename T1>
__simd_callee__ inline void Hif8Fp4Pad(__ubuf__ T1 *yByteAddr, uint32_t rowByteBase, uint32_t concatCol,
                                       uint32_t padCol, RegTensor<uint8_t> &zeroU8, UnalignRegForStore &upadReg)
{
    if (padCol > 0) {
        __ubuf__ T1 *padAddr = yByteAddr + rowByteBase + concatCol;
        StoreUnAlign(padAddr, (RegTensor<T1> &)zeroU8, upadReg, padCol);
        StoreUnAlignPost(padAddr, upadReg, 0);
    }
}

template <typename T0, typename T1>
__simd_vf__ inline void VFProcessHif8Fp4G64VF(__ubuf__ T1 *yByteAddr, __ubuf__ T0 *xAddr, const Hif8Fp4RowDesc desc,
                                              const uint16_t curRowNum)
{
    static constexpr AscendC::Reg::DivSpecificMode divMode = {AscendC::Reg::MaskMergeMode::ZEROING, false};
    {
        __ubuf__ hifloat8_t *yHif8Addr = reinterpret_cast<__ubuf__ hifloat8_t *>(yByteAddr);
        __ubuf__ T0 *yBf16Addr = reinterpret_cast<__ubuf__ T0 *>(yByteAddr);

        RegTensor<float> xF32;
        RegTensor<float> xAbs;
        RegTensor<float> amax;
        RegTensor<float> scaleF32;
        RegTensor<float> invScaleF32;
        RegTensor<float> six;
        RegTensor<T0> scaleBf16;
        RegTensor<T0> invScaleBf16;
        RegTensor<T0> invDupFull;
        RegTensor<T0> zeroBf16;
        RegTensor<T0> xChunkBf16;
        RegTensor<T0> xqChunk;
        RegTensor<T0> fp4MaxBf16;
        RegTensor<T0> fp4MinBf16;
        RegTensor<fp4x2_e2m1_t> fp4Out;
        RegTensor<uint8_t> zeroU8;
        UnalignRegForStore upadReg;
        MaskReg preg1 = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        MaskReg pregRope = CreateMask<float>();
        MaskReg pregFullU8 = CreateMask<uint8_t>();
        MaskReg pregFullB16 = CreateMask<T0>();

        Duplicate(six, FP4_E2M1_MAX_VALUE, preg1);
        Duplicate(zeroU8, static_cast<uint8_t>(0), pregFullU8);
        Duplicate(fp4MaxBf16, static_cast<T0>(FP4_E2M1_MAX_VALUE), pregFullB16);
        Duplicate(fp4MinBf16, static_cast<T0>(-FP4_E2M1_MAX_VALUE), pregFullB16);

        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t rowByteBase = static_cast<uint32_t>(i) * desc.dstRowStride;
            uint32_t rowElemBase = static_cast<uint32_t>(i) * desc.xRowAlign;

            Hif8Fp4Rope<T0, T1>(yHif8Addr, xAddr, rowByteBase, rowElemBase, desc.nopeElems, desc.scalesAttr, pregRope);

            for (uint32_t c = 0; c < desc.numChunks; c++) {
                uint32_t chunkElemBase = rowElemBase + c * FP4_CHUNK_ELEMS;
                uint32_t s64 = FP4_CHUNK_ELEMS;
                MaskReg mask64 = UpdateMask<T0>(s64);
                MaskReg maskGf32 = CreateMask<float>();

                LoadInputData<T0>(xF32, xAddr, maskGf32, chunkElemBase);
                Abs(xAbs, xF32, maskGf32);
                Reduce<AscendC::Reg::ReduceType::MAX>(amax, xAbs, maskGf32);

                Muls(scaleF32, amax, static_cast<float>(1.0f / FP4_E2M1_MAX_VALUE), preg1);
                MaskReg cmpNZ;
                Compares<float, AscendC::CMPMODE::NE>(cmpNZ, amax, 0.0f, preg1);
                Div<float, &divMode>(invScaleF32, six, amax, cmpNZ);

                Cast<T0, float, castTraitB322B16Even>(scaleBf16, scaleF32, preg1);
                StoreAlign<T0, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(
                    yBf16Addr + (rowByteBase + desc.scaleByteOff) / 2 + c, scaleBf16, preg1);

                Cast<T0, float, castTraitB322B16Even>(invScaleBf16, invScaleF32, preg1);
                Duplicate(invDupFull, invScaleBf16, mask64);

                LoadAlign(xChunkBf16, xAddr + chunkElemBase);
                Mul(xqChunk, xChunkBf16, invDupFull, mask64);
                Min(xqChunk, xqChunk, fp4MaxBf16, mask64);
                Max(xqChunk, xqChunk, fp4MinBf16, mask64);
                Cast<fp4x2_e2m1_t, T0, castTraitBf16toFp4>(fp4Out, xqChunk, mask64);
                StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    reinterpret_cast<__ubuf__ int8_t *>(yByteAddr + rowByteBase + desc.fp4ByteOff +
                                                        c * (FP4_CHUNK_ELEMS / 2)),
                    (RegTensor<int8_t> &)fp4Out, mask64);
            }

            Hif8Fp4Pad<T1>(yByteAddr, rowByteBase, desc.concatCol, desc.padCol, zeroU8, upadReg);
        }
    }
}

template <typename T0, typename T1, uint32_t GROUPS_PER_CHUNK>
__simd_callee__ inline void Hif8Fp4ScaleToScratch(__ubuf__ T0 *yBf16Addr, __ubuf__ T0 *scratchChunkAddr,
                                                  uint32_t scaleElemBase, RegTensor<uint16_t> &amaxBits,
                                                  RegTensor<float> &six, MaskReg maskK, MaskReg maskScratch,
                                                  UnalignRegForStore &uScale)
{
    static constexpr AscendC::Reg::DivSpecificMode divMode = {AscendC::Reg::MaskMergeMode::ZEROING, false};
    RegTensor<uint16_t> amaxSpread;
    RegTensor<uint16_t> dummy;
    RegTensor<float> amaxF32;
    RegTensor<float> scaleF32;
    RegTensor<float> invScaleF32;
    RegTensor<T0> scaleBf16;
    RegTensor<T0> invScaleBf16;
    RegTensor<T0> scaleConsec;
    RegTensor<T0> invConsec;
    RegTensor<T0> scratchVals;
    RegTensor<T0> scratchHi;

    Interleave(amaxSpread, dummy, amaxBits, amaxBits);
    Cast<float, T0, castTraitB162B32Even>(amaxF32, (RegTensor<T0> &)amaxSpread, maskK);
    Muls(scaleF32, amaxF32, static_cast<float>(1.0f / FP4_E2M1_MAX_VALUE), maskK);
    MaskReg cmpNZ;
    Compares<float, AscendC::CMPMODE::NE>(cmpNZ, amaxF32, 0.0f, maskK);
    Div<float, &divMode>(invScaleF32, six, amaxF32, cmpNZ);

    Cast<T0, float, castTraitB322B16Even>(scaleBf16, scaleF32, maskK);
    DeInterleave(scaleConsec, (RegTensor<T0> &)dummy, scaleBf16, scaleBf16);
    __ubuf__ T0 *scaleAddr = yBf16Addr + scaleElemBase;
    StoreUnAlign(scaleAddr, scaleConsec, uScale, GROUPS_PER_CHUNK);
    StoreUnAlignPost(scaleAddr, uScale, 0);

    Cast<T0, float, castTraitB322B16Even>(invScaleBf16, invScaleF32, maskK);
    DeInterleave(invConsec, (RegTensor<T0> &)dummy, invScaleBf16, invScaleBf16);
    if constexpr (GROUPS_PER_CHUNK == 4) {
        StoreAlign(scratchChunkAddr, invConsec, maskScratch);
    } else {
        Interleave(scratchVals, scratchHi, invConsec, invConsec);
        StoreAlign(scratchChunkAddr, scratchVals, maskScratch);
    }
}

template <typename T0, typename T1>
__simd_vf__ inline void VFProcessHif8Fp4G16VF(__ubuf__ T1 *yByteAddr, __ubuf__ T0 *xAddr, __ubuf__ T0 *scratchAddr,
                                              const Hif8Fp4RowDesc desc, const uint16_t curRowNum)
{
    {
        __ubuf__ hifloat8_t *yHif8Addr = reinterpret_cast<__ubuf__ hifloat8_t *>(yByteAddr);
        __ubuf__ T0 *yBf16Addr = reinterpret_cast<__ubuf__ T0 *>(yByteAddr);

        RegTensor<float> six;
        RegTensor<T0> xChunkBf16;
        RegTensor<uint16_t> absBits;
        RegTensor<uint16_t> amaxBits;
        RegTensor<uint16_t> absMaskReg;
        RegTensor<T0> invBroadcast;
        RegTensor<T0> xqChunk;
        RegTensor<T0> fp4MaxBf16;
        RegTensor<T0> fp4MinBf16;
        RegTensor<fp4x2_e2m1_t> fp4Out;
        RegTensor<uint8_t> zeroU8;
        UnalignRegForStore upadReg;
        UnalignRegForStore uScale;
        MaskReg pregRope = CreateMask<float>();
        MaskReg pregFullU8 = CreateMask<uint8_t>();
        MaskReg pregFullB16 = CreateMask<T0>();
        MaskReg pregFullF32 = CreateMask<float>();

        Duplicate(six, FP4_E2M1_MAX_VALUE, pregFullF32);
        Duplicate(zeroU8, static_cast<uint8_t>(0), pregFullU8);
        Duplicate(fp4MaxBf16, static_cast<T0>(FP4_E2M1_MAX_VALUE), pregFullB16);
        Duplicate(fp4MinBf16, static_cast<T0>(-FP4_E2M1_MAX_VALUE), pregFullB16);
        Duplicate(absMaskReg, BF16_ABS_MASK, pregFullB16);

        const uint32_t scaleElemBaseRow0 = desc.scaleByteOff / 2;

        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t rowByteBase = static_cast<uint32_t>(i) * desc.dstRowStride;
            uint32_t rowElemBase = static_cast<uint32_t>(i) * desc.xRowAlign;
            uint32_t rowScratchBase = static_cast<uint32_t>(i) * desc.scratchRowB16;

            Hif8Fp4Rope<T0, T1>(yHif8Addr, xAddr, rowByteBase, rowElemBase, desc.nopeElems, desc.scalesAttr, pregRope);

            for (uint32_t c = 0; c < desc.numChunks; c++) {
                uint32_t chunkElemBase = rowElemBase + c * FP4_CHUNK_ELEMS;
                uint32_t s64 = FP4_CHUNK_ELEMS;
                MaskReg mask64 = UpdateMask<T0>(s64);
                uint32_t s4 = FOUR_UNFOLD_FP4;
                MaskReg maskK = UpdateMask<float>(s4);
                uint32_t s4b16 = FOUR_UNFOLD_FP4;
                MaskReg maskScratch = UpdateMask<T0>(s4b16);

                LoadAlign(xChunkBf16, xAddr + chunkElemBase);
                And(absBits, (RegTensor<uint16_t> &)xChunkBf16, absMaskReg, mask64);
                ReduceDataBlock<AscendC::Reg::ReduceType::MAX, uint16_t>(amaxBits, absBits, mask64);

                uint32_t scaleElemBase = rowByteBase / 2 + scaleElemBaseRow0 + c * FOUR_UNFOLD_FP4;
                __ubuf__ T0 *scratchChunkAddr = scratchAddr + rowScratchBase + c * SCRATCH_BLK_PER_CHUNK;
                Hif8Fp4ScaleToScratch<T0, T1, 4>(yBf16Addr, scratchChunkAddr, scaleElemBase, amaxBits, six, maskK,
                                                 maskScratch, uScale);
            }

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            // ---- pass2: E2B_B16 把 scratch 每个 bf16 invScale 广播到 16 lane datablock, 量化写 fp4 ----
            for (uint32_t c = 0; c < desc.numChunks; c++) {
                uint32_t chunkElemBase = rowElemBase + c * FP4_CHUNK_ELEMS;
                uint32_t s64 = FP4_CHUNK_ELEMS;
                MaskReg mask64 = UpdateMask<T0>(s64);
                __ubuf__ T0 *scratchChunkAddr = scratchAddr + rowScratchBase + c * SCRATCH_BLK_PER_CHUNK;

                LoadAlign(xChunkBf16, xAddr + chunkElemBase);
                LoadAlign<T0, AscendC::Reg::LoadDist::DIST_E2B_B16>(invBroadcast, scratchChunkAddr);
                Mul(xqChunk, xChunkBf16, invBroadcast, mask64);
                Min(xqChunk, xqChunk, fp4MaxBf16, mask64);
                Max(xqChunk, xqChunk, fp4MinBf16, mask64);
                Cast<fp4x2_e2m1_t, T0, castTraitBf16toFp4>(fp4Out, xqChunk, mask64);
                StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    reinterpret_cast<__ubuf__ int8_t *>(yByteAddr + rowByteBase + desc.fp4ByteOff +
                                                        c * (FP4_CHUNK_ELEMS / 2)),
                    (RegTensor<int8_t> &)fp4Out, mask64);
            }

            Hif8Fp4Pad<T1>(yByteAddr, rowByteBase, desc.concatCol, desc.padCol, zeroU8, upadReg);
        }
    }
}

template <typename T0, typename T1>
__simd_vf__ inline void VFProcessHif8Fp4G32VF(__ubuf__ T1 *yByteAddr, __ubuf__ T0 *xAddr, __ubuf__ T0 *scratchAddr,
                                              const Hif8Fp4RowDesc desc, const uint16_t curRowNum)
{
    {
        __ubuf__ hifloat8_t *yHif8Addr = reinterpret_cast<__ubuf__ hifloat8_t *>(yByteAddr);
        __ubuf__ T0 *yBf16Addr = reinterpret_cast<__ubuf__ T0 *>(yByteAddr);

        RegTensor<float> six;
        RegTensor<T0> xChunkBf16;
        RegTensor<uint16_t> absBits;
        RegTensor<uint16_t> deEven;
        RegTensor<uint16_t> deOdd;
        RegTensor<uint16_t> pairMax;
        RegTensor<uint16_t> amaxBits;
        RegTensor<uint16_t> absMaskReg;
        RegTensor<T0> invBroadcast;
        RegTensor<T0> xqChunk;
        RegTensor<T0> fp4MaxBf16;
        RegTensor<T0> fp4MinBf16;
        RegTensor<fp4x2_e2m1_t> fp4Out;
        RegTensor<uint8_t> zeroU8;
        UnalignRegForStore upadReg;
        UnalignRegForStore uScale;
        MaskReg pregRope = CreateMask<float>();
        MaskReg pregFullU8 = CreateMask<uint8_t>();
        MaskReg pregFullB16 = CreateMask<T0>();
        MaskReg pregFullF32 = CreateMask<float>();

        Duplicate(six, FP4_E2M1_MAX_VALUE, pregFullF32);
        Duplicate(zeroU8, static_cast<uint8_t>(0), pregFullU8);
        Duplicate(fp4MaxBf16, static_cast<T0>(FP4_E2M1_MAX_VALUE), pregFullB16);
        Duplicate(fp4MinBf16, static_cast<T0>(-FP4_E2M1_MAX_VALUE), pregFullB16);
        Duplicate(absMaskReg, BF16_ABS_MASK, pregFullB16);

        const uint32_t scaleElemBaseRow0 = desc.scaleByteOff / 2;
        constexpr uint32_t GROUPS_PER_CHUNK_G32 = 2;

        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t rowByteBase = static_cast<uint32_t>(i) * desc.dstRowStride;
            uint32_t rowElemBase = static_cast<uint32_t>(i) * desc.xRowAlign;
            uint32_t rowScratchBase = static_cast<uint32_t>(i) * desc.scratchRowB16;

            Hif8Fp4Rope<T0, T1>(yHif8Addr, xAddr, rowByteBase, rowElemBase, desc.nopeElems, desc.scalesAttr, pregRope);

            for (uint32_t c = 0; c < desc.numChunks; c++) {
                uint32_t chunkElemBase = rowElemBase + c * FP4_CHUNK_ELEMS;
                uint32_t s64 = FP4_CHUNK_ELEMS;
                MaskReg mask64 = UpdateMask<T0>(s64);
                uint32_t s32 = FP4_CHUNK_ELEMS / 2;
                MaskReg mask32 = UpdateMask<T0>(s32);
                uint32_t s2 = GROUPS_PER_CHUNK_G32;
                MaskReg maskK = UpdateMask<float>(s2);
                uint32_t s4b16 = FOUR_UNFOLD_FP4;
                MaskReg maskScratch = UpdateMask<T0>(s4b16);

                LoadAlign(xChunkBf16, xAddr + chunkElemBase);
                And(absBits, (RegTensor<uint16_t> &)xChunkBf16, absMaskReg, mask64);
                DeInterleave(deEven, deOdd, absBits, absBits);
                Max(pairMax, deEven, deOdd, mask32);
                ReduceDataBlock<AscendC::Reg::ReduceType::MAX, uint16_t>(amaxBits, pairMax, mask32);

                uint32_t scaleElemBase = rowByteBase / 2 + scaleElemBaseRow0 + c * GROUPS_PER_CHUNK_G32;
                __ubuf__ T0 *scratchChunkAddr = scratchAddr + rowScratchBase + c * SCRATCH_BLK_PER_CHUNK;
                Hif8Fp4ScaleToScratch<T0, T1, GROUPS_PER_CHUNK_G32>(yBf16Addr, scratchChunkAddr, scaleElemBase,
                                                                    amaxBits, six, maskK, maskScratch, uScale);
            }

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            for (uint32_t c = 0; c < desc.numChunks; c++) {
                uint32_t chunkElemBase = rowElemBase + c * FP4_CHUNK_ELEMS;
                uint32_t s64 = FP4_CHUNK_ELEMS;
                MaskReg mask64 = UpdateMask<T0>(s64);
                __ubuf__ T0 *scratchChunkAddr = scratchAddr + rowScratchBase + c * SCRATCH_BLK_PER_CHUNK;

                LoadAlign(xChunkBf16, xAddr + chunkElemBase);
                LoadAlign<T0, AscendC::Reg::LoadDist::DIST_E2B_B16>(invBroadcast, scratchChunkAddr);
                Mul(xqChunk, xChunkBf16, invBroadcast, mask64);
                Min(xqChunk, xqChunk, fp4MaxBf16, mask64);
                Max(xqChunk, xqChunk, fp4MinBf16, mask64);
                Cast<fp4x2_e2m1_t, T0, castTraitBf16toFp4>(fp4Out, xqChunk, mask64);
                StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                    reinterpret_cast<__ubuf__ int8_t *>(yByteAddr + rowByteBase + desc.fp4ByteOff +
                                                        c * (FP4_CHUNK_ELEMS / 2)),
                    (RegTensor<int8_t> &)fp4Out, mask64);
            }

            Hif8Fp4Pad<T1>(yByteAddr, rowByteBase, desc.concatCol, desc.padCol, zeroU8, upadReg);
        }
    }
}

template <typename T0, typename T1>
__aicore__ inline void VFProcessHif8Fp4(const LocalTensor<T1> &yLocal, const LocalTensor<T0> &xLocal,
                                        const LocalTensor<T0> &scratchLocal, float scalesAttr, const uint16_t curRowNum,
                                        const uint32_t d, const uint32_t nGroup, const uint32_t groupSize,
                                        const uint32_t dstRowStride, const uint32_t concatCol, const uint32_t padCol)
{
    __ubuf__ T1 *yByteAddr = (__ubuf__ T1 *)yLocal.GetPhyAddr();
    __ubuf__ T0 *xAddr = (__ubuf__ T0 *)xLocal.GetPhyAddr();
    __ubuf__ T0 *scratchAddr = (__ubuf__ T0 *)scratchLocal.GetPhyAddr();

    Hif8Fp4RowDesc desc;
    desc.nopeElems = d - ROPE_HIF8_COLS;
    desc.numChunks = desc.nopeElems / FP4_CHUNK_ELEMS;
    desc.xRowAlign = RoundUp<T0>(d);
    desc.dstRowStride = dstRowStride;
    desc.fp4ByteOff = ROPE_HIF8_COLS;
    desc.scaleByteOff = ROPE_HIF8_COLS + desc.nopeElems / 2;
    desc.scratchRowB16 = desc.numChunks * SCRATCH_BLK_PER_CHUNK;
    desc.concatCol = concatCol;
    desc.padCol = padCol;
    desc.scalesAttr = scalesAttr;

    if (groupSize == 16) {
        VFProcessHif8Fp4G16VF<T0, T1>(yByteAddr, xAddr, scratchAddr, desc, curRowNum);
    } else if (groupSize == 32) {
        VFProcessHif8Fp4G32VF<T0, T1>(yByteAddr, xAddr, scratchAddr, desc, curRowNum);
    } else {
        VFProcessHif8Fp4G64VF<T0, T1>(yByteAddr, xAddr, desc, curRowNum);
    }
}

} // namespace KvCompressEpilogOps

#endif
