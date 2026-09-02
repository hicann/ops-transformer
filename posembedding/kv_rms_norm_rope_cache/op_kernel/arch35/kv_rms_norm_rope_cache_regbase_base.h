/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_rms_norm_rope_cache_regbase_base.h
 * \brief
 */

#ifndef KV_RMS_NORM_ROPE_CACHE_REGBASE_BASE_H
#define KV_RMS_NORM_ROPE_CACHE_REGBASE_BASE_H

#include "kernel_operator.h"
#include "kv_rms_norm_rope_cache_regbase_platform.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

constexpr static uint32_t VL_FP32 = static_cast<int64_t>(platform::GetVRegSize()) / sizeof(float);

constexpr static AscendC::MicroAPI::CastTrait CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::MicroAPI::CastTrait CAST_FP16_TO_INT8 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};

constexpr static AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::MicroAPI::CastTrait CAST_INT32_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_INT16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::MicroAPI::CastTrait CAST_INT16_TO_FP16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FLOAT8 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_HIFLOAT8 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

constexpr int64_t BLOCK_SIZE = 32;
// Constants for buffer counts to replace magic numbers
constexpr int32_t BUFFER_COUNT_SINGLE = 1;
constexpr int32_t BUFFER_COUNT_DOUBLE = 2;

// Constants for channel counts in buffer size calculations to replace magic numbers
constexpr int32_t COS_SIN_CHANNEL_COUNT = 4;        // Real/Imaginary for Cos/Sin
constexpr int32_t K_SCALE_OFFSET_CHANNEL_COUNT = 4; // Real/Imaginary for Scale/Offset (K)
constexpr int32_t V_SCALE_OFFSET_CHANNEL_COUNT = 2; // Scale/Offset (V)
constexpr int32_t WORKSPACE_FLOAT_VECTOR_COUNT = 2; // Number of float vectors in workspace

constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static float CONST_MINUS_ONE = -1.0;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_THREE = 3;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_FIVE = 5;
constexpr static int64_t CONST_SIX = 6;
constexpr static int64_t CONST_SEVEN = 7;
constexpr static int64_t CONST_EIGHT = 8;
constexpr static int64_t CONST_SIXTY_THREE = 63;

constexpr static int64_t NORM_CACHE_MODE = 0;
constexpr static int64_t PA_CACHE_MODE = 1;
constexpr static int64_t PA_NZ_CACHE_MODE = 2;
constexpr static int64_t PA_BLK_BNSD_CACHE_MODE = 3;
constexpr static int64_t PA_BLK_NZ_CACHE_MODE = 4;

template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
class KvRmsNormRopeCacheRegbase {
public:
    __aicore__ inline KvRmsNormRopeCacheRegbase() {}

protected:
    /* global memory address */
    GlobalTensor<T_KV> kvGm;
    GlobalTensor<T_KV> gammaGm;
    GlobalTensor<T_KV> cosGm;
    GlobalTensor<T_KV> sinGm;
    GlobalTensor<int64_t> indexGm;
    GlobalTensor<T_K_CACHE> kCacheGm;
    GlobalTensor<T_V_CACHE> vCacheGm;
    GlobalTensor<T_KV> kOutGm, vOutGm;
    GlobalTensor<float> kScaleGm, vScaleGm;
    GlobalTensor<float> kOffsetGm, vOffsetGm;

    /* variable */
    float epsilon = 1e-5;
    float reciprocal = 1.0;

    /* ascendc variable */
    uint32_t blockIdx = GetBlockIdx();
    uint32_t useCoreNum = GetBlockNum();

    int64_t ubFactor = 1;
    int64_t ubLoop = 1;
    int64_t ubTail = 1;

    /* tilingdata*/
    int64_t dv = 0;
    int64_t dvAlign = 0;
    int64_t dvB8Align = 0;
    int64_t dkAlign = 0;
    int64_t dk = 0;
    int64_t dvLoopCount = 0;

protected:
    __aicore__ inline void RmsNormBasicComputeVF(const LocalTensor<T_KV> &xLocal, const LocalTensor<T_KV> &gammaLocal,
                                                 const LocalTensor<float> &wsLocal, int64_t aSize, int64_t &foldPoint,
                                                 int64_t &outerLoopDstStride);
    __aicore__ inline void RmsNormPostVF(const LocalTensor<T_KV> &dstTensor, const LocalTensor<T_KV> &xTensor,
                                         const LocalTensor<T_KV> &gammaLocal,
                                         const LocalTensor<float> &reduceSumTempTensor, const int64_t aSize,
                                         const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormAsymQuantWithKvPostVF(
        const LocalTensor<T_KV> &dstTensor, const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xTensor,
        const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &reduceSumTempTensor,
        const LocalTensor<float> &vScaleTensor, const LocalTensor<float> &vOffsetTensor, const int64_t aSize,
        const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormSymQuantWithKvPostVF(
        const LocalTensor<T_KV> &dstTensor, const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xTensor,
        const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &reduceSumTempTensor,
        const LocalTensor<float> &vScaleTensor, const int64_t aSize, const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormAsymQuantPostVF(const LocalTensor<T_V_CACHE> &quantLocal,
                                                  const LocalTensor<T_KV> &xTensor, const LocalTensor<T_KV> &gammaLocal,
                                                  const LocalTensor<float> &reduceSumTempTensor,
                                                  const LocalTensor<float> &vScaleTensor,
                                                  const LocalTensor<float> &vOffsetTensor, const int64_t aSize,
                                                  const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormSymQuantPostVF(const LocalTensor<T_V_CACHE> &quantLocal,
                                                 const LocalTensor<T_KV> &xTensor, const LocalTensor<T_KV> &gammaLocal,
                                                 const LocalTensor<float> &reduceSumTempTensor,
                                                 const LocalTensor<float> &vScaleTensor, const int64_t aSize,
                                                 const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormVF(const LocalTensor<T_KV> &outVLocal, const LocalTensor<T_KV> &xLocal,
                                     const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &wsLocal,
                                     int64_t calcRow);
    __aicore__ inline void RmsNormAsymQuantWithKvVF(
        const LocalTensor<T_KV> &outVLocal, const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xLocal,
        const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &wsLocal, const LocalTensor<float> &vScaleTensor,
        const LocalTensor<float> &vOffsetTensor, int64_t calcRow);
    __aicore__ inline void RmsNormSymQuantWithKvVF(const LocalTensor<T_KV> &outVLocal,
                                                   const LocalTensor<T_V_CACHE> &quantLocal,
                                                   const LocalTensor<T_KV> &xLocal, const LocalTensor<T_KV> &gammaLocal,
                                                   const LocalTensor<float> &wsLocal,
                                                   const LocalTensor<float> &vScaleTensor, int64_t calcRow);
    __aicore__ inline void RmsNormAsymQuantVF(const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xLocal,
                                              const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &wsLocal,
                                              const LocalTensor<float> &vScaleTensor,
                                              const LocalTensor<float> &vOffsetTensor, int64_t calcRow);
    __aicore__ inline void RmsNormSymQuantVF(const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xLocal,
                                             const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &wsLocal,
                                             const LocalTensor<float> &vScaleTensor, int64_t calcRow);
    __aicore__ inline void RmsNormPostWithMul(const LocalTensor<float> &dstTensor, const LocalTensor<T_KV> &xTensor,
                                              const LocalTensor<T_KV> &grammaTensor, const int64_t aSize,
                                              const int64_t rSize, const int64_t stride);
};

__aicore__ inline int64_t FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - ScalarCountLeadingZero(num);
        return (CONST_ONE << pow);
    }
}

template <typename T>
__aicore__ inline void LoadTensorForDtypeT(__ubuf__ T *input, AscendC::MicroAPI::RegTensor<float> &dst,
                                           AscendC::MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        AscendC::MicroAPI::RegTensor<half> xFp16;
        AscendC::MicroAPI::LoadAlign<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
            xFp16, ((__ubuf__ half *)(input) + (offset)));
        Cast<float, half, CAST_B16_TO_B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        AscendC::MicroAPI::RegTensor<bfloat16_t> xBf16;
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
            xBf16, ((__ubuf__ bfloat16_t *)(input) + (offset)));
        Cast<float, bfloat16_t, CAST_B16_TO_B32>(dst, xBf16, preg);
    } else {
        LoadAlign(dst, ((__ubuf__ float *)(input) + (offset)));
    }
}

template <typename T>
__aicore__ inline void StoreTensorForDtypeTOut(__ubuf__ T *dst, AscendC::MicroAPI::RegTensor<float> &src,
                                               AscendC::MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::MicroAPI::RegTensor<T> xFp16;
        Cast<T, float, CAST_FP32_TO_FP16>(xFp16, src, preg);
        AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

// store 非对齐的float32类型的src(寄存器)数据到output(ub)中
// output数据类型支持int8, hifloat8, float8_e5m2, float8_e4m3fn, bfloat16, float16, float32
template <typename T>
__aicore__ inline void StoreUnAlignOneTensor(__ubuf__ T *&output, MicroAPI::RegTensor<float> &src,
                                             MicroAPI::UnalignRegForStore &uValue, MicroAPI::MaskReg &preg,
                                             uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        MicroAPI::RegTensor<half> xFp16Pack;
        Cast<half, float, CAST_FP32_TO_FP16>(xFp16, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t> &)xFp16Pack, (MicroAPI::RegTensor<uint32_t> &)xFp16);
        StoreUnAlign(output, xFp16Pack, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::RegTensor<bfloat16_t> xBf16Pack;
        Cast<bfloat16_t, float, CAST_FP32_TO_FP16>(xBf16, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t> &)xBf16Pack, (MicroAPI::RegTensor<uint32_t> &)xBf16);
        StoreUnAlign(output, xBf16Pack, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, int8_t>::value) {
        // 量化为int8
        AscendC::MicroAPI::RegTensor<half> tmpHalf;
        AscendC::MicroAPI::RegTensor<int16_t> tmpInt16;
        AscendC::MicroAPI::RegTensor<int8_t> quantInt8;
        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmpInt16, src, preg);
        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmpHalf, tmpInt16, preg);
        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quantInt8, tmpHalf, preg);
        Pack((MicroAPI::RegTensor<uint16_t> &)tmpInt16, (MicroAPI::RegTensor<uint32_t> &)quantInt8);
        Pack((MicroAPI::RegTensor<uint8_t> &)quantInt8, (MicroAPI::RegTensor<uint16_t> &)tmpInt16);
        StoreUnAlign(output, quantInt8, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, hifloat8_t>::value) {
        // 量化为hifloat8
        AscendC::MicroAPI::RegTensor<hifloat8_t> quantHifloat8Pack;
        AscendC::MicroAPI::RegTensor<hifloat8_t> quantHifloat8;
        AscendC::MicroAPI::Cast<hifloat8_t, float, CAST_FP32_TO_HIFLOAT8>(quantHifloat8, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t> &)quantHifloat8Pack, (MicroAPI::RegTensor<uint32_t> &)quantHifloat8);
        Pack((MicroAPI::RegTensor<uint8_t> &)quantHifloat8, (MicroAPI::RegTensor<uint16_t> &)quantHifloat8Pack);
        StoreUnAlign(output, quantHifloat8, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, fp8_e5m2_t>::value) {
        // 量化fp8_e5m2
        AscendC::MicroAPI::RegTensor<fp8_e5m2_t> quantFloat8E5m2Pack;
        AscendC::MicroAPI::RegTensor<fp8_e5m2_t> quantFloat8E5m2;
        AscendC::MicroAPI::Cast<fp8_e5m2_t, float, CAST_FP32_TO_FLOAT8>(quantFloat8E5m2, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t> &)quantFloat8E5m2Pack, (MicroAPI::RegTensor<uint32_t> &)quantFloat8E5m2);
        Pack((MicroAPI::RegTensor<uint8_t> &)quantFloat8E5m2, (MicroAPI::RegTensor<uint16_t> &)quantFloat8E5m2Pack);
        StoreUnAlign(output, quantFloat8E5m2, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
        // 量化为fp8_e4m3fn
        AscendC::MicroAPI::RegTensor<fp8_e4m3fn_t> quantFloat8E4m3fnPack;
        AscendC::MicroAPI::RegTensor<fp8_e4m3fn_t> quantFloat8E4m3fn;
        AscendC::MicroAPI::Cast<fp8_e4m3fn_t, float, CAST_FP32_TO_FLOAT8>(quantFloat8E4m3fn, src, preg);
        Pack((MicroAPI::RegTensor<uint16_t> &)quantFloat8E4m3fnPack,
             (MicroAPI::RegTensor<uint32_t> &)quantFloat8E4m3fn);
        Pack((MicroAPI::RegTensor<uint8_t> &)quantFloat8E4m3fn, (MicroAPI::RegTensor<uint16_t> &)quantFloat8E4m3fnPack);
        StoreUnAlign(output, quantFloat8E4m3fn, uValue, postUpdateStride);
    } else {
        StoreUnAlign(output, src, uValue, postUpdateStride);
    }
}

/*
gm->ub拷入
lineNum：拷入的行数
srcLineLength：原Gm中数据一行的长度
copyLineLength：从原Gm中实际需要拷入的长度
拷入完成后dstUb中的数据为lineNum*copyLineLengthPadAlign
*/
template <typename T>
__aicore__ inline void CopyInLineAlign(const LocalTensor<T> &dstUb, const GlobalTensor<T> &srcGm, int64_t lineNum,
                                       int64_t srcLineLength, int64_t copyLineLength)
{
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = lineNum;
    copyInParams.blockLen = copyLineLength * sizeof(T);
    copyInParams.srcStride = (srcLineLength - copyLineLength) * sizeof(T);
    copyInParams.dstStride = 0;
    DataCopyPad(dstUb, srcGm, copyInParams, padParams);
}

/*
ub->gm拷出
lineNum：拷入的行数
dstLineLength：Gm中实际一行的长度
copyLineLength：从原ub中实际需要拷入Gm的长度
拷入完成后dstGm中的数据为lineNum*copyLineLengthPadAlign
*/
template <typename T>
__aicore__ inline void CopyLineAlignOut(const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t lineNum,
                                        int64_t dstLineLength, int64_t copyLineLength)
{
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = lineNum;
    copyOutParams.blockLen = copyLineLength * sizeof(T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = (dstLineLength - copyLineLength) * sizeof(T);
    DataCopyPad(dstGm, srcUb, copyOutParams);
}

template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormPostWithMul(
    const LocalTensor<float> &dstTensor, const LocalTensor<T_KV> &xTensor, const LocalTensor<T_KV> &grammaTensor,
    const int64_t aSize, const int64_t rSize, const int64_t stride)
{
    if (aSize <= 0) {
        return;
    }
    if (rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;

    if (rSize <= VL_FP32) {
        __ubuf__ float *dst = (__ubuf__ float *)dstTensor.GetPhyAddr();
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)grammaTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 行
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadTensorForDtypeT<T_KV>(x, reg0, pMask, i * stride);
                LoadTensorForDtypeT<T_KV>(gamma, reg1, pMask, 0);
                AscendC::MicroAPI::Mul(reg2, reg0, reg0, pMask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg2, reg2, pMask);
                AscendC::MicroAPI::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg4, reg3, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);
                AscendC::MicroAPI::Div(reg6, reg0, reg5, pMask);
                AscendC::MicroAPI::Mul(reg7, reg1, reg6, pMask);
                StoreTensorForDtypeTOut<float>(dst, reg7, pMask, i * stride);
            }
        }
    } else {
        __ubuf__ float *dst = (__ubuf__ float *)dstTensor.GetPhyAddr();
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *x_1 = (__ubuf__ T_KV *)xTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)grammaTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma_1 = (__ubuf__ T_KV *)grammaTensor.GetPhyAddr() + VL_FP32;
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg0_1, reg1_1, reg2, reg2_1;
            AscendC::MicroAPI::RegTensor<float> reg3, reg4, reg5, reg6, reg7, reg8, reg9;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> regqunat12, regqunat12_1;
            AscendC::MicroAPI::RegTensor<T_KV> reg12, reg12_1;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 行循环
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadTensorForDtypeT<T_KV>(x, reg0, pFull, i * stride);
                LoadTensorForDtypeT<T_KV>(x_1, reg0_1, pMask, i * stride);
                LoadTensorForDtypeT<T_KV>(gamma, reg1, pFull, 0);
                LoadTensorForDtypeT<T_KV>(gamma_1, reg1_1, pMask, 0);
                AscendC::MicroAPI::Mul(reg2, reg0, reg0, pFull);
                AscendC::MicroAPI::Mul(reg2_1, reg0_1, reg0_1, pMask);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg2_1, reg2, reg2_1, pMask);
                AscendC::MicroAPI::Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg2, reg2_1, pMask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg2, reg2, pFull);

                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                AscendC::MicroAPI::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg4, reg3, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);
                // 因为输入分为两段，所以这里也要分为两段
                AscendC::MicroAPI::Div(reg6, reg0, reg5, pFull);
                AscendC::MicroAPI::Mul(reg7, reg1, reg6, pFull);
                StoreTensorForDtypeTOut<float>(dst, reg7, pFull, i * stride);
                // 剩余的一部分
                AscendC::MicroAPI::Div(reg8, reg0_1, reg5, pMask);
                AscendC::MicroAPI::Mul(reg9, reg1_1, reg8, pMask);
                StoreTensorForDtypeTOut<float>(dst, reg9, pMask, i * stride + VL_FP32);
            }
        }
    }
}

template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormBasicComputeVF(
    const LocalTensor<T_KV> &xLocal, const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &wsLocal,
    int64_t aSize, int64_t &foldPoint, int64_t &outerLoopDstStride)
{
    int64_t rSize = dv;
    int64_t stride = dvAlign;
    if (aSize <= 0 || rSize <= 0) {
        return;
    }

    int64_t ceilVLCount =
        ops::CeilDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    int64_t floorVLCount =
        ops::FloorDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    foldPoint = FindNearestPower2(ceilVLCount);

    uint16_t outerLoopTimes = aSize;
    uint16_t tailFoldLoopTimes = ceilVLCount - floorVLCount;
    uint32_t tailFoldElemCount = static_cast<uint32_t>(rSize - floorVLCount * VL_FP32);
    uint16_t mainFoldLoopTimes = floorVLCount - foldPoint;
    uint16_t unFoldLoopTimes = foldPoint + foldPoint - ceilVLCount;
    uint32_t outerLoopStride = stride;
    uint32_t innerLoopStride = VL_FP32;
    outerLoopDstStride =
        ops::Aligned(static_cast<int64_t>(foldPoint), static_cast<int64_t>(platform::GetUbBlockSize() / sizeof(float)));

    int64_t foldSrcBOffset = foldPoint * VL_FP32;
    int64_t tailSrcAOffset = mainFoldLoopTimes * VL_FP32;
    int64_t tailSrcBOffset = floorVLCount * VL_FP32;
    int64_t unFoldSrcOffset = (mainFoldLoopTimes + tailFoldLoopTimes) * VL_FP32;

    __ubuf__ float *dst = (__ubuf__ float *)wsLocal.GetPhyAddr();

    __ubuf__ T_KV *foldSrcX0A = (__ubuf__ T_KV *)xLocal.GetPhyAddr();
    __ubuf__ T_KV *foldSrcX0B = (__ubuf__ T_KV *)xLocal.GetPhyAddr() + foldSrcBOffset;
    __ubuf__ T_KV *tailSrcX0A = (__ubuf__ T_KV *)xLocal.GetPhyAddr() + tailSrcAOffset;
    __ubuf__ T_KV *tailSrcX0B = (__ubuf__ T_KV *)xLocal.GetPhyAddr() + tailSrcBOffset;
    __ubuf__ T_KV *unFoldX0 = (__ubuf__ T_KV *)xLocal.GetPhyAddr() + unFoldSrcOffset;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg pFull = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignRegForStore UReg;
        // 每一行循环一次
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            dst = (__ubuf__ float *)wsLocal.GetPhyAddr() + i * outerLoopDstStride;
            // 完全能折叠
            for (uint16_t j = 0; j < mainFoldLoopTimes; ++j) {
                AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg2_1;
                LoadTensorForDtypeT<T_KV>(foldSrcX0A, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeT<T_KV>(foldSrcX0B, reg1, pFull, i * outerLoopStride + j * innerLoopStride);

                AscendC::MicroAPI::Mul(reg2, reg0, reg0, pFull);
                AscendC::MicroAPI::Mul(reg2_1, reg1, reg1, pFull);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg2, reg2, reg2_1, pFull);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg2, reg2, pFull);
                AscendC::MicroAPI::StoreUnAlign((__ubuf__ float *&)dst, reg2, UReg, 1);
            }
            // 一个能折叠，另外一个短了
            for (uint16_t j = 0; j < tailFoldLoopTimes; ++j) {
                uint32_t count = static_cast<uint32_t>(tailFoldElemCount);
                AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg2_1;
                AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                LoadTensorForDtypeT<T_KV>(tailSrcX0A, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeT<T_KV>(tailSrcX0B, reg1, pMask, i * outerLoopStride + j * innerLoopStride);
                AscendC::MicroAPI::Mul(reg2, reg0, reg0, pFull);
                AscendC::MicroAPI::Mul(reg2_1, reg1, reg1, pMask);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg2_1, reg2, reg2_1, pMask);
                AscendC::MicroAPI::Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg2, reg2_1, pMask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg2, reg2, pFull);
                AscendC::MicroAPI::StoreUnAlign((__ubuf__ float *&)dst, reg2, UReg, 1);
            }
            // 无法折叠
            for (uint16_t j = 0; j < unFoldLoopTimes; ++j) {
                AscendC::MicroAPI::RegTensor<float> reg0, reg1;
                LoadTensorForDtypeT<T_KV>(unFoldX0, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                AscendC::MicroAPI::Mul(reg1, reg0, reg0, pFull);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg1, reg1, pFull);
                AscendC::MicroAPI::StoreUnAlign((__ubuf__ float *&)dst, reg1, UReg, 1);
            }
            AscendC::MicroAPI::StoreUnAlignPost((__ubuf__ float *&)dst, UReg, 0);
        }
    }
}

// 普通场景
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormPostVF(
    const LocalTensor<T_KV> &dstTensor, const LocalTensor<T_KV> &xTensor, const LocalTensor<T_KV> &gammaLocal,
    const LocalTensor<float> &reduceSumTempTensor, const int64_t aSize, const int64_t rSize, const int64_t stride)
{
    if (aSize <= 0 || rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    uint16_t rLoopCount = dvLoopCount;
    uint32_t oriR = static_cast<uint32_t>(dv);
    uint32_t oriRAligned = static_cast<uint32_t>(dvAlign);

    if (rSize <= VL_FP32) {
        __ubuf__ T_KV *dst = (__ubuf__ T_KV *)dstTensor.GetPhyAddr();
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
        __ubuf__ float *sumTmp = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 每行循环一次
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(reg0, (__ubuf__ float *)sumTmp + i * stride);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg1, reg0, pMask);
                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                AscendC::MicroAPI::Muls(reg2, reg1, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg3, reg2, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg4, reg3, pFull);
                Duplicate(reg5, reg4, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::MicroAPI::Div(reg6, reg1, reg5, maskOri);
                    // gramma只有一行，每次都是取一样的数值
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(reg8, reg7, reg6, maskOri);
                    StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);
                }
            }
        }
    } else {
        __ubuf__ T_KV *dst = (__ubuf__ T_KV *)dstTensor.GetPhyAddr();
        __ubuf__ float *sumTmpA = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float *sumTmpB = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 行
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(reg0, (__ubuf__ float *)sumTmpA + i * stride);
                LoadAlign(reg1, (__ubuf__ float *)sumTmpB + i * stride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
                AscendC::MicroAPI::Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, reg1, pMask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg2, reg0, pFull);
                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                // reducesum后的结果就一个数，所以这几个都是一个数在做计算
                AscendC::MicroAPI::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg4, reg3, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::MicroAPI::Div(reg6, reg1, reg5, maskOri);
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(reg8, reg7, reg6, maskOri);
                    StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);
                }
            }
        }
    }
}
// 需要输出中间结果：scale + offset
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormAsymQuantWithKvPostVF(
    const LocalTensor<T_KV> &dstTensor, const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xTensor,
    const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &reduceSumTempTensor,
    const LocalTensor<float> &vScaleTensor, const LocalTensor<float> &vOffsetTensor, const int64_t aSize,
    const int64_t rSize, const int64_t stride)
{
    if (aSize <= 0 || rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    uint16_t rLoopCount = dvLoopCount;
    uint32_t oriR = static_cast<uint32_t>(dv);
    uint32_t oriRAligned = static_cast<uint32_t>(dvAlign);
    uint32_t quantOutOffset = static_cast<uint32_t>(dvB8Align);
    if (rSize <= VL_FP32) {
        __ubuf__ T_KV *dst = (__ubuf__ T_KV *)dstTensor.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ float *vOffset = (__ubuf__ float *)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
        __ubuf__ float *sumTmp = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::MicroAPI::RegTensor<float> scale, offset;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;

            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 每行循环一次
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(reg0, (__ubuf__ float *)sumTmp + i * stride);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg1, reg0, pMask);
                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                AscendC::MicroAPI::Muls(reg2, reg1, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg3, reg2, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg4, reg3, pFull);
                Duplicate(reg5, reg4, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::MicroAPI::Div(reg6, reg1, reg5, maskOri);
                    // gramma只有一行，每次都是取一样的数值
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(reg8, reg7, reg6, maskOri);
                    StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);
                    LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
                    LoadTensorForDtypeT<float>(vOffset, offset, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, reg8, scale, maskOri);
                    AscendC::MicroAPI::Add(offset, offset, scale, maskOri);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, maskOri);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, maskOri);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
                }
            }
        }
    } else {
        __ubuf__ T_KV *dst = (__ubuf__ T_KV *)dstTensor.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ float *vOffset = (__ubuf__ float *)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __ubuf__ float *sumTmpA = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float *sumTmpB = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::MicroAPI::RegTensor<float> scale, offset;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 行
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(reg0, (__ubuf__ float *)sumTmpA + i * stride);
                LoadAlign(reg1, (__ubuf__ float *)sumTmpB + i * stride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
                AscendC::MicroAPI::Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, reg1, pMask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg2, reg0, pFull);
                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                // reducesum后的结果就一个数，所以这几个都是一个数在做计算
                AscendC::MicroAPI::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg4, reg3, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::MicroAPI::Div(reg6, reg1, reg5, maskOri);
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(reg8, reg7, reg6, maskOri);
                    StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);
                    LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
                    LoadTensorForDtypeT<float>(vOffset, offset, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, reg8, scale, maskOri);
                    AscendC::MicroAPI::Add(offset, offset, scale, maskOri);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, maskOri);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, maskOri);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
                }
            }
        }
    }
}
// 需要输出中间结果：scale
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormSymQuantWithKvPostVF(
    const LocalTensor<T_KV> &dstTensor, const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xTensor,
    const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &reduceSumTempTensor,
    const LocalTensor<float> &vScaleTensor, const int64_t aSize, const int64_t rSize, const int64_t stride)
{
    if (aSize <= 0 || rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    uint16_t rLoopCount = dvLoopCount;
    uint32_t oriR = static_cast<uint32_t>(dv);
    uint32_t oriRAligned = static_cast<uint32_t>(dvAlign);
    uint32_t quantOutOffset = static_cast<uint32_t>(dvB8Align);
    if (rSize <= VL_FP32) {
        __ubuf__ T_KV *dst = (__ubuf__ T_KV *)dstTensor.GetPhyAddr();
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
        __ubuf__ float *sumTmp = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::MicroAPI::RegTensor<float> scale;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;

            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 每行循环一次
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(reg0, (__ubuf__ float *)sumTmp + i * stride);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg1, reg0, pMask);
                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                AscendC::MicroAPI::Muls(reg2, reg1, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg3, reg2, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg4, reg3, pFull);
                Duplicate(reg5, reg4, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::MicroAPI::Div(reg6, reg1, reg5, maskOri);
                    // gramma只有一行，每次都是取一样的数值
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(reg8, reg7, reg6, maskOri);
                    StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);

                    LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, reg8, scale, maskOri);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, maskOri);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, maskOri);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
                }
            }
        }
    } else {
        __ubuf__ T_KV *dst = (__ubuf__ T_KV *)dstTensor.GetPhyAddr();
        __ubuf__ float *sumTmpA = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float *sumTmpB = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::MicroAPI::RegTensor<float> scale;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 行
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(reg0, (__ubuf__ float *)sumTmpA + i * stride);
                LoadAlign(reg1, (__ubuf__ float *)sumTmpB + i * stride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
                AscendC::MicroAPI::Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, reg1, pMask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg2, reg0, pFull);
                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                // reducesum后的结果就一个数，所以这几个都是一个数在做计算
                AscendC::MicroAPI::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg4, reg3, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::MicroAPI::Div(reg6, reg1, reg5, maskOri);
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(reg8, reg7, reg6, maskOri);
                    StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);

                    LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, reg8, scale, maskOri);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, maskOri);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, maskOri);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
                }
            }
        }
    }
}

template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormAsymQuantPostVF(
    const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xTensor, const LocalTensor<T_KV> &gammaLocal,
    const LocalTensor<float> &reduceSumTempTensor, const LocalTensor<float> &vScaleTensor,
    const LocalTensor<float> &vOffsetTensor, const int64_t aSize, const int64_t rSize, const int64_t stride)
{
    if (aSize <= 0 || rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    uint16_t rLoopCount = dvLoopCount;
    uint32_t oriR = static_cast<uint32_t>(dv);
    uint32_t oriRAligned = static_cast<uint32_t>(dvAlign);
    uint32_t quantOutOffset = static_cast<uint32_t>(dvB8Align);
    if (rSize <= VL_FP32) {
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
        __ubuf__ float *sumTmp = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ float *vOffset = (__ubuf__ float *)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::MicroAPI::RegTensor<float> scale, offset;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 每行循环一次
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(reg0, (__ubuf__ float *)sumTmp + i * stride);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg1, reg0, pMask);
                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                AscendC::MicroAPI::Muls(reg2, reg1, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg3, reg2, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg4, reg3, pFull);
                Duplicate(reg5, reg4, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::MicroAPI::Div(reg6, reg1, reg5, maskOri);
                    // gramma只有一行，每次都是取一样的数值
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(reg8, reg7, reg6, maskOri);
                    LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
                    LoadTensorForDtypeT<float>(vOffset, offset, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, reg8, scale, maskOri);
                    AscendC::MicroAPI::Add(offset, offset, scale, maskOri);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, maskOri);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, maskOri);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
                }
            }
        }
    } else {
        __ubuf__ float *sumTmpA = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float *sumTmpB = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ float *vOffset = (__ubuf__ float *)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::MicroAPI::RegTensor<float> scale, offset;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 行
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(reg0, (__ubuf__ float *)sumTmpA + i * stride);
                LoadAlign(reg1, (__ubuf__ float *)sumTmpB + i * stride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
                AscendC::MicroAPI::Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, reg1, pMask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg2, reg0, pFull);
                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                // reducesum后的结果就一个数，所以这几个都是一个数在做计算
                AscendC::MicroAPI::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg4, reg3, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::MicroAPI::Div(reg6, reg1, reg5, maskOri);
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(reg8, reg7, reg6, maskOri);
                    LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
                    LoadTensorForDtypeT<float>(vOffset, offset, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, reg8, scale, maskOri);
                    AscendC::MicroAPI::Add(offset, offset, scale, maskOri);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, maskOri);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, maskOri);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
                }
            }
        }
    }
}

template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormSymQuantPostVF(
    const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xTensor, const LocalTensor<T_KV> &gammaLocal,
    const LocalTensor<float> &reduceSumTempTensor, const LocalTensor<float> &vScaleTensor, const int64_t aSize,
    const int64_t rSize, const int64_t stride)
{
    if (aSize <= 0 || rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    uint16_t rLoopCount = dvLoopCount;
    uint32_t oriR = static_cast<uint32_t>(dv);
    uint32_t oriRAligned = static_cast<uint32_t>(dvAlign);
    uint32_t quantOutOffset = static_cast<uint32_t>(dvB8Align);
    if (rSize <= VL_FP32) {
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
        __ubuf__ float *sumTmp = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::MicroAPI::RegTensor<float> scale;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 每行循环一次
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(reg0, (__ubuf__ float *)sumTmp + i * stride);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg1, reg0, pMask);
                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                AscendC::MicroAPI::Muls(reg2, reg1, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg3, reg2, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg4, reg3, pFull);
                Duplicate(reg5, reg4, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::MicroAPI::Div(reg6, reg1, reg5, maskOri);
                    // gramma只有一行，每次都是取一样的数值
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(reg8, reg7, reg6, maskOri);
                    LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, reg8, scale, maskOri);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, maskOri);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, maskOri);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
                }
            }
        }
    } else {
        __ubuf__ float *sumTmpA = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float *sumTmpB = (__ubuf__ float *)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV *x = (__ubuf__ T_KV *)xTensor.GetPhyAddr();
        __ubuf__ T_KV *gamma = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::MicroAPI::RegTensor<float> scale;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            // 行
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(reg0, (__ubuf__ float *)sumTmpA + i * stride);
                LoadAlign(reg1, (__ubuf__ float *)sumTmpB + i * stride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
                AscendC::MicroAPI::Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, reg1, pMask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(reg2, reg0, pFull);
                // Calc: xSum = xSum * reciprocal
                // Calc: xSum = xSum + epsilon
                // Calc: xSum = sqrt(xSum)
                // reducesum后的结果就一个数，所以这几个都是一个数在做计算
                AscendC::MicroAPI::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::MicroAPI::Adds(reg4, reg3, epsilon, pFull);
                AscendC::MicroAPI::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::MicroAPI::Div(reg6, reg1, reg5, maskOri);
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(reg8, reg7, reg6, maskOri);
                    LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, reg8, scale, maskOri);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, maskOri);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, maskOri);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, maskOri);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
                }
            }
        }
    }
}

// 普通场景
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormVF(
    const LocalTensor<T_KV> &outVLocal, const LocalTensor<T_KV> &xLocal, const LocalTensor<T_KV> &gammaLocal,
    const LocalTensor<float> &wsLocal, int64_t calcRow)
{
    int64_t stride = dvAlign;
    uint16_t row = static_cast<uint16_t>(calcRow);
    uint32_t actualCol = static_cast<uint32_t>(dv);
    uint32_t alignedCol = static_cast<uint32_t>(dvAlign);
    uint16_t colLoopCount = static_cast<uint16_t>(ops::CeilDiv(static_cast<uint32_t>(actualCol), VL_FP32));
    if (row <= 0 || actualCol <= 0) {
        return;
    }
    if (actualCol <= CONST_TWO * VL_FP32) {
        RmsNormPostWithMul(wsLocal, xLocal, gammaLocal, row, actualCol, stride);
        __ubuf__ T_KV *dst = (__ubuf__ T_KV *)outVLocal.GetPhyAddr();
        __ubuf__ float *ws = (__ubuf__ float *)wsLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x;
            AscendC::MicroAPI::MaskReg mask;
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = actualCol;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
                    StoreTensorForDtypeTOut<T_KV>(dst, x, mask, i * VL_FP32 + rowId * alignedCol);
                }
            }
        }
        return;
    }
    int64_t foldPoint = 0;
    int64_t outerLoopDstStride = 0;
    RmsNormBasicComputeVF(xLocal, gammaLocal, wsLocal, calcRow, foldPoint, outerLoopDstStride);
    RmsNormPostVF(outVLocal, xLocal, gammaLocal, wsLocal, row, foldPoint, outerLoopDstStride);
}

// 需要输出中间结果：scale + offset
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormAsymQuantWithKvVF(
    const LocalTensor<T_KV> &outVLocal, const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xLocal,
    const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &wsLocal, const LocalTensor<float> &vScaleTensor,
    const LocalTensor<float> &vOffsetTensor, int64_t calcRow)
{
    int64_t stride = dvAlign;
    uint16_t row = static_cast<uint16_t>(calcRow);
    uint32_t actualCol = static_cast<uint32_t>(dv);
    uint32_t alignedCol = static_cast<uint32_t>(dvAlign);
    uint32_t quantOutOffset = static_cast<uint32_t>(dvB8Align);
    uint16_t colLoopCount = static_cast<uint16_t>(ops::CeilDiv(static_cast<uint32_t>(actualCol), VL_FP32));
    if (row <= 0 || actualCol <= 0) {
        return;
    }
    if (actualCol <= CONST_TWO * VL_FP32) {
        RmsNormPostWithMul(wsLocal, xLocal, gammaLocal, row, actualCol, stride);
        __ubuf__ T_KV *dst = (__ubuf__ T_KV *)outVLocal.GetPhyAddr();
        __ubuf__ float *ws = (__ubuf__ float *)wsLocal.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ float *vOffset = (__ubuf__ float *)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x, out, scale, offset;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;
            AscendC::MicroAPI::MaskReg mask;
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = actualCol;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
                    StoreTensorForDtypeTOut<T_KV>(dst, x, mask, i * VL_FP32 + rowId * alignedCol);
                    LoadTensorForDtypeT<float>(vScale, scale, mask, i * VL_FP32);
                    LoadTensorForDtypeT<float>(vOffset, offset, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, x, scale, mask);
                    AscendC::MicroAPI::Add(offset, offset, scale, mask);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, mask);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, mask);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, mask);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, mask);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, mask);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + i * VL_FP32 + rowId * quantOutOffset, quant, mask);
                }
            }
        }
        return;
    }
    int64_t foldPoint = 0;
    int64_t outerLoopDstStride = 0;
    this->RmsNormBasicComputeVF(xLocal, gammaLocal, wsLocal, calcRow, foldPoint, outerLoopDstStride);
    this->RmsNormAsymQuantWithKvPostVF(outVLocal, quantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor, vOffsetTensor,
                                       row, foldPoint, outerLoopDstStride);
}

// 需要输出中间结果：量化
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormSymQuantWithKvVF(
    const LocalTensor<T_KV> &outVLocal, const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xLocal,
    const LocalTensor<T_KV> &gammaLocal, const LocalTensor<float> &wsLocal, const LocalTensor<float> &vScaleTensor,
    int64_t calcRow)
{
    int64_t stride = dvAlign;
    uint16_t row = static_cast<uint16_t>(calcRow);
    uint32_t actualCol = static_cast<uint32_t>(dv);
    uint16_t alignedCol = static_cast<uint16_t>(dvAlign);
    uint16_t quantOutOffset = static_cast<uint16_t>(dvB8Align);
    uint16_t colLoopCount = static_cast<uint16_t>(ops::CeilDiv(static_cast<uint32_t>(actualCol), VL_FP32));
    if (row <= 0 || actualCol <= 0) {
        return;
    }
    if (actualCol <= CONST_TWO * VL_FP32) {
        RmsNormPostWithMul(wsLocal, xLocal, gammaLocal, row, actualCol, stride);
        __ubuf__ T_KV *dst = (__ubuf__ T_KV *)outVLocal.GetPhyAddr();
        __ubuf__ float *ws = (__ubuf__ float *)wsLocal.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x, scale, offset;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;
            AscendC::MicroAPI::MaskReg mask;
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = actualCol;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
                    StoreTensorForDtypeTOut<T_KV>(dst, x, mask, i * VL_FP32 + rowId * alignedCol);
                    LoadTensorForDtypeT<float>(vScale, scale, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, x, scale, mask);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, mask);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, mask);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, mask);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, mask);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, mask);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + i * VL_FP32 + rowId * quantOutOffset, quant, mask);
                }
            }
        }
        return;
    }
    int64_t foldPoint = 0;
    int64_t outerLoopDstStride = 0;
    this->RmsNormBasicComputeVF(xLocal, gammaLocal, wsLocal, calcRow, foldPoint, outerLoopDstStride);
    this->RmsNormSymQuantWithKvPostVF(outVLocal, quantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor, row, foldPoint,
                                      outerLoopDstStride);
}

// 不需要输出中间结果：scale + offset
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormAsymQuantVF(
    const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xLocal, const LocalTensor<T_KV> &gammaLocal,
    const LocalTensor<float> &wsLocal, const LocalTensor<float> &vScaleTensor, const LocalTensor<float> &vOffsetTensor,
    int64_t calcRow)
{
    int64_t stride = dvAlign;
    uint16_t row = static_cast<uint16_t>(calcRow);
    uint32_t actualCol = static_cast<uint32_t>(dv);
    uint16_t alignedCol = static_cast<uint16_t>(dvAlign);
    uint16_t quantOutOffset = static_cast<uint16_t>(dvB8Align);
    uint16_t colLoopCount = static_cast<uint16_t>(ops::CeilDiv(static_cast<uint32_t>(actualCol), VL_FP32));
    if (row <= 0 || actualCol <= 0) {
        return;
    }
    if (actualCol <= CONST_TWO * VL_FP32) {
        RmsNormPostWithMul(wsLocal, xLocal, gammaLocal, row, actualCol, stride);
        __ubuf__ float *ws = (__ubuf__ float *)wsLocal.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ float *vOffset = (__ubuf__ float *)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x, out, scale, offset;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;
            AscendC::MicroAPI::MaskReg mask;
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = actualCol;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
                    LoadTensorForDtypeT<float>(vScale, scale, mask, i * VL_FP32);
                    LoadTensorForDtypeT<float>(vOffset, offset, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, x, scale, mask);
                    AscendC::MicroAPI::Add(offset, offset, scale, mask);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, mask);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, mask);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, mask);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, mask);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, mask);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + i * VL_FP32 + rowId * quantOutOffset, quant, mask);
                }
            }
        }
        return;
    }
    int64_t foldPoint = 0;
    int64_t outerLoopDstStride = 0;
    this->RmsNormBasicComputeVF(xLocal, gammaLocal, wsLocal, calcRow, foldPoint, outerLoopDstStride);
    this->RmsNormAsymQuantPostVF(quantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor, vOffsetTensor, row, foldPoint,
                                 outerLoopDstStride);
}

// 不需要中间结果：offset
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormSymQuantVF(
    const LocalTensor<T_V_CACHE> &quantLocal, const LocalTensor<T_KV> &xLocal, const LocalTensor<T_KV> &gammaLocal,
    const LocalTensor<float> &wsLocal, const LocalTensor<float> &vScaleTensor, int64_t calcRow)
{
    int64_t stride = dvAlign;
    uint16_t row = static_cast<uint16_t>(calcRow);
    uint32_t actualCol = static_cast<uint32_t>(dv);
    uint16_t alignedCol = static_cast<uint16_t>(dvAlign);
    uint16_t quantOutOffset = static_cast<uint16_t>(dvB8Align);
    uint16_t colLoopCount = static_cast<uint16_t>(ops::CeilDiv(static_cast<uint32_t>(actualCol), VL_FP32));
    if (row <= 0 || actualCol <= 0) {
        return;
    }
    if (actualCol <= CONST_TWO * VL_FP32) {
        RmsNormPostWithMul(wsLocal, xLocal, gammaLocal, row, actualCol, stride);
        __ubuf__ float *ws = (__ubuf__ float *)wsLocal.GetPhyAddr();
        __ubuf__ float *vScale = (__ubuf__ float *)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE *vQuant = (__ubuf__ T_V_CACHE *)quantLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x, out, scale, offset;
            AscendC::MicroAPI::RegTensor<half> tmp;
            AscendC::MicroAPI::RegTensor<int16_t> tmp_int16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> quant;
            AscendC::MicroAPI::MaskReg mask;
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = actualCol;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
                    LoadTensorForDtypeT<float>(vScale, scale, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale, x, scale, mask);

                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                        AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, mask);
                        AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, mask);
                        AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, mask);
                    } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, mask);
                    } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                         IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::MicroAPI::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, mask);
                    }
                    AscendC::MicroAPI::StoreAlign<T_V_CACHE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        vQuant + i * VL_FP32 + rowId * quantOutOffset, quant, mask);
                }
            }
        }
        return;
    }
    int64_t foldPoint = 0;
    int64_t outerLoopDstStride = 0;
    this->RmsNormBasicComputeVF(xLocal, gammaLocal, wsLocal, calcRow, foldPoint, outerLoopDstStride);
    this->RmsNormSymQuantPostVF(quantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor, row, foldPoint,
                                outerLoopDstStride);
}

} // namespace KvRmsNormRopeCache
#endif // KV_RMS_NORM_ROPE_CACHE_REGBASE_BASE_H
