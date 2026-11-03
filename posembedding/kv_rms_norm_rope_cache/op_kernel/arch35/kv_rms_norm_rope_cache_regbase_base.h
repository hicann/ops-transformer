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
#include "kv_rms_norm_rope_cache_regbase_vf.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;

constexpr int64_t BLOCK_SIZE = 32;
// Constants for buffer counts to replace magic numbers
constexpr int32_t BUFFER_COUNT_SINGLE = 1;
constexpr int32_t BUFFER_COUNT_DOUBLE = 2;

// Constants for channel counts in buffer size calculations to replace magic numbers
constexpr int32_t COS_SIN_CHANNEL_COUNT = 4;        // Real/Imaginary for Cos/Sin
constexpr int32_t K_SCALE_OFFSET_CHANNEL_COUNT = 4; // Real/Imaginary for Scale/Offset (K)
constexpr int32_t V_SCALE_OFFSET_CHANNEL_COUNT = 2; // Scale/Offset (V)
constexpr int32_t WORKSPACE_FLOAT_VECTOR_COUNT = 2; // Number of float vectors in workspace

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
    __aicore__ inline void RmsNormBasicComputeVF(const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal,
                                                 const LocalTensor<float>& wsLocal, int64_t aSize, int64_t& foldPoint,
                                                 int64_t& outerLoopDstStride);
    __aicore__ inline void RmsNormPostVF(const LocalTensor<T_KV>& dstTensor, const LocalTensor<T_KV>& xTensor,
                                         const LocalTensor<T_KV>& gammaLocal,
                                         const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize,
                                         const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormAsymQuantWithKvPostVF(
        const LocalTensor<T_KV>& dstTensor, const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xTensor,
        const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& reduceSumTempTensor,
        const LocalTensor<float>& vScaleTensor, const LocalTensor<float>& vOffsetTensor, const int64_t aSize,
        const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormSymQuantWithKvPostVF(
        const LocalTensor<T_KV>& dstTensor, const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xTensor,
        const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& reduceSumTempTensor,
        const LocalTensor<float>& vScaleTensor, const int64_t aSize, const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormAsymQuantPostVF(const LocalTensor<T_V_CACHE>& quantLocal,
                                                  const LocalTensor<T_KV>& xTensor, const LocalTensor<T_KV>& gammaLocal,
                                                  const LocalTensor<float>& reduceSumTempTensor,
                                                  const LocalTensor<float>& vScaleTensor,
                                                  const LocalTensor<float>& vOffsetTensor, const int64_t aSize,
                                                  const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormSymQuantPostVF(const LocalTensor<T_V_CACHE>& quantLocal,
                                                 const LocalTensor<T_KV>& xTensor, const LocalTensor<T_KV>& gammaLocal,
                                                 const LocalTensor<float>& reduceSumTempTensor,
                                                 const LocalTensor<float>& vScaleTensor, const int64_t aSize,
                                                 const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormVF(const LocalTensor<T_KV>& outVLocal, const LocalTensor<T_KV>& xLocal,
                                     const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& wsLocal,
                                     int64_t calcRow);
    __aicore__ inline void RmsNormAsymQuantWithKvVF(
        const LocalTensor<T_KV>& outVLocal, const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xLocal,
        const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& wsLocal, const LocalTensor<float>& vScaleTensor,
        const LocalTensor<float>& vOffsetTensor, int64_t calcRow);
    __aicore__ inline void RmsNormSymQuantWithKvVF(const LocalTensor<T_KV>& outVLocal,
                                                   const LocalTensor<T_V_CACHE>& quantLocal,
                                                   const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal,
                                                   const LocalTensor<float>& wsLocal,
                                                   const LocalTensor<float>& vScaleTensor, int64_t calcRow);
    __aicore__ inline void RmsNormAsymQuantVF(const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xLocal,
                                              const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& wsLocal,
                                              const LocalTensor<float>& vScaleTensor,
                                              const LocalTensor<float>& vOffsetTensor, int64_t calcRow);
    __aicore__ inline void RmsNormSymQuantVF(const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xLocal,
                                             const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& wsLocal,
                                             const LocalTensor<float>& vScaleTensor, int64_t calcRow);
    __aicore__ inline void RmsNormPostWithMul(const LocalTensor<float>& dstTensor, const LocalTensor<T_KV>& xTensor,
                                              const LocalTensor<T_KV>& grammaTensor, const int64_t aSize,
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

/*
gm->ub拷入
lineNum：拷入的行数
srcLineLength：原Gm中数据一行的长度
copyLineLength：从原Gm中实际需要拷入的长度
拷入完成后dstUb中的数据为lineNum*copyLineLengthPadAlign
*/
template <typename T>
__aicore__ inline void CopyInLineAlign(const LocalTensor<T>& dstUb, const GlobalTensor<T>& srcGm, int64_t lineNum,
                                       int64_t srcLineLength, int64_t copyLineLength)
{
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = lineNum;
    copyInParams.blockLen = copyLineLength * sizeof(T);
    copyInParams.srcStride = (srcLineLength - copyLineLength) * sizeof(T);
    copyInParams.dstStride = 0;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    DataCopyPad(dstUb, srcGm, copyInParams, padParams, false);
#else
    DataCopyPad(dstUb, srcGm, copyInParams, padParams);
#endif
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
template <typename T>
__aicore__ inline void BroadcastQuantParamNddma(const LocalTensor<T>& dstUb, const GlobalTensor<T>& srcGm,
                                                uint32_t broadcastCount)
{
    constexpr uint8_t NDDMA_DIM = 1;
    MultiCopyParams<T, NDDMA_DIM> params{};
    params.constantValue = static_cast<T>(0);
    params.loopInfo.loopSize[0] = broadcastCount;
    params.loopInfo.loopSrcStride[0] = 0;
    params.loopInfo.loopDstStride[0] = 1;
    params.loopInfo.loopLpSize[0] = 0;
    params.loopInfo.loopRpSize[0] = 0;
    AscendC::NdDmaDci();
    DataCopy<T, NDDMA_DIM, kDefaultNdDmaConfig>(dstUb, srcGm, params, false);
}
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
template <typename T>
__aicore__ inline void CopyInPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                 const DataCopyParams& copyParams, const DataCopyPadParams& padParams,
                                 bool enablePreAllocation)
{
    DataCopyPad(dst, src, copyParams, padParams, enablePreAllocation);
}
#else
template <typename T>
__aicore__ inline void CopyInPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                 const DataCopyParams& copyParams, const DataCopyPadParams& padParams, bool)
{
    DataCopyPad(dst, src, copyParams, padParams);
}
#endif

/*
ub->gm拷出
lineNum：拷入的行数
dstLineLength：Gm中实际一行的长度
copyLineLength：从原ub中实际需要拷入Gm的长度
拷入完成后dstGm中的数据为lineNum*copyLineLengthPadAlign
*/
template <typename T>
__aicore__ inline void CopyLineAlignOut(const GlobalTensor<T>& dstGm, const LocalTensor<T>& srcUb, int64_t lineNum,
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
    const LocalTensor<float>& dstTensor, const LocalTensor<T_KV>& xTensor, const LocalTensor<T_KV>& grammaTensor,
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
        __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)grammaTensor.GetPhyAddr();
        asc_vf_call<RmsNormPostWithMulSingleVF<T_KV>>(dst, x, gamma, loopTimes, rSize, stride, reciprocal, epsilon);
    } else {
        __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* x_1 = (__ubuf__ T_KV*)xTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)grammaTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma_1 = (__ubuf__ T_KV*)grammaTensor.GetPhyAddr() + VL_FP32;
        asc_vf_call<RmsNormPostWithMulDoubleVF<T_KV>>(dst, x, x_1, gamma, gamma_1, loopTimes, rSize, stride, reciprocal,
                                                      epsilon);
    }
}

template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormBasicComputeVF(
    const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& wsLocal,
    int64_t aSize, int64_t& foldPoint, int64_t& outerLoopDstStride)
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

    __ubuf__ float* wsBase = (__ubuf__ float*)wsLocal.GetPhyAddr();

    __ubuf__ T_KV* foldSrcX0A = (__ubuf__ T_KV*)xLocal.GetPhyAddr();
    __ubuf__ T_KV* foldSrcX0B = (__ubuf__ T_KV*)xLocal.GetPhyAddr() + foldSrcBOffset;
    __ubuf__ T_KV* tailSrcX0A = (__ubuf__ T_KV*)xLocal.GetPhyAddr() + tailSrcAOffset;
    __ubuf__ T_KV* tailSrcX0B = (__ubuf__ T_KV*)xLocal.GetPhyAddr() + tailSrcBOffset;
    __ubuf__ T_KV* unFoldX0 = (__ubuf__ T_KV*)xLocal.GetPhyAddr() + unFoldSrcOffset;

    asc_vf_call<RmsNormBasicComputeImplVF<T_KV>>(
        wsBase, foldSrcX0A, foldSrcX0B, tailSrcX0A, tailSrcX0B, unFoldX0, outerLoopTimes, mainFoldLoopTimes,
        tailFoldLoopTimes, tailFoldElemCount, unFoldLoopTimes, outerLoopStride, innerLoopStride, outerLoopDstStride);
}

// 普通场景
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormPostVF(
    const LocalTensor<T_KV>& dstTensor, const LocalTensor<T_KV>& xTensor, const LocalTensor<T_KV>& gammaLocal,
    const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize, const int64_t rSize, const int64_t stride)
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
        __ubuf__ T_KV* dst = (__ubuf__ T_KV*)dstTensor.GetPhyAddr();
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)gammaLocal.GetPhyAddr();
        __ubuf__ float* sumTmp = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        asc_vf_call<RmsNormPostSingleVF<T_KV>>(dst, x, gamma, sumTmp, loopTimes, rLoopCount, oriR, oriRAligned, stride,
                                               rSize, reciprocal, epsilon);
    } else {
        __ubuf__ T_KV* dst = (__ubuf__ T_KV*)dstTensor.GetPhyAddr();
        __ubuf__ float* sumTmpA = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float* sumTmpB = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)gammaLocal.GetPhyAddr();
        asc_vf_call<RmsNormPostDoubleVF<T_KV>>(dst, sumTmpA, sumTmpB, x, gamma, loopTimes, rLoopCount, oriR,
                                               oriRAligned, stride, rSize, reciprocal, epsilon);
    }
}
// 需要输出中间结果：scale + offset
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormAsymQuantWithKvPostVF(
    const LocalTensor<T_KV>& dstTensor, const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xTensor,
    const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& reduceSumTempTensor,
    const LocalTensor<float>& vScaleTensor, const LocalTensor<float>& vOffsetTensor, const int64_t aSize,
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
        __ubuf__ T_KV* dst = (__ubuf__ T_KV*)dstTensor.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ float* vOffset = (__ubuf__ float*)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)gammaLocal.GetPhyAddr();
        __ubuf__ float* sumTmp = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        asc_vf_call<RmsNormAsymQuantWithKvPostSingleVF<T_KV, T_V_CACHE>>(
            dst, vScale, vOffset, vQuant, x, gamma, sumTmp, loopTimes, rLoopCount, oriR, oriRAligned, quantOutOffset,
            stride, rSize, reciprocal, epsilon);
    } else {
        __ubuf__ T_KV* dst = (__ubuf__ T_KV*)dstTensor.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ float* vOffset = (__ubuf__ float*)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        __ubuf__ float* sumTmpA = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float* sumTmpB = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)gammaLocal.GetPhyAddr();
        asc_vf_call<RmsNormAsymQuantWithKvPostDoubleVF<T_KV, T_V_CACHE>>(
            dst, vScale, vOffset, vQuant, sumTmpA, sumTmpB, x, gamma, loopTimes, rLoopCount, oriR, oriRAligned,
            quantOutOffset, stride, rSize, reciprocal, epsilon);
    }
}
// 需要输出中间结果：scale
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormSymQuantWithKvPostVF(
    const LocalTensor<T_KV>& dstTensor, const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xTensor,
    const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& reduceSumTempTensor,
    const LocalTensor<float>& vScaleTensor, const int64_t aSize, const int64_t rSize, const int64_t stride)
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
        __ubuf__ T_KV* dst = (__ubuf__ T_KV*)dstTensor.GetPhyAddr();
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)gammaLocal.GetPhyAddr();
        __ubuf__ float* sumTmp = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        asc_vf_call<RmsNormSymQuantWithKvPostSingleVF<T_KV, T_V_CACHE>>(
            dst, x, gamma, sumTmp, vScale, vQuant, loopTimes, rLoopCount, oriR, oriRAligned, quantOutOffset, stride,
            rSize, reciprocal, epsilon);
    } else {
        __ubuf__ T_KV* dst = (__ubuf__ T_KV*)dstTensor.GetPhyAddr();
        __ubuf__ float* sumTmpA = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float* sumTmpB = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)gammaLocal.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        asc_vf_call<RmsNormSymQuantWithKvPostDoubleVF<T_KV, T_V_CACHE>>(
            dst, sumTmpA, sumTmpB, x, gamma, vScale, vQuant, loopTimes, rLoopCount, oriR, oriRAligned, quantOutOffset,
            stride, rSize, reciprocal, epsilon);
    }
}

template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormAsymQuantPostVF(
    const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xTensor, const LocalTensor<T_KV>& gammaLocal,
    const LocalTensor<float>& reduceSumTempTensor, const LocalTensor<float>& vScaleTensor,
    const LocalTensor<float>& vOffsetTensor, const int64_t aSize, const int64_t rSize, const int64_t stride)
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
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)gammaLocal.GetPhyAddr();
        __ubuf__ float* sumTmp = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ float* vOffset = (__ubuf__ float*)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        asc_vf_call<RmsNormAsymQuantPostSingleVF<T_KV, T_V_CACHE>>(x, gamma, sumTmp, vScale, vOffset, vQuant, loopTimes,
                                                                   rLoopCount, oriR, oriRAligned, quantOutOffset,
                                                                   stride, rSize, reciprocal, epsilon);
    } else {
        __ubuf__ float* sumTmpA = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float* sumTmpB = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)gammaLocal.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ float* vOffset = (__ubuf__ float*)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        asc_vf_call<RmsNormAsymQuantPostDoubleVF<T_KV, T_V_CACHE>>(sumTmpA, sumTmpB, x, gamma, vScale, vOffset, vQuant,
                                                                   loopTimes, rLoopCount, oriR, oriRAligned,
                                                                   quantOutOffset, stride, rSize, reciprocal, epsilon);
    }
}

template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormSymQuantPostVF(
    const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xTensor, const LocalTensor<T_KV>& gammaLocal,
    const LocalTensor<float>& reduceSumTempTensor, const LocalTensor<float>& vScaleTensor, const int64_t aSize,
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
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)gammaLocal.GetPhyAddr();
        __ubuf__ float* sumTmp = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        asc_vf_call<RmsNormSymQuantPostSingleVF<T_KV, T_V_CACHE>>(x, gamma, sumTmp, vScale, vQuant, loopTimes,
                                                                  rLoopCount, oriR, oriRAligned, quantOutOffset, stride,
                                                                  rSize, reciprocal, epsilon);
    } else {
        __ubuf__ float* sumTmpA = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float* sumTmpB = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __ubuf__ T_KV* x = (__ubuf__ T_KV*)xTensor.GetPhyAddr();
        __ubuf__ T_KV* gamma = (__ubuf__ T_KV*)gammaLocal.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        asc_vf_call<RmsNormSymQuantPostDoubleVF<T_KV, T_V_CACHE>>(sumTmpA, sumTmpB, x, gamma, vScale, vQuant, loopTimes,
                                                                  rLoopCount, oriR, oriRAligned, quantOutOffset, stride,
                                                                  rSize, reciprocal, epsilon);
    }
}

// 普通场景
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>::RmsNormVF(
    const LocalTensor<T_KV>& outVLocal, const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal,
    const LocalTensor<float>& wsLocal, int64_t calcRow)
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
        __ubuf__ T_KV* dst = (__ubuf__ T_KV*)outVLocal.GetPhyAddr();
        __ubuf__ float* ws = (__ubuf__ float*)wsLocal.GetPhyAddr();
        asc_vf_call<RmsNormSmallColCastVF<T_KV>>(dst, ws, row, colLoopCount, actualCol, alignedCol);
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
    const LocalTensor<T_KV>& outVLocal, const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xLocal,
    const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& wsLocal, const LocalTensor<float>& vScaleTensor,
    const LocalTensor<float>& vOffsetTensor, int64_t calcRow)
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
        __ubuf__ T_KV* dst = (__ubuf__ T_KV*)outVLocal.GetPhyAddr();
        __ubuf__ float* ws = (__ubuf__ float*)wsLocal.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ float* vOffset = (__ubuf__ float*)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        asc_vf_call<RmsNormAsymQuantWithKvSmallColVF<T_KV, T_V_CACHE>>(
            dst, ws, vScale, vOffset, vQuant, row, colLoopCount, actualCol, alignedCol, quantOutOffset);
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
    const LocalTensor<T_KV>& outVLocal, const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xLocal,
    const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& wsLocal, const LocalTensor<float>& vScaleTensor,
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
        __ubuf__ T_KV* dst = (__ubuf__ T_KV*)outVLocal.GetPhyAddr();
        __ubuf__ float* ws = (__ubuf__ float*)wsLocal.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        asc_vf_call<RmsNormSymQuantWithKvSmallColVF<T_KV, T_V_CACHE>>(dst, ws, vScale, vQuant, row, colLoopCount,
                                                                      actualCol, alignedCol, quantOutOffset);
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
    const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal,
    const LocalTensor<float>& wsLocal, const LocalTensor<float>& vScaleTensor, const LocalTensor<float>& vOffsetTensor,
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
        __ubuf__ float* ws = (__ubuf__ float*)wsLocal.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ float* vOffset = (__ubuf__ float*)vOffsetTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        asc_vf_call<RmsNormAsymQuantSmallColVF<T_KV, T_V_CACHE>>(ws, vScale, vOffset, vQuant, row, colLoopCount,
                                                                 actualCol, alignedCol, quantOutOffset);
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
    const LocalTensor<T_V_CACHE>& quantLocal, const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal,
    const LocalTensor<float>& wsLocal, const LocalTensor<float>& vScaleTensor, int64_t calcRow)
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
        __ubuf__ float* ws = (__ubuf__ float*)wsLocal.GetPhyAddr();
        __ubuf__ float* vScale = (__ubuf__ float*)vScaleTensor.GetPhyAddr();
        __ubuf__ T_V_CACHE* vQuant = (__ubuf__ T_V_CACHE*)quantLocal.GetPhyAddr();
        asc_vf_call<RmsNormSymQuantSmallColVF<T_KV, T_V_CACHE>>(ws, vScale, vQuant, row, colLoopCount, actualCol,
                                                                alignedCol, quantOutOffset);
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
