/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_SINKHORN_REGBASE_BASE_H
#define MHC_PRE_SINKHORN_REGBASE_BASE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace MhcPreSinkhornNs {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UnalignRegForLoad;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t VL_FP32 = 64;
constexpr int32_t C0_SIZE = 8;
constexpr int32_t FOUR_UNFOLD = 4;
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr MatmulConfig MM_CFG = GetMDLConfig();

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b)
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

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float> &dst, __ubuf__ T *src, MaskReg pregLoop, uint32_t srcOffset)
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
__aicore__ inline void StoreOutputData(__ubuf__ T *dst, RegTensor<float> &src, MaskReg pregLoop, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign(dst + dstOffset, src, pregLoop);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadInputDataWithBrc(RegTensor<float> &dst, __ubuf__ T *src, MaskReg pregLoop,
                                            uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        LoadAlign<T, AscendC::Reg::LoadDist::DIST_BRC_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadInputDataUnalign(RegTensor<float> &dst, __ubuf__ T *&src, UnalignRegForLoad &uSrc,
                                            MaskReg pregLoop, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadUnAlign(dst, uSrc, src, postUpdateStride);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        RegTensor<T> tmpUnPack;
        LoadUnAlign(tmp, uSrc, src, postUpdateStride);
        UnPack((RegTensor<uint32_t> &)tmpUnPack, (RegTensor<uint16_t> &)tmp);
        Cast<float, T, castTraitB162B32Even>(dst, tmpUnPack, pregLoop);
    }
}

__aicore__ inline void VFSigmoid(RegTensor<float> &y, RegTensor<float> &x, RegTensor<float> &one, MaskReg pregLoop)
{
    Muls(x, x, static_cast<float>(-1), pregLoop);
    Exp(x, x, pregLoop);
    Adds(x, x, static_cast<float>(1), pregLoop);
    Div(y, one, x, pregLoop);
}

__aicore__ inline void VFTransND2NZ(const LocalTensor<float> &yLocal, const LocalTensor<float> &xLocal,
                                    const uint16_t curRowNum, const uint16_t curColNum)
{
    __ubuf__ float *yLocalAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    __ubuf__ float *xLocalAddr = (__ubuf__ float *)xLocal.GetPhyAddr();
    // NZ分享要求M,N方向按照16对齐
    uint16_t curRowNumAlign = CeilAlign(curRowNum, C0_SIZE);
    uint16_t c1Size = BLOCK_SIZE / sizeof(float);
    uint16_t curColNumAlign = RoundUp<float>(curColNum);
    uint16_t curRowMainCount = curRowNum / C0_SIZE;
    uint16_t curRowReminder = curRowNum % C0_SIZE;
    uint16_t tailBaseOffset = curRowReminder * c1Size;
    uint32_t dataBlockStride = curColNumAlign / c1Size + 1;
    uint16_t loopCount = curColNumAlign / c1Size;
    if (curRowReminder == 0) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            MaskReg pregMain = CreateMask<float>();
            for (uint16_t i = 0; i < curRowMainCount; i++) {
                for (uint16_t j = 0; j < loopCount; j++) {
                    LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                        x, xLocalAddr + i * C0_SIZE * (curColNumAlign + BLOCK_SIZE / sizeof(float)) + j * c1Size,
                        dataBlockStride, pregMain);
                    StoreAlign(yLocalAddr + i * C0_SIZE * c1Size + j * curRowNumAlign * c1Size, x, pregMain);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            MaskReg pregMain = CreateMask<float>();
            uint32_t sreg = curRowReminder * C0_SIZE;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            for (uint16_t i = 0; i < curRowMainCount; i++) {
                for (uint16_t j = 0; j < loopCount; j++) {
                    LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                        x, xLocalAddr + i * C0_SIZE * (curColNumAlign + BLOCK_SIZE / sizeof(float)) + j * c1Size,
                        dataBlockStride, pregMain);
                    StoreAlign(yLocalAddr + i * C0_SIZE * c1Size + j * curRowNumAlign * c1Size, x, pregMain);
                }
            }
            xLocalAddr = xLocalAddr + curRowMainCount * C0_SIZE * (curColNumAlign + BLOCK_SIZE / sizeof(float));
            yLocalAddr = yLocalAddr + curRowMainCount * C0_SIZE * c1Size;
            for (uint16_t i = 0; i < loopCount; i++) {
                LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(x, xLocalAddr + i * c1Size,
                                                                              dataBlockStride, pregLoop);
                StoreAlign(yLocalAddr + i * curRowNumAlign * c1Size, x, pregLoop);
            }
        }
    }
}

template <typename T>
__aicore__ inline void VFProcessCast(const LocalTensor<float> &yLocal, const LocalTensor<T> &xLocal,
                                     const uint16_t curRowNum, const uint16_t curColNum)
{
    __ubuf__ float *yLocalAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    __ubuf__ T *xLocalAddr = (__ubuf__ T *)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint16_t curColNumAlign = RoundUp<T>(curColNum);
    uint16_t dstCurColNumAlign = RoundUp<float>(curColNum) + BLOCK_SIZE / sizeof(float);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            MaskReg pregLoop;
            uint32_t sreg;
            for (uint16_t i = 0; i < curRowNum; i++) {
                sreg = curColNum;
                for (uint16_t j = 0; j < loopCount; j++) {
                    pregLoop = UpdateMask<float>(sreg);
                    LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign + j * VL_FP32);
                    StoreOutputData(yLocalAddr, x, pregLoop, i * dstCurColNumAlign + j * VL_FP32);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            MaskReg pregLoop = CreateMask<float>();
            for (uint16_t i = 0; i < curRowNum; i++) {
                LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign);
                StoreOutputData(yLocalAddr, x, pregLoop, i * dstCurColNumAlign);
            }
        }
    }
}

template <typename T, bool WithUbReduce = false>
__aicore__ inline void VFProcessCastAndInvRmsPart1(const LocalTensor<float> &rmsNormLocal,
                                                   const LocalTensor<float> &xCastLocal, const LocalTensor<T> &xLocal,
                                                   float coeff, const uint16_t curRowNum, const uint16_t curColNum)
{
    __ubuf__ float *rmsNormLocalAddr = (__ubuf__ float *)rmsNormLocal.GetPhyAddr();
    __ubuf__ float *xCastLocalAddr = (__ubuf__ float *)xCastLocal.GetPhyAddr();
    __ubuf__ T *xLocalAddr = (__ubuf__ T *)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint16_t curColNumAlign = RoundUp<T>(curColNum);
    uint16_t dstCurColNumAlign = RoundUp<float>(curColNum) + BLOCK_SIZE / sizeof(float);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> x1;
            RegTensor<float> sum;
            RegTensor<float> one;
            RegTensor<float> y;
            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float>();
            MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            uint32_t sreg;
            for (uint16_t i = 0; i < curRowNum; i++) {
                Duplicate(sum, 0.0f);
                if constexpr (WithUbReduce) {
                    LoadInputDataWithBrc<float>(y, rmsNormLocalAddr, pregMerge, i);
                }
                sreg = curColNum;
                for (uint16_t j = 0; j < loopCount; j++) {
                    pregLoop = UpdateMask<float>(sreg);
                    LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign + j * VL_FP32);
                    Mul(x1, x, x, pregLoop);
                    Add(sum, sum, x1, pregMain);
                    StoreOutputData(xCastLocalAddr, x, pregLoop, i * dstCurColNumAlign + j * VL_FP32);
                }
                Muls(sum, sum, coeff, pregMain);
                Reduce<ReduceType::SUM>(sum, sum, pregMain);
                if constexpr (WithUbReduce) {
                    Add(y, y, sum, pregMerge);
                    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rmsNormLocalAddr + i, y,
                                                                                       pregMerge);
                } else {
                    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rmsNormLocalAddr + i, sum,
                                                                                       pregMerge);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> x1;
            RegTensor<float> sum;
            RegTensor<float> one;
            RegTensor<float> y;
            MaskReg pregMain = CreateMask<float>();
            MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            uint32_t sreg = curColNum;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            for (uint16_t i = 0; i < curRowNum; i++) {
                Duplicate(sum, 0.0f);
                if constexpr (WithUbReduce) {
                    LoadInputDataWithBrc<float>(y, rmsNormLocalAddr, pregMerge, i);
                }
                LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign);
                StoreOutputData(xCastLocalAddr, x, pregLoop, i * dstCurColNumAlign);
                Mul(x1, x, x, pregLoop);
                Add(sum, sum, x1, pregLoop);
                Muls(sum, sum, coeff, pregLoop);
                Reduce<ReduceType::SUM>(sum, sum, pregLoop);
                if constexpr (WithUbReduce) {
                    Add(y, y, sum, pregMerge);
                    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rmsNormLocalAddr + i, y,
                                                                                       pregMerge);
                } else {
                    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rmsNormLocalAddr + i, sum,
                                                                                       pregMerge);
                }
            }
        }
    }
}

template <bool WithUbReduce = false>
__aicore__ inline void VFProcessInvRmsPart1(const LocalTensor<float> &yLocal, const LocalTensor<float> &xLocal,
                                            float coeff, uint16_t curRowNum, uint32_t curColNum)
{
    __ubuf__ float *yLocalAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    __ubuf__ float *xLocalAddr = (__ubuf__ float *)xLocal.GetPhyAddr();

    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint16_t curColNumAlign = RoundUp<float>(curColNum) + BLOCK_SIZE / sizeof(float);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> sum;
            RegTensor<float> one;
            RegTensor<float> y;
            MaskReg pregLoop;
            MaskReg pregMain = CreateMask<float>();
            MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            uint32_t sreg;
            for (uint16_t i = 0; i < curRowNum; i++) {
                Duplicate(sum, 0.0f);
                if constexpr (WithUbReduce) {
                    LoadInputDataWithBrc<float>(y, yLocalAddr, pregMerge, i);
                }
                sreg = curColNum;
                for (uint16_t j = 0; j < loopCount; j++) {
                    pregLoop = UpdateMask<float>(sreg);
                    LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign + j * VL_FP32);
                    Mul(x, x, x, pregLoop);
                    Add(sum, sum, x, pregMain);
                }
                Muls(sum, sum, coeff, pregMain);
                Reduce<ReduceType::SUM>(sum, sum, pregMain);
                if constexpr (WithUbReduce) {
                    Add(y, y, sum, pregMerge);
                    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, y, pregMerge);
                } else {
                    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, sum, pregMerge);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> sum;
            RegTensor<float> one;
            RegTensor<float> y;
            MaskReg pregMain = CreateMask<float>();
            MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            uint32_t sreg = curColNum;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            for (uint16_t i = 0; i < curRowNum; i++) {
                Duplicate(sum, 0.0f);
                if constexpr (WithUbReduce) {
                    LoadInputDataWithBrc<float>(y, yLocalAddr, pregMerge, i);
                }
                LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign);
                Mul(x, x, x, pregLoop);
                Add(sum, sum, x, pregLoop);
                Muls(sum, sum, coeff, pregLoop);
                Reduce<ReduceType::SUM>(sum, sum, pregLoop);
                if constexpr (WithUbReduce) {
                    Add(y, y, sum, pregMerge);
                    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, y, pregMerge);
                } else {
                    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, sum, pregMerge);
                }
            }
        }
    }
}

// (bs, k) --> (bs, 1)
// k 为Matmul K轴切分时的分核数，必然小于64
__aicore__ inline void VFProcessInvRmsPart2(const LocalTensor<float> &yLocal, const LocalTensor<float> &xLocal,
                                            float eps, uint16_t curRowNum, uint32_t curColNum)
{
    __ubuf__ float *yLocalAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    __ubuf__ float *xLocalAddr = (__ubuf__ float *)xLocal.GetPhyAddr();
    uint16_t curColNumAlign = RoundUp<float>(curColNum);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> sum;
        RegTensor<float> one;
        RegTensor<float> y;
        uint32_t sreg = curColNum;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        Duplicate(one, static_cast<float>(1.0), pregMerge);
        for (uint16_t i = 0; i < curRowNum; i++) {
            LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign);
            Reduce<ReduceType::SUM>(sum, x, pregLoop);
            Adds(sum, sum, eps, pregMerge);
            Sqrt(sum, sum, pregMerge);
            Div(y, one, sum, pregMerge);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, y, pregMerge);
        }
    }
}

// (k, bs, hc_mix) * (k, bs, 1) = (bs, hc_mix)
// for循环组织形式如下:
/*
    for (i, 0, bs)

        for (j, 0, k)
            for (h, 0, hc_mix)
*/
// hcMix小于64，因此直接去掉内层for循环
__aicore__ inline void VFProcessInvRmsPart3WithGroupReduce(const LocalTensor<float> &yLocal,
                                                           const LocalTensor<float> &mmLocal,
                                                           const LocalTensor<float> &xLocal, float eps, uint16_t groupK,
                                                           uint16_t bs, uint32_t hcMix)
{
    __ubuf__ float *yLocalAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    __ubuf__ float *mmLocalAddr = (__ubuf__ float *)mmLocal.GetPhyAddr();
    __ubuf__ float *xLocalAddr = (__ubuf__ float *)xLocal.GetPhyAddr();
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    uint32_t bsAlign = RoundUp<float>(bs);
    uint16_t fourLoopNum = groupK / FOUR_UNFOLD;
    uint16_t tailLoopNum = groupK % FOUR_UNFOLD;
    if (groupK < 4) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> sum1;
            RegTensor<float> sum2;
            RegTensor<float> one;
            RegTensor<float> rsqrt;
            RegTensor<float> y;
            RegTensor<float> mm;
            uint32_t sreg = hcMix;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            Duplicate(one, static_cast<float>(1.0), pregMerge);
            for (uint16_t i = 0; i < bs; i++) {
                Duplicate(sum1, static_cast<float>(0.0f), pregMerge);
                Duplicate(sum2, static_cast<float>(0.0f), pregLoop);
                for (uint16_t j = 0; j < groupK; j++) {
                    LoadInputDataWithBrc<float>(x, xLocalAddr, pregMerge, i + j * bsAlign);
                    Add(sum1, sum1, x, pregMerge);
                    LoadInputData<float>(mm, mmLocalAddr, pregLoop, i * hcMixAlign + j * bs * hcMixAlign);
                    Add(sum2, sum2, mm, pregLoop);
                }
                Adds(sum1, sum1, eps, pregMerge);
                Sqrt(sum1, sum1, pregMerge);
                Div(rsqrt, one, sum1, pregMerge);
                Duplicate(rsqrt, rsqrt, pregLoop);
                Mul(y, sum2, rsqrt, pregLoop);
                StoreOutputData(yLocalAddr, y, pregLoop, i * hcMixAlign);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;
            RegTensor<float> mm1;
            RegTensor<float> mm2;
            RegTensor<float> mm3;
            RegTensor<float> mm4;
            RegTensor<float> sumX1;
            RegTensor<float> sumX2;
            RegTensor<float> sumX3;
            RegTensor<float> sumX4;
            RegTensor<float> sumM1;
            RegTensor<float> sumM2;
            RegTensor<float> sumM3;
            RegTensor<float> sumM4;
            RegTensor<float> one;
            RegTensor<float> rsqrt;
            RegTensor<float> y;
            uint32_t sreg = hcMix;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            Duplicate(one, static_cast<float>(1.0), pregMerge);
            for (uint16_t i = 0; i < bs; i++) {
                Duplicate(sumX1, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumX2, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumX3, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumX4, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumM1, static_cast<float>(0.0f), pregLoop);
                Duplicate(sumM2, static_cast<float>(0.0f), pregLoop);
                Duplicate(sumM3, static_cast<float>(0.0f), pregLoop);
                Duplicate(sumM4, static_cast<float>(0.0f), pregLoop);
                for (uint16_t j = 0; j < fourLoopNum; j++) {
                    LoadInputDataWithBrc<float>(x1, xLocalAddr, pregMerge, i + 4 * j * bsAlign);
                    Add(sumX1, sumX1, x1, pregMerge);
                    LoadInputData<float>(mm1, mmLocalAddr, pregLoop, i * hcMixAlign + 4 * j * bs * hcMixAlign);
                    Add(sumM1, sumM1, mm1, pregLoop);

                    LoadInputDataWithBrc<float>(x2, xLocalAddr, pregMerge, i + (4 * j + 1) * bsAlign);
                    Add(sumX2, sumX2, x2, pregMerge);
                    LoadInputData<float>(mm2, mmLocalAddr, pregLoop, i * hcMixAlign + (4 * j + 1) * bs * hcMixAlign);
                    Add(sumM2, sumM2, mm2, pregLoop);

                    LoadInputDataWithBrc<float>(x3, xLocalAddr, pregMerge, i + (4 * j + 2) * bsAlign);
                    Add(sumX3, sumX3, x3, pregMerge);
                    LoadInputData<float>(mm3, mmLocalAddr, pregLoop, i * hcMixAlign + (4 * j + 2) * bs * hcMixAlign);
                    Add(sumM3, sumM3, mm3, pregLoop);

                    LoadInputDataWithBrc<float>(x4, xLocalAddr, pregMerge, i + (4 * j + 3) * bsAlign);
                    Add(sumX4, sumX4, x4, pregMerge);
                    LoadInputData<float>(mm4, mmLocalAddr, pregLoop, i * hcMixAlign + (4 * j + 3) * bs * hcMixAlign);
                    Add(sumM4, sumM4, mm4, pregLoop);
                }
                for (uint16_t j = 0; j < tailLoopNum; j++) {
                    LoadInputDataWithBrc<float>(x1, xLocalAddr, pregMerge,
                                                i + (fourLoopNum * FOUR_UNFOLD + j) * bsAlign);
                    Add(sumX1, sumX1, x1, pregMerge);
                    LoadInputData<float>(mm1, mmLocalAddr, pregLoop,
                                         i * hcMixAlign + (fourLoopNum * FOUR_UNFOLD + j) * bs * hcMixAlign);
                    Add(sumM1, sumM1, mm1, pregLoop);
                }
                Add(sumX1, sumX1, sumX4, pregMerge);
                Add(sumX2, sumX2, sumX3, pregMerge);
                Add(sumX1, sumX1, sumX2, pregMerge);
                Add(sumM1, sumM1, sumM4, pregLoop);
                Add(sumM2, sumM2, sumM3, pregLoop);
                Add(sumM1, sumM1, sumM2, pregLoop);

                Adds(sumX1, sumX1, eps, pregMerge);
                Sqrt(sumX1, sumX1, pregMerge);
                Div(rsqrt, one, sumX1, pregMerge);
                Duplicate(rsqrt, rsqrt, pregLoop);
                Mul(y, sumM1, rsqrt, pregLoop);
                StoreOutputData(yLocalAddr, y, pregLoop, i * hcMixAlign);
            }
        }
    }
}

__aicore__ inline void VFProcessInvRmsPart3WithGroupReduceGradout(const LocalTensor<float> &yLocal,
                                                                  const LocalTensor<float> &hcBeforeNormLocal,
                                                                  const LocalTensor<float> &invRmsOutLocal,
                                                                  const LocalTensor<float> &mmLocal,
                                                                  const LocalTensor<float> &xLocal, float eps,
                                                                  uint16_t groupK, uint16_t bs, uint32_t hcMix)
{
    __ubuf__ float *yLocalAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    __ubuf__ float *mmLocalAddr = (__ubuf__ float *)mmLocal.GetPhyAddr();
    __ubuf__ float *xLocalAddr = (__ubuf__ float *)xLocal.GetPhyAddr();
    __ubuf__ float *hcBeforeNormLocalAddr = (__ubuf__ float *)hcBeforeNormLocal.GetPhyAddr();
    __ubuf__ float *invRmsOutLocalAddr = (__ubuf__ float *)invRmsOutLocal.GetPhyAddr();
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    uint32_t bsAlign = RoundUp<float>(bs);
    uint16_t fourLoopNum = groupK / FOUR_UNFOLD;
    uint16_t tailLoopNum = groupK % FOUR_UNFOLD;
    if (groupK < 4) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> sum1;
            RegTensor<float> sum2;
            RegTensor<float> one;
            RegTensor<float> rsqrt;
            RegTensor<float> y;
            RegTensor<float> mm;
            uint32_t sreg = hcMix;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            Duplicate(one, static_cast<float>(1.0), pregMerge);
            for (uint16_t i = 0; i < bs; i++) {
                Duplicate(sum1, static_cast<float>(0.0f), pregMerge);
                Duplicate(sum2, static_cast<float>(0.0f), pregLoop);
                for (uint16_t j = 0; j < groupK; j++) {
                    LoadInputDataWithBrc<float>(x, xLocalAddr, pregMerge, i + j * bsAlign);
                    Add(sum1, sum1, x, pregMerge);
                    LoadInputData<float>(mm, mmLocalAddr, pregLoop, i * hcMixAlign + j * bs * hcMixAlign);
                    Add(sum2, sum2, mm, pregLoop);
                }
                Adds(sum1, sum1, eps, pregMerge);
                Sqrt(sum1, sum1, pregMerge);
                Div(rsqrt, one, sum1, pregMerge);
                // store invRmsOut
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(invRmsOutLocalAddr + i, rsqrt,
                                                                                   pregMerge);
                Duplicate(rsqrt, rsqrt, pregLoop);
                Mul(y, sum2, rsqrt, pregLoop);
                // store hcBeforeNorm
                StoreOutputData(hcBeforeNormLocalAddr, sum2, pregLoop, i * hcMixAlign);
                StoreOutputData(yLocalAddr, y, pregLoop, i * hcMixAlign);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;
            RegTensor<float> mm1;
            RegTensor<float> mm2;
            RegTensor<float> mm3;
            RegTensor<float> mm4;
            RegTensor<float> sumX1;
            RegTensor<float> sumX2;
            RegTensor<float> sumX3;
            RegTensor<float> sumX4;
            RegTensor<float> sumM1;
            RegTensor<float> sumM2;
            RegTensor<float> sumM3;
            RegTensor<float> sumM4;
            RegTensor<float> one;
            RegTensor<float> rsqrt;
            RegTensor<float> y;
            uint32_t sreg = hcMix;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            Duplicate(one, static_cast<float>(1.0), pregMerge);
            for (uint16_t i = 0; i < bs; i++) {
                Duplicate(sumX1, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumX2, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumX3, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumX4, static_cast<float>(0.0f), pregMerge);
                Duplicate(sumM1, static_cast<float>(0.0f), pregLoop);
                Duplicate(sumM2, static_cast<float>(0.0f), pregLoop);
                Duplicate(sumM3, static_cast<float>(0.0f), pregLoop);
                Duplicate(sumM4, static_cast<float>(0.0f), pregLoop);
                for (uint16_t j = 0; j < fourLoopNum; j++) {
                    LoadInputDataWithBrc<float>(x1, xLocalAddr, pregMerge, i + 4 * j * bsAlign);
                    Add(sumX1, sumX1, x1, pregMerge);
                    LoadInputData<float>(mm1, mmLocalAddr, pregLoop, i * hcMixAlign + 4 * j * bs * hcMixAlign);
                    Add(sumM1, sumM1, mm1, pregLoop);

                    LoadInputDataWithBrc<float>(x2, xLocalAddr, pregMerge, i + (4 * j + 1) * bsAlign);
                    Add(sumX2, sumX2, x2, pregMerge);
                    LoadInputData<float>(mm2, mmLocalAddr, pregLoop, i * hcMixAlign + (4 * j + 1) * bs * hcMixAlign);
                    Add(sumM2, sumM2, mm2, pregLoop);

                    LoadInputDataWithBrc<float>(x3, xLocalAddr, pregMerge, i + (4 * j + 2) * bsAlign);
                    Add(sumX3, sumX3, x3, pregMerge);
                    LoadInputData<float>(mm3, mmLocalAddr, pregLoop, i * hcMixAlign + (4 * j + 2) * bs * hcMixAlign);
                    Add(sumM3, sumM3, mm3, pregLoop);

                    LoadInputDataWithBrc<float>(x4, xLocalAddr, pregMerge, i + (4 * j + 3) * bsAlign);
                    Add(sumX4, sumX4, x4, pregMerge);
                    LoadInputData<float>(mm4, mmLocalAddr, pregLoop, i * hcMixAlign + (4 * j + 3) * bs * hcMixAlign);
                    Add(sumM4, sumM4, mm4, pregLoop);
                }
                for (uint16_t j = 0; j < tailLoopNum; j++) {
                    LoadInputDataWithBrc<float>(x1, xLocalAddr, pregMerge,
                                                i + (fourLoopNum * FOUR_UNFOLD + j) * bsAlign);
                    Add(sumX1, sumX1, x1, pregMerge);
                    LoadInputData<float>(mm1, mmLocalAddr, pregLoop,
                                         i * hcMixAlign + (fourLoopNum * FOUR_UNFOLD + j) * bs * hcMixAlign);
                    Add(sumM1, sumM1, mm1, pregLoop);
                }
                Add(sumX1, sumX1, sumX4, pregMerge);
                Add(sumX2, sumX2, sumX3, pregMerge);
                Add(sumX1, sumX1, sumX2, pregMerge);
                Add(sumM1, sumM1, sumM4, pregLoop);
                Add(sumM2, sumM2, sumM3, pregLoop);
                Add(sumM1, sumM1, sumM2, pregLoop);

                Adds(sumX1, sumX1, eps, pregMerge);
                Sqrt(sumX1, sumX1, pregMerge);
                Div(rsqrt, one, sumX1, pregMerge);
                // store invRmsOut
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(invRmsOutLocalAddr + i, rsqrt,
                                                                                   pregMerge);
                Duplicate(rsqrt, rsqrt, pregLoop);
                Mul(y, sumM1, rsqrt, pregLoop);
                // store hcBeforeNorm
                StoreOutputData(hcBeforeNormLocalAddr, sumM1, pregLoop, i * hcMixAlign);
                StoreOutputData(yLocalAddr, y, pregLoop, i * hcMixAlign);
            }
        }
    }
}

__aicore__ inline void VFProcessInvRmsPart3(const LocalTensor<float> &yLocal, const LocalTensor<float> &mmLocal,
                                            const LocalTensor<float> &xLocal, float eps, uint16_t bs, uint32_t hcMix)
{
    __ubuf__ float *yLocalAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    __ubuf__ float *mmLocalAddr = (__ubuf__ float *)mmLocal.GetPhyAddr();
    __ubuf__ float *xLocalAddr = (__ubuf__ float *)xLocal.GetPhyAddr();
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    uint32_t bsAlign = RoundUp<float>(bs);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> sum1;
        RegTensor<float> sum2;
        RegTensor<float> one;
        RegTensor<float> rsqrt;
        RegTensor<float> y;
        RegTensor<float> mm;
        uint32_t sreg = hcMix;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        Duplicate(one, static_cast<float>(1.0), pregLoop);
        for (uint16_t i = 0; i < bs; i++) {
            LoadInputDataWithBrc<float>(x, xLocalAddr, pregLoop, i);
            LoadInputData<float>(mm, mmLocalAddr, pregLoop, i * hcMixAlign);
            Adds(x, x, eps, pregLoop);
            Sqrt(x, x, pregLoop);
            Div(rsqrt, one, x, pregLoop);
            Mul(y, mm, rsqrt, pregLoop);
            StoreOutputData(yLocalAddr, y, pregLoop, i * hcMixAlign);
        }
    }
}

__aicore__ inline void VFProcessInvRmsPart3Gradout(const LocalTensor<float> &invRmsOutLocal,
                                                   const LocalTensor<float> &yLocal, const LocalTensor<float> &mmLocal,
                                                   const LocalTensor<float> &xLocal, float eps, uint16_t bs,
                                                   uint32_t hcMix)
{
    __ubuf__ float *invRmsOutLocalAddr = (__ubuf__ float *)invRmsOutLocal.GetPhyAddr();
    __ubuf__ float *yLocalAddr = (__ubuf__ float *)yLocal.GetPhyAddr();
    __ubuf__ float *mmLocalAddr = (__ubuf__ float *)mmLocal.GetPhyAddr();
    __ubuf__ float *xLocalAddr = (__ubuf__ float *)xLocal.GetPhyAddr();
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    uint32_t bsAlign = RoundUp<float>(bs);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> sum1;
        RegTensor<float> sum2;
        RegTensor<float> one;
        RegTensor<float> rsqrt;
        RegTensor<float> y;
        RegTensor<float> mm;
        uint32_t sreg = hcMix;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        Duplicate(one, static_cast<float>(1.0), pregLoop);
        MaskReg pregOne = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        for (uint16_t i = 0; i < bs; i++) {
            LoadInputDataWithBrc<float>(x, xLocalAddr, pregLoop, i);
            LoadInputData<float>(mm, mmLocalAddr, pregLoop, i * hcMixAlign);
            Adds(x, x, eps, pregLoop);
            Sqrt(x, x, pregLoop);
            Div(rsqrt, one, x, pregLoop);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(invRmsOutLocalAddr + i, rsqrt, pregOne);
            Mul(y, mm, rsqrt, pregLoop);
            StoreOutputData(yLocalAddr, y, pregLoop, i * hcMixAlign);
        }
    }
}

// Matmul的结果会直接FixPipe到UB上，不会在搬运时完成Split动作，因此需要在UB内完成Split动作
__aicore__ inline void VFProcessPre(const LocalTensor<float> &preLocal, const LocalTensor<float> &mixLocal,
                                    const LocalTensor<float> &hcBaseLocal, float scale, float eps, uint16_t curRowNum,
                                    uint16_t curColNum, uint16_t hcMix)
{
    __ubuf__ float *preLocalAddr = (__ubuf__ float *)preLocal.GetPhyAddr();
    __ubuf__ float *mixLocalAddr = (__ubuf__ float *)mixLocal.GetPhyAddr();
    __ubuf__ float *hcBaseLocalAddr = (__ubuf__ float *)hcBaseLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> one;
            MaskReg pregLoop = CreateMask<float>();
            uint32_t sreg = curColNum;
            Duplicate(one, static_cast<float>(1), pregLoop);
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, i * VL_FP32);
                for (uint16_t j = 0; j < curRowNum; j++) {
                    LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * VL_FP32 + j * hcMixAlign);
                    Muls(mix, mix, scale, pregLoop);
                    Add(mix, mix, base, pregLoop);
                    VFSigmoid(mix, mix, one, pregLoop);
                    Adds(mix, mix, eps, pregLoop);
                    StoreOutputData(preLocalAddr, mix, pregLoop, i * VL_FP32 + j * hcMixAlign);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> one;
            uint32_t sreg = curColNum;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            Duplicate(one, static_cast<float>(1), pregLoop);
            LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, 0);
            for (uint16_t i = 0; i < curRowNum; i++) {
                LoadInputData<float>(mix, mixLocalAddr, pregLoop, i * hcMixAlign);
                Muls(mix, mix, scale, pregLoop);
                Add(mix, mix, base, pregLoop);
                VFSigmoid(mix, mix, one, pregLoop);
                Adds(mix, mix, eps, pregLoop);
                StoreOutputData(preLocalAddr, mix, pregLoop, i * hcMixAlign);
            }
        }
    }
}

__aicore__ inline void VFProcessPost(const LocalTensor<float> &postLocal, const LocalTensor<float> &mixLocal,
                                     const LocalTensor<float> &hcBaseLocal, float scale, float eps, uint16_t curRowNum,
                                     uint16_t curColNum, uint16_t hcMix)
{
    __ubuf__ float *postLocalAddr = (__ubuf__ float *)postLocal.GetPhyAddr();
    __ubuf__ float *mixOriginLocalAddr = (__ubuf__ float *)mixLocal.GetPhyAddr();
    __ubuf__ float *hcBaseLocalAddr = (__ubuf__ float *)hcBaseLocal.GetPhyAddr();
    __ubuf__ float *mixLocalAddr = mixOriginLocalAddr;
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> one;
            UnalignRegForLoad uMix;
            MaskReg pregLoop = CreateMask<float>();
            uint32_t sreg = curColNum;
            Duplicate(one, static_cast<float>(1), pregLoop);
            LoadUnAlignPre<float>(uMix, mixLocalAddr);
            for (uint16_t i = 0; i < loopCount; i++) {
                mixLocalAddr = mixOriginLocalAddr + i * VL_FP32;
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, i * VL_FP32);
                for (uint16_t j = 0; j < curRowNum; j++) {
                    LoadInputDataUnalign(mix, mixLocalAddr, uMix, pregLoop, hcMixAlign);
                    Muls(mix, mix, scale, pregLoop);
                    Add(mix, mix, base, pregLoop);
                    VFSigmoid(mix, mix, one, pregLoop);
                    Muls(mix, mix, static_cast<float>(2.0), pregLoop);
                    StoreOutputData(postLocalAddr, mix, pregLoop, i * VL_FP32 + j * curColNumAlign);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> mix;
            RegTensor<float> base;
            RegTensor<float> one;
            UnalignRegForLoad uMix;
            uint32_t sreg = curColNum;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            Duplicate(one, static_cast<float>(1), pregLoop);
            LoadUnAlignPre<float>(uMix, mixLocalAddr);
            LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, 0);
            for (uint16_t i = 0; i < curRowNum; i++) {
                LoadInputDataUnalign(mix, mixLocalAddr, uMix, pregLoop, hcMixAlign);
                Muls(mix, mix, scale, pregLoop);
                Add(mix, mix, base, pregLoop);
                VFSigmoid(mix, mix, one, pregLoop);
                Muls(mix, mix, static_cast<float>(2.0), pregLoop);
                StoreOutputData(postLocalAddr, mix, pregLoop, i * curColNumAlign);
            }
        }
    }
}

// dim2是R轴，R轴小于64, 不需要回写UB
__aicore__ inline void VFProcessCombFragRLessVL(const LocalTensor<float> &combFragLocal,
                                                const LocalTensor<float> &mixLocal,
                                                const LocalTensor<float> &hcBaseLocal, float scale, float eps,
                                                uint16_t iters, uint16_t dim0, uint16_t dim1, uint16_t dim2,
                                                uint16_t hcMix)
{
    __ubuf__ float *combFragLocalAddr = (__ubuf__ float *)combFragLocal.GetPhyAddr();
    __ubuf__ float *mixLocalOriginAddr = (__ubuf__ float *)mixLocal.GetPhyAddr();
    __ubuf__ float *hcBaseLocalAddr = (__ubuf__ float *)hcBaseLocal.GetPhyAddr();
    __ubuf__ float *mixLocalAddr = mixLocalOriginAddr;
    uint32_t dim2Align = RoundUp<float>(dim2);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    __VEC_SCOPE__
    {
        RegTensor<float> base;
        RegTensor<float> mix;
        RegTensor<float> rsqrt;
        RegTensor<float> max;
        RegTensor<float> sum;
        RegTensor<float> sum1;
        UnalignRegForLoad uMix;
        uint32_t sreg = dim2;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        LoadUnAlignPre<float>(uMix, mixLocalAddr);
        for (uint16_t i = 0; i < dim0; i++) {
            Duplicate(sum1, static_cast<float>(0), pregLoop);
            for (uint16_t j = 0; j < dim1; j++) {
                mixLocalAddr = mixLocalOriginAddr + i * hcMixAlign + j * dim2;
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, j * dim2Align);
                LoadInputDataUnalign<float>(mix, mixLocalAddr, uMix, pregLoop, VL_FP32);
                Muls(mix, mix, scale, pregLoop);
                Add(mix, mix, base, pregLoop);
                Reduce<ReduceType::MAX>(max, mix, pregLoop);
                Duplicate(max, max, pregLoop);
                Sub(mix, mix, max, pregLoop);
                Exp(mix, mix, pregLoop);
                Reduce<ReduceType::SUM>(sum, mix, pregLoop);
                Duplicate(sum, sum, pregLoop);
                Div(mix, mix, sum, pregLoop);
                Adds(mix, mix, eps, pregLoop);
                Add(sum1, sum1, mix, pregLoop);
                StoreOutputData(combFragLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            Adds(sum1, sum1, eps, pregLoop);
            for (uint16_t j = 0; j < dim1; j++) {
                LoadInputData<float>(mix, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + j * dim2Align);
                Div(mix, mix, sum1, pregLoop);
                StoreOutputData(combFragLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
            }
        }
        for (uint16_t i = 0; i < iters; i++) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            for (uint16_t j = 0; j < dim0; j++) {
                Duplicate(sum1, static_cast<float>(0), pregLoop);
                for (uint16_t k = 0; k < dim1; k++) {
                    LoadInputData<float>(mix, combFragLocalAddr, pregLoop, j * dim1 * dim2Align + k * dim2Align);
                    Reduce<ReduceType::SUM>(sum, mix, pregLoop);
                    Duplicate(sum, sum, pregLoop);
                    Adds(sum, sum, eps, pregLoop);
                    Div(mix, mix, sum, pregLoop);
                    Add(sum1, sum1, mix, pregLoop);
                    StoreOutputData(combFragLocalAddr, mix, pregLoop, j * dim1 * dim2Align + k * dim2Align);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                Adds(sum1, sum1, eps, pregLoop);
                for (uint16_t k = 0; k < dim1; k++) {
                    LoadInputData<float>(mix, combFragLocalAddr, pregLoop, j * dim1 * dim2Align + k * dim2Align);
                    Div(mix, mix, sum1, pregLoop);
                    StoreOutputData(combFragLocalAddr, mix, pregLoop, j * dim1 * dim2Align + k * dim2Align);
                }
            }
        }
    }
}

__aicore__ inline void VFProcessIteration(RegTensor<float> &sum0, RegTensor<float> &sum1, RegTensor<float> &mix,
                                          float eps, MaskReg pregLoop)
{
    Reduce<ReduceType::SUM>(sum1, mix, pregLoop);
    Duplicate(sum1, sum1, pregLoop);
    Adds(sum1, sum1, eps, pregLoop);
    Div(mix, mix, sum1, pregLoop);
    Add(sum0, sum0, mix, pregLoop);
}

__aicore__ inline void VFProcessCombFragRLessVLUseFourUnfold(const LocalTensor<float> &combFragLocal,
                                                             const LocalTensor<float> &mixLocal,
                                                             const LocalTensor<float> &hcBaseLocal, float scale,
                                                             float eps, uint16_t iters, uint16_t dim0, uint16_t dim1,
                                                             uint16_t dim2, uint16_t hcMix)
{
    __ubuf__ float *combFragLocalAddr = (__ubuf__ float *)combFragLocal.GetPhyAddr();
    __ubuf__ float *mixLocalOriginAddr = (__ubuf__ float *)mixLocal.GetPhyAddr();
    __ubuf__ float *hcBaseLocalAddr = (__ubuf__ float *)hcBaseLocal.GetPhyAddr();
    __ubuf__ float *mixLocalAddr = mixLocalOriginAddr;
    uint32_t dim2Align = RoundUp<float>(dim2);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    __VEC_SCOPE__
    {
        RegTensor<float> base;
        RegTensor<float> mix;
        RegTensor<float> mix1;
        RegTensor<float> mix2;
        RegTensor<float> mix3;
        RegTensor<float> mix4;
        RegTensor<float> max;
        RegTensor<float> sum;
        RegTensor<float> sum1;
        RegTensor<float> sum2;
        RegTensor<float> sum3;
        RegTensor<float> sum4;
        UnalignRegForLoad uMix;
        uint32_t sreg = dim2;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        LoadUnAlignPre<float>(uMix, mixLocalAddr);
        for (uint16_t i = 0; i < dim0; i++) {
            Duplicate(sum1, static_cast<float>(0), pregLoop);
            for (uint16_t j = 0; j < dim1; j++) {
                mixLocalAddr = mixLocalOriginAddr + i * hcMixAlign + j * dim2;
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, j * dim2Align);
                LoadInputDataUnalign<float>(mix, mixLocalAddr, uMix, pregLoop, VL_FP32);
                Muls(mix, mix, scale, pregLoop);
                Add(mix, mix, base, pregLoop);
                Reduce<ReduceType::MAX>(max, mix, pregLoop);
                Duplicate(max, max, pregLoop);
                Sub(mix, mix, max, pregLoop);
                Exp(mix, mix, pregLoop);
                Reduce<ReduceType::SUM>(sum, mix, pregLoop);
                Duplicate(sum, sum, pregLoop);
                Div(mix, mix, sum, pregLoop);
                Adds(mix, mix, eps, pregLoop);
                Add(sum1, sum1, mix, pregLoop);
                StoreOutputData(combFragLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            Adds(sum1, sum1, eps, pregLoop);
            for (uint16_t j = 0; j < dim1; j++) {
                LoadInputData<float>(mix, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + j * dim2Align);
                Div(mix, mix, sum1, pregLoop);
                StoreOutputData(combFragLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < dim0; i++) {
            LoadInputData<float>(mix1, combFragLocalAddr, pregLoop, i * dim1 * dim2Align);
            LoadInputData<float>(mix2, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + 1 * dim2Align);
            LoadInputData<float>(mix3, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + 2 * dim2Align);
            LoadInputData<float>(mix4, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + 3 * dim2Align);
            for (uint16_t j = 0; j < iters; j++) {
                Duplicate(sum, static_cast<float>(0), pregLoop);
                VFProcessIteration(sum, sum1, mix1, eps, pregLoop);
                VFProcessIteration(sum, sum2, mix2, eps, pregLoop);
                VFProcessIteration(sum, sum3, mix3, eps, pregLoop);
                VFProcessIteration(sum, sum4, mix4, eps, pregLoop);
                Adds(sum, sum, eps, pregLoop);
                Div(mix1, mix1, sum, pregLoop);
                Div(mix2, mix2, sum, pregLoop);
                Div(mix3, mix3, sum, pregLoop);
                Div(mix4, mix4, sum, pregLoop);
            }
            StoreOutputData(combFragLocalAddr, mix1, pregLoop, i * dim1 * dim2Align);
            StoreOutputData(combFragLocalAddr, mix2, pregLoop, i * dim1 * dim2Align + 1 * dim2Align);
            StoreOutputData(combFragLocalAddr, mix3, pregLoop, i * dim1 * dim2Align + 2 * dim2Align);
            StoreOutputData(combFragLocalAddr, mix4, pregLoop, i * dim1 * dim2Align + 3 * dim2Align);
        }
    }
}

__aicore__ inline void VFProcessIterationGradout(RegTensor<float> &sum, RegTensor<float> &sum1, RegTensor<float> &mix1,
                                                 float eps, MaskReg pregLoop, MaskReg pregOne, uint16_t i, uint16_t j,
                                                 uint16_t curDim1, uint16_t dim0, uint16_t dim1, uint16_t dim2Align,
                                                 __ubuf__ float *sumOutLocalAddr, __ubuf__ float *normOutLocalAddr)
{
    Reduce<ReduceType::SUM>(sum1, mix1, pregLoop);
    Duplicate(sum1, sum1, pregLoop);
    Adds(sum1, sum1, eps, pregLoop);
    // store sumOut[2j][dim1] dim1:0,1,2,3
    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        sumOutLocalAddr + 2 * j * dim0 * dim2Align + i * dim2Align + curDim1, sum1, pregOne);
    Div(mix1, mix1, sum1, pregLoop);
    // store normOut[2j][dim1] dim1:0,1,2,3
    StoreOutputData(normOutLocalAddr, mix1, pregLoop,
                    2 * j * dim0 * dim1 * dim2Align + i * dim1 * dim2Align + curDim1 * dim2Align);
    Add(sum, sum, mix1, pregLoop);
}

__aicore__ inline void VFProcessCombFragRLessVLUseFourUnfoldGradout(const LocalTensor<float> &combFragLocal,
                                                                    const LocalTensor<float> &mixLocal,
                                                                    const LocalTensor<float> &hcBaseLocal, float scale,
                                                                    float eps, uint16_t iters, uint16_t dim0,
                                                                    uint16_t dim1, uint16_t dim2, uint16_t hcMix,
                                                                    uint16_t rowInnerFactor)
{
    __ubuf__ float *combFragLocalAddr = (__ubuf__ float *)combFragLocal.GetPhyAddr();
    __ubuf__ float *mixLocalOriginAddr = (__ubuf__ float *)mixLocal.GetPhyAddr();
    __ubuf__ float *hcBaseLocalAddr = (__ubuf__ float *)hcBaseLocal.GetPhyAddr();
    __ubuf__ float *mixLocalAddr = mixLocalOriginAddr;
    uint32_t dim2Align = RoundUp<float>(dim2);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    // normOutLocal shape [2*iter, rowInnerFactor, dim1, dim2Align]
    // sumOutLocal shape [2*iter, rowInnerFactor, dim2Align]
    __ubuf__ float *normOutLocalAddr = combFragLocalAddr + rowInnerFactor * dim2 * dim2Align;
    __ubuf__ float *sumOutLocalAddr = normOutLocalAddr + iters * 2 * rowInnerFactor * dim2 * dim2Align;
    __VEC_SCOPE__
    {
        RegTensor<float> base;
        RegTensor<float> mix;
        RegTensor<float> mix1;
        RegTensor<float> mix2;
        RegTensor<float> mix3;
        RegTensor<float> mix4;
        RegTensor<float> max;
        RegTensor<float> sum;
        RegTensor<float> sum1;
        RegTensor<float> sum2;
        RegTensor<float> sum3;
        RegTensor<float> sum4;
        RegTensor<float> sumOut;
        UnalignRegForLoad uMix;
        uint32_t sreg = dim2;
        MaskReg pregLoop = UpdateMask<float>(sreg);
        LoadUnAlignPre<float>(uMix, mixLocalAddr);
        MaskReg pregOne = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        for (uint16_t i = 0; i < dim0; i++) {
            Duplicate(sum1, static_cast<float>(0), pregLoop);
            for (uint16_t j = 0; j < dim1; j++) {
                mixLocalAddr = mixLocalOriginAddr + i * hcMixAlign + j * dim2;
                LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, j * dim2Align);
                LoadInputDataUnalign<float>(mix, mixLocalAddr, uMix, pregLoop, VL_FP32);
                Muls(mix, mix, scale, pregLoop);
                Add(mix, mix, base, pregLoop);
                Reduce<ReduceType::MAX>(max, mix, pregLoop);
                Duplicate(max, max, pregLoop);
                Sub(mix, mix, max, pregLoop);
                Exp(mix, mix, pregLoop);
                Reduce<ReduceType::SUM>(sum, mix, pregLoop);
                // store sumOut[0][j] j -> 1~dim1
                Adds(sumOut, sum, eps, pregOne);
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumOutLocalAddr + i * dim2Align + j,
                                                                                   sumOut, pregOne);
                Duplicate(sum, sum, pregLoop);
                Div(mix, mix, sum, pregLoop);
                Adds(mix, mix, eps, pregLoop);
                // store normOut[0][j] j-> 1~dim1
                StoreOutputData(normOutLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
                Add(sum1, sum1, mix, pregLoop);
                StoreOutputData(combFragLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            Adds(sum1, sum1, eps, pregLoop);
            // store sumOut[1]
            StoreOutputData(sumOutLocalAddr, sum1, pregLoop, dim0 * dim2Align + i * dim2Align);
            for (uint16_t j = 0; j < dim1; j++) {
                LoadInputData<float>(mix, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + j * dim2Align);
                Div(mix, mix, sum1, pregLoop);
                // store normOut[1]
                StoreOutputData(normOutLocalAddr, mix, pregLoop,
                                dim0 * dim1 * dim2Align + i * dim1 * dim2Align + j * dim2Align);
                StoreOutputData(combFragLocalAddr, mix, pregLoop, i * dim1 * dim2Align + j * dim2Align);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < dim0; i++) {
            LoadInputData<float>(mix1, combFragLocalAddr, pregLoop, i * dim1 * dim2Align);
            LoadInputData<float>(mix2, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + 1 * dim2Align);
            LoadInputData<float>(mix3, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + 2 * dim2Align);
            LoadInputData<float>(mix4, combFragLocalAddr, pregLoop, i * dim1 * dim2Align + 3 * dim2Align);
            for (uint16_t j = 1; j < iters; j++) { // j: [1~iter-1]
                Duplicate(sum, static_cast<float>(0), pregLoop);

                VFProcessIterationGradout(sum, sum1, mix1, eps, pregLoop, pregOne, i, j, 0, dim0, dim1, dim2Align,
                                          sumOutLocalAddr, normOutLocalAddr);
                VFProcessIterationGradout(sum, sum2, mix2, eps, pregLoop, pregOne, i, j, 1, dim0, dim1, dim2Align,
                                          sumOutLocalAddr, normOutLocalAddr);
                VFProcessIterationGradout(sum, sum3, mix3, eps, pregLoop, pregOne, i, j, 2, dim0, dim1, dim2Align,
                                          sumOutLocalAddr, normOutLocalAddr);
                VFProcessIterationGradout(sum, sum4, mix4, eps, pregLoop, pregOne, i, j, 3, dim0, dim1, dim2Align,
                                          sumOutLocalAddr, normOutLocalAddr);

                Adds(sum, sum, eps, pregLoop);
                // store sumOut[2j + 1]
                StoreOutputData(sumOutLocalAddr, sum, pregLoop, (2 * j + 1) * dim0 * dim2Align + i * dim2Align);
                Div(mix1, mix1, sum, pregLoop);
                Div(mix2, mix2, sum, pregLoop);
                Div(mix3, mix3, sum, pregLoop);
                Div(mix4, mix4, sum, pregLoop);
                // store normOut[2j + 1]
                StoreOutputData(normOutLocalAddr, mix1, pregLoop,
                                (2 * j + 1) * dim0 * dim1 * dim2Align + i * dim1 * dim2Align + 0 * dim2Align);
                StoreOutputData(normOutLocalAddr, mix2, pregLoop,
                                (2 * j + 1) * dim0 * dim1 * dim2Align + i * dim1 * dim2Align + 1 * dim2Align);
                StoreOutputData(normOutLocalAddr, mix3, pregLoop,
                                (2 * j + 1) * dim0 * dim1 * dim2Align + i * dim1 * dim2Align + 2 * dim2Align);
                StoreOutputData(normOutLocalAddr, mix4, pregLoop,
                                (2 * j + 1) * dim0 * dim1 * dim2Align + i * dim1 * dim2Align + 3 * dim2Align);
            }
            StoreOutputData(combFragLocalAddr, mix1, pregLoop, i * dim1 * dim2Align);
            StoreOutputData(combFragLocalAddr, mix2, pregLoop, i * dim1 * dim2Align + 1 * dim2Align);
            StoreOutputData(combFragLocalAddr, mix3, pregLoop, i * dim1 * dim2Align + 2 * dim2Align);
            StoreOutputData(combFragLocalAddr, mix4, pregLoop, i * dim1 * dim2Align + 3 * dim2Align);
        }
    }
}

template <typename T>
__aicore__ inline void VFProcessY(const LocalTensor<T> &yLocal, const LocalTensor<float> &mixLocal,
                                  const LocalTensor<T> &xLocal, uint16_t bs, uint16_t hcMult, uint16_t d,
                                  uint16_t hcMix)
{
    __ubuf__ T *yLocalAddr = (__ubuf__ T *)yLocal.GetPhyAddr();
    __ubuf__ float *mixLocalAddr = (__ubuf__ float *)mixLocal.GetPhyAddr();
    __ubuf__ T *xLocalAddr = (__ubuf__ T *)xLocal.GetPhyAddr();
    uint32_t dAlign = RoundUp<T>(d);
    uint16_t loopCount = CeilDiv(d, VL_FP32);
    uint32_t hcMixAlign = RoundUp<float>(hcMix);
    if (loopCount > 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> mix;
            RegTensor<float> sum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < bs; i++) {
                uint32_t sreg = d;
                for (uint16_t j = 0; j < loopCount; j++) {
                    pregLoop = UpdateMask<float>(sreg);
                    Duplicate(sum, static_cast<float>(0), pregLoop);
                    for (uint16_t k = 0; k < hcMult; k++) {
                        LoadInputDataWithBrc<float>(mix, mixLocalAddr, pregLoop, i * hcMixAlign + k);
                        LoadInputData<T>(x, xLocalAddr, pregLoop, i * hcMult * dAlign + j * VL_FP32 + k * dAlign);
                        Mul(x, mix, x, pregLoop);
                        Add(sum, sum, x, pregLoop);
                    }
                    StoreOutputData(yLocalAddr, sum, pregLoop, i * dAlign + j * VL_FP32);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> mix;
            RegTensor<float> sum;
            uint32_t sreg = d;
            MaskReg pregLoop = UpdateMask<float>(sreg);
            for (uint16_t i = 0; i < bs; i++) {
                Duplicate(sum, static_cast<float>(0), pregLoop);
                for (uint16_t j = 0; j < hcMult; j++) {
                    LoadInputDataWithBrc<float>(mix, mixLocalAddr, pregLoop, i * hcMixAlign + j);
                    LoadInputData<T>(x, xLocalAddr, pregLoop, i * hcMult * dAlign + j * dAlign);
                    Mul(x, mix, x, pregLoop);
                    Add(sum, sum, x, pregLoop);
                }
                StoreOutputData(yLocalAddr, sum, pregLoop, i * dAlign);
            }
        }
    }
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

template <typename T>
__aicore__ inline void CopyToL1(const LocalTensor<T> &srcTensor, const LocalTensor<T> &dstTensor,
                                const DataCopyParams dataCopyXParams)
{
    DataCopy(dstTensor, srcTensor, dataCopyXParams);
}

template <typename T>
__aicore__ inline void CopyInWithLoopMode(const GlobalTensor<T> &inputGm, const LocalTensor<T> &inputTensor,
                                          const uint16_t outerLoop, const uint16_t nBurst, const uint32_t copyLen,
                                          const uint32_t gmLastDim, uint32_t srcStride = 0)
{
    uint16_t copyLenAlign = RoundUp<T>(copyLen);
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = outerLoop;
    loopParams.loop2SrcStride = 0;
    loopParams.loop1SrcStride = gmLastDim * sizeof(T);
    loopParams.loop2DstStride = 0;
    loopParams.loop1DstStride = nBurst * copyLenAlign * sizeof(T);

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
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
}

template <typename T>
__aicore__ inline void CopyOut(const LocalTensor<T> &outputTensor, const GlobalTensor<T> &outputGm,
                               const uint16_t nBurst, const uint32_t copyLen, uint32_t dstStride = 0,
                               uint32_t srcStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = srcStride;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(outputGm, outputTensor, dataCopyParams);
}

template <typename T>
__aicore__ inline void CopyOut(const LocalTensor<T> &outputTensor, const LocalTensor<T> &outputGm,
                               const uint16_t nBurst, const uint32_t copyLen, uint32_t dstStride = 0,
                               uint32_t srcStride = 0)
{
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = CeilDiv(copyLen * sizeof(T), BLOCK_SIZE);
    dataCopyParams.srcStride = srcStride;
    dataCopyParams.dstStride = dstStride;
    DataCopy(outputGm, outputTensor, dataCopyParams);
}

template <typename T>
__aicore__ inline void CopyOutWithLoopMode(const LocalTensor<T> &outputTensor, const GlobalTensor<T> &outputGm,
                                           const uint32_t outerLoop, const uint64_t outerLoopDstStride,
                                           const uint32_t innerLoop, const uint64_t innerLoopDstStride,
                                           const uint16_t nBurst, const uint32_t copyLen, uint32_t srcStride = 0,
                                           uint32_t dstStride = 0)
{
    uint16_t copyLenAlign = RoundUp<T>(copyLen);
    LoopModeParams loopParams;
    loopParams.loop2Size = outerLoop;
    loopParams.loop1Size = innerLoop;
    loopParams.loop2SrcStride = innerLoop * nBurst * copyLenAlign * sizeof(T);
    loopParams.loop1SrcStride = nBurst * copyLenAlign * sizeof(T);
    loopParams.loop2DstStride = outerLoopDstStride * sizeof(T);
    loopParams.loop1DstStride = innerLoopDstStride * sizeof(T);

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = dstStride * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyPad(outputGm, outputTensor, dataCoptExtParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

// Compact and element-major Sinkhorn vector paths.

template <bool MaxReduce = false>
__aicore__ inline void VFCompactRowReduce(RegTensor<float> &result, RegTensor<float> &value,
                                          RegTensor<uint32_t> &index0, RegTensor<uint32_t> &index1,
                                          RegTensor<uint32_t> &index2, RegTensor<uint32_t> &index3, MaskReg &mask)
{
    RegTensor<float> part1;
    RegTensor<float> part2;
    RegTensor<float> part3;
    Gather(result, value, index0);
    Gather(part1, value, index1);
    Gather(part2, value, index2);
    Gather(part3, value, index3);
    if constexpr (MaxReduce) {
        Max(result, result, part1, mask);
        Max(part2, part2, part3, mask);
        Max(result, result, part2, mask);
    } else {
        Add(result, result, part1, mask);
        Add(part2, part2, part3, mask);
        Add(result, result, part2, mask);
    }
}

__aicore__ inline void VFCompactColumnReduce(RegTensor<float> &result, RegTensor<float> &value,
                                             RegTensor<uint32_t> &index0, RegTensor<uint32_t> &index1,
                                             RegTensor<uint32_t> &index2, RegTensor<uint32_t> &index3, MaskReg &mask)
{
    RegTensor<float> part;
    Gather(result, value, index0);
    Gather(part, value, index1);
    Add(result, result, part, mask);
    Gather(part, value, index2);
    Add(result, result, part, mask);
    Gather(part, value, index3);
    Add(result, result, part, mask);
}

__aicore__ inline void VFProcessCombFragCompact(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                                const LocalTensor<float> &bias, float scale, float eps, uint16_t iters,
                                                uint16_t rows, uint16_t hcMix)
{
    __ubuf__ float *outputAddr = (__ubuf__ float *)output.GetPhyAddr();
    __ubuf__ float *inputAddr = (__ubuf__ float *)input.GetPhyAddr();
    __ubuf__ float *biasAddr = (__ubuf__ float *)bias.GetPhyAddr();
    uint32_t inputStride = RoundUp<float>(hcMix);
    constexpr uint32_t MATRIX_ELEMENTS = 16;
    constexpr uint32_t MATRICES_PER_REG = VL_FP32 / MATRIX_ELEMENTS;
    constexpr uint32_t PADDED_MATRIX_ELEMENTS = 32;
    uint16_t loops = CeilDiv(rows, MATRICES_PER_REG);
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> order;
        RegTensor<uint32_t> bits;
        RegTensor<uint32_t> element;
        RegTensor<uint32_t> token;
        RegTensor<uint32_t> inputIndex;
        RegTensor<uint32_t> outputIndex;
        RegTensor<uint32_t> biasIndex;
        RegTensor<uint32_t> col;
        RegTensor<uint32_t> row0;
        RegTensor<uint32_t> row1;
        RegTensor<uint32_t> row2;
        RegTensor<uint32_t> row3;
        RegTensor<uint32_t> col0;
        RegTensor<uint32_t> col1;
        RegTensor<uint32_t> col2;
        RegTensor<uint32_t> col3;
        MaskReg fullMask = CreateMask<float>();
        Arange((RegTensor<int32_t> &)order, static_cast<int32_t>(0));
        Duplicate(bits, static_cast<uint32_t>(15));
        And(element, order, bits, fullMask);
        ShiftRights(token, order, static_cast<int16_t>(4), fullMask);
        Muls(inputIndex, token, inputStride, fullMask);
        Add(inputIndex, inputIndex, element, fullMask);

        Duplicate(bits, static_cast<uint32_t>(3));
        And(col, element, bits, fullMask);
        ShiftRights(biasIndex, element, static_cast<int16_t>(2), fullMask);
        ShiftLefts(biasIndex, biasIndex, static_cast<int16_t>(3), fullMask);
        Add(biasIndex, biasIndex, col, fullMask);
        Muls(outputIndex, token, PADDED_MATRIX_ELEMENTS, fullMask);
        Add(outputIndex, outputIndex, biasIndex, fullMask);

        Duplicate(bits, static_cast<uint32_t>(60));
        And(row0, order, bits, fullMask);
        Adds(row1, row0, static_cast<uint32_t>(1), fullMask);
        Adds(row2, row0, static_cast<uint32_t>(2), fullMask);
        Adds(row3, row0, static_cast<uint32_t>(3), fullMask);
        Duplicate(bits, static_cast<uint32_t>(51));
        And(col0, order, bits, fullMask);
        Adds(col1, col0, static_cast<uint32_t>(4), fullMask);
        Adds(col2, col0, static_cast<uint32_t>(8), fullMask);
        Adds(col3, col0, static_cast<uint32_t>(12), fullMask);

        RegTensor<float> base;
        RegTensor<float> mix;
        RegTensor<float> sum;
        Gather(base, biasAddr, biasIndex, fullMask);
        uint32_t remaining = rows * MATRIX_ELEMENTS;
        for (uint16_t batch = 0; batch < loops; ++batch) {
            MaskReg mask = UpdateMask<float>(remaining);
            // Keep each 4x4 matrix compact through softmax and all Sinkhorn iterations.
            Gather(mix, inputAddr + batch * MATRICES_PER_REG * inputStride, inputIndex, mask);
            Muls(mix, mix, scale, mask);
            Add(mix, mix, base, mask);
            VFCompactRowReduce<true>(sum, mix, row0, row1, row2, row3, mask);
            Sub(mix, mix, sum, mask);
            Exp(mix, mix, mask);
            VFCompactRowReduce(sum, mix, row0, row1, row2, row3, mask);
            Div(mix, mix, sum, mask);
            Adds(mix, mix, eps, mask);
            VFCompactColumnReduce(sum, mix, col0, col1, col2, col3, mask);
            Adds(sum, sum, eps, mask);
            Div(mix, mix, sum, mask);
            for (uint16_t iter = 0; iter < iters; ++iter) {
                VFCompactRowReduce(sum, mix, row0, row1, row2, row3, mask);
                Adds(sum, sum, eps, mask);
                Div(mix, mix, sum, mask);
                VFCompactColumnReduce(sum, mix, col0, col1, col2, col3, mask);
                Adds(sum, sum, eps, mask);
                Div(mix, mix, sum, mask);
            }
            Scatter(outputAddr + batch * MATRICES_PER_REG * PADDED_MATRIX_ELEMENTS, mix, outputIndex, mask);
        }
    }
}

template <bool Pairwise>
__aicore__ inline void VFElementMajorSum(RegTensor<float> &sum, RegTensor<float> &a, RegTensor<float> &b,
                                         RegTensor<float> &c, RegTensor<float> &d, MaskReg &mask)
{
    Add(sum, a, b, mask);
    if constexpr (Pairwise) {
        RegTensor<float> tail;
        Add(tail, c, d, mask);
        Add(sum, sum, tail, mask);
    } else {
        Add(sum, sum, c, mask);
        Add(sum, sum, d, mask);
    }
}

template <bool Pairwise>
__aicore__ inline void VFElementMajorNormalize(RegTensor<float> &a, RegTensor<float> &b, RegTensor<float> &c,
                                               RegTensor<float> &d, float eps, MaskReg &mask)
{
    RegTensor<float> sum;
    VFElementMajorSum<Pairwise>(sum, a, b, c, d, mask);
    Adds(sum, sum, eps, mask);
    Div(a, a, sum, mask);
    Div(b, b, sum, mask);
    Div(c, c, sum, mask);
    Div(d, d, sum, mask);
}

__aicore__ inline void VFElementMajorSoftmax(RegTensor<float> &a, RegTensor<float> &b, RegTensor<float> &c,
                                             RegTensor<float> &d, float eps, MaskReg &mask)
{
    RegTensor<float> maximum;
    RegTensor<float> tail;
    RegTensor<float> sum;
    Max(maximum, a, b, mask);
    Max(tail, c, d, mask);
    Max(maximum, maximum, tail, mask);
    Sub(a, a, maximum, mask);
    Sub(b, b, maximum, mask);
    Sub(c, c, maximum, mask);
    Sub(d, d, maximum, mask);
    Exp(a, a, mask);
    Exp(b, b, mask);
    Exp(c, c, mask);
    Exp(d, d, mask);
    VFElementMajorSum<true>(sum, a, b, c, d, mask);
    Div(a, a, sum, mask);
    Div(b, b, sum, mask);
    Div(c, c, sum, mask);
    Div(d, d, sum, mask);
    Adds(a, a, eps, mask);
    Adds(b, b, eps, mask);
    Adds(c, c, eps, mask);
    Adds(d, d, eps, mask);
}

template <bool NormalizeInput, uint32_t Element>
__aicore__ inline void VFElementMajorLoad(RegTensor<float> &value, __ubuf__ float *input, __ubuf__ float *bias,
                                          RegTensor<uint32_t> &index, RegTensor<float> &invRms, float scale,
                                          MaskReg &mask)
{
    RegTensor<float> base;
    Gather(value, input + (NormalizeInput ? 8 : 0) + Element, index, mask);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(base, bias + Element / 4 * 8 + Element % 4);
    if constexpr (NormalizeInput) {
        Mul(value, value, invRms, mask);
    }
    Muls(value, value, scale, mask);
    Add(value, value, base, mask);
}

template <bool NormalizeInput = true>
__aicore__ inline void VFProcessCombFragElementMajor(const LocalTensor<float> &mm, const LocalTensor<float> &rms,
                                                     const LocalTensor<float> &bias, float scale, float eps,
                                                     float normEps, uint16_t iters, uint16_t rows)
{
    __ubuf__ float *mmAddr = (__ubuf__ float *)mm.GetPhyAddr();
    __ubuf__ float *rmsAddr = (__ubuf__ float *)rms.GetPhyAddr();
    __ubuf__ float *biasAddr = (__ubuf__ float *)bias.GetPhyAddr();
    uint16_t loops = CeilDiv(rows, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> lane;
        RegTensor<uint32_t> inputIndex;
        RegTensor<uint32_t> outputIndex;
        MaskReg fullMask = CreateMask<float>();
        Arange((RegTensor<int32_t> &)lane, static_cast<int32_t>(0));
        constexpr uint32_t INPUT_STRIDE = NormalizeInput ? 24 : 16;
        Muls(inputIndex, lane, INPUT_STRIDE, fullMask);
        Muls(outputIndex, lane, static_cast<uint32_t>(16), fullMask);
        uint32_t remaining = rows;
        for (uint16_t batch = 0; batch < loops; ++batch) {
            MaskReg mask = UpdateMask<float>(remaining);
            RegTensor<float> invRms;
            RegTensor<float> one;
            if constexpr (NormalizeInput) {
                Gather(invRms, rmsAddr + batch * VL_FP32, lane, mask);
                Adds(invRms, invRms, normEps, mask);
                Sqrt(invRms, invRms, mask);
                Duplicate(one, static_cast<float>(1), mask);
                Div(invRms, one, invRms, mask);
            }

            RegTensor<float> m00, m01, m02, m03;
            RegTensor<float> m10, m11, m12, m13;
            RegTensor<float> m20, m21, m22, m23;
            RegTensor<float> m30, m31, m32, m33;
            __ubuf__ float *input = mmAddr + batch * VL_FP32 * INPUT_STRIDE;
            VFElementMajorLoad<NormalizeInput, 0>(m00, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 1>(m01, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 2>(m02, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 3>(m03, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 4>(m10, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 5>(m11, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 6>(m12, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 7>(m13, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 8>(m20, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 9>(m21, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 10>(m22, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 11>(m23, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 12>(m30, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 13>(m31, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 14>(m32, input, biasAddr, inputIndex, invRms, scale, mask);
            VFElementMajorLoad<NormalizeInput, 15>(m33, input, biasAddr, inputIndex, invRms, scale, mask);

            VFElementMajorSoftmax(m00, m01, m02, m03, eps, mask);
            VFElementMajorSoftmax(m10, m11, m12, m13, eps, mask);
            VFElementMajorSoftmax(m20, m21, m22, m23, eps, mask);
            VFElementMajorSoftmax(m30, m31, m32, m33, eps, mask);
            VFElementMajorNormalize<false>(m00, m10, m20, m30, eps, mask);
            VFElementMajorNormalize<false>(m01, m11, m21, m31, eps, mask);
            VFElementMajorNormalize<false>(m02, m12, m22, m32, eps, mask);
            VFElementMajorNormalize<false>(m03, m13, m23, m33, eps, mask);
            for (uint16_t iter = 0; iter < iters; ++iter) {
                VFElementMajorNormalize<true>(m00, m01, m02, m03, eps, mask);
                VFElementMajorNormalize<true>(m10, m11, m12, m13, eps, mask);
                VFElementMajorNormalize<true>(m20, m21, m22, m23, eps, mask);
                VFElementMajorNormalize<true>(m30, m31, m32, m33, eps, mask);
                VFElementMajorNormalize<false>(m00, m10, m20, m30, eps, mask);
                VFElementMajorNormalize<false>(m01, m11, m21, m31, eps, mask);
                VFElementMajorNormalize<false>(m02, m12, m22, m32, eps, mask);
                VFElementMajorNormalize<false>(m03, m13, m23, m33, eps, mask);
            }

            // All input elements of this batch have been loaded. Compact output
            // ends before the next batch's 24-element input, so in-place reuse is safe.
            __ubuf__ float *output = mmAddr + batch * VL_FP32 * 16;
            Scatter(output + 0, m00, outputIndex, mask);
            Scatter(output + 1, m01, outputIndex, mask);
            Scatter(output + 2, m02, outputIndex, mask);
            Scatter(output + 3, m03, outputIndex, mask);
            Scatter(output + 4, m10, outputIndex, mask);
            Scatter(output + 5, m11, outputIndex, mask);
            Scatter(output + 6, m12, outputIndex, mask);
            Scatter(output + 7, m13, outputIndex, mask);
            Scatter(output + 8, m20, outputIndex, mask);
            Scatter(output + 9, m21, outputIndex, mask);
            Scatter(output + 10, m22, outputIndex, mask);
            Scatter(output + 11, m23, outputIndex, mask);
            Scatter(output + 12, m30, outputIndex, mask);
            Scatter(output + 13, m31, outputIndex, mask);
            Scatter(output + 14, m32, outputIndex, mask);
            Scatter(output + 15, m33, outputIndex, mask);
        }
    }
}

} // namespace MhcPreSinkhornNs

#endif
