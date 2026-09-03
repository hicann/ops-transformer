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
 * \file mhc_pre_sinkhorn_backward_deterministic.h
 * \brief
 */
#ifndef MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_SIMD_VF_H
#define MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_SIMD_VF_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

using namespace AscendC;

constexpr uint64_t DOUBLE_BUFFER = 2;

struct BsCLoopInfo {
    int64_t bsLen = 0;
    int64_t cLen = 0;
    int64_t cLenXTAlign = 0;
    int64_t cLenUAlign = 0;
    int64_t cLenGRADHINTAlign = 0;
};

constexpr static AscendC::Reg::CastTrait castTrait16ToFloat = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait castTraitFloatTo16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename T>
constexpr T AlignUp(T value, T align)
{
    return (value + align - 1) / align * align;
}

template <typename PARAM_T>
__aicore__ inline void CopyIn(const LocalTensor<PARAM_T> &dstTensor, const GlobalTensor<PARAM_T> &srcTensor,
                              int64_t repTime, int64_t dataLen, uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(repTime), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                    static_cast<uint32_t>(gmStride), static_cast<uint32_t>(ubStride),
                                    static_cast<uint32_t>(0)};
    DataCopyPadExtParams<PARAM_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                               static_cast<PARAM_T>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
}

template <typename PARAM_T>
__aicore__ inline void CopyOut(const GlobalTensor<PARAM_T> &dstTensor, const LocalTensor<PARAM_T> &srcTensor,
                               int64_t repTime, int64_t dataLen, uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(repTime), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                    static_cast<uint32_t>(ubStride), static_cast<uint32_t>(gmStride),
                                    static_cast<uint32_t>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams);
}

template <typename U, bool ISPRE>
__aicore__ inline void SigmoidGrad(__local_mem__ U *gradHAddr, __local_mem__ U *zAddr, __local_mem__ U *gradZAddr,
                                   uint64_t bsLen, uint64_t n, uint64_t vRegSize, float hcEps)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (bsLen * n + vfLen - 1) / vfLen;
    float hcEpsNeg = hcEps * (-1);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> onesReg;
        AscendC::Reg::RegTensor<U> zReg;
        AscendC::Reg::RegTensor<U> gradHReg;
        AscendC::Reg::RegTensor<U> sigmaReg;
        AscendC::Reg::RegTensor<U> dysigmaReg;
        AscendC::Reg::RegTensor<U> dysigma2Reg;
        AscendC::Reg::RegTensor<U> gradZReg;
        AscendC::Reg::MaskReg maskReg;
        uint32_t maskLen = static_cast<uint32_t>(bsLen * n);
        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::Reg::UpdateMask<U>(maskLen);
            Reg::Duplicate(onesReg, U(1), maskReg);
            AscendC::Reg::AddrReg zOfst = AscendC::Reg::CreateAddrReg<U>(i, vfLen);
            AscendC::Reg::AddrReg gradHOfst = AscendC::Reg::CreateAddrReg<U>(i, vfLen);
            AscendC::Reg::AddrReg gradZOfst = AscendC::Reg::CreateAddrReg<U>(i, vfLen);
            AscendC::Reg::DataCopy(zReg, zAddr, zOfst);
            AscendC::Reg::DataCopy(gradHReg, gradHAddr, gradHOfst);
            AscendC::Reg::Neg(zReg, zReg, maskReg);
            AscendC::Reg::Exp(zReg, zReg, maskReg);
            AscendC::Reg::Add(zReg, zReg, onesReg, maskReg);
            AscendC::Reg::Div(sigmaReg, onesReg, zReg, maskReg);
            if constexpr (ISPRE) {
                AscendC::Reg::Adds(sigmaReg, sigmaReg, hcEpsNeg, maskReg);
            }
            AscendC::Reg::Sub(dysigma2Reg, onesReg, sigmaReg, maskReg);
            AscendC::Reg::Mul(dysigmaReg, sigmaReg, gradHReg, maskReg);
            AscendC::Reg::Mul(gradZReg, dysigmaReg, dysigma2Reg, maskReg);
            if constexpr (!ISPRE) {
                AscendC::Reg::Muls(gradZReg, gradZReg, U(2), maskReg);
            }
            AscendC::Reg::DataCopy(gradZAddr, gradZReg, gradZOfst, maskReg);
        }
    }
}

template <typename U>
__aicore__ inline void ComputeGradBias(__local_mem__ U *gradBiasAddr, __local_mem__ U *gradZAddr, uint64_t bsLen,
                                       uint64_t colLen, uint64_t vRegSize)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> gradZReg;
        AscendC::Reg::RegTensor<U> gradBiasReg;
        AscendC::Reg::MaskReg maskReg;
        AscendC::Reg::UnalignReg u0;
        AscendC::Reg::MaskReg allMaskReg = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            auto gradBiasStart = gradBiasAddr + i * vfLen;
            maskReg = AscendC::Reg::UpdateMask<U>(maskLen);
            Reg::Duplicate(gradBiasReg, U(0), allMaskReg);
            for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
                auto gradZAddrStart = gradZAddr + j * colLen + i * vfLen;
                AscendC::Reg::LoadUnAlignPre(u0, gradZAddrStart);
                AscendC::Reg::LoadUnAlign(gradZReg, u0, gradZAddrStart);
                AscendC::Reg::Add(gradBiasReg, gradBiasReg, gradZReg, maskReg);
            }
            AscendC::Reg::DataCopy(gradBiasStart, gradBiasReg, maskReg);
        }
    }
}

template <typename U>
__aicore__ inline void ComputeGradAlpha(__local_mem__ U *normOutForwardPartAddr, __local_mem__ U *gradZAddr,
                                        __local_mem__ U *gradAlphaAddr, uint64_t bsLen, uint64_t colLen,
                                        uint64_t vRegSize)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (bsLen * colLen + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> normOutForwardPartReg;
        AscendC::Reg::RegTensor<U> gradZReg;
        AscendC::Reg::RegTensor<U> gradAlphaReg;
        AscendC::Reg::RegTensor<U> gradAlphaLoopSumReg;
        AscendC::Reg::RegTensor<U> gradAlphaSumReg;
        AscendC::Reg::MaskReg maskReg;
        AscendC::Reg::MaskReg OneMaskReg = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::VL1>();
        Reg::Duplicate(gradAlphaSumReg, U(0), OneMaskReg);
        uint32_t maskLen = static_cast<uint32_t>(bsLen * colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::Reg::UpdateMask<U>(maskLen);
            AscendC::Reg::AddrReg normOutForwardPartOfst = AscendC::Reg::CreateAddrReg<U>(i, vfLen);
            AscendC::Reg::AddrReg gradZOfst = AscendC::Reg::CreateAddrReg<U>(i, vfLen);
            AscendC::Reg::DataCopy(normOutForwardPartReg, normOutForwardPartAddr, normOutForwardPartOfst);
            AscendC::Reg::DataCopy(gradZReg, gradZAddr, gradZOfst);
            AscendC::Reg::Mul(gradAlphaReg, normOutForwardPartReg, gradZReg, maskReg);
            AscendC::Reg::Reduce<Reg::ReduceType::SUM, U, U>(gradAlphaLoopSumReg, gradAlphaReg,
                                                             maskReg); // reduce_sum
            AscendC::Reg::Add(gradAlphaSumReg, gradAlphaSumReg, gradAlphaLoopSumReg, OneMaskReg);
        }
        AscendC::Reg::DataCopy(gradAlphaAddr, gradAlphaSumReg, OneMaskReg);
    }
}

template <typename U>
__aicore__ inline void ComputeZAndForwardPart(__local_mem__ U *zAddr, __local_mem__ U *normOutForwardAddr,
                                              __local_mem__ U *biasAddr, uint64_t bsLen, uint64_t colLen,
                                              uint64_t cLenAlign, uint64_t vRegSize, const U &alpha)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> zReg;
        AscendC::Reg::RegTensor<U> normOutForwardReg;
        AscendC::Reg::RegTensor<U> biasReg;
        AscendC::Reg::MaskReg valueMaskReg;
        AscendC::Reg::UnalignRegForStore unStoreZReg;

        for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
            AscendC::Reg::UnalignReg unAlignNormOutForwardReg;
            AscendC::Reg::UnalignReg unAlignBiasReg;
            auto biasAddrStart = biasAddr;
            auto zAddrStart = zAddr + j * colLen;
            auto normOutForwardAddrStart = normOutForwardAddr + j * colLen;
            uint32_t maskLen = static_cast<uint32_t>(colLen);
            for (uint16_t i = 0; i < loopCnt; i++) {
                valueMaskReg = AscendC::Reg::UpdateMask<U>(maskLen);
                AscendC::Reg::LoadUnAlignPre(unAlignNormOutForwardReg, normOutForwardAddrStart + i * vfLen);
                AscendC::Reg::LoadUnAlign(normOutForwardReg, unAlignNormOutForwardReg,
                                          normOutForwardAddrStart + i * vfLen);
                AscendC::Reg::LoadUnAlignPre(unAlignBiasReg, biasAddrStart + i * vfLen);
                AscendC::Reg::LoadUnAlign(biasReg, unAlignBiasReg, biasAddrStart + i * vfLen);
                AscendC::Reg::Muls(zReg, normOutForwardReg, alpha, valueMaskReg);
                AscendC::Reg::Add(zReg, zReg, biasReg, valueMaskReg);
                Reg::StoreUnAlign<U>(zAddrStart, zReg, unStoreZReg, vfLen);
            }
            Reg::StoreUnAlignPost<U>(zAddrStart, unStoreZReg, 0);
        }
    }
}

template <typename U>
__aicore__ inline void ComputeForwardPart(__local_mem__ U *hcBeformNormAddr, __local_mem__ U *invRmsAddr,
                                          __local_mem__ U *normOutForwardAddr, uint64_t bsLen, uint64_t colLen,
                                          uint64_t cLenAlign, uint64_t vRegSize, const U &alpha)
{
    uint32_t vfLen = vRegSize / sizeof(U);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> hcBeformNormReg;
        AscendC::Reg::RegTensor<U> normOutForwardReg;
        AscendC::Reg::MaskReg valueMaskReg;
        AscendC::Reg::UnalignRegForStore unStoreNormOutForwardReg;

        for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
            auto normOutForwardAddrStart = normOutForwardAddr + j * colLen;
            auto hcBeformNormAddrStart = hcBeformNormAddr + j * cLenAlign;
            U invRms = *(invRmsAddr + j);
            uint32_t maskLen = static_cast<uint32_t>(colLen);
            for (uint16_t i = 0; i < loopCnt; i++) {
                valueMaskReg = AscendC::Reg::UpdateMask<U>(maskLen);
                AscendC::Reg::AddrReg hcBeformNormOfst = AscendC::Reg::CreateAddrReg<U>(i, vfLen);
                AscendC::Reg::DataCopy(hcBeformNormReg, hcBeformNormAddrStart, hcBeformNormOfst);
                AscendC::Reg::Muls(normOutForwardReg, hcBeformNormReg, invRms, valueMaskReg);
                Reg::StoreUnAlign<U>(normOutForwardAddrStart, normOutForwardReg, unStoreNormOutForwardReg, vfLen);
            }
            Reg::StoreUnAlignPost<U>(normOutForwardAddrStart, unStoreNormOutForwardReg, 0);
        }
    }
}
/*****************************************************************************
 * 计算GradNormOut和GradHcBeforeNorm的值
 * 注意事项：
 *   1、需要将该函数插入到GradNormPre,GradNormPre,GradNormRes每核每次BS计算完成后
 *   2、计算完成GradHcBeforeNorm后需添加syncAll，保证matmul的确定性。
 ******************************************************************************/
template <typename T>
__aicore__ inline void ComputeMulScalar(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, T scalar,
                                        uint32_t count, uint32_t offset)
{
    auto srcAddr = (__local_mem__ T *)srcLocal.GetPhyAddr() + offset;
    auto dstAddr = (__local_mem__ T *)dstLocal.GetPhyAddr();
    uint32_t repeatCount = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t maskCount = count;
    uint16_t repeatTimes = Ops::Base::CeilDiv(maskCount, repeatCount);

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> reg0;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            Reg::MaskReg preg = Reg::UpdateMask<T>(maskCount);
            Reg::UnalignReg u0;
            Reg::DataCopyUnAlignPre(u0, srcAddr);
            Reg::DataCopyUnAlign(reg0, u0, srcAddr, repeatCount);
            Reg::Muls(reg0, reg0, scalar, preg);
            Reg::DataCopy(dstAddr + i * repeatCount, reg0, preg);
        }
    }
}

template <typename T>
__aicore__ inline void ComputeGradInvRms(const LocalTensor<T> &gradNormOutLocal,
                                         const LocalTensor<T> &hcBeforeNormLocal, int64_t count, T &gradInvRms)
{
    Mul(hcBeforeNormLocal, gradNormOutLocal, hcBeforeNormLocal, count);
    ReduceSum<T>(hcBeforeNormLocal, hcBeforeNormLocal, hcBeforeNormLocal, count);
    gradInvRms = hcBeforeNormLocal(0);
}

template <typename T>
__aicore__ inline void ComputeGradXFromRmsVF(const LocalTensor<T> &xFp32Local, const LocalTensor<T> &xFp32LocalOut,
                                             T gradInvRms, T invRms, int64_t count, int64_t ncCount)
{
    __local_mem__ T *xFp32Addr = (__local_mem__ T *)xFp32Local.GetPhyAddr();
    __local_mem__ T *xFp32AddrOut = (__local_mem__ T *)xFp32LocalOut.GetPhyAddr();
    uint32_t repeatCount = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t maskCount = count;
    uint16_t repeatTimes = Ops::Base::CeilDiv(maskCount, repeatCount);

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xFp32Reg;
        Reg::RegTensor<T> ncCountReg;
        Reg::Duplicate(ncCountReg, static_cast<T>(ncCount));

        for (uint16_t i = 0; i < repeatTimes; i++) {
            Reg::MaskReg preg = Reg::UpdateMask<T>(maskCount);
            Reg::AddrReg offset = Reg::CreateAddrReg<T>(i, repeatCount);
            Reg::DataCopy(xFp32Reg, (__local_mem__ T *)xFp32Addr, offset);
            Reg::Muls(xFp32Reg, xFp32Reg, static_cast<T>(-invRms), preg);
            Reg::Muls(xFp32Reg, xFp32Reg, gradInvRms, preg);
            Reg::Div(xFp32Reg, xFp32Reg, ncCountReg, preg);
            Reg::DataCopy(xFp32AddrOut + i * repeatCount, xFp32Reg, preg);
        }
    }
}

#endif //  MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_SIMD_VF_H
