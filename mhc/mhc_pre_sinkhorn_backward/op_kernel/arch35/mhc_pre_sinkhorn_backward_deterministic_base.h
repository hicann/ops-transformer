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

constexpr static AscendC::MicroAPI::CastTrait castTrait16ToFloat = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTraitFloatTo16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
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
        AscendC::MicroAPI::RegTensor<U> onesReg;
        AscendC::MicroAPI::RegTensor<U> zReg;
        AscendC::MicroAPI::RegTensor<U> gradHReg;
        AscendC::MicroAPI::RegTensor<U> sigmaReg;
        AscendC::MicroAPI::RegTensor<U> dysigmaReg;
        AscendC::MicroAPI::RegTensor<U> dysigma2Reg;
        AscendC::MicroAPI::RegTensor<U> gradZReg;
        AscendC::MicroAPI::MaskReg maskReg;
        uint32_t maskLen = static_cast<uint32_t>(bsLen * n);
        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            MicroAPI::Duplicate(onesReg, U(1), maskReg);
            AscendC::MicroAPI::AddrReg zOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg gradHOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg gradZOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::DataCopy(zReg, zAddr, zOfst);
            AscendC::MicroAPI::DataCopy(gradHReg, gradHAddr, gradHOfst);
            AscendC::MicroAPI::Neg(zReg, zReg, maskReg);
            AscendC::MicroAPI::Exp(zReg, zReg, maskReg);
            AscendC::MicroAPI::Add(zReg, zReg, onesReg, maskReg);
            AscendC::MicroAPI::Div(sigmaReg, onesReg, zReg, maskReg);
            if constexpr (ISPRE) {
                AscendC::MicroAPI::Adds(sigmaReg, sigmaReg, hcEpsNeg, maskReg);
            }
            AscendC::MicroAPI::Sub(dysigma2Reg, onesReg, sigmaReg, maskReg);
            AscendC::MicroAPI::Mul(dysigmaReg, sigmaReg, gradHReg, maskReg);
            AscendC::MicroAPI::Mul(gradZReg, dysigmaReg, dysigma2Reg, maskReg);
            if constexpr (!ISPRE) {
                AscendC::MicroAPI::Muls(gradZReg, gradZReg, U(2), maskReg);
            }
            AscendC::MicroAPI::DataCopy(gradZAddr, gradZReg, gradZOfst, maskReg);
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
        AscendC::MicroAPI::RegTensor<U> gradZReg;
        AscendC::MicroAPI::RegTensor<U> gradBiasReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::MaskReg allMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            auto gradBiasStart = gradBiasAddr + i * vfLen;
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            MicroAPI::Duplicate(gradBiasReg, U(0), allMaskReg);
            for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
                auto gradZAddrStart = gradZAddr + j * colLen + i * vfLen;
                AscendC::MicroAPI::LoadUnAlignPre(u0, gradZAddrStart);
                AscendC::MicroAPI::LoadUnAlign(gradZReg, u0, gradZAddrStart);
                AscendC::MicroAPI::Add(gradBiasReg, gradBiasReg, gradZReg, maskReg);
            }
            AscendC::MicroAPI::DataCopy(gradBiasStart, gradBiasReg, maskReg);
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
        AscendC::MicroAPI::RegTensor<U> normOutForwardPartReg;
        AscendC::MicroAPI::RegTensor<U> gradZReg;
        AscendC::MicroAPI::RegTensor<U> gradAlphaReg;
        AscendC::MicroAPI::RegTensor<U> gradAlphaLoopSumReg;
        AscendC::MicroAPI::RegTensor<U> gradAlphaSumReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();
        MicroAPI::Duplicate(gradAlphaSumReg, U(0), OneMaskReg);
        uint32_t maskLen = static_cast<uint32_t>(bsLen * colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
            AscendC::MicroAPI::AddrReg normOutForwardPartOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::AddrReg gradZOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
            AscendC::MicroAPI::DataCopy(normOutForwardPartReg, normOutForwardPartAddr, normOutForwardPartOfst);
            AscendC::MicroAPI::DataCopy(gradZReg, gradZAddr, gradZOfst);
            AscendC::MicroAPI::Mul(gradAlphaReg, normOutForwardPartReg, gradZReg, maskReg);
            AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradAlphaLoopSumReg, gradAlphaReg,
                                                                       maskReg); // reduce_sum
            AscendC::MicroAPI::Add(gradAlphaSumReg, gradAlphaSumReg, gradAlphaLoopSumReg, OneMaskReg);
        }
        AscendC::MicroAPI::DataCopy(gradAlphaAddr, gradAlphaSumReg, OneMaskReg);
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
        AscendC::MicroAPI::RegTensor<U> zReg;
        AscendC::MicroAPI::RegTensor<U> normOutForwardReg;
        AscendC::MicroAPI::RegTensor<U> biasReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::UnalignRegForStore unStoreZReg;

        for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
            AscendC::MicroAPI::UnalignReg unAlignNormOutForwardReg;
            AscendC::MicroAPI::UnalignReg unAlignBiasReg;
            auto biasAddrStart = biasAddr;
            auto zAddrStart = zAddr + j * colLen;
            auto normOutForwardAddrStart = normOutForwardAddr + j * colLen;
            uint32_t maskLen = static_cast<uint32_t>(colLen);
            for (uint16_t i = 0; i < loopCnt; i++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::LoadUnAlignPre(unAlignNormOutForwardReg, normOutForwardAddrStart + i * vfLen);
                AscendC::MicroAPI::LoadUnAlign(normOutForwardReg, unAlignNormOutForwardReg,
                                               normOutForwardAddrStart + i * vfLen);
                AscendC::MicroAPI::LoadUnAlignPre(unAlignBiasReg, biasAddrStart + i * vfLen);
                AscendC::MicroAPI::LoadUnAlign(biasReg, unAlignBiasReg, biasAddrStart + i * vfLen);
                AscendC::MicroAPI::Muls(zReg, normOutForwardReg, alpha, valueMaskReg);
                AscendC::MicroAPI::Add(zReg, zReg, biasReg, valueMaskReg);
                MicroAPI::StoreUnAlign<U>(zAddrStart, zReg, unStoreZReg, vfLen);
            }
            MicroAPI::StoreUnAlignPost<U>(zAddrStart, unStoreZReg, 0);
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
        AscendC::MicroAPI::RegTensor<U> hcBeformNormReg;
        AscendC::MicroAPI::RegTensor<U> normOutForwardReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::UnalignRegForStore unStoreNormOutForwardReg;

        for (uint16_t j = 0; j < static_cast<uint16_t>(bsLen); j++) {
            auto normOutForwardAddrStart = normOutForwardAddr + j * colLen;
            auto hcBeformNormAddrStart = hcBeformNormAddr + j * cLenAlign;
            U invRms = *(invRmsAddr + j);
            uint32_t maskLen = static_cast<uint32_t>(colLen);
            for (uint16_t i = 0; i < loopCnt; i++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::AddrReg hcBeformNormOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
                AscendC::MicroAPI::DataCopy(hcBeformNormReg, hcBeformNormAddrStart, hcBeformNormOfst);
                AscendC::MicroAPI::Muls(normOutForwardReg, hcBeformNormReg, invRms, valueMaskReg);
                MicroAPI::StoreUnAlign<U>(normOutForwardAddrStart, normOutForwardReg, unStoreNormOutForwardReg, vfLen);
            }
            MicroAPI::StoreUnAlignPost<U>(normOutForwardAddrStart, unStoreNormOutForwardReg, 0);
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
        MicroAPI::RegTensor<T> reg0;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(maskCount);
            MicroAPI::UnalignReg u0;
            MicroAPI::DataCopyUnAlignPre(u0, srcAddr);
            MicroAPI::DataCopyUnAlign(reg0, u0, srcAddr, repeatCount);
            MicroAPI::Muls(reg0, reg0, scalar, preg);
            MicroAPI::DataCopy(dstAddr + i * repeatCount, reg0, preg);
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
        MicroAPI::RegTensor<T> xFp32Reg;
        MicroAPI::RegTensor<T> ncCountReg;
        MicroAPI::Duplicate(ncCountReg, static_cast<T>(ncCount));

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(maskCount);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, repeatCount);
            MicroAPI::DataCopy(xFp32Reg, (__local_mem__ T *)xFp32Addr, offset);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, static_cast<T>(-invRms), preg);
            MicroAPI::Muls(xFp32Reg, xFp32Reg, gradInvRms, preg);
            MicroAPI::Div(xFp32Reg, xFp32Reg, ncCountReg, preg);
            MicroAPI::DataCopy(xFp32AddrOut + i * repeatCount, xFp32Reg, preg);
        }
    }
}

#endif //  MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_SIMD_VF_H
