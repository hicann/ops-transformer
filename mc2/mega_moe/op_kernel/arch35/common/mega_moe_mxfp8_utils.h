/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_MXFP8_UTILS_H
#define MEGA_MOE_MXFP8_UTILS_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "mega_moe_constants.h"
#if __has_include("../../../common/quantize_functions.h")
#include "../../../common/quantize_functions.h"
#else
#include "../../../../common/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

namespace Mxfp8 {

template <typename InputType, typename Fp8Type>
__aicore__ inline void ComputeFp8Token(__ubuf__ InputType *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                       __ubuf__ uint16_t *mxScaleAddr, __ubuf__ uint16_t *halfScaleAddr,
                                       __ubuf__ int8_t *outDataAddr, uint32_t processLen, uint32_t scaleNum)
{
    Quant::ComputeMaxExp(srcAddr, maxExpAddr, processLen);
    Quant::ComputeScale<Fp8Type>(maxExpAddr, mxScaleAddr, halfScaleAddr, scaleNum);
    Quant::ComputeFp8Data<InputType, Fp8Type, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
        srcAddr, halfScaleAddr, outDataAddr, processLen);
}

// 将一行 BF16 token 量化为带 padding 的 MXFP8 data + scale 记录。
template <uint8_t QuantMode, typename ExpandXType>
__aicore__ inline void QuantMxFp8(LocalTensor<ExpandXType> &outLocal, LocalTensor<ExpandXType> &inLocal,
                                  LocalTensor<float> &floatTemp, int32_t processLen)
{
    uint32_t mxScaleNum = Ops::Base::CeilAlign(
        Ops::Base::CeilDiv(static_cast<uint32_t>(processLen), static_cast<uint32_t>(ALIGN_32)), 2U);
    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
    LocalTensor<Fp8Type> castFp8LocalTensor = outLocal.template ReinterpretCast<Fp8Type>();
    __ubuf__ ExpandXType *srcAddr = (__ubuf__ ExpandXType *)inLocal.GetPhyAddr();
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)floatTemp.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr =
        (__ubuf__ uint16_t *)floatTemp[Ops::Base::CeilAlign(mxScaleNum, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr();
    __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)castFp8LocalTensor.GetPhyAddr();
    uint32_t tokenStorageElementCount =
        Ops::Base::CeilAlign(static_cast<uint32_t>(processLen), static_cast<uint32_t>(ALIGN_256));
    __ubuf__ uint16_t *mxScaleLocalAddr =
        (__ubuf__ uint16_t *)castFp8LocalTensor[tokenStorageElementCount].GetPhyAddr();
    ComputeFp8Token<ExpandXType, Fp8Type>(srcAddr, maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, outLocalAddr,
                                          static_cast<uint32_t>(processLen), mxScaleNum);
}

// 将一条 MXFP8 token 记录反量化为 FP32，供 Unpermute 累加。
template <typename T, typename XType>
__aicore__ inline void DeQuantMxFp8(LocalTensor<XType> &inLocal, LocalTensor<float> &sumTensor,
                                    LocalTensor<bfloat16_t> &scaleBf16Tensor, LocalTensor<float> &scaleFP32Tensor,
                                    uint32_t scaleLen, uint32_t tokenLen)
{
    LocalTensor<T> castFp8LocalTensor_ = inLocal.template ReinterpretCast<T>();
    LocalTensor<fp8_e8m0_t> scaleDivFp8Tensor_ =
        inLocal[Ops::Base::CeilAlign(tokenLen, static_cast<uint32_t>(ALIGN_256)) / 2]
            .template ReinterpretCast<fp8_e8m0_t>();
    __ubuf__ bfloat16_t *dyScaleBf16Ptr = (__ubuf__ bfloat16_t *)scaleBf16Tensor.GetPhyAddr();
    __ubuf__ float *dyScaleFp32Ptr = (__ubuf__ float *)scaleFP32Tensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *srcPtr0 = (__ubuf__ fp8_e8m0_t *)scaleDivFp8Tensor_.GetPhyAddr();
    __ubuf__ T *tokenPtr0 = (__ubuf__ T *)castFp8LocalTensor_.GetPhyAddr();
    __ubuf__ float *sumDstPtr = (__ubuf__ float *)sumTensor.GetPhyAddr();
    uint32_t bf16RepeatSize = Quant::GetVRegSizeDispatch() / sizeof(bfloat16_t);
    uint32_t fp32RepeatSize = Quant::GetVRegSizeDispatch() / sizeof(float);
    uint16_t repeatTimes = Ops::Base::CeilDiv(scaleLen, bf16RepeatSize);
    uint16_t fp32RepeatTimes = Ops::Base::CeilDiv(tokenLen, fp32RepeatSize);
    uint16_t repeatTimes2 = Ops::Base::CeilDiv(scaleLen * 2, fp32RepeatSize);
    uint32_t quantCount2 = scaleLen * 2;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<fp8_e8m0_t> vSrcReg;
        AscendC::MicroAPI::RegTensor<T> tokenSrcReg;
        AscendC::MicroAPI::RegTensor<float> tokenFp32SrcReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vDstReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> dyScaleBf16Reg;
        AscendC::MicroAPI::RegTensor<float> dyScaleFp32Reg;
        AscendC::MicroAPI::RegTensor<float> sumDstReg;
        AscendC::MicroAPI::RegTensor<float> sumLocalDstReg;
        static constexpr AscendC::MicroAPI::CastTrait FP82BF16CastTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait FP162FP32CastTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg maskReg1;
        AscendC::MicroAPI::MaskReg maskReg2;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<bfloat16_t>(scaleLen);
            MicroAPI::DataCopy<fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(vSrcReg, srcPtr0 + i * bf16RepeatSize);
            MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, FP82BF16CastTraitZero>(vDstReg, vSrcReg, maskReg);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(dyScaleBf16Ptr + i * bf16RepeatSize * 2,
                                                                                vDstReg, vDstReg, maskReg);
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < repeatTimes2; i++) {
            maskReg1 = AscendC::MicroAPI::UpdateMask<float>(quantCount2);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(dyScaleBf16Reg,
                                                                                dyScaleBf16Ptr + i * fp32RepeatSize);
            MicroAPI::Cast<float, bfloat16_t, FP162FP32CastTraitZero>(dyScaleFp32Reg, dyScaleBf16Reg, maskReg1);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(dyScaleFp32Ptr + i * fp32RepeatSize * 2,
                                                                           dyScaleFp32Reg, dyScaleFp32Reg, maskReg1);
        }
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < fp32RepeatTimes; i++) {
            maskReg2 = AscendC::MicroAPI::UpdateMask<float>(tokenLen);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(dyScaleFp32Reg, dyScaleFp32Ptr + i * 8);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(tokenSrcReg, tokenPtr0 + i * fp32RepeatSize);
            MicroAPI::Cast<float, T, FP82BF16CastTraitZero>(tokenFp32SrcReg, tokenSrcReg, maskReg2);
            MicroAPI::Mul(sumLocalDstReg, dyScaleFp32Reg, tokenFp32SrcReg, maskReg2);
            MicroAPI::DataCopy(sumDstPtr + i * fp32RepeatSize, sumLocalDstReg, maskReg2);
        }
    }
}

} // namespace Mxfp8
} // namespace MegaMoeImpl

#endif // MEGA_MOE_MXFP8_UTILS_H
