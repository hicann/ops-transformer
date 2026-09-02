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
 * \file ring_attention_update_regbase_common.h
 * \brief
 */
#ifndef _RING_ATTENTION_UPDATE_REGBASE_COMMON_H_
#define _RING_ATTENTION_UPDATE_REGBASE_COMMON_H_
#include "kernel_operator.h"

namespace RingAttentionUpdateRegbase {

using namespace AscendC;

/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

constexpr static uint32_t VL_FP32 = GetVRegSize() / sizeof(float);
constexpr static AscendC::MicroAPI::CastTrait castTrait0 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::MicroAPI::CastTrait castTrait1 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (unlikely(b == 0)) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline void SoftmaxCompute(LocalTensor<float> prevCurSoftmaxMaxLocal,
                                      LocalTensor<float> prevCurSoftmaxSumLocal, LocalTensor<float> softmaxMaxLocal,
                                      LocalTensor<float> softmaxSumLocal, LocalTensor<float> tempFp32Buf1Local,
                                      LocalTensor<float> tempFp32Buf2Local, int64_t softmaxEleNumLoop, int64_t calNum)
{
    auto prevSoftmaxMaxUbAddr = (__ubuf__ float *)prevCurSoftmaxMaxLocal.GetPhyAddr();
    auto curSoftmaxMaxUbAddr = (__ubuf__ float *)prevCurSoftmaxMaxLocal.GetPhyAddr() + softmaxEleNumLoop;
    auto prevSoftmaxSumUbAddr = (__ubuf__ float *)prevCurSoftmaxSumLocal.GetPhyAddr();
    auto curSoftmaxSumUbAddr = (__ubuf__ float *)prevCurSoftmaxSumLocal.GetPhyAddr() + softmaxEleNumLoop;
    auto softmaxMaxUbAddr = (__ubuf__ float *)softmaxMaxLocal.GetPhyAddr();
    auto softmaxSumUbAddr = (__ubuf__ float *)softmaxSumLocal.GetPhyAddr();
    auto tempFp32Buf1UbAddr = (__ubuf__ float *)tempFp32Buf1Local.GetPhyAddr(); // prev_out_scale
    auto tempFp32Buf2UbAddr = (__ubuf__ float *)tempFp32Buf2Local.GetPhyAddr(); // cur_out_scale
    uint16_t repeatTimes = (uint16_t)CeilDiv(calNum, (int64_t)VL_FP32);         // 实际计算量
    uint32_t count = uint32_t(calNum);                                          // 实际计算量
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::RegTensor<float> vreg0;
        AscendC::MicroAPI::RegTensor<float> vreg1;
        AscendC::MicroAPI::RegTensor<float> vreg2;
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> vreg4;
        AscendC::MicroAPI::RegTensor<float> vreg5;
        AscendC::MicroAPI::RegTensor<float> vreg6;
        AscendC::MicroAPI::RegTensor<float> vreg7;
        AscendC::MicroAPI::RegTensor<float> vreg8;
        AscendC::MicroAPI::RegTensor<float> vreg9;
        AscendC::MicroAPI::RegTensor<float> vreg10;
        AscendC::MicroAPI::RegTensor<float> vreg11;
        AscendC::MicroAPI::RegTensor<float> outreg0;
        AscendC::MicroAPI::RegTensor<float> outreg1;
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true};

        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskReg = AscendC::MicroAPI::UpdateMask<float>(count);
            // copyin prevSoftmaxMax curSoftmaxMax ub->regTensor
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg0, prevSoftmaxMaxUbAddr,
                                                                                        VL_FP32);
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg1, curSoftmaxMaxUbAddr,
                                                                                        VL_FP32);

            AscendC::MicroAPI::Max(outreg0, vreg0, vreg1, maskReg);

            // copyout softmaxMax regTensor->ub
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(softmaxMaxUbAddr, outreg0,
                                                                                        VL_FP32, maskReg);

            AscendC::MicroAPI::Sub(vreg2, vreg0, outreg0, maskReg);
            AscendC::MicroAPI::Sub(vreg3, vreg1, outreg0, maskReg);
            // prev_scale = torch.exp(prev_softmax_max - softmax_max)
            AscendC::MicroAPI::Exp(vreg4, vreg2, maskReg);
            // cur_scale = torch.exp(cur_softmax_max - softmax_max)
            AscendC::MicroAPI::Exp(vreg5, vreg3, maskReg);

            // copyin prevSoftmaxSum curSoftmaxSum ub->regTensor
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg6, prevSoftmaxSumUbAddr,
                                                                                        VL_FP32);
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg7, curSoftmaxSumUbAddr,
                                                                                        VL_FP32);

            // prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
            AscendC::MicroAPI::Mul(vreg8, vreg6, vreg4, maskReg);
            // cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
            AscendC::MicroAPI::Mul(vreg9, vreg7, vreg5, maskReg);
            // softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled
            AscendC::MicroAPI::Add(outreg1, vreg8, vreg9, maskReg);

            // copyout softmaxSum regTensor->ub
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(softmaxSumUbAddr, outreg1,
                                                                                        VL_FP32, maskReg);

            // cur_out_scale = cur_softmax_sum_scaled / softmax_sum
            AscendC::MicroAPI::Div<float, &mode>(vreg11, vreg9, outreg1, maskReg);
            // prev_out_scale = prev_softmax_sum_scaled / softmax_sum
            AscendC::MicroAPI::Div<float, &mode>(vreg10, vreg8, outreg1, maskReg);

            // copyout prev_out_scale cur_out_scale regTensor->ub
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tempFp32Buf1UbAddr, vreg10,
                                                                                        VL_FP32, maskReg);
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tempFp32Buf2UbAddr, vreg11,
                                                                                        VL_FP32, maskReg);
        }
    }
}

template <typename T>
__aicore__ inline void AttnComputeFp32(LocalTensor<T> prevCurAttnOutLocal, LocalTensor<T> attnOutLocal,
                                       LocalTensor<float> tempFp32Buf1Local, LocalTensor<float> tempFp32Buf2Local,
                                       int64_t attnEleNumLoop, int64_t loopNum, int64_t calNum, int64_t headDimLoopEach,
                                       int64_t softmaxTailSize)
{
    auto prevAttnOutUbAddr = (__ubuf__ T *)prevCurAttnOutLocal.GetPhyAddr();
    auto curAttnOutUbAddr = (__ubuf__ T *)prevCurAttnOutLocal.GetPhyAddr() + attnEleNumLoop;
    auto tempFp32Buf1UbAddr = (__ubuf__ float *)tempFp32Buf1Local.GetPhyAddr(); // prev_out_scale
    auto tempFp32Buf2UbAddr = (__ubuf__ float *)tempFp32Buf2Local.GetPhyAddr(); // cur_out_scale
    auto attnOutUbAddr = (__ubuf__ float *)attnOutLocal.GetPhyAddr();
    uint16_t loop = (uint16_t)loopNum;                                           // 实际计算量
    uint16_t repeatTimes = (uint16_t)((uint32_t)calNum + VL_FP32 - 1) / VL_FP32; // 实际计算量对齐，冗余的最后不会搬出

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg maskReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<float> vreg0;
        AscendC::MicroAPI::RegTensor<float> vreg1;
        AscendC::MicroAPI::RegTensor<float> vreg2;
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> vreg4;
        AscendC::MicroAPI::RegTensor<float> vreg5;
        AscendC::MicroAPI::RegTensor<float> outreg0;

        for (uint16_t idx = 0; idx < loop; idx++) { // 外循环
            // copyin prev_out_scale cur_out_scale ub->regTensor
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg2, tempFp32Buf1UbAddr,
                                                                                   softmaxTailSize); // prev_out_scale
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg3, tempFp32Buf2UbAddr,
                                                                                   softmaxTailSize); // cur_out_scale
            for (uint16_t i = 0; i < repeatTimes; i++) {
                AscendC::MicroAPI::AddrReg srcAddrReg =
                    AscendC::MicroAPI::CreateAddrReg<T>(idx, headDimLoopEach, i, VL_FP32);
                // copyin prev_attn_out cur_attn_out ub->regTensor
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0, prevAttnOutUbAddr,
                                                                                           srcAddrReg);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg1, curAttnOutUbAddr,
                                                                                           srcAddrReg);

                // prev_attn_out * prev_out_scale
                AscendC::MicroAPI::Mul(vreg4, vreg0, vreg2, maskReg);
                // cur_attn_out * cur_out_scale
                AscendC::MicroAPI::Mul(vreg5, vreg1, vreg3, maskReg);
                // cur_attn_out * cur_out_scale + prev_attn_out * prev_out_scale
                AscendC::MicroAPI::Add(outreg0, vreg4, vreg5, maskReg);

                // copyout attn_out regTensor->ub
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(attnOutUbAddr, outreg0,
                                                                                                srcAddrReg, maskReg);
            }
        }
    }
}

template <typename T>
__aicore__ inline void AttnComputeBf16Fp16(LocalTensor<T> prevCurAttnOutLocal, LocalTensor<T> attnOutLocal,
                                           LocalTensor<float> tempFp32Buf1Local, LocalTensor<float> tempFp32Buf2Local,
                                           int64_t attnEleNumLoop, int64_t loopNum, int64_t calNum,
                                           int64_t headDimLoopEach, int64_t softmaxTailSize)
{
    auto prevAttnOutUbAddr = (__ubuf__ T *)prevCurAttnOutLocal.GetPhyAddr();
    auto curAttnOutUbAddr = (__ubuf__ T *)prevCurAttnOutLocal.GetPhyAddr() + attnEleNumLoop;
    auto tempFp32Buf1UbAddr = (__ubuf__ float *)tempFp32Buf1Local.GetPhyAddr(); // prev_out_scale
    auto tempFp32Buf2UbAddr = (__ubuf__ float *)tempFp32Buf2Local.GetPhyAddr(); // cur_out_scale
    auto attnOutUbAddr = (__ubuf__ T *)attnOutLocal.GetPhyAddr();
    uint16_t loop = (uint16_t)loopNum;                                           // 实际计算量
    uint16_t repeatTimes = (uint16_t)((uint32_t)calNum + VL_FP32 - 1) / VL_FP32; // 实际计算量对齐，冗余的最后不会搬出

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg maskReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<T> vreg0;
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<float> vreg2;
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> vreg4;
        AscendC::MicroAPI::RegTensor<float> vreg5;
        AscendC::MicroAPI::RegTensor<float> vreg6;
        AscendC::MicroAPI::RegTensor<float> vreg7;
        AscendC::MicroAPI::RegTensor<float> vreg8;
        AscendC::MicroAPI::RegTensor<T> outreg0;

        for (uint16_t idx = 0; idx < loop; idx++) { // 外循环
            // copyin prev_out_scale cur_out_scale ub->regTensor
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg3, tempFp32Buf1UbAddr,
                                                                                   softmaxTailSize); // prev_out_scale
            AscendC::MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg4, tempFp32Buf2UbAddr,
                                                                                   softmaxTailSize); // cur_out_scale
            for (uint16_t i = 0; i < repeatTimes; i++) {
                AscendC::MicroAPI::AddrReg srcAddrReg =
                    AscendC::MicroAPI::CreateAddrReg<T>(idx, headDimLoopEach, i, VL_FP32);
                // copyin prev_attn_out cur_attn_out ub->regTensor
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, prevAttnOutUbAddr,
                                                                                             srcAddrReg);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, curAttnOutUbAddr,
                                                                                             srcAddrReg);

                // prev_attn_out bf16/fp16 -> fp32
                AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg2, vreg0, maskReg);
                // prev_attn_out * prev_out_scale
                AscendC::MicroAPI::Mul(vreg6, vreg2, vreg3, maskReg);
                // cur_attn_out bf16/fp16 -> fp32
                AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg5, vreg1, maskReg);
                // cur_attn_out * cur_out_scale
                AscendC::MicroAPI::Mul(vreg7, vreg4, vreg5, maskReg);
                // cur_attn_out * cur_out_scale + prev_attn_out * prev_out_scale
                AscendC::MicroAPI::Add(vreg8, vreg6, vreg7, maskReg);
                // attn_out fp32 -> bf16/fp16
                AscendC::MicroAPI::Cast<T, float, castTrait1>(outreg0, vreg8, maskReg);

                // copyout attn_out regTensor->ub
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(attnOutUbAddr, outreg0,
                                                                                            srcAddrReg, maskReg);
            }
        }
    }
}

} // namespace RingAttentionUpdateRegbase
#endif // _RING_ATTENTION_UPDATE_REGBASE_COMMON_H_
