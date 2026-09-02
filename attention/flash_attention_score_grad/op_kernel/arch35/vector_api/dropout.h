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
 * \file dropout.h
 */

#ifndef DROPOUT_H
#define DROPOUT_H

#include "../flash_attention_score_grad_common.h"
#include "../../../../common/op_kernel/arch35/dropmask.h"

using namespace AscendC;
using namespace AscendC::MicroAPI;
using namespace commondef;

#ifndef __CCE_KT_TEST__
__simd_vf__ inline void CalculateDropoutVF(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t maskLocalInt,
                                           float divValue, const uint32_t dataSize)
{
    RegTensor<float> vreg0;
    RegTensor<float> vreg1;
    RegTensor<float> vreg2;
    RegTensor<float> vreg3;
    RegTensor<float> vreg4;
    RegTensor<float> vreg5;
    RegTensor<float> vreg6;
    Duplicate(vreg1, 0);
    uint32_t sreg = (uint32_t)dataSize;
    MaskReg preg0;
    MaskReg preg1;
    MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
    MaskReg preg3;
    MaskReg preg4;
    uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t repeatTimes = CeilDivision(dataSize, repeatElm);

    for (uint16_t i = 0; i < (uint16_t)(repeatTimes / UNROLL_FACTOR); ++i) {
        LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_US>(preg1, (__ubuf__ uint32_t *&)maskLocalInt + i * 4);
        MaskInterleave<half>(preg3, preg4, preg1, preg2);
        preg0 = UpdateMask<float>(sreg);
        LoadAlign(vreg0, (__ubuf__ float *&)srcLocalInt + i * 2 * repeatElm);
        Select(vreg2, vreg0, vreg1, preg3);
        Muls(vreg5, vreg2, divValue, preg0);
        StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float *&)dstLocalInt + i * 2 * repeatElm, vreg5,
                                                              preg0);
        preg0 = UpdateMask<float>(sreg);
        LoadAlign(vreg3, (__ubuf__ float *&)srcLocalInt + (i * 2 + 1) * repeatElm);
        Select(vreg4, vreg3, vreg1, preg4);
        Muls(vreg6, vreg4, divValue, preg0);
        StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float *&)dstLocalInt + (i * 2 + 1) * repeatElm,
                                                              vreg6, preg0);
    }

    RegTensor<float> vreg7;
    RegTensor<float> vreg8;
    RegTensor<float> vreg9;
    MaskReg preg7;
    MaskReg preg8;
    uint32_t offset0 = (repeatTimes / UNROLL_FACTOR) * 2 * repeatElm;
    uint32_t offset2 = (repeatTimes / UNROLL_FACTOR) * 2 * repeatElm;
    uint32_t selOffset = (repeatTimes / UNROLL_FACTOR) * 4;
    for (uint16_t i = 0; i < (uint16_t)(repeatTimes % UNROLL_FACTOR); ++i) {
        preg0 = UpdateMask<float>(sreg);
        LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_US>(preg7, (__ubuf__ uint32_t *&)maskLocalInt + selOffset);
        UnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(preg8, preg7);
        LoadAlign(vreg7, (__ubuf__ float *&)srcLocalInt + offset0);
        Select(vreg8, vreg7, vreg1, preg8);
        Muls(vreg9, vreg8, divValue, preg0);
        StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float *&)dstLocalInt + offset2, vreg9, preg0);
    }
}

template <typename T2, const uint32_t IS_DROP = 0, uint32_t srcN>
__aicore__ inline void CalculateDropout(FagConstInfo &constInfo, FagRunInfo &runInfo, DropMaskInfo &dropInfo,
                                        LocalTensor<T2> &srcTensor, LocalTensor<T2> &dstTensor, TBuf<> &dropMaskBuf)
{
    if constexpr (IS_DROP) {
        static_assert(std::is_same<T2, float>::value, "DropOutVf API only support float type.");
        float divValue = 1.0;
        divValue = divValue / constInfo.commonConstInfo.keepProb;
        const uint32_t dataSize = runInfo.commonRunInfo.halfS1RealSize * srcN;
        LocalTensor<uint8_t> dropTensor = dropMaskBuf.template Get<uint8_t>();

        uint64_t srcLocalInt = srcTensor.GetPhyAddr();
        uint64_t dstLocalInt = dstTensor.GetPhyAddr();
        uint64_t maskLocalInt = dropTensor.GetPhyAddr();

        CalculateDropoutVF(srcLocalInt, dstLocalInt, maskLocalInt, divValue, dataSize);
    }
}

#else
template <typename T2, const uint32_t IS_DROP = 0, uint32_t srcN>
__aicore__ inline void CalculateDropout(FagConstInfo &constInfo, FagRunInfo &runInfo, DropMaskInfo &dropInfo,
                                        LocalTensor<T2> &srcTensor, LocalTensor<T2> &dstTensor, TBuf<> &dropMaskBuf)
{}
#endif
#endif
