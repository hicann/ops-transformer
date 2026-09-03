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
 * \file mx_quant_compute_scale.h
 * \brief MX 分组最大指数和 Scale 计算。
 */

#ifndef MEGA_MOE_ARCH35_MX_QUANT_COMPUTE_SCALE_H
#define MEGA_MOE_ARCH35_MX_QUANT_COMPUTE_SCALE_H

#if defined(__DAV_C310__)
#include "mx_quant_common.h"

namespace MegaMoeImpl {
namespace MxQuant {

__aicore__ inline void ComputeGroupMaxExp(__ubuf__ bfloat16_t *input, __ubuf__ uint16_t *maxExp, uint32_t dataCount,
                                          uint16_t dataLoopCount)
{
    int64_t inputElementCountPerLoop = DATA_ELEMENT_COUNT_PER_LOOP;
    int64_t scaleElementCountPerLoop = SCALE_ELEMENT_COUNT_PER_DATA_LOOP;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<bfloat16_t> inputReg0, inputReg1;
        AscendC::Reg::RegTensor<uint16_t> exponentReg0, exponentReg1;
        AscendC::Reg::RegTensor<uint16_t> exponentMaskReg, maxExponentReg;
        AscendC::Reg::Duplicate(exponentMaskReg, BF16_EXPONENT_MASK);
        AscendC::Reg::MaskReg dataMask0, dataMask1;
        AscendC::Reg::UnalignReg unalignReg;
        for (uint16_t loopIndex = 0; loopIndex < dataLoopCount; loopIndex++) {
            dataMask0 = AscendC::Reg::UpdateMask<bfloat16_t>(dataCount);
            // DINTLV_B16 每轮消费两个 BF16 寄存器，第二次调用必须保留以推进剩余计数。
            dataMask1 = AscendC::Reg::UpdateMask<bfloat16_t>(dataCount);
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(inputReg0, inputReg1, input,
                                                                            inputElementCountPerLoop);
            AscendC::Reg::And(exponentReg0, (AscendC::Reg::RegTensor<uint16_t> &)inputReg0, exponentMaskReg, dataMask0);
            AscendC::Reg::And(exponentReg1, (AscendC::Reg::RegTensor<uint16_t> &)inputReg1, exponentMaskReg, dataMask0);
            AscendC::Reg::Max(maxExponentReg, exponentReg0, exponentReg1, dataMask0);
            AscendC::Reg::ReduceMaxWithDataBlock(maxExponentReg, maxExponentReg, dataMask0);
            AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExp, maxExponentReg, unalignReg, scaleElementCountPerLoop);
        }
        AscendC::Reg::DataCopyUnAlignPost(maxExp, unalignReg, 0);
    }
}

template <typename OutputType>
__aicore__ inline void ComputeMxScale(__ubuf__ uint16_t *maxExp, __ubuf__ uint16_t *outputScale,
                                      __ubuf__ uint16_t *reciprocalScale, uint32_t scaleCount, uint16_t scaleLoopCount)
{
    const uint16_t outputMaxExponent = GetOutputMaxExponent<OutputType>();
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> exponentMaskReg, maxExponentReg;
        AscendC::Reg::Duplicate(exponentMaskReg, BF16_EXPONENT_MASK);
        AscendC::Reg::MaskReg validExponentMask, nonzeroMask, scaleMask;
        AscendC::Reg::RegTensor<uint16_t> outputMaxExponentReg, sharedExponentReg, outputScaleReg;
        AscendC::Reg::RegTensor<uint16_t> exponentBiasReg, reciprocalScaleReg;
        AscendC::Reg::Duplicate(outputMaxExponentReg, outputMaxExponent);
        AscendC::Reg::Duplicate(exponentBiasReg, BF16_EXPONENT_BIAS);
        AscendC::Reg::RegTensor<uint16_t> fp8NanReg, zeroReg, bf16NanReg;
        AscendC::Reg::Duplicate(fp8NanReg, FP8_NAN_EXPONENT);
        AscendC::Reg::Duplicate(zeroReg, 0);
        AscendC::Reg::Duplicate(bf16NanReg, BF16_NAN_VALUE);
        AscendC::Reg::MaskReg belowMaxExponentMask, specialExponentMask;
        AscendC::Reg::RegTensor<uint16_t> specialExponentReg;
        AscendC::Reg::Duplicate(specialExponentReg, SPECIAL_EXPONENT_THRESHOLD);
        for (uint16_t loopIndex = 0; loopIndex < scaleLoopCount; loopIndex++) {
            scaleMask = AscendC::Reg::UpdateMask<uint16_t>(scaleCount);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExponentReg, maxExp, SCALE_ELEMENT_COUNT_PER_VECTOR);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(validExponentMask, maxExponentReg, exponentMaskReg,
                                                                  scaleMask);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(nonzeroMask, maxExponentReg, zeroReg, scaleMask);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(belowMaxExponentMask, maxExponentReg,
                                                                  outputMaxExponentReg, scaleMask);
            AscendC::Reg::Select<uint16_t>(maxExponentReg, outputMaxExponentReg, maxExponentReg, belowMaxExponentMask);
            AscendC::Reg::Sub(sharedExponentReg, maxExponentReg, outputMaxExponentReg, scaleMask);
            AscendC::Reg::ShiftRights(outputScaleReg, sharedExponentReg, BF16_EXPONENT_SHIFT, scaleMask);
            AscendC::Reg::Select<uint16_t>(outputScaleReg, outputScaleReg, fp8NanReg, validExponentMask);
            AscendC::Reg::Select<uint16_t>(outputScaleReg, outputScaleReg, zeroReg, nonzeroMask);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK_B16>(outputScale, outputScaleReg,
                                                                           SCALE_PACK_ELEMENT_COUNT, scaleMask);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialExponentMask, sharedExponentReg,
                                                                  exponentBiasReg, scaleMask);
            AscendC::Reg::Sub(reciprocalScaleReg, exponentBiasReg, sharedExponentReg, scaleMask);
            AscendC::Reg::Select<uint16_t>(reciprocalScaleReg, reciprocalScaleReg, bf16NanReg, validExponentMask);
            AscendC::Reg::Select<uint16_t>(reciprocalScaleReg, reciprocalScaleReg, zeroReg, nonzeroMask);
            AscendC::Reg::Select<uint16_t>(reciprocalScaleReg, specialExponentReg, reciprocalScaleReg,
                                           specialExponentMask);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                reciprocalScale, reciprocalScaleReg, SCALE_ELEMENT_COUNT_PER_VECTOR, scaleMask);
        }
    }
}

} // namespace MxQuant
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_MX_QUANT_COMPUTE_SCALE_H
