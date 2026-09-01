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
 * \file mxfp8_quant.h
 * \brief MX FP8 数据量化。
 */

#ifndef MEGA_MOE_ARCH35_MXFP8_QUANT_H
#define MEGA_MOE_ARCH35_MXFP8_QUANT_H

#if defined(__DAV_C310__)
#include "mx_quant_common.h"

namespace MegaMoeImpl {
namespace MxQuant {

template <typename OutputType>
__aicore__ inline void QuantizeMxFp8Data(__ubuf__ bfloat16_t *input, __ubuf__ uint16_t *reciprocalScale,
                                         __ubuf__ int8_t *output, uint32_t dataCount, uint16_t dataLoopCount)
{
    (void)dataCount;
    int64_t scaleElementCountPerLoop = SCALE_ELEMENT_COUNT_PER_DATA_LOOP;
    int64_t inputElementCountPerLoop = DATA_ELEMENT_COUNT_PER_LOOP;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> reciprocalScaleReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> inputReg0, inputReg1;
        AscendC::MicroAPI::RegTensor<float> inputFp32Layout0Reg0, inputFp32Layout1Reg0;
        AscendC::MicroAPI::RegTensor<float> inputFp32Layout0Reg1, inputFp32Layout1Reg1;
        AscendC::MicroAPI::RegTensor<OutputType> outputLayout0Reg0, outputLayout1Reg0;
        AscendC::MicroAPI::RegTensor<OutputType> outputLayout0Reg1, outputLayout1Reg1;
        AscendC::MicroAPI::MaskReg fullB16Mask =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg fullB8Mask =
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t loopIndex = 0; loopIndex < dataLoopCount; loopIndex++) {
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(inputReg0, inputReg1, input,
                                                                                      inputElementCountPerLoop);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(reciprocalScaleReg, reciprocalScale,
                                                                                   scaleElementCountPerLoop);
            AscendC::MicroAPI::Mul(inputReg0, inputReg0, (AscendC::MicroAPI::RegTensor<bfloat16_t> &)reciprocalScaleReg,
                                   fullB16Mask);
            AscendC::MicroAPI::Mul(inputReg1, inputReg1, (AscendC::MicroAPI::RegTensor<bfloat16_t> &)reciprocalScaleReg,
                                   fullB16Mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_TO_FP32_ZERO_LAYOUT>(inputFp32Layout0Reg0, inputReg0,
                                                                                      fullB16Mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_TO_FP32_ONE_LAYOUT>(inputFp32Layout1Reg0, inputReg0,
                                                                                     fullB16Mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_TO_FP32_ZERO_LAYOUT>(inputFp32Layout0Reg1, inputReg1,
                                                                                      fullB16Mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_TO_FP32_ONE_LAYOUT>(inputFp32Layout1Reg1, inputReg1,
                                                                                     fullB16Mask);
            AscendC::MicroAPI::Cast<OutputType, float, CAST_FP32_TO_FP8_LAYOUT_ZERO>(outputLayout0Reg0,
                                                                                     inputFp32Layout0Reg0, fullB16Mask);
            AscendC::MicroAPI::Cast<OutputType, float, CAST_FP32_TO_FP8_LAYOUT_TWO>(outputLayout1Reg0,
                                                                                    inputFp32Layout1Reg0, fullB16Mask);
            AscendC::MicroAPI::Cast<OutputType, float, CAST_FP32_TO_FP8_LAYOUT_ONE>(outputLayout0Reg1,
                                                                                    inputFp32Layout0Reg1, fullB16Mask);
            AscendC::MicroAPI::Cast<OutputType, float, CAST_FP32_TO_FP8_LAYOUT_THREE>(
                outputLayout1Reg1, inputFp32Layout1Reg1, fullB16Mask);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)outputLayout0Reg0,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)outputLayout0Reg0,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)outputLayout1Reg0, fullB8Mask);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)outputLayout0Reg0,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)outputLayout0Reg0,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)outputLayout0Reg1, fullB8Mask);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)outputLayout0Reg0,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)outputLayout0Reg0,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)outputLayout1Reg1, fullB8Mask);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                output, (AscendC::MicroAPI::RegTensor<int8_t> &)outputLayout0Reg0, inputElementCountPerLoop,
                fullB8Mask);
        }
    }
}

} // namespace MxQuant
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_MXFP8_QUANT_H
