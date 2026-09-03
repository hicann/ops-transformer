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
 * \file mxfp4_quant.h
 * \brief MX FP4 数据量化。
 */

#ifndef MEGA_MOE_ARCH35_MXFP4_QUANT_H
#define MEGA_MOE_ARCH35_MXFP4_QUANT_H

#if defined(__DAV_C310__)
#include "mx_quant_common.h"

namespace MegaMoeImpl {
namespace MxQuant {

template <typename OutputType>
__aicore__ inline void QuantizeMxFp4Data(__ubuf__ bfloat16_t *input, __ubuf__ uint16_t *reciprocalScale,
                                         __ubuf__ int8_t *output, uint32_t dataCount, uint16_t dataLoopCount)
{
    int64_t scaleElementCountPerLoop = SCALE_ELEMENT_COUNT_PER_DATA_LOOP;
    int64_t inputElementCountPerLoop = DATA_ELEMENT_COUNT_PER_LOOP;
    int64_t outputElementCountPerStore = FP4_OUTPUT_ELEMENT_COUNT_PER_STORE;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg dataMask;
        AscendC::Reg::RegTensor<uint16_t> reciprocalScaleReg;
        AscendC::Reg::RegTensor<bfloat16_t> inputReg0, inputReg1;
        AscendC::Reg::RegTensor<OutputType> outputReg0, outputReg1;
        for (uint16_t loopIndex = 0; loopIndex < dataLoopCount; loopIndex++) {
            dataMask = AscendC::Reg::UpdateMask<bfloat16_t>(dataCount);
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(inputReg0, inputReg1, input,
                                                                            inputElementCountPerLoop);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_E2B_B16>(reciprocalScaleReg, reciprocalScale,
                                                                         scaleElementCountPerLoop);
            AscendC::Reg::Mul(inputReg0, inputReg0, (AscendC::Reg::RegTensor<bfloat16_t> &)reciprocalScaleReg,
                              dataMask);
            AscendC::Reg::Mul(inputReg1, inputReg1, (AscendC::Reg::RegTensor<bfloat16_t> &)reciprocalScaleReg,
                              dataMask);
            AscendC::Reg::Interleave(inputReg0, inputReg1, inputReg0, inputReg1);
            AscendC::Reg::Cast<OutputType, bfloat16_t, CAST_BF16_TO_FP4>(outputReg0, inputReg0, dataMask);
            AscendC::Reg::Cast<OutputType, bfloat16_t, CAST_BF16_TO_FP4>(outputReg1, inputReg1, dataMask);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                output, (AscendC::Reg::RegTensor<int8_t> &)outputReg0, outputElementCountPerStore, dataMask);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                output, (AscendC::Reg::RegTensor<int8_t> &)outputReg1, outputElementCountPerStore, dataMask);
        }
    }
}

} // namespace MxQuant
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_MXFP4_QUANT_H
