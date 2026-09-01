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
        AscendC::MicroAPI::MaskReg dataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> reciprocalScaleReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> inputReg0, inputReg1;
        AscendC::MicroAPI::RegTensor<OutputType> outputReg0, outputReg1;
        for (uint16_t loopIndex = 0; loopIndex < dataLoopCount; loopIndex++) {
            dataMask = AscendC::MicroAPI::UpdateMask<bfloat16_t>(dataCount);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(inputReg0, inputReg1, input,
                                                                                      inputElementCountPerLoop);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(reciprocalScaleReg, reciprocalScale,
                                                                                   scaleElementCountPerLoop);
            AscendC::MicroAPI::Mul(inputReg0, inputReg0, (AscendC::MicroAPI::RegTensor<bfloat16_t> &)reciprocalScaleReg,
                                   dataMask);
            AscendC::MicroAPI::Mul(inputReg1, inputReg1, (AscendC::MicroAPI::RegTensor<bfloat16_t> &)reciprocalScaleReg,
                                   dataMask);
            AscendC::MicroAPI::Interleave(inputReg0, inputReg1, inputReg0, inputReg1);
            AscendC::MicroAPI::Cast<OutputType, bfloat16_t, CAST_BF16_TO_FP4>(outputReg0, inputReg0, dataMask);
            AscendC::MicroAPI::Cast<OutputType, bfloat16_t, CAST_BF16_TO_FP4>(outputReg1, inputReg1, dataMask);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                output, (AscendC::MicroAPI::RegTensor<int8_t> &)outputReg0, outputElementCountPerStore, dataMask);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                output, (AscendC::MicroAPI::RegTensor<int8_t> &)outputReg1, outputElementCountPerStore, dataMask);
        }
    }
}

} // namespace MxQuant
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_MXFP4_QUANT_H
