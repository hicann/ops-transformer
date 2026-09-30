/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CGMCT_EPILOGUE_MX_QUANT_REDUCE_H
#define CGMCT_EPILOGUE_MX_QUANT_REDUCE_H

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Cgmct {
namespace Gemm {
namespace Block {
namespace MxQuantDetail {
// Share only the reduction: scale rounding and zero handling remain with each epilogue.
__aicore__ inline void ReduceBf16MaxExponent(__ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr,
                                             uint32_t validCount, uint16_t loopNum, uint32_t registerElements,
                                             uint32_t reducedElements)
{
    constexpr uint16_t BF16_EXPONENT_MASK = 0x7f80;
    constexpr uint32_t INTERLEAVED_REGISTERS = 2;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<bfloat16_t> input0, input1;
        AscendC::Reg::RegTensor<uint16_t> exponent0, exponent1, exponentMask, maximum;
        AscendC::Reg::Duplicate(exponentMask, BF16_EXPONENT_MASK);
        AscendC::Reg::UnalignReg unaligned;
        for (uint16_t i = 0; i < loopNum; ++i) {
            auto mask = AscendC::Reg::UpdateMask<bfloat16_t>(validCount);
            // Both loads advance validCount, matching the existing deinterleaved reduction.
            (void)AscendC::Reg::UpdateMask<bfloat16_t>(validCount);
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(input0, input1, srcAddr,
                                                                            registerElements * INTERLEAVED_REGISTERS);
            AscendC::Reg::And(exponent0, (AscendC::Reg::RegTensor<uint16_t>&)input0, exponentMask, mask);
            AscendC::Reg::And(exponent1, (AscendC::Reg::RegTensor<uint16_t>&)input1, exponentMask, mask);
            AscendC::Reg::Max(maximum, exponent0, exponent1, mask);
            AscendC::Reg::ReduceMaxWithDataBlock(maximum, maximum, mask);
            AscendC::Reg::DataCopyUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, maximum, unaligned, reducedElements);
        }
        AscendC::Reg::DataCopyUnAlignPost(maxExpAddr, unaligned, 0);
    }
}
} // namespace MxQuantDetail
} // namespace Block
} // namespace Gemm
} // namespace Cgmct
#endif
#endif
