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
 * \file quant_reduce_scatter_utils.h
 * \brief
 */

#ifndef QUANT_REDUCE_SCATTER_UTILS_H
#define QUANT_REDUCE_SCATTER_UTILS_H

#include "../../common/op_kernel/mc2_kernel_utils.h"
namespace AscendC {

// 向上取整除法：计算a除以b的向上取整结果
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

// uint32_t版本的向上取整除法
__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

// uint32_t版本的向上对齐
__aicore__ inline uint32_t CeilAlignU32(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return CeilDivU32(a, b) * b;
};

// 尾块模运算：计算a对b取模，模为0时返回b（用于数据块尾块的计算）
__aicore__ inline uint64_t BlockAlignMod(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }

    uint64_t c = a % b;
    return c ? c : b;
}

// 向下取整对齐：将a向下对齐到b的倍数
__aicore__ inline uint64_t FloorAlign(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a / b) * b;
}

// 分数除法：total / (base + num/den)，避免 base + num/den 整数截断为 base
__aicore__ inline uint64_t FracDiv(uint64_t total, uint64_t base, uint32_t num, uint32_t den)
{
    return total * num / (base * num + den);
}

static constexpr AscendC::MicroAPI::CastTrait castTrait = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE};

// fp8_e8m0_t数据类型到bf16_t数据类型转换
static __aicore__ inline void CastVf(__local_mem__ bfloat16_t *dstPtr, __local_mem__ fp8_e8m0_t *srcPtr, uint32_t count)
{
    AscendC::MicroAPI::RegTensor<fp8_e8m0_t> srcReg;
    AscendC::MicroAPI::RegTensor<fp8_e8m0_t> srcZeroReg;
    AscendC::MicroAPI::RegTensor<fp8_e8m0_t> dstReg0;
    AscendC::MicroAPI::RegTensor<fp8_e8m0_t> dstReg1;
    AscendC::MicroAPI::RegTensor<bfloat16_t> bf16DstReg;
    AscendC::MicroAPI::MaskReg maskReg;
    maskReg = AscendC::MicroAPI::UpdateMask<bfloat16_t>(count);
    AscendC::MicroAPI::DataCopy(srcReg, srcPtr);
    AscendC::MicroAPI::Interleave(dstReg0, dstReg1, srcReg, srcZeroReg);
    AscendC::MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTrait>(bf16DstReg, dstReg0, maskReg);
    AscendC::MicroAPI::DataCopy(dstPtr, bf16DstReg, maskReg);
}

} // namespace AscendC

#endif
