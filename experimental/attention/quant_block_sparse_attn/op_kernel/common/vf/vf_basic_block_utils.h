/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_basic_block_utils.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UTILS_H
#define VF_BASIC_BLOCK_UTILS_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../quant_block_sparse_attn_arch_select.h"

namespace FaVectorApi {
constexpr uint32_t floatRepSize = 64;
constexpr uint32_t halfRepSize = 128;
constexpr uint32_t blockBytesU8 = 32;
constexpr float fp8e4m3MaxValue = 448.0f;
constexpr float int8MaxValue = 127.0f;
constexpr float hifp8MaxValue = 32768.0f;
constexpr float floatEps = 2.220446049250313e-16;
// MX V1 沿用 common softmax 技巧：先用 log2 转换以配合 exp2 指令，
// 再通过 ln2 转回自然指数语义。MXFP8 DN VF 实现需要这两个常量。
constexpr float LN2 = static_cast<float>(0.6931471806f);
constexpr float INV_LN2 = static_cast<float>(1.4426950409F);
/* **************************************************************************************************
 * Muls + Select(optional) + SoftmaxFlashV2 + Cast(fp32->fp16/bf16) + ND2NZ
 * ************************************************************************************************* */
using namespace Reg;

// Dynamic MX PScale only: alpha has already been rounded from FP32 to BF16.
// DAV_9201/9202 use explicit exponent encoding instead of bf162fp8e8m0 vcvt.
// Keep the arch35 trait and its original Cast call sites unchanged.
#if QBSA_ARCH_9102R1
__simd_callee__ inline void EncodeMxPScaleE8M0Zero(RegTensor<fp8_e8m0_t>& dst, RegTensor<bfloat16_t>& src,
                                                   MaskReg& mask)
{
    // For finite nonnegative alpha, this matches the CPU golden's biased-exponent
    // extraction after BF16 rounding. Zero/subnormal BF16 -> 0x00 (E8M0 2^-127),
    // alpha=1 -> 0x7f. Exponent 0xff stays 0xff: do not hide NaN/Inf from upstream.
    // Each B16 lane becomes [E8M0 code, 0], preserving RegLayout::ZERO byte placement
    // and the ZERO/ONE -> Or -> DeInterleave packing at both MX V1 call sites.
    RegTensor<uint16_t> exponentMask;
    Duplicate(exponentMask, static_cast<uint16_t>(0x7f80U));
    And((RegTensor<uint16_t>&)dst, (RegTensor<uint16_t>&)src, exponentMask, mask);
    ShiftRights((RegTensor<uint16_t>&)dst, (RegTensor<uint16_t>&)dst, static_cast<int16_t>(7), mask);
}
#else
constexpr static AscendC::RoundMode pScaleE8M0RoundMode = AscendC::RoundMode::CAST_NONE;
constexpr static AscendC::Reg::SatMode pScaleE8M0SatMode = AscendC::Reg::SatMode::UNKNOWN;

constexpr static AscendC::Reg::CastTrait castTraitE8M0Zero = {
    // 这里不是 cast 输入的 quantScale1；quantScale1 传入时已经是 fp8_e8m0。
    // VF 会根据 softmax/update 过程重新生成给 BMM2 使用的 per-group PScale，
    // 生成过程先得到 bf16 lane 形式的 scale 编码，再 cast 成 fp8_e8m0 写入 UB/L1。
    AscendC::Reg::RegLayout::ZERO,
    pScaleE8M0SatMode,
    AscendC::Reg::MaskMergeMode::ZEROING,
    pScaleE8M0RoundMode,
};
#endif

constexpr static AscendC::Reg::CastTrait castTraitZero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait castTraitOne = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait castTraitTwo = {
    AscendC::Reg::RegLayout::TWO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait castTraitThree = {
    AscendC::Reg::RegLayout::THREE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait castTraitRintZero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::Reg::CastTrait castTraitRintOne = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::Reg::CastTrait castTraitRintTwo = {
    AscendC::Reg::RegLayout::TWO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::Reg::CastTrait castTraitRintThree = {
    AscendC::Reg::RegLayout::THREE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

#define USE_MLA_FULLQUANT_V1_P(vreg_exp, vreg_rowmax_p, MaskReg) \
    do { \
        Muls(vreg_exp, vreg_exp, fp8e4m3MaxValue, MaskReg); \
        Div(vreg_exp, vreg_exp, vreg_rowmax_p, MaskReg); \
    } while (0)

#define USE_MLA_FULLQUANT_V1_P_INT8(vreg_exp, vreg_rowmax_p, MaskReg) \
    do { \
        Muls(vreg_exp, vreg_exp, int8MaxValue, MaskReg); \
        Div(vreg_exp, vreg_exp, vreg_rowmax_p, MaskReg); \
    } while (0)

#define USE_MLA_FULLQUANT_V1_P_HIFP8(vreg_exp, vreg_rowmax_p, MaskReg) \
    do { \
        Muls(vreg_exp, vreg_exp, hifp8MaxValue, MaskReg); \
        Div(vreg_exp, vreg_exp, vreg_rowmax_p, MaskReg); \
    } while (0)
} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_UTILS_H
