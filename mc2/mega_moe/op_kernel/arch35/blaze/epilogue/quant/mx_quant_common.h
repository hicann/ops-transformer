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
 * \file mx_quant_common.h
 * \brief MX 量化公共参数、常量和转换属性。
 */

#ifndef MEGA_MOE_ARCH35_MX_QUANT_COMMON_H
#define MEGA_MOE_ARCH35_MX_QUANT_COMMON_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace MegaMoeImpl {
namespace MxQuant {

constexpr uint16_t BF16_EXPONENT_MASK = 0x7f80;
constexpr uint16_t FP8_NAN_EXPONENT = 0x00ff;
constexpr uint16_t BF16_EXPONENT_BIAS = 0x7f00;
constexpr int16_t BF16_EXPONENT_SHIFT = 7;
constexpr uint16_t BF16_NAN_VALUE = 0x7f81;
constexpr uint16_t SPECIAL_EXPONENT_THRESHOLD = 0x0040;

constexpr uint16_t FP8_E4M3_MAX_EXPONENT = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXPONENT = 0x0780;
constexpr uint16_t FP4_E2M1_MAX_EXPONENT = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXPONENT = 0x0000;

constexpr int64_t DATA_ELEMENT_COUNT_PER_LOOP = 256;
constexpr int64_t SCALE_ELEMENT_COUNT_PER_DATA_LOOP = 8;
constexpr int64_t SCALE_ELEMENT_COUNT_PER_VECTOR = 128;
constexpr int64_t SCALE_PACK_ELEMENT_COUNT = 64;
constexpr int64_t FP4_OUTPUT_ELEMENT_COUNT_PER_STORE = 64;

constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP32_ZERO_LAYOUT = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP32_ONE_LAYOUT = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP8_LAYOUT_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP8_LAYOUT_ONE = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP8_LAYOUT_TWO = {
    AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP8_LAYOUT_THREE = {
    AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename OutputType>
__aicore__ inline uint16_t GetOutputMaxExponent()
{
    if constexpr (AscendC::IsSameType<OutputType, fp8_e4m3fn_t>::value) {
        return FP8_E4M3_MAX_EXPONENT;
    } else if constexpr (AscendC::IsSameType<OutputType, fp8_e5m2_t>::value) {
        return FP8_E5M2_MAX_EXPONENT;
    } else if constexpr (AscendC::IsSameType<OutputType, fp4x2_e2m1_t>::value) {
        return FP4_E2M1_MAX_EXPONENT;
    } else {
        return FP4_E1M2_MAX_EXPONENT;
    }
}

} // namespace MxQuant
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_MX_QUANT_COMMON_H
