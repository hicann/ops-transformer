/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_basic_block_utils.h
 * \brief arch92 mxfp8 fp16 路径 VF 层公用的 CastTrait 与常量定义
 */
#ifndef VF_BASIC_BLOCK_UTILS_H
#define VF_BASIC_BLOCK_UTILS_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace FaVectorApi {
constexpr float LN2 = static_cast<float>(0.6931471806f);
constexpr float INV_LN2 = static_cast<float>(1.4426950409F);

using namespace MicroAPI;

constexpr static AscendC::MicroAPI::CastTrait castTraitZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitNoneZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitNoneOne = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC, // 只支持CAST_TRUNC和CAST_CEIL两种模式
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_UTILS_H
