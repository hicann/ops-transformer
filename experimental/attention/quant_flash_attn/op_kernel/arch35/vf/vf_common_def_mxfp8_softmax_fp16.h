/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_common_def_mxfp8_softmax_fp16.h
 * \brief VectorBlock 共享常量（CastTrait / 数学常量），镜像 MxFP4 参考 vf_common_def.h
 */

#ifndef VF_COMMON_DEF_H_
#define VF_COMMON_DEF_H_

#include "kernel_tensor.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

// ===== CastTrait（宽度变化时的元素路由控制，非舍入模式） =====
// h2i 前缀 = "half to int"；Zero/One 指 RegLayout 编号（偶数位/奇数位元素抽取）

// 宽化 Cast（half→int32）：取偶数位元素（0, 2, 4, ...）
constexpr static AscendC::Reg::CastTrait h2iZero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

// 宽化 Cast（half→int32）：取奇数位元素（1, 3, 5, ...）
constexpr static AscendC::Reg::CastTrait h2iOne = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

// ===== 窄化 CastTrait（half→uint8，2:1，带 SAT 饱和——MxFP4 mxscale 同款） =====

// 窄化 Cast（half→uint8）：取偶数位元素（0, 2, 4, ...），饱和到 [0,255]
constexpr static AscendC::Reg::CastTrait castTraitZero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

// 窄化 Cast（half→uint8）：取奇数位元素（1, 3, 5, ...），饱和到 [0,255]
// 注意：RoundMode 在 AscendC 顶层命名空间（非 AscendC::Reg——RegLayout/SatMode/MaskMergeMode 才在 Reg）
constexpr static AscendC::Reg::CastTrait castTraitOne = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

// ===== fp8 窄化 CastTrait（half→fp8_e4m3fn，CAST_RINT 舍入——mxfp8 高精 L292-295 同款） =====

// fp8 窄化：偶数位元素（0, 2, 4, ...），饱和（e4m3 max=448 防溢出）
constexpr static AscendC::Reg::CastTrait castTraitRintZero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// fp8 窄化：奇数位元素（1, 3, 5, ...），饱和
constexpr static AscendC::Reg::CastTrait castTraitRintOne = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// float→fp8 窄化 4:1 第 2 位元素（float→fp8 配对用），饱和
constexpr static AscendC::Reg::CastTrait castTraitRintTwo = {
    AscendC::Reg::RegLayout::TWO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// float→fp8 窄化 4:1 第 3 位元素（float→fp8 配对用），饱和
constexpr static AscendC::Reg::CastTrait castTraitRintThree = {
    AscendC::Reg::RegLayout::THREE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// ===== 数学常量（log2 整数域量化用） =====
constexpr half NUM_127 = static_cast<half>(127.0f);  // fp32/e8m0 指数偏置
constexpr half ZERO_VALUE = static_cast<half>(0.0f); // 防御负指数
constexpr int16_t SHIFT_VALUE = 23;                  // fp32 尾数域位宽（移位目标）

constexpr half LN2 = static_cast<half>(0.6931471806f);     // P 归一化用：k·ln2
constexpr half INV_LN2 = static_cast<half>(1.4426950409f); // 量化用：max/ln2
constexpr half MIN_VALUE = static_cast<half>(-65504.0f);   // fp16 最小值（accMax 复位）

constexpr uint32_t QFA_UB_PSCALE_SLOT = 768; // 单 pscale 槽字节数（L1 网格影像同构）
constexpr uint32_t QFA_UB_PSCALE_RINGS = 6;  // ring 槽数（loop×2+subLoopIdx，周期 6）
constexpr uint32_t QFA_UB_SMAX_SLOT = 128;   // 单 max 槽元素数（S1 半宽 [128]）

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_COMMON_DEF_H_
