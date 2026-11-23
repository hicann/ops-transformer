/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
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
 * \file vf_mxfp4_attenout_dn.h
 * \brief
 */

#ifndef VF_COMMON_DEF_H_
#define VF_COMMON_DEF_H_
#include "kernel_tensor.h"

namespace Mxfp4Api {
using namespace AscendC;
using namespace Reg;

#define VMULSCVT false
#define DROPOUT false

constexpr static AscendC::Reg::CastTrait h2iZero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait h2iOne = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

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

constexpr half NUM_127 = static_cast<half>(127.0f);
constexpr half NUM_NEG_125 = static_cast<half>(-125.0f);
constexpr half ZERO_VALUE = static_cast<half>(0.0f);
constexpr int16_t SHIFT_VALUE = 23;

constexpr uint8_t NUM_128 = static_cast<uint8_t>(128);
constexpr int16_t NUM_2 = static_cast<int16_t>(2);
constexpr int8_t indexSubLength = static_cast<int8_t>(32);

constexpr half LN2 = static_cast<half>(0.6931471806f);
constexpr half INV_LN2 = static_cast<half>(1.4426950409f);
constexpr half NEG_TWO_VALE = static_cast<half>(-2.0f);
constexpr half TWO_VALE = static_cast<half>(2.0f);
constexpr half MIN_VALUE = static_cast<half>(-65504.0f);
// CAUSAL: groupMax snap 链下限 (有限值, 防 -inf 进 Sub 链产生 NaN)
constexpr half CAUSAL_CLAMP = static_cast<half>(-30000.0f);

// CAUSAL 行前缀 mask: 可见 ⇔ lane >= t (t = (maskBase + kvRow) * causalScale - causalRem),
// 前缀 lane < t 置 -inf。
// lane 序号取 lane_idx_reg (0..S1Base-1); t 夹到 [0, S1Base]: 全可见时 Select 原值, 全不可见时全 -inf。
// 调用处按 tile 全可见与否选择是否实例化 (hasAtten 模板参数): VF 内标量 if 会被谓词化, 不省向量指令。
// causalScale: M 行号 → s1 行号的折算 (S1G 合轴 r=t*G+g ⇒ gSize; 头轴摊平 ⇒ 1)。
// causalRem: tile 起始行 gS1*S1Base 对 causalScale 取余 (S1G 合轴时 = gS1*S1Base % gSize)。
//            逐行阈值按全局行号 r = gS1*S1Base + lane 折算 ⇒ lane >= t*G - (gS1*S1Base % G);
//            tile 起点整除 G 时余数为 0, 与旧口径一致。
// 依赖使用处局部量: neg_inf_reg, preg_causal, preg_all_16bit, lane_idx_reg, S1Base (模板参数),
//                   causalScale / causalRem (入参)
#define CAUSAL_ROW_MASK(reg, t) \
    do { \
        int32_t causal_t_ = static_cast<int32_t>(t) * causalScale - causalRem; \
        causal_t_ = (causal_t_ < 0) ? 0 : causal_t_; \
        causal_t_ = (causal_t_ > static_cast<int32_t>(S1Base)) ? static_cast<int32_t>(S1Base) : causal_t_; \
        Compares<int16_t, CMPMODE::GE>(preg_causal, lane_idx_reg, static_cast<int16_t>(causal_t_), preg_all_16bit); \
        Select((reg), (reg), neg_inf_reg, preg_causal); \
    } while (0)

} // namespace Mxfp4Api
#endif // VF_COMMON_DEF_H_
