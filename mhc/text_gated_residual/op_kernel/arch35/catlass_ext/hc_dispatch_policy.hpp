/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */
/**
 * Operator-local Catlass policies missing from ops-transformer third_party catlass v2.0.0.
 * Do not copy the whole catlass tree; these structs only enable BlockMmadTla specializations
 * in this directory (MmadPingpongPreLoad / MmadAscend950BasicSplitK).
 */
#pragma once

#include <type_traits>
#include "catlass/gemm/dispatch_policy.hpp"

namespace Catlass::Gemm {

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
          bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2,
          uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadPingpongPreLoad : public MmadBase<ArchTag_, true> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = true, uint32_t L0C_STAGES_ = 2, uint32_t L1_STAGES_ = 2,
          uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2, bool ENABLE_MM_LAYOUT_TRANSFORM_ = true,
          bool ENABLE_INNER_SPLIT_K_ = true>
struct MmadAscend950BasicSplitK : public MmadBase<ArchTag_, false> {
    static_assert(std::is_same_v<ArchTag_, Arch::Ascend950>, "MmadAscend950BasicSplitK only supports Arch::Ascend950");
    static constexpr uint32_t L1_STAGES = L1_STAGES_;
    static constexpr uint32_t L1A_STAGES = L1_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_MM_LAYOUT_TRANSFORM = ENABLE_MM_LAYOUT_TRANSFORM_;
    static constexpr bool ENABLE_INNER_SPLIT_K = ENABLE_INNER_SPLIT_K_;
    static constexpr uint64_t FP32_SPLIT_K_BASE1 = 1024;
    static constexpr uint64_t FP32_SPLIT_K_BASE2 = 8192;
    static constexpr uint64_t FP32_K_SWITCH_BASE = 268435456;
};

} // namespace Catlass::Gemm
