/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLASH_ATTENTION_SCORE_GRAD_QUANT_CONFIG_ARCH92_H_
#define FLASH_ATTENTION_SCORE_GRAD_QUANT_CONFIG_ARCH92_H_
#include <cstdint>
#if defined(__NPU_ARCH__)
#define FAG_QUANT_CONFIG_INLINE __aicore__ inline
#else
#define FAG_QUANT_CONFIG_INLINE inline
#endif
namespace FagBaseApi {
namespace quant {
// Fixed VBS + dQ BF16 Fixpipe implementation; no experiment switches.
// Architecture and tile constants shared by Host and Kernel.
constexpr uint32_t QUANT_CV_CORE_RATIO = 1;
constexpr uint32_t QUANT_CUBE_BASE = 128;
constexpr uint32_t QUANT_INNER_LOOP_NUM = 4;
constexpr uint32_t QUANT_K_SIZE = 256;
constexpr uint32_t QUANT_BASEK = 128;
constexpr uint16_t QUANT_S1_HALF = 64;
constexpr uint32_t QUANT_SP_PASS_NUM = 2;
constexpr uint32_t UB_SIZE_ARCH92 = 384 * 1024;
constexpr uint32_t QUANT_SP_SLAB_ELEM = QUANT_CUBE_BASE * QUANT_CUBE_BASE;

constexpr uint32_t SP_SLOTS = 2;
constexpr uint32_t SP_ELEMS = 128 * 128;
constexpr uint32_t GRAD_CREDITS = 3; // Completion credits, not gradient UB slots.
constexpr uint32_t STATS_BYTES = 8 * 1024;
constexpr uint32_t PDS_OUTPUT_ELEMS = 128 * 128;
constexpr uint32_t MAIN_UB_BYTES =
    SP_SLOTS * SP_ELEMS * sizeof(float) * 2 + SP_SLOTS * PDS_OUTPUT_ELEMS * 2 + STATS_BYTES;
constexpr uint32_t ARCH92_USABLE_UB_BYTES = (384 - 8) * 1024;
constexpr uint32_t BALANCED_REDUCE_ELEMS = 64 * 128;
constexpr uint32_t BALANCED_REDUCE_BYTES = BALANCED_REDUCE_ELEMS * 2 * sizeof(float);
static_assert(MAIN_UB_BYTES == 328 * 1024, "Unexpected quant UB layout");
static_assert(MAIN_UB_BYTES <= ARCH92_USABLE_UB_BYTES, "Quant main buffers exceed usable UB");
static_assert(BALANCED_REDUCE_BYTES <= ARCH92_USABLE_UB_BYTES, "Balanced reduction exceeds usable UB");

FAG_QUANT_CONFIG_INLINE bool UseDqFixBf16(uint64_t s2Outer)
{
    return s2Outer == 1;
}

FAG_QUANT_CONFIG_INLINE bool UseBalanced(uint64_t bn, uint64_t s1, uint64_t s2, uint64_t cores)
{
    return s2 > 0 && s2 <= 512 && s1 > 512 && bn > 0 && bn < cores;
}

FAG_QUANT_CONFIG_INLINE uint64_t TaskBegin(uint64_t total, uint64_t core, uint64_t cores)
{
    return total / cores * core + (total % cores < core ? total % cores : core);
}

FAG_QUANT_CONFIG_INLINE uint64_t PartialOffset(uint64_t sfmgOffset, uint64_t bn, uint64_t s1)
{
    // Host sfmg allocation includes a padded last head and BIT_NUMS=8 (the existing host allocation).
    uint64_t elements = (bn - 1) * s1 + (s1 + 127) / 128 * 128;
    return (sfmgOffset + elements * 8 * 4 + 512) / 512 * 512;
}

FAG_QUANT_CONFIG_INLINE uint64_t PartialBytes(uint64_t cores, uint64_t s2)
{
    // A contiguous core interval is <= one head's Q-block count, hence intersects <=2 heads.
    return cores * 2 * 2 * s2 * 128 * 4;
}

} // namespace quant
} // namespace FagBaseApi
#undef FAG_QUANT_CONFIG_INLINE
#endif
