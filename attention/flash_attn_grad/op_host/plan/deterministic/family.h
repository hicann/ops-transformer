/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FAMILY_H_
#define FLASH_ATTN_GRAD_DETER_FAMILY_H_

#include <cstdint>

namespace optiling {
// Shared Host contract for compile-time decoders. Keep values synchronized with
// op_kernel/fag_tiling.py:DeterFamily; 27..31 remain reserved for future families.
namespace DeterFamily {
constexpr int64_t LEGACY = 0;
constexpr int64_t FIXED_MHA_DENSE = 1;
constexpr int64_t FIXED_MHA_CAUSAL = 2;
constexpr int64_t FIXED_MHA_BAND = 3;
constexpr int64_t FIXED_GQA_DENSE = 4;
constexpr int64_t FIXED_GQA_CAUSAL = 5;
constexpr int64_t FIXED_GQA_BAND_PIECEWISE = 6;
constexpr int64_t FIXED_GQA_BAND_COLUMN = 7;
constexpr int64_t FIXED_GQA_BAND_BALANCED = 8;
constexpr int64_t FIXED_GQA_BAND_TRANSLATED = 9;
constexpr int64_t FIXED_GQA_BAND_FOLDED = 10;
constexpr int64_t FIXED_GQA_BAND_OVERLAP = 11;
constexpr int64_t TND_MHA_DENSE = 12;
constexpr int64_t TND_MHA_SPARSE = 13;
constexpr int64_t TND_GQA_DENSE = 14;
constexpr int64_t TND_GQA_MIRROR = 15;
constexpr int64_t TND_GQA_COLUMN_C4 = 16;
// Separate capabilities when normalized effective Q/KV lengths may differ.
constexpr int64_t TND_GQA_MIRROR_OR_BAND = 17;
constexpr int64_t TND_GQA_COLUMN_C4_OR_BAND = 18;
constexpr int64_t TND_GQA_CAUSAL_COMPAT = 19;
constexpr int64_t TND_GQA_CAUSAL_OR_BAND_COMPAT = 20;
// Source-compatible fixed fallback schedules, isolated from the legacy kernel.
constexpr int64_t FIXED_MHA_DENSE_BASELINE = 21;
constexpr int64_t FIXED_MHA_CAUSAL_BASELINE = 22;
constexpr int64_t FIXED_MHA_BAND_BASELINE = 23;
constexpr int64_t FIXED_GQA_BAND_BASELINE = 24;
constexpr int64_t FIXED_GQA_CAUSAL_GROUPED = 25;
constexpr int64_t FIXED_MHA_BAND_GROUPED = 26;
} // namespace DeterFamily
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FAMILY_H_
