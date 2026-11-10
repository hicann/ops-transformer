/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_MXFP8_CONSTANTS_H
#define GBSA_MXFP8_CONSTANTS_H

// Keep the original unsigned literal types and values: these constants name
// existing template choices and serialized key fields, without changing either.
namespace GbsaMxScaleMode {
constexpr auto PIPELINE = 4U;           // Grouped K-scale streaming.
constexpr auto COALESCED_PIPELINE = 5U; // Retained prefill dispatch alias of grouped streaming.
} // namespace GbsaMxScaleMode

namespace GbsaTilingKeyEncoding {
constexpr auto DECIMAL_RADIX = 10U;
constexpr auto MASK_DIVISOR = 1000U;
constexpr auto KV_LAYOUT_DIVISOR = 10000000ULL;

// Serialized additives used by host key generation. Do not change the key ABI.
constexpr auto BF16_DTYPE_FIELD = 22220ULL;
constexpr auto FP8_OUT_FP16_FIELD = 10ULL;
constexpr auto FP8_OUT_BF16_FIELD = 20ULL;
constexpr auto KV_PA_BBND_FIELD = 70000000ULL;
constexpr auto KV_PA_BNBD_FIELD = 80000000ULL;
constexpr auto KV_BNSD_FIELD = 10000000ULL;
constexpr auto KV_BSND_FIELD = 20000000ULL;
constexpr auto KV_TND_FIELD = 30000000ULL;
constexpr auto PAGED_FIELD = 1000000ULL;
constexpr auto LOW_PRECISION_SOFTMAX_FIELD = 100000ULL;
constexpr auto MASK_FIELD = 1000ULL;
constexpr auto Q_TND_FIELD = 2ULL;
constexpr auto Q_BNSD_FIELD = 3ULL;
constexpr auto Q_BSND_FIELD = 4ULL;
} // namespace GbsaTilingKeyEncoding

#endif // GBSA_MXFP8_CONSTANTS_H
