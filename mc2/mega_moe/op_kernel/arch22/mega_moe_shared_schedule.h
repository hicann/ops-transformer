/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SHARED_SCHEDULE_H
#define MEGA_MOE_SHARED_SCHEDULE_H
#include <cstdint>

namespace MegaMoeSharedGmm2 {
constexpr uint32_t CHUNK = 1024;
// All ranks must make the same scheduling decision from immutable call parameters.
__aicore__ inline bool CanOverlap(uint32_t m, uint32_t shared, bool supportedType, bool mask, uint64_t roundsMax,
                                  uint64_t budget, uint32_t ep, uint32_t topK, uint32_t epr)
{
    if (shared == 0 || !supportedType || mask || m == 0 || roundsMax != 1 || ep == 0 || topK == 0 || epr == 0) {
        return false;
    }
    const uint64_t worst = static_cast<uint64_t>(CHUNK) * ep * (topK < epr ? topK : epr);
    return budget >= worst;
}
struct Range {
    uint32_t base;
    uint32_t rows;
};
struct Schedule {
    bool activation;
    bool gmm2;
};
__aicore__ inline Schedule ForRound(uint32_t shared, bool chunked, uint32_t chunk, uint64_t round)
{
    const bool first = shared > 0 && chunk == 0 && round == 0;
    return {first, shared > 0 && round == 0 && (chunk == 0 || chunked)};
}
__aicore__ inline Range TokenRange(uint32_t m, uint32_t chunk)
{
    const uint64_t base = static_cast<uint64_t>(chunk) * CHUNK;
    if (base >= m)
        return {m, 0};
    return {static_cast<uint32_t>(base), m - base < CHUNK ? static_cast<uint32_t>(m - base) : CHUNK};
}
__aicore__ inline uint64_t ExpertOffset(uint32_t expert, uint32_t fullM, uint32_t base, uint32_t width)
{
    return (static_cast<uint64_t>(expert) * fullM + base) * width;
}
} // namespace MegaMoeSharedGmm2
#endif
