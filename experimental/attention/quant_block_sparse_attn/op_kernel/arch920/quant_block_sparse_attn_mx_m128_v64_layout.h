/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root.
 */
#ifndef QBSA_MX_M128_V64_LAYOUT_H_
#define QBSA_MX_M128_V64_LAYOUT_H_

#include <cstdint>

// [M128_V64] Compile-time A/B switch, scoped to the 9102 MX entry only.
#ifndef QBSA_9102_MX_M128_V64
#define QBSA_9102_MX_M128_V64 1
#endif
static_assert(QBSA_9102_MX_M128_V64 == 0 || QBSA_9102_MX_M128_V64 == 1, "QBSA_9102_MX_M128_V64 must be 0 or 1");

// [M128_P_PINGPONG] Compile-time A/B switch within the M128/V64 experiment.
// 0 restores the aa69207df packed-P stage selection; this is not a runtime flag.
#ifndef QBSA_9102_MX_M128_P_PINGPONG
#define QBSA_9102_MX_M128_P_PINGPONG 1
#endif
static_assert(QBSA_9102_MX_M128_P_PINGPONG == 0 || QBSA_9102_MX_M128_P_PINGPONG == 1,
              "QBSA_9102_MX_M128_P_PINGPONG must be 0 or 1");

// [M128_C2_DB] Trade one 64KiB C1 slot for one additional 32KiB C2 slot.
#ifndef QBSA_9102_MX_M128_C2_DB
#define QBSA_9102_MX_M128_C2_DB 1
#endif
static_assert(QBSA_9102_MX_M128_C2_DB == 0 || QBSA_9102_MX_M128_C2_DB == 1, "QBSA_9102_MX_M128_C2_DB must be 0 or 1");

// [M128_MASK_PREFETCH] Isolated task-local mask DMA lookahead. Rebuild with 0 for baseline.
#ifndef QBSA_9102_MX_M128_MASK_PREFETCH
#define QBSA_9102_MX_M128_MASK_PREFETCH 1
#endif
static_assert(QBSA_9102_MX_M128_MASK_PREFETCH == 0 || QBSA_9102_MX_M128_MASK_PREFETCH == 1,
              "mask prefetch must be 0 or 1");

// [M128_MTE3_SCOPE] Protect only actual accumulator output DMA, independently per row slice.
// 0 restores the mask-prefetch baseline's shared, unconditional V2 MTE3 fence.
#ifndef QBSA_9102_MX_M128_MTE3_SCOPE
#define QBSA_9102_MX_M128_MTE3_SCOPE 1
#endif
static_assert(QBSA_9102_MX_M128_MTE3_SCOPE == 0 || QBSA_9102_MX_M128_MTE3_SCOPE == 1, "V2 MTE3 scope must be 0 or 1");

namespace BaseApi {
struct MxM128V64Layout {
    static constexpr uint32_t CUBE_M = 128U;
    static constexpr uint32_t VEC_M = 64U;
    static constexpr uint32_t ROW_SLICES = CUBE_M / VEC_M;
    static constexpr uint32_t S2 = 512U;
    static constexpr uint32_t HALF_S2 = 256U;
    static constexpr uint32_t D = 128U;
    static constexpr uint32_t C1_SLOTS = QBSA_9102_MX_M128_C2_DB ? 2U : 3U;
    static constexpr uint32_t C2_SLOTS = QBSA_9102_MX_M128_C2_DB ? 2U : 1U;
    static_assert(2U * (C1_SLOTS + C2_SLOTS) + 3U == 11U, "Cross-core ID budget");
    static constexpr uint32_t C1_UB_BYTES = VEC_M * HALF_S2 * 4U;
    static constexpr uint32_t C2_UB_BYTES = VEC_M * D * 4U;
    static constexpr uint32_t ACC_UB_BYTES = CUBE_M * D * 4U;
    static constexpr uint32_t P_DATA_BYTES = CUBE_M * S2;
    static constexpr uint32_t P_SCALE_HALF_BYTES = CUBE_M * HALF_S2 / 32U;
    static constexpr uint32_t P_SLOT_BYTES = P_DATA_BYTES + P_DATA_BYTES / 32U;
    static constexpr uint32_t L1_BYTES =
        2U * (CUBE_M * D * 33U / 32U) + 6U * (HALF_S2 * D * 33U / 32U) + 3U * P_SLOT_BYTES;
    // C1*C1_SLOTS + C2*C2_SLOTS + accumulator + (packed-P + dynamic scale)*2 +
    // (sum/max/exp)*3 + preMax + mask + padded LSE + DN index + optional indices.
    static constexpr uint32_t MAX_UB_BYTES =
        C1_SLOTS * C1_UB_BYTES + C2_SLOTS * C2_UB_BYTES + ACC_UB_BYTES + 2U * VEC_M * (HALF_S2 + HALF_S2 / 32U) +
        2U * VEC_M * HALF_S2 / 32U + 10U * CUBE_M * 4U + (1U + QBSA_9102_MX_M128_MASK_PREFETCH) * VEC_M * HALF_S2 +
        VEC_M * 4U * 8U + 256U + 8192U;
    static_assert(L1_BYTES == 429U * 1024U && L1_BYTES <= 512U * 1024U, "M128 L1 budget");
    static_assert(MAX_UB_BYTES ==
                          (QBSA_9102_MX_M128_C2_DB ? 328960U : 361728U) + QBSA_9102_MX_M128_MASK_PREFETCH * 16384U &&
                      MAX_UB_BYTES <= 376U * 1024U,
                  "M128 UB budget");
    static_assert(2U * HALF_S2 * CUBE_M * 4U == 256U * 1024U, "M128 L0C double buffer");

    __aicore__ static inline uint32_t Align16(uint32_t n)
    {
        return (n + 15U) / 16U * 16U;
    }
    // Offsets below are elements, not bytes. C1 slices N, C2 slices M in NZ [N/16,M,16].
    __aicore__ static inline uint64_t C1SourceOffset(uint32_t rowSlice, uint32_t s2Real)
    {
        return static_cast<uint64_t>(rowSlice) * VEC_M * Align16(s2Real);
    }
    __aicore__ static inline uint64_t C2SourceOffset(uint32_t rowSlice)
    {
        return static_cast<uint64_t>(rowSlice) * VEC_M * 16U;
    }
    // P is packed as two [S2=256, Q=128] NZ regions. Each contains two contiguous Q=64 regions.
    __aicore__ static inline uint64_t PDataOffset(uint32_t subLoop, uint32_t rowSlice)
    {
        return static_cast<uint64_t>(subLoop) * HALF_S2 * CUBE_M + static_cast<uint64_t>(rowSlice) * HALF_S2 * VEC_M;
    }
};
} // namespace BaseApi
#endif
