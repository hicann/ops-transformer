/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_CONSTANTS_H
#define MEGA_MOE_CONSTANTS_H

#include <cstdint>

namespace MegaMoeImpl {

enum class MegaMoeActMode : uint8_t {
    SWIGLU = 0U,
    SITU = 1U,
    SWIGLU_STEP = 2U,
    SWIGLU_OAI = 3U,
};

enum class MegaMoeActSubMode : uint8_t {
    DEFAULT = 0U,
    LINEAR = 1U,
};

constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint64_t IDX_GMM1_OFFSET = 12UL;
constexpr uint64_t IDX_GMM2_OFFSET = 13UL;

constexpr int32_t INT_CACHELINE = 16;
constexpr uint64_t INT32_PER_256B = 8U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t DOUBLE_BUFFER = 2U;
constexpr uint32_t RANK_ID = 0U;
constexpr uint32_t TOKEN_ID = 1U;
constexpr uint32_t TOPK_INDEX = 2U;
constexpr uint32_t WEIGHT_INDEX = 3U;
constexpr uint32_t SYNC_EVENT_ID0 = 0;
constexpr uint32_t SYNC_EVENT_ID1 = 1;
constexpr uint32_t SYNC_EVENT_ID2 = 2;
constexpr uint32_t SYNC_EVENT_ID3 = 3;
constexpr uint32_t SYNC_EVENT_ID4 = 4;
constexpr uint32_t SYNC_EVENT_ID5 = 5;
constexpr int64_t GM_FLAG_POLL_BACKOFF_CYCLES = 200;
constexpr int64_t E5M2_QUANT = 3U;
constexpr int64_t E2M1_QUANT = 5U;
constexpr int64_t OVERFLOW_MODE_CTRL = 60U;
constexpr uint8_t COMBINE_NO_QUANT = 0;
constexpr uint8_t MXFP8_E4M3_COMM_QUANT = 4;
constexpr uint32_t META_INFO_SIZE = 8U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W8_NZ = 2U;
constexpr uint8_t GROUPED_MATMUL_MODE_A4W4_NZ = 4U;

constexpr uint32_t L1_TILE_M_256 = 256U;
constexpr uint32_t L1_TILE_M_128 = 128U;
constexpr uint32_t L1_TILE_N = 256U;
constexpr uint32_t L0_TILE_K = 128U;
constexpr uint32_t ACTIVATION_N_HALF = 2U;
} // namespace MegaMoeImpl

#endif // MEGA_MOE_CONSTANTS_H
