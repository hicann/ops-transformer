/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_mat_mul_allto_allv_mte_tiling.h
 * \brief AIV-driven MTE communication parameters for GroupedMatMulAlltoAllvV2.
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_MTE_TILING_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_MTE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace MC2KernelTemplate {

constexpr uint32_t GMMA2AV_MTE_MAX_COUNT_NUM = 1024U;
constexpr uint32_t GMMA2AV_MTE_MAX_RANK_SIZE = 128U;

// Zero closes only at the final expert, including any trailing empty experts.
constexpr uint32_t GMMA2AV_MTE_SINGLE_CHUNK = 0U;
constexpr uint32_t GMMA2AV_MTE_DENSE_EXPERT_CHUNK_ROWS = 1536U;

// The AIV-driven MTE path only consumes the fields below. Keep this definition
// local instead of inheriting TaskTilingInfo from the legacy implementation.
struct GroupedMatMulAlltoAllvMteTaskTilingInfo {
    uint64_t BSK;
    uint64_t BS;
    uint64_t H1;
    uint64_t H2;
    uint64_t A;
    uint64_t N1;
    uint64_t N2;
    uint64_t epWorldSize;
    uint64_t e;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    int32_t sendCnt[GMMA2AV_MTE_MAX_COUNT_NUM];
    int32_t recvCnt[GMMA2AV_MTE_MAX_COUNT_NUM];
};

struct GmmA2avCoCTiling {
    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t ubMoveNum;
    int32_t swizzlCount;
    int32_t swizzlDirect;
};

// Serialized only for the MTE path selected by the AIV tiling keys. Other keys
// continue to serialize QuantGmmA2avTilingData directly and do not share this ABI.
struct GroupedMatMulAlltoAllvMteTilingData {
    // NNopbase reads this header from offset zero to allocate the communicator
    // context passed to the kernel. It is a runtime ABI requirement, not a
    // dependency on the legacy quant tiling wrapper.
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    GroupedMatMulAlltoAllvMteTaskTilingInfo taskTilingInfo;
    uint64_t commBufferSize;
    uint64_t cumsumWorkspaceOffset;
    // Ascend 910_93 and Ascend 910B expose different HCCL window-context
    // layouts for the AIV-driven MTE path.
    uint32_t isA3;
    // Preserve the serialized size and following member offsets.
    uint32_t reserved;
    // Expert-overlap pipeline granularity: consecutive experts are batched
    // into one synchronization stage until their cumulative local GMM rows
    // reach this threshold (boundary snapped to an expert end). Zero selects
    // one chunk ending at the final expert. Host tiling derives this from counts.
    uint32_t expertChunkRows;
    GmmA2avCoCTiling cocTiling;
};

// The correctness-oriented AIV-driven MTE path uses fixed CoC values. These
// constants are the single source of truth shared by CATLASS and MTE communication until
// host-side tile tuning is introduced.
namespace Gmma2avMteTiling {
constexpr int32_t L1_TILE_M = 128;
constexpr int32_t L1_TILE_K = 256;
constexpr int32_t L1_TILE_N = 256;
constexpr int32_t L0_TILE_M = 128;
constexpr int32_t L0_TILE_K = 64;
constexpr int32_t L0_TILE_N = 256;
// Tile tuning is intentionally deferred. Host tiling selects one complete N1
// row per peer-memory move and validates it against this payload region.
constexpr uint64_t PAYLOAD_UB_REGION_BYTES = 64UL * 1024UL;
constexpr uint32_t UB_MOVE_ELEMENTS = 4U * 1024U;
constexpr int32_t SWIZZLE_COUNT = 1;
constexpr int32_t SWIZZLE_DIRECTION = 0;
constexpr uint64_t WORKSPACE_ALIGNMENT = 512UL;
// Keep peer-memory accesses away from the end of the CCL mapping.  The two
// mature arch22 AllToAll+Matmul kernels place their flags at 180 MiB in the
// default 200 MiB window.  Express the same contract from the tail so the
// layout also works when HCCL_BUFFSIZE is enlarged.
constexpr uint64_t MIN_CCL_BUFFER_BYTES = 200UL * 1024UL * 1024UL;
constexpr uint64_t CCL_TAIL_SAFETY_BYTES = 20UL * 1024UL * 1024UL;

// Control data is reverse-packed below the synchronization flags.  Its
// addresses therefore do not depend on the communication dtype/data size and
// cannot move into the flag region when different cases share a CCL window.
// At the default buffer size the data-control and flag regions start at
// 179 MiB and 180 MiB respectively.
constexpr uint64_t CONTROL_REGION_BYTES = 21UL * 1024UL * 1024UL;
constexpr uint64_t SEND_COUNT_STAGING_FROM_TAIL = CONTROL_REGION_BYTES;
// Every source publishes one packed send-count table in its own window.  The
// table is separated from payload/control storage on a workspace boundary.
constexpr uint64_t COUNT_TABLE_ALIGNMENT_BYTES = WORKSPACE_ALIGNMENT;
constexpr uint64_t SYNC_REGION_FROM_TAIL = CCL_TAIL_SAFETY_BYTES;
constexpr uint64_t SYNC_SLOT_BYTES = 64UL;
constexpr uint32_t MAX_RANK_SIZE = GMMA2AV_MTE_MAX_RANK_SIZE;
constexpr uint32_t MAX_COUNT_NUM = GMMA2AV_MTE_MAX_COUNT_NUM;
// Version 2 keeps the byte offsets but stores an int64_t epoch in each slot.
constexpr uint32_t FIXED_SYNC_LAYOUT_VERSION = 2U;
constexpr int64_t MAX_SYNC_EPOCH = 0x7ffffffffffffffeLL;

// Byte offsets within the fixed synchronization region.  Region bases do not
// depend on the current rank size or expert count, so persistent HCCL windows
// can safely be reused by cases with different shapes.
constexpr uint64_t EPOCH_BASE = 0UL;
constexpr uint64_t COUNT_READY_BASE = EPOCH_BASE + SYNC_SLOT_BYTES;
constexpr uint64_t EXPERT_READY_BASE = COUNT_READY_BASE + static_cast<uint64_t>(MAX_RANK_SIZE) * SYNC_SLOT_BYTES;
constexpr uint64_t COMPLETION_BASE = EXPERT_READY_BASE + static_cast<uint64_t>(MAX_COUNT_NUM) * SYNC_SLOT_BYTES;
constexpr uint64_t ACK_BASE = COMPLETION_BASE + static_cast<uint64_t>(MAX_RANK_SIZE) * SYNC_SLOT_BYTES;
constexpr uint64_t FIXED_SYNC_BYTES = ACK_BASE + static_cast<uint64_t>(MAX_RANK_SIZE) * SYNC_SLOT_BYTES;

static_assert(SYNC_SLOT_BYTES == 64UL, "Every synchronization slot must occupy one cache line");
static_assert(COUNT_READY_BASE % SYNC_SLOT_BYTES == 0UL, "Count-ready region must be slot aligned");
static_assert(EXPERT_READY_BASE % SYNC_SLOT_BYTES == 0UL, "Expert-ready region must be slot aligned");
static_assert(COMPLETION_BASE % SYNC_SLOT_BYTES == 0UL, "Completion region must be slot aligned");
static_assert(ACK_BASE % SYNC_SLOT_BYTES == 0UL, "Acknowledgement region must be slot aligned");
static_assert(FIXED_SYNC_BYTES <= SYNC_REGION_FROM_TAIL,
              "Fixed synchronization layout must fit in the reserved CCL tail region");
} // namespace Gmma2avMteTiling

} // namespace MC2KernelTemplate

// GET_TILING_DATA_WITH_STRUCT expands the selected type as an unqualified name
// in some framework versions.
using GroupedMatMulAlltoAllvMteTilingData = MC2KernelTemplate::GroupedMatMulAlltoAllvMteTilingData;

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_MTE_TILING_H
