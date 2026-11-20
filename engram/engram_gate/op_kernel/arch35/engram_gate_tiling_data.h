/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_GATE_TILING_DATA_H
#define ENGRAM_GATE_TILING_DATA_H

#include <cstdint>

namespace EngramGateTiling {
constexpr uint32_t VECTOR_LANES = 64;
constexpr uint32_t CHUNK_FIXED_BYTES = 32 + 3 * VECTOR_LANES * sizeof(float) + 4 * 32;
constexpr uint32_t CHUNK_STREAM_BYTES_PER_ELEMENT = 22;
constexpr uint32_t CHUNK_RESIDENT_BYTES_PER_ELEMENT = 14;
constexpr uint32_t WEIGHT_RESIDENT_UB_DIVISOR = 3;
} // namespace EngramGateTiling

// ============================================================================
// EngramGateTilingData — host-to-device ABI for the pure-path layout.
// All offsets are byte offsets from the TPipe root buffer base.
// Per-slot sizes are derived in-kernel from rowsPerSlot and dim.
// ============================================================================
struct EngramGateTilingData {
    // ── Shared (all paths) ──
    int64_t heads;      // H = 4 (fixed)
    int64_t dim;        // D = hidden_size (multiple of 64)
    int64_t rows;       // T × H total rows
    uint32_t chunk;     // chunked-path chunk size
    uint32_t saveStats; // 1 = write backward stats
    float eps;
    float clampValue;
    float invDim;       // 1/D (host precomputed)
    float invSqrtDim;   // 1/√D (host precomputed)
    uint32_t usedCores; // blockDim

    // ── Pure path (unified row-batch layout) ──
    uint32_t pureSlots;       // S: 2 (double-buffer) or 1 (fallback)
    uint32_t pureRowsPerSlot; // K: rows per slot (1, 2 or multiple of 4)
    uint32_t pureUbBytes;     // total layout bytes (TPipe alloc size)
    uint32_t pureXOff;        // x slots region start
    uint32_t pureKvOff;       // kv slots region start
    uint32_t pureOutOff;      // out slots region start
    uint32_t pureWOff;        // resident w start
    uint32_t pureStatsOff;    // stats window start
    uint32_t pureBypass;      // 1 = output uses L2 bypass store

    // ── Chunked path ──
    uint32_t chunkedEnabled;
    uint32_t ubLayoutBytes;
    uint32_t weightResident;
};

#endif
