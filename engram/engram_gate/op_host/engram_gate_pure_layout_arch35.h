/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// EngramGate pure-path UB layout solver
// ============================================================================
// Model: S uniform slots, each holding K rows (a head, a head pair or whole
// tokens). Input (x, kv) and output (out) live in separate regions so
// their slot lifetimes are independent:
//
//   Input slot lifecycle:  MTE2 writes → VF reads  → released (V_MTE2 credit)
//   Output slot lifecycle: VF writes  → MTE3 reads → released (MTE3_V credit)
//
// GM kv layout per token: [k0 | k1 | k2 | k3 | v] — k and v contiguous (5D).
// For K >= 4 (whole tokens) the kv region stores full 5D token blocks and is
// copied in one contiguous burst per token. For K == 2 (half token, large D)
// the kv region stores [k-pair(2 rows) | v(1 row)] = 3 rows. For K == 1
// it stores [k_head | v] = 2 rows.
//
// UB layout (byte offsets from TPipe root):
//   [x:    S slots × K rows  × D × 2B]   ← input x
//   [kv:   S slots × KV rows × D × 2B]   ← input k + v
//   [out:  S slots × K rows  × D × 2B]   ← output
//   [w:    4 rows × D × 4B]              ← resident weight address space
//   [stats: window ≤ 64 × 4 × 32B]       ← save=true only
// ============================================================================

#ifndef ENGRAM_GATE_PURE_LAYOUT_ARCH35_H
#define ENGRAM_GATE_PURE_LAYOUT_ARCH35_H

#include <cstdint>
#include <limits>

namespace EngramGatePure {

namespace Policy {
constexpr int64_t VF_LANES = 64;       // one 256B FP32 vector
constexpr uint32_t HEADS = 4;          // H: heads per token (fixed)
constexpr uint32_t KV_ROWS_FULL = 5;   // k0..k3 + v per token
constexpr uint32_t KV_ROWS_HALF = 3;   // k-pair(2) + v(1) per half-token
constexpr uint32_t KV_ROWS_SINGLE = 2; // one k and the shared v
constexpr uint32_t BF16_BYTES = 2;
constexpr uint32_t FP32_BYTES = 4;
// K ceiling: S×K must ≤ STATS_MAX_WINDOW(64) when save=true, else stats overflow.
// For save=true with S=2: K ≤ 32. For S=1: K ≤ 64. Use 32 as universal safe ceiling.
constexpr uint32_t MAX_ROWS_PER_SLOT = 32;
constexpr uint32_t STATS_CLASSES = 4;
constexpr uint32_t STATS_SLOT_BYTES = 32;
constexpr uint32_t STATS_MAX_WINDOW = 64;
constexpr uint64_t FIELD_MAX = std::numeric_limits<uint32_t>::max();
} // namespace Policy

struct Layout {
    uint32_t slots = 0;       // S: ping-pong depth (2 or 1)
    uint32_t rowsPerSlot = 0; // K: rows per slot (1, 2 or multiple of 4)
    uint32_t kvRows = 0;      // rows in kv region per slot (2, 3 or 5×K/4)
    uint32_t xBytes = 0;      // per-slot x region size
    uint32_t kvBytes = 0;     // per-slot kv region size
    uint32_t outBytes = 0;    // per-slot out region size
    uint32_t wBytes = 0;      // resident w size
    uint32_t total = 0;       // total layout bytes
    // Byte offsets from UB base
    uint32_t xOff = 0;     // start of x slots
    uint32_t kvOff = 0;    // start of kv slots
    uint32_t outOff = 0;   // start of out slots
    uint32_t wOff = 0;     // start of resident w
    uint32_t statsOff = 0; // start of stats window
};

// Reserve four non-aliasing statistics windows for the output copies.
inline bool TryStatsBytes(uint64_t ub, uint64_t fixedBytes, uint32_t K, uint32_t S, uint64_t& statsBytes)
{
    const uint64_t statsStride = (static_cast<uint64_t>(K) + 7U) & ~7ULL;
    const uint64_t minimumBytes = 4U * Policy::STATS_CLASSES * statsStride * sizeof(float);
    const uint64_t minEntries = static_cast<uint64_t>(S) * K;
    const uint64_t capacity = (ub - fixedBytes) / (Policy::STATS_CLASSES * Policy::STATS_SLOT_BYTES);
    if (capacity < minEntries || ub - fixedBytes < minimumBytes) {
        return false;
    }
    const uint64_t capped = capacity > Policy::STATS_MAX_WINDOW ? Policy::STATS_MAX_WINDOW : capacity;
    uint64_t window = 1;
    while (window * 2 <= capped) {
        window *= 2;
    }
    statsBytes = window * Policy::STATS_CLASSES * Policy::STATS_SLOT_BYTES;
    return statsBytes >= minimumBytes;
}

// Try one (K, S) combination. Returns false if it exceeds the UB budget.
inline bool TryLayout(int64_t dim, bool save, uint64_t ub, uint32_t K, uint32_t S, Layout& out)
{
    const uint64_t d = static_cast<uint64_t>(dim);
    const uint64_t xBytesPerSlot = K * d * Policy::BF16_BYTES;
    const uint64_t kvRowsPerSlot = (K >= Policy::HEADS) ?
                                       (K / Policy::HEADS) * Policy::KV_ROWS_FULL // whole tokens: 5D each
                                       :
                                       (K == 1 ? Policy::KV_ROWS_SINGLE : Policy::KV_ROWS_HALF);
    const uint64_t kvBytesPerSlot = kvRowsPerSlot * d * Policy::BF16_BYTES;
    const uint64_t outBytesPerSlot = K * d * Policy::BF16_BYTES;
    // Keep the natural four-row weight address space. The kernel may transfer
    // only the active K=2 head pair into its original offset without changing
    // the wUb addressing contract for whole-token slots.
    const uint64_t weightRows = Policy::HEADS;
    const uint64_t wBytes = weightRows * d * Policy::FP32_BYTES;

    const uint64_t inputBytes = S * (xBytesPerSlot + kvBytesPerSlot);
    const uint64_t outputBytes = S * outBytesPerSlot;
    const uint64_t fixedBytes = inputBytes + outputBytes + wBytes;
    if (fixedBytes > ub || fixedBytes > Policy::FIELD_MAX) {
        return false;
    }
    uint64_t statsBytes = 0;
    if (save && !TryStatsBytes(ub, fixedBytes, K, S, statsBytes)) {
        return false;
    }

    const uint64_t total = fixedBytes + statsBytes;
    if (total > ub || total > Policy::FIELD_MAX || xBytesPerSlot > Policy::FIELD_MAX ||
        kvBytesPerSlot > Policy::FIELD_MAX) {
        return false;
    }

    out.slots = S;
    out.rowsPerSlot = K;
    out.kvRows = static_cast<uint32_t>(kvRowsPerSlot);
    out.xBytes = static_cast<uint32_t>(xBytesPerSlot);
    out.kvBytes = static_cast<uint32_t>(kvBytesPerSlot);
    out.outBytes = static_cast<uint32_t>(outBytesPerSlot);
    out.wBytes = static_cast<uint32_t>(wBytes);
    out.total = static_cast<uint32_t>(total);
    out.xOff = 0;
    out.kvOff = static_cast<uint32_t>(S * xBytesPerSlot);
    out.outOff = static_cast<uint32_t>(S * xBytesPerSlot + S * kvBytesPerSlot);
    out.wOff = static_cast<uint32_t>(S * xBytesPerSlot + S * kvBytesPerSlot + S * outBytesPerSlot);
    out.statsOff = out.wOff + out.wBytes;
    return true;
}

// Solve the layout. Double buffering (S=2) is the top priority because it hides
// MTE2 latency behind VF compute (total ≈ max(vec, mte2) instead of vec + mte2).
// K=4 halves both the batch count and copies-per-batch vs K=2, so it's preferred
// within each S tier. K=2 is the S=2 fallback for large D where K=4/S=2 doesn't fit.
//   1. S=2, K=4m  — overlap + whole token (2 copies/batch)
//   2. S=2, K=2   — overlap + half token (2-3 copies/batch, for large D)
//   3. S=2, K=1   — overlap + single head when each core keeps the same head
//   4. S=1, K=4m  — serial + whole token (fallback)
//   5. S=1, K=2   — last resort
inline bool ComputeLayout(int64_t dim, bool save, uint64_t ub, Layout& out, bool allowSingleRow = true)
{
    out = Layout{};
    if (dim <= 0 || dim % Policy::VF_LANES != 0 || ub == 0 || ub > Policy::FIELD_MAX) {
        return false;
    }
    // Tier 1: S=2 with whole tokens.
    for (uint32_t K = Policy::MAX_ROWS_PER_SLOT; K >= Policy::HEADS; K -= Policy::HEADS) {
        if (TryLayout(dim, save, ub, K, 2, out)) {
            return true;
        }
    }
    // Tier 2: S=2 with half token (large D fallback).
    if (TryLayout(dim, save, ub, 2, 2, out)) {
        return true;
    }
    // A single-head slot is valid only when the host can keep each core on
    // one head (at least four cores and a core stride divisible by four).
    if (allowSingleRow && TryLayout(dim, save, ub, 1, 2, out)) {
        return true;
    }
    // Tier 3: S=1 with whole tokens.
    for (uint32_t K = Policy::MAX_ROWS_PER_SLOT; K >= Policy::HEADS; K -= Policy::HEADS) {
        if (TryLayout(dim, save, ub, K, 1, out)) {
            return true;
        }
    }
    // Tier 4: S=1 with half token.
    if (TryLayout(dim, save, ub, 2, 1, out)) {
        return true;
    }
    return false;
}

} // namespace EngramGatePure

#endif // ENGRAM_GATE_PURE_LAYOUT_ARCH35_H
