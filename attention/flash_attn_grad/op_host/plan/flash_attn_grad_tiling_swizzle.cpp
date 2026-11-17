/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_swizzle.h"
#include "deterministic/flash_attn_grad_tiling_deter.h"
#include "deterministic/flash_attn_grad_tiling_deter_tnd.h"

#include <algorithm>
#include <limits>
#include <set>
#include <vector>

namespace optiling {
namespace {
constexpr int64_t CUBE_BASE = 128;
constexpr uint64_t INPUT_BYTES = 2;
constexpr uint64_t FP32_BYTES = 4;
constexpr uint64_t COMPRESSED_MASK_BYTES = 2048 * 2048;

int64_t CeilDivSigned(int64_t value, int64_t divisor)
{
    return value >= 0 ? (value + divisor - 1) / divisor : value / divisor;
}

// Same effective columns as padded_block_valid / col_prefix_body. The ASC
// GetSparseBlockInfo partition is a contiguous interval in this packed order.
// Keep int64_t offsets: std::set<int> in the old ASC estimate can alias large grids.
struct ValidColumns {
    std::vector<int64_t> begin;
    std::vector<int64_t> prefix;
};

ValidColumns BuildValidColumns(const FagParsedInfo& info, int64_t m, int64_t n)
{
    ValidColumns columns;
    columns.begin.resize(n);
    columns.prefix.resize(n + 1, 0);
    const int64_t p =
        info.maskMode == MASK_MODE_CAUSAL ? m - 1 : std::min(CeilDivSigned(info.s1Token, CUBE_BASE), m - 1);
    const int64_t q = std::min(CeilDivSigned(info.s2Token, CUBE_BASE), n - 1);
    for (int64_t c = 0; c < n; ++c) {
        int64_t lo = 0;
        int64_t hi = m - 1;
        if (info.maskMode != MASK_MODE_NO_MASK) {
            lo = std::max(c - q, int64_t{0});
            hi = std::min(c + p, m - 1);
            if (c == n - 1 && info.maskMode == MASK_MODE_WINDOW) {
                const int64_t alignedEnd = CeilDivSigned(info.s2, 64) * 64;
                hi = std::min(CeilDivSigned(alignedEnd + info.s1Token, CUBE_BASE) - 1, m - 1);
            }
        }
        columns.begin[c] = lo;
        columns.prefix[c + 1] = columns.prefix[c] + std::max(hi - lo + 1, int64_t{0});
    }
    return columns;
}

bool ProductFits(int64_t lhs, int64_t rhs)
{
    return lhs >= 0 && rhs > 0 && lhs <= std::numeric_limits<int64_t>::max() / rhs;
}

// Port of ASC CheckExceedL2Cache: interleave one effective tile per active core,
// accumulate distinct Q and KV addresses, and stop only after exceeding L2.
// D and Dv are independent: y/dout use Dv, q/dq and k/dk use D.
bool CheckExceedL2(const FagParsedInfo& info, const ValidColumns& columns, int64_t m, int64_t n,
                   FagSchedulePlan& schedule)
{
    const int64_t perHead = schedule.normalValidPerHead;
    const int64_t total = perHead * info.b * info.n2 * info.g;
    const uint64_t qBytes = CUBE_BASE * ((info.d + 2 * info.dv) * INPUT_BYTES + info.d * FP32_BYTES);
    const uint64_t kvBytes = CUBE_BASE * (info.d + info.dv) * (INPUT_BYTES + FP32_BYTES);
    std::set<int64_t> qAddresses;
    std::set<int64_t> kvAddresses;
    uint64_t usedBytes = 0;
    for (int64_t round = 0; round < schedule.normalMaxValidBlocks; ++round) {
        for (int64_t core = 0; core < schedule.normalBlockOuter; ++core) {
            const int64_t packed = core * schedule.normalMaxValidBlocks + round;
            if (packed >= total) {
                continue;
            }
            const int64_t head = packed / perHead;
            const int64_t within = packed % perHead;
            const auto end = std::upper_bound(columns.prefix.begin(), columns.prefix.end(), within);
            const int64_t col = static_cast<int64_t>(end - columns.prefix.begin()) - 1;
            const int64_t row = columns.begin[col] + within - columns.prefix[col];
            if (qAddresses.insert(head * m + row).second) {
                usedBytes += qBytes;
            }
            if (kvAddresses.insert((head / info.g) * n + col).second) {
                usedBytes += kvBytes;
            }
            if (usedBytes > info.l2CacheSize) {
                schedule.normalWorkingSetBytes = usedBytes;
                return true;
            }
        }
    }
    if (info.maskMode != MASK_MODE_NO_MASK) {
        usedBytes += COMPRESSED_MASK_BYTES;
    }
    schedule.normalWorkingSetBytes = usedBytes;
    return usedBytes > info.l2CacheSize;
}
} // namespace

void DecideSwizzle(const FagParsedInfo& info, const FagKernelPlan& kernel, FagSchedulePlan& schedule,
                   FagRouteTrace& trace)
{
    schedule.swizzle = 0;
    // The deterministic family planner and its key encoding remain independent.
    if (info.isDeterministic) {
        if (info.isTnd) {
            DecideTndDeterministicSchedule(info, kernel, schedule, trace);
        } else {
            DecideFixedDeterministicSchedule(info, kernel, schedule, trace);
        }
        return;
    }
    if (kernel.tmpl != TMPL_BN2GS1S2) {
        trace.swizzleReject = "not-bn2gs1s2";
        return;
    }
    if (info.hasSeqused) {
        trace.swizzleReject = "seqused-variable-grid";
        return;
    }
    if (info.isTnd && (!info.tndAllSame || info.maxSeqlenQ <= 0 || info.maxSeqlenKv <= 0)) {
        trace.swizzleReject = "tnd-not-uniform";
        return;
    }
    if (info.layout != LAYOUT_NOT_TND && info.layout != LAYOUT_TND) {
        trace.swizzleReject = "unsupported-layout";
        return;
    }
    if (info.l2CacheSize == 0) {
        trace.swizzleReject = "unknown-l2";
        return;
    }
    const int64_t m = (info.s1 + CUBE_BASE - 1) / CUBE_BASE;
    const int64_t n = (info.s2 + CUBE_BASE - 1) / CUBE_BASE;
    if (m <= 0 || n <= 0 || info.b <= 0 || info.n2 <= 0 || info.g <= 0 || info.aicNum == 0) {
        trace.swizzleReject = "degenerate-grid";
        return;
    }
    const ValidColumns columns = BuildValidColumns(info, m, n);
    schedule.normalValidPerHead = columns.prefix.back();
    int64_t total = schedule.normalValidPerHead;
    if (!ProductFits(total, info.b) || !ProductFits(total * info.b, info.n2) ||
        !ProductFits(total * info.b * info.n2, info.g)) {
        trace.swizzleReject = "grid-overflow";
        return;
    }
    total *= info.b * info.n2 * info.g;
    if (total == 0) {
        trace.swizzleReject = "empty-effective-grid";
        return;
    }
    const int64_t cores = static_cast<int64_t>(info.aicNum);
    schedule.normalMaxValidBlocks = total / cores + (total % cores != 0);
    schedule.normalBlockOuter = (total - 1) / schedule.normalMaxValidBlocks + 1;
    const int64_t lengthLimit = info.s1 < 16384 ? 6 : (info.s1 / 16384) * 8;
    schedule.normalContinuousBlocks = std::min(schedule.normalMaxValidBlocks, lengthLimit);
    // ASC SetSparseParams also treats a full-window BAND as dense after token
    // correction. Keep this local to normal swizzle; do not alter other routes.
    const bool normalDense = info.sparseType == FAG_SPARSE_TYPE_DENSE ||
                             (info.maskMode == MASK_MODE_WINDOW && info.s1Token >= info.s1 && info.s2Token >= info.s2);
    // Equal-length TND only uses the ordinary mapping; storage/LSE stay TND.
    if (!info.isTnd && normalDense && info.s1 >= 32768 && info.s2 >= 32768) {
        schedule.normalSwizzleKind = 1;
    } else if (!info.isTnd && info.sparseType == FAG_SPARSE_TYPE_CASUAL && info.s1 == info.s2 && info.s1 >= 32768) {
        schedule.normalSwizzleKind = 2;
    }
    if (schedule.usedCube != info.aicNum || schedule.normalBlockOuter != cores) {
        trace.swizzleReject = "partial-effective-chip";
        return;
    }
    // ASC's large-invalid-block exception is LEFT_UP_CAUSAL only. This API
    // exposes RIGHT_DOWN_CAUSAL and BAND; neither inherits that exception.
    if (!CheckExceedL2(info, columns, m, n, schedule)) {
        trace.swizzleReject = "working-set-fits";
        return;
    }
    schedule.swizzle = 1;
    trace.swizzleReject = "accepted";
}
} // namespace optiling
