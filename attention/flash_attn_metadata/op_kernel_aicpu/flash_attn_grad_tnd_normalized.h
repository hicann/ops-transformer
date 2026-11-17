/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLASH_ATTN_GRAD_TND_NORMALIZED_H_
#define FLASH_ATTN_GRAD_TND_NORMALIZED_H_
#include "../../flash_attn_grad/op_host/plan/deterministic/fixed/planner.h"
#include <array>
#include <vector>
namespace aicpu {
namespace fag_pto_normalized {
constexpr int64_t MAGIC = 0x464e5032;
constexpr int64_t HEADER_I64 = 16;
constexpr int64_t PLAN_I64 = 91;
constexpr int64_t TOTAL_I64 = HEADER_I64 + 2 * PLAN_I64;
constexpr int64_t WORDS = 2 * TOTAL_I64;
using Wire = std::array<int64_t, TOTAL_I64>;
inline void SavePlan(Wire& wire, int64_t slot, const optiling::FagDeterPlan& plan)
{
    int64_t off = HEADER_I64 + slot * PLAN_I64;
    auto put = [&](int64_t value) { wire[off++] = value; };
    put(plan.family);
    put(plan.sparseKind);
    put(plan.maxRound);
    put(plan.denseRound);
    put(plan.causalRoundsPerGroup);
    put(plan.bandMode);
    put(plan.m);
    put(plan.n);
    put(plan.p);
    put(plan.q);
    put(plan.rowOffset);
    put(plan.colOffset);
    const auto& band = plan.band;
    put(band.phaseCount);
    put(band.tailPhaseCount);
    put(band.k);
    put(band.b);
    put(band.g);
    put(band.groupsPerChunk);
    put(band.fullRounds);
    put(band.tailRounds);
    put(band.totalRounds);
    for (int side = 0; side < 2; ++side) {
        const auto* phases = side == 0 ? band.phases : band.tailPhases;
        for (int i = 0; i < 5; ++i) {
            const auto& p = phases[i];
            put(p.kind);
            put(p.m);
            put(p.n);
            put(p.rowOffset);
            put(p.colOffset);
            put(p.fullRounds);
            put(p.tailRounds);
        }
    }
}
inline optiling::FagDeterPlan BuildFixed(const optiling::FagParsedInfo& info, const optiling::FagSchedulePlan& schedule)
{
    int64_t swizzle = 0;
    return optiling::fag_pto_fixed::BuildFixedCandidate(info, schedule, swizzle, false);
}
inline Wire Build(const std::vector<int64_t>& q, const std::vector<int64_t>& kv, int64_t n2, int64_t g, int64_t d,
                  int64_t dv, int64_t cores, int64_t mask, int64_t left, int64_t right)
{
    Wire wire{};
    wire[0] = MAGIC;
    wire[1] = 1;
    wire[2] = WORDS;
    if (q.empty() || q.size() != kv.size()) {
        wire[3] = 2;
        return wire;
    }
    for (size_t i = 1; i < q.size(); ++i) {
        if (q[i] != q[0] || kv[i] != kv[0]) {
            return wire;
        }
    }
    wire[3] = 2;
    if (q[0] <= 0 || kv[0] <= 0) {
        return wire;
    }
    optiling::FagParsedInfo info;
    info.b = q.size();
    info.n2 = n2;
    info.g = g;
    info.s1 = q[0];
    info.s2 = kv[0];
    info.d = d;
    info.dv = dv;
    info.aicNum = cores;
    info.aivNum = cores * 2;
    info.maskMode = mask;
    info.isDeterministic = true;
    info.s1Token = (mask == 3 || left < 0 ? INT32_MAX : left) + info.s1 - info.s2;
    info.s2Token = (mask == 3 ? 0 : (right < 0 ? INT32_MAX : right)) - info.s1 + info.s2;
    if (mask == 0) {
        info.s1Token = INT32_MAX;
        info.s2Token = INT32_MAX;
    }
    optiling::FagSchedulePlan shape;
    shape.s1Outer = (info.s1 + 127) / 128;
    shape.s2Outer = (info.s2 + 127) / 128;
    const auto footprint = optiling::fag_pto_fixed::CountFixedFootprint(info, shape.s1Outer, shape.s2Outer);
    const __int128 tiles = static_cast<__int128>(footprint.validTiles) * info.b * n2 * g;
    const __int128 bytes = static_cast<__int128>(footprint.queryRows) * info.b * n2 * g * 128 * d * 10 +
                           static_cast<__int128>(footprint.kvColumns) * info.b * n2 * 128 * d * 12 +
                           (mask == 0 ? 0 : 2048 * 2048);
    if (tiles <= 0 || tiles > INT64_MAX / 64 || bytes > INT64_MAX) {
        return wire;
    }
    info.l2CacheSize = INT64_MAX;
    const auto baseline = BuildFixed(info, shape);
    info.l2CacheSize = 0;
    const auto candidate = BuildFixed(info, shape);
    if (baseline.maxRound <= 0 || candidate.maxRound <= 0 || baseline.maxRound >= UINT32_MAX ||
        candidate.maxRound >= UINT32_MAX) {
        return wire;
    }
    wire[3] = 1;
    wire[4] = info.s1;
    wire[5] = info.s2;
    wire[6] = static_cast<int64_t>(bytes);
    wire[7] = optiling::fag_pto_fixed::FixedDenseBlockOuter(static_cast<int64_t>(tiles), cores);
    wire[8] = info.b;
    wire[9] = n2;
    wire[10] = g;
    wire[11] = cores;
    wire[12] = mask;
    wire[13] = info.s1Token;
    wire[14] = info.s2Token;
    SavePlan(wire, 0, baseline);
    SavePlan(wire, 1, candidate);
    return wire;
}
} // namespace fag_pto_normalized
} // namespace aicpu
#endif
