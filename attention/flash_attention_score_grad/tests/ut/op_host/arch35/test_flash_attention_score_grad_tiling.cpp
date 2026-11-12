/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "tiling/platform/platform_ascendc.h"
#include "../../../../common/include/op_host/tiling_base.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/flash_attention_score_grad_tiling_common_regbase.h"
#include "../../../../op_kernel/arch35/deter_gqa_band_common.h"

using namespace std;

std::string A5SocInfo = "{\n"
                        "  \"hardware_info\": {\n"
                        "    \"BT_SIZE\": 0,\n"
                        "    \"load3d_constraints\": \"1\",\n"
                        "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                        "    \"Intrinsic_data_move_l12ub\": true,\n"
                        "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                        "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                        "    \"UB_SIZE\": 262144,\n"
                        "    \"L2_SIZE\": 134217728,\n"
                        "    \"L1_SIZE\": 524288,\n"
                        "    \"L0A_SIZE\": 65536,\n"
                        "    \"L0B_SIZE\": 65536,\n"
                        "    \"L0C_SIZE\": 262144,\n"
                        "    \"CORE_NUM\": 32,\n"
                        "    \"socVersion\": \"Ascend950\"\n"
                        "  }\n"
                        "}";

// SelectDeterBandSchedule requires coreNum == aicNum * 2. Default A5SocInfo only sets
// CORE_NUM, so the faker maps both AIC/AIV to 32. cube/vector counts make AIV=2*AIC.
std::string A5SocInfoHybrid = "{\n"
                              "  \"hardware_info\": {\n"
                              "    \"BT_SIZE\": 0,\n"
                              "    \"load3d_constraints\": \"1\",\n"
                              "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                              "    \"Intrinsic_data_move_l12ub\": true,\n"
                              "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                              "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                              "    \"UB_SIZE\": 262144,\n"
                              "    \"L2_SIZE\": 134217728,\n"
                              "    \"L1_SIZE\": 524288,\n"
                              "    \"L0A_SIZE\": 65536,\n"
                              "    \"L0B_SIZE\": 65536,\n"
                              "    \"L0C_SIZE\": 262144,\n"
                              "    \"CORE_NUM\": 32,\n"
                              "    \"cube_core_cnt\": 32,\n"
                              "    \"vector_core_cnt\": 64,\n"
                              "    \"socVersion\": \"Ascend950\"\n"
                              "  }\n"
                              "}";

class FlashAttentionScoreGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreGradTiling TearDown" << std::endl;
    }
};

TEST_F(FlashAttentionScoreGradTiling, DeterGqaCausalGroupedPreservesTilingLayout)
{
    using namespace optiling::fag;
    // All group fields occupy former padding; unrelated serialized tiling fields must not move.
    static_assert(sizeof(FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase) == 200,
                  "Base tiling size must remain unchanged");
    static_assert(offsetof(FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase, reservedGqaCausal) == 193,
                  "Reserved CAUSAL byte must retain its original offset");
    static_assert(offsetof(FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase, deterGqaCausalGroups) == 195,
                  "GQA group count must occupy the final former padding byte");
    static_assert(sizeof(BaseDeterParamRegbase) == 32 + 2 * MAX_CORE_NUM * sizeof(uint64_t),
                  "Deterministic tiling size must remain unchanged");
    static_assert(offsetof(BaseDeterParamRegbase, deterGqaCausalRoundsPerGroup) == sizeof(uint32_t),
                  "GQA rounds must occupy the former reserved uint32 field");
    static_assert(offsetof(BaseDeterParamRegbase, deterMaxRound) == sizeof(uint64_t),
                  "Maximum round offset must remain unchanged");
    static_assert(offsetof(BaseDeterParamRegbase, dqIsNeedDeter) == 32,
                  "Synchronization array offset must remain unchanged");

    FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase baseParams{};
    BaseDeterParamRegbase deterParams{};
    EXPECT_EQ(baseParams.reservedGqaCausal, 0);
    EXPECT_EQ(deterParams.get_deterGqaCausalRoundsPerGroup(), 0);
    EXPECT_EQ(baseParams.get_deterGqaCausalGroups(), 0);
    deterParams.set_deterGqaCausalRoundsPerGroup(1040);
    baseParams.set_deterGqaCausalGroups(2);
    EXPECT_EQ(baseParams.reservedGqaCausal, 0);
    EXPECT_EQ(deterParams.get_deterGqaCausalRoundsPerGroup(), 1040);
    EXPECT_EQ(baseParams.get_deterGqaCausalGroups(), 2);
}

// Complete-domain owner oracle adapter. Keep every serialized owner kind explicit;
// an unrecognized tag must remain an invalid slot instead of silently using dense indexing.
static void MapGqaBandOwnerScheduleForTest(const optiling::fag::GqaBandSchedule& plan, int64_t core, int64_t round,
                                           int64_t& w, int64_t& x, int64_t& y)
{
    using namespace optiling::fag;
    w = -1;
    const int64_t h = plan.groupsPerChunk;
    if (h <= 0 || plan.fullRounds <= 0 || plan.phaseCount != 1 || plan.tailPhaseCount != 0 || core < 1 ||
        core > plan.k || round < 1 || round > plan.totalRounds) {
        return;
    }
    const int64_t fullChunks = plan.b / h;
    const int64_t fullEnd = fullChunks * plan.fullRounds;
    const bool tail = round > fullEnd;
    const int64_t chunk = tail ? fullChunks : (round - 1) / plan.fullRounds;
    const int64_t groups = tail ? plan.b % h : h;
    const int64_t local = tail ? round - fullEnd : round - chunk * plan.fullRounds;
    const auto& phase = plan.phases[0];
    switch (phase.kind) {
        case GqaBandPhaseKind::BAND_COLUMN_OWNER:
        case GqaBandPhaseKind::BAND_BALANCED_COLUMN_OWNER:
            GqaBandColumnOwnerIndex(plan.k, groups, plan.g, phase.m, phase.n, phase.rowOffset, phase.colOffset, core,
                                    local, w, x, y, phase.kind == GqaBandPhaseKind::BAND_BALANCED_COLUMN_OWNER);
            break;
        case GqaBandPhaseKind::LOWER_TRIANGLE_COLUMN_OWNER:
            GqaBandColumnOwnerIndex(plan.k, groups, plan.g, phase.m, phase.n, phase.m, 1, core, local, w, x, y);
            if (w > 0) {
                x += phase.rowOffset;
                y += phase.colOffset;
            }
            break;
        case GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER:
            GqaBandFoldedColumnOwnerIndex(plan.k, groups, plan.g, phase.m, phase.n, phase.rowOffset, phase.colOffset,
                                          core, local, w, x, y);
            break;
        case GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER:
            GqaBandFoldedColumnOwnerIndex(plan.k, groups, plan.g, phase.m, phase.n, 1, phase.rowOffset, core, local, w,
                                          x, y);
            if (w > 0) {
                y += phase.colOffset;
            }
            break;
        case GqaBandPhaseKind::FOLDED_LOWER_COLUMN_OWNER:
            GqaBandFoldedColumnOwnerIndex(plan.k, groups, plan.g, phase.m, phase.n, phase.colOffset, 1, core, local, w,
                                          x, y);
            if (w > 0) {
                x += phase.rowOffset;
            }
            break;
        default:
            return;
    }
    if (w > 0) {
        w += chunk * h * plan.g;
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandDiagonalColumnOwnerCoverageAndConflicts)
{
    using namespace optiling::fag;
    constexpr int64_t k = 32;
    for (int64_t groups : {1, 2, 3, 33}) {
        for (int64_t size : {1, 7, 31, 32, 33}) {
            for (int64_t g : {3, 7, 8}) {
                GqaBandPlan plan{};
                AppendGqaBandPhase(plan, GqaBandPhaseKind::DIAGONAL_COLUMN_OWNER, size, size, 0, 0);
                SetGqaBandGroups(k, groups, g, groups, plan);
                EXPECT_EQ(GqaBandKvRunUpperBound(plan), groups * size);
                std::vector<int64_t> seen(groups * g * size, 0);
                std::vector<int64_t> owners(groups * size, -1);
                for (int64_t r = 1; r <= plan.totalRounds; ++r) {
                    std::vector<bool> qSeen(groups * g * size, false);
                    std::vector<bool> kvSeen(groups * size, false);
                    for (int64_t j = 1; j <= k; ++j) {
                        int64_t w = -1, x = 0, y = 0;
                        GqaBandDiagonalColumnIndex(k, size, groups, g, j, r, w, x, y);
                        if (w < 0) {
                            continue;
                        }
                        ASSERT_GE(w, 1);
                        ASSERT_LE(w, groups * g);
                        ASSERT_EQ(x, y);
                        ASSERT_GE(y, 1);
                        ASSERT_LE(y, size);
                        const int64_t q = (w - 1) * size + x - 1;
                        const int64_t kv = (w - 1) / g * size + y - 1;
                        EXPECT_FALSE(qSeen[q]);
                        EXPECT_FALSE(kvSeen[kv]);
                        qSeen[q] = kvSeen[kv] = true;
                        ++seen[q];
                        if (owners[kv] < 0) {
                            owners[kv] = j;
                        }
                        EXPECT_EQ(owners[kv], j);
                    }
                }
                for (int64_t visits : seen) {
                    EXPECT_EQ(visits, 1);
                }
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandDiagonalColumnOwnerDrainAndIdleSlots)
{
    using namespace optiling::fag;
    const int64_t cases[][6] = {
        {32, 33, 1, 7, 0, 1}, {32, 33, 1, 7, 33, 1}, {32, 33, 1, 7, 1, 0}, {32, 33, 1, 7, 1, 15}, {32, 33, 1, 7, 2, 8},
        {0, 33, 1, 7, 1, 1},  {32, 0, 1, 7, 1, 1},   {32, 33, 0, 7, 1, 1}, {32, 33, 1, 1, 1, 1},
    };
    for (const auto& c : cases) {
        int64_t w = 99, x = 0, y = 0;
        GqaBandDiagonalColumnIndex(c[0], c[1], c[2], c[3], c[4], c[5], w, x, y);
        EXPECT_EQ(w, -1);
    }
}

// Independent enumeration uses the band inequalities, rather than the
// mapper's prefix formula. Include non-power-of-two heads/cores and tail columns.
TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandColumnOwnerProperties)
{
    using namespace optiling::fag;
    uint32_t seed = 1729;
    const auto next = [&seed]() -> uint32_t {
        seed = seed * 1664525U + 1013904223U;
        return seed;
    };
    for (int trial = 0; trial < 2048; ++trial) {
        const int64_t k = 1 + next() % 36;
        const int64_t b = 1 + next() % k;
        const int64_t g = 2 + next() % 17;
        const int64_t m = 1 + next() % 47;
        const int64_t n = 1 + next() % 51;
        const int64_t p = 1 + next() % m;
        const int64_t q = 1 + next() % n;
        SCOPED_TRACE(::testing::Message() << k << "," << b << "," << g << "," << m << "," << n << "," << p << "," << q);
        const int64_t lanes = GqaBandColumnOwnerLanes(k, b, g);
        ASSERT_GT(lanes, 0);
        const int64_t rounds = GqaBandColumnOwnerRounds(k, b, g, m, n, p, q);
        std::vector<uint8_t> seen(b * g * m * n, 0);
        std::vector<int64_t> owners(b * n, -1);
        std::vector<int64_t> previous(k, -1);
        std::vector<uint8_t> closed(b * n, 0);
        int64_t lastActive = 0;
        for (int64_t round = 1; round <= rounds + 1; ++round) {
            std::vector<uint8_t> dq(b * g * m, 0), dkv(b * n, 0);
            for (int64_t core = 1; core <= k; ++core) {
                int64_t w = 99, x = 0, y = 0;
                GqaBandColumnOwnerIndex(k, b, g, m, n, p, q, core, round, w, x, y);
                if (w < 0) {
                    continue;
                }
                ASSERT_LE(round, rounds);
                ASSERT_GE(w, 1);
                ASSERT_LE(w, b * g);
                ASSERT_GE(x, 1);
                ASSERT_LE(x, m);
                ASSERT_GE(y, 1);
                ASSERT_LE(y, n);
                ASSERT_LE(x - y, p - 1);
                ASSERT_LE(y - x, q - 1);
                const int64_t query = (w - 1) * m + x - 1;
                const int64_t kv = (w - 1) / g * n + y - 1;
                const int64_t task = query * n + y - 1;
                ASSERT_EQ(dq[query]++, 0);
                ASSERT_EQ(dkv[kv]++, 0);
                ASSERT_EQ(seen[task]++, 0);
                if (owners[kv] < 0) {
                    owners[kv] = core;
                }
                ASSERT_EQ(owners[kv], core);
                if (previous[core - 1] != kv) {
                    if (previous[core - 1] >= 0) {
                        closed[previous[core - 1]] = 1;
                    }
                    ASSERT_EQ(closed[kv], 0);
                    previous[core - 1] = kv;
                }
                lastActive = round;
            }
        }
        EXPECT_EQ(lastActive, rounds);
        for (int64_t w = 1; w <= b * g; ++w) {
            for (int64_t x = 1; x <= m; ++x) {
                for (int64_t y = 1; y <= n; ++y) {
                    EXPECT_EQ(seen[((w - 1) * m + x - 1) * n + y - 1], x - y < p && y - x < q ? 1 : 0);
                }
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandColumnOwnerPrefixOracle)
{
    using namespace optiling::fag;
    for (int64_t lanes : {1, 3, 5, 8, 13}) {
        for (int64_t m : {1, 7, 31, 64, 129}) {
            for (int64_t n : {1, 9, 32, 65, 130}) {
                for (int64_t p : {int64_t(1), (m + 1) / 2, m}) {
                    for (int64_t q : {int64_t(1), (n + 1) / 2, n}) {
                        for (int64_t lane = 0; lane < lanes; ++lane) {
                            int64_t expected = 0;
                            for (int64_t pair = 0; pair <= (n + 2 * lanes - 1) / (2 * lanes); ++pair) {
                                ASSERT_EQ(GqaBandColumnOwnerPairPrefix(m, n, p, q, lanes, lane, pair), expected);
                                for (int64_t col : {2 * lanes * pair + lane + 1, 2 * lanes * pair + 2 * lanes - lane}) {
                                    if (col > n) {
                                        continue;
                                    }
                                    for (int64_t row = 1; row <= m; ++row) {
                                        expected += row - col < p && col - row < q ? 1 : 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandColumnOwnerInvalidAndDrain)
{
    using namespace optiling::fag;
    const int64_t invalid[][7] = {{0, 1, 3, 5, 7, 2, 3}, {7, 0, 3, 5, 7, 2, 3}, {7, 8, 3, 5, 7, 2, 3},
                                  {7, 1, 1, 5, 7, 2, 3}, {7, 1, 3, 0, 7, 2, 3}, {7, 1, 3, 5, 0, 2, 3},
                                  {7, 1, 3, 5, 7, 0, 3}, {7, 1, 3, 5, 7, 2, 0}, {7, 1, 3, 5, 7, 6, 3},
                                  {7, 1, 3, 5, 7, 2, 8}};
    for (const auto& v : invalid) {
        int64_t w = 1, x = 0, y = 0;
        EXPECT_EQ(GqaBandColumnOwnerRounds(v[0], v[1], v[2], v[3], v[4], v[5], v[6]), 0);
        GqaBandColumnOwnerIndex(v[0], v[1], v[2], v[3], v[4], v[5], v[6], 1, 1, w, x, y);
        EXPECT_EQ(w, -1);
    }
    for (int64_t core : {0, 1, 7, 8}) {
        for (int64_t round : {0, 1000000}) {
            int64_t w = 1, x = 0, y = 0;
            GqaBandColumnOwnerIndex(7, 2, 5, 19, 23, 7, 3, core, round, w, x, y);
            EXPECT_EQ(w, -1);
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandBalancedColumnPrefixIndependentOracle)
{
    using namespace optiling::fag;
    uint32_t seed = 2917;
    const auto next = [&seed]() -> uint32_t {
        seed = seed * 1664525U + 1013904223U;
        return seed >> 8;
    };
    for (int trial = 0; trial < 4096; ++trial) {
        const int64_t m = 1 + next() % 53, n = 1 + next() % 67;
        const int64_t p = 1 + next() % m, q = 1 + next() % n, lanes = 1 + next() % 32;
        std::vector<int64_t> heights;
        for (int64_t y = 1; y <= n; ++y) {
            const int64_t height = std::max<int64_t>(0, std::min(m, y + p - 1) - std::max<int64_t>(1, y - q + 1) + 1);
            if (height > 0) {
                heights.push_back(height);
            }
        }
        std::sort(heights.begin(), heights.end(), [](int64_t a, int64_t b) { return a > b; });
        ASSERT_EQ(GqaBandBalancedColumnCount(m, n, q), heights.size());
        std::vector<uint8_t> seen(n, 0);
        for (int64_t rank = 0; rank < static_cast<int64_t>(heights.size()); ++rank) {
            const int64_t y = GqaBandBalancedPhysicalColumn(m, n, p, q, rank);
            ASSERT_GE(y, 1);
            ASSERT_LE(y, n);
            ASSERT_EQ(seen[y - 1]++, 0);
            ASSERT_EQ(heights[rank], std::min(m, y + p - 1) - std::max<int64_t>(1, y - q + 1) + 1);
        }
        for (int64_t lane = 0; lane < lanes; ++lane) {
            int64_t expected = 0;
            for (int64_t pairs = 0; pairs <= (n + 2 * lanes - 1) / (2 * lanes) + 1; ++pairs) {
                ASSERT_EQ(GqaBandBalancedPairPrefix(m, n, p, q, lanes, lane, pairs), expected);
                for (int64_t rank : {2 * lanes * pairs + lane, 2 * lanes * pairs + 2 * lanes - 1 - lane}) {
                    if (rank < static_cast<int64_t>(heights.size())) {
                        expected += heights[rank];
                    }
                }
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandOwnerChunksCoverageAndConflicts)
{
    using namespace optiling::fag;
    uint32_t seed = 589;
    const auto next = [&seed]() -> uint32_t {
        seed = seed * 1664525U + 1013904223U;
        return seed >> 8;
    };
    for (int trial = 0; trial < 512; ++trial) {
        const int64_t k = 1 + next() % 36, b = 1 + next() % (2 * k + 3), g = 2 + next() % 11;
        const int64_t m = 1 + next() % 19, n = 1 + next() % 23, p = 1 + next() % m, q = 1 + next() % n;
        const int64_t h = 1 + next() % std::min(k, b);
        const bool balanced = trial % 2 != 0;
        GqaBandPlan plan{};
        AppendGqaBandPhase(
            plan, balanced ? GqaBandPhaseKind::BAND_BALANCED_COLUMN_OWNER : GqaBandPhaseKind::BAND_COLUMN_OWNER, m, n,
            p, q);
        SetGqaBandGroups(k, b, g, h, plan);
        GqaBandSchedule schedule{};
        SaveGqaBandSchedule(plan, schedule);
        EXPECT_EQ(schedule.totalRounds, (b / h) * schedule.fullRounds + schedule.tailRounds);
        std::vector<uint8_t> seen(b * g * m * n, 0);
        std::vector<int64_t> owners(b * n, -1), previous(k, -1);
        std::vector<uint8_t> closed(b * n, 0);
        int64_t lastActive = 0;
        for (int64_t r = 1; r <= schedule.totalRounds + 1; ++r) {
            std::vector<uint8_t> dq(b * g * m, 0), dkv(b * n, 0);
            const int64_t full = b / h * schedule.fullRounds;
            const bool tail = r > full;
            const int64_t chunk = tail ? b / h : (r - 1) / schedule.fullRounds;
            const int64_t groups = tail ? b % h : h;
            const int64_t local = tail ? r - full : (r - 1) % schedule.fullRounds + 1;
            for (int64_t core = 1; core <= k; ++core) {
                int64_t w = -1, x = 0, y = 0;
                GqaBandColumnOwnerIndex(k, groups, g, m, n, p, q, core, local, w, x, y, balanced);
                if (w < 0) {
                    continue;
                }
                ASSERT_LE(r, schedule.totalRounds);
                w += chunk * h * g;
                ASSERT_GE(w, 1);
                ASSERT_LE(w, b * g);
                ASSERT_GE(x, 1);
                ASSERT_LE(x, m);
                ASSERT_GE(y, 1);
                ASSERT_LE(y, n);
                const int64_t query = (w - 1) * m + x - 1, kv = (w - 1) / g * n + y - 1;
                ASSERT_EQ(dq[query]++, 0);
                ASSERT_EQ(dkv[kv]++, 0);
                ASSERT_EQ(seen[query * n + y - 1]++, 0);
                if (owners[kv] < 0) {
                    owners[kv] = core;
                }
                ASSERT_EQ(owners[kv], core);
                if (previous[core - 1] != kv) {
                    if (previous[core - 1] >= 0) {
                        closed[previous[core - 1]] = 1;
                    }
                    ASSERT_EQ(closed[kv], 0);
                    previous[core - 1] = kv;
                }
                lastActive = r;
            }
        }
        EXPECT_EQ(lastActive, schedule.totalRounds);
        for (int64_t w = 1; w <= b * g; ++w) {
            for (int64_t x = 1; x <= m; ++x) {
                for (int64_t y = 1; y <= n; ++y) {
                    EXPECT_EQ(seen[((w - 1) * m + x - 1) * n + y - 1], x - y < p && y - x < q ? 1 : 0);
                }
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandFoldedOwnerIndependentCoverageAndConflicts)
{
    using namespace optiling::fag;
    const int64_t geometries[][2] = {{7, 11}, {17, 9}, {31, 31}, {33, 35}, {37, 29}, {5, 19}};
    for (int64_t k : {32, 36}) {
        for (int64_t b : std::vector<int64_t>{1, 3, k + 1}) {
            for (int64_t g : {2, 4, 7}) {
                for (const auto& geometry : geometries) {
                    const int64_t m = geometry[0], n = geometry[1];
                    const int64_t bands[][2] = {
                        {1, 1}, {1, n}, {m, 1}, {std::min<int64_t>(m, 4), std::min<int64_t>(n, 5)}};
                    for (const auto& band : bands) {
                        const int64_t p = band[0], q = band[1];
                        SCOPED_TRACE(::testing::Message()
                                     << k << "," << b << "," << g << "," << m << "," << n << "," << p << "," << q);
                        const int64_t columns = b * std::min(m, n);
                        const int64_t expectedRounds = ((columns + k - 1) / k) * (p + q - 1) * g;
                        ASSERT_EQ(GqaBandFoldedColumnOwnerRounds(k, b, g, m, n, p, q), expectedRounds);
                        std::vector<uint8_t> seen(b * g * m * n, 0), closed(b * n, 0);
                        std::vector<int64_t> owners(b * n, -1), previous(k, -1);
                        for (int64_t round = 1; round <= expectedRounds + 1; ++round) {
                            std::vector<uint8_t> dq(b * g * m, 0), dkv(b * n, 0);
                            for (int64_t core = 1; core <= k; ++core) {
                                int64_t w = 0, x = 0, y = 0;
                                GqaBandFoldedColumnOwnerIndex(k, b, g, m, n, p, q, core, round, w, x, y);
                                if (round > expectedRounds) {
                                    ASSERT_EQ(w, -1);
                                }
                                if (w < 0) {
                                    continue;
                                }
                                ASSERT_GE(w, 1);
                                ASSERT_LE(w, b * g);
                                ASSERT_GE(x, 1);
                                ASSERT_LE(x, m);
                                ASSERT_GE(y, 1);
                                ASSERT_LE(y, n);
                                ASSERT_LT(x - y, p);
                                ASSERT_LT(y - x, q);
                                const int64_t query = (w - 1) * m + x - 1, kv = (w - 1) / g * n + y - 1;
                                ASSERT_EQ(dq[query]++, 0);
                                ASSERT_EQ(dkv[kv]++, 0);
                                ASSERT_EQ(seen[query * n + y - 1]++, 0);
                                if (owners[kv] < 0) {
                                    owners[kv] = core;
                                }
                                ASSERT_EQ(owners[kv], core);
                                if (previous[core - 1] != kv) {
                                    if (previous[core - 1] >= 0) {
                                        closed[previous[core - 1]] = 1;
                                    }
                                    ASSERT_EQ(closed[kv], 0);
                                    previous[core - 1] = kv;
                                }
                                if ((round - 1) % g != 0) {
                                    int64_t pw = 0, px = 0, py = 0;
                                    GqaBandFoldedColumnOwnerIndex(k, b, g, m, n, p, q, core, round - 1, pw, px, py);
                                    ASSERT_EQ(w, pw + 1);
                                    ASSERT_EQ(x, px);
                                    ASSERT_EQ(y, py);
                                }
                            }
                        }
                        // Independent dense-domain enumeration: no virtual-column or modulo formula is reused.
                        for (int64_t w = 1; w <= b * g; ++w) {
                            for (int64_t x = 1; x <= m; ++x) {
                                for (int64_t y = 1; y <= n; ++y) {
                                    ASSERT_EQ(seen[((w - 1) * m + x - 1) * n + y - 1], x - y < p && y - x < q ? 1 : 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandFoldedOwnerInvalidAndDrain)
{
    using namespace optiling::fag;
    const int64_t invalid[][7] = {{0, 3, 2, 17, 19, 1, 4},   {32, 0, 2, 17, 19, 1, 4}, {32, 3, 1, 17, 19, 1, 4},
                                  {32, 3, 2, 0, 19, 1, 4},   {32, 3, 2, 17, 0, 1, 4},  {32, 3, 2, 17, 19, 0, 4},
                                  {32, 3, 2, 17, 19, 18, 4}, {32, 3, 2, 17, 19, 1, 0}, {32, 3, 2, 17, 19, 1, 20}};
    for (const auto& v : invalid) {
        int64_t w = 42, x = 0, y = 0;
        EXPECT_EQ(GqaBandFoldedColumnOwnerRounds(v[0], v[1], v[2], v[3], v[4], v[5], v[6]), 0);
        GqaBandFoldedColumnOwnerIndex(v[0], v[1], v[2], v[3], v[4], v[5], v[6], 1, 1, w, x, y);
        EXPECT_EQ(w, -1);
    }
    for (int64_t core : {0, 1, 32, 33}) {
        for (int64_t round : {0, 1, 28, 29}) {
            int64_t w = 42, x = 0, y = 0;
            GqaBandFoldedColumnOwnerIndex(32, 3, 7, 5, 9, 1, 4, core, round, w, x, y);
            if (core < 1 || core > 32 || round < 1 || round > 28 || core > 15) {
                EXPECT_EQ(w, -1);
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandFoldedOwnerChunksOffsetsAndKvRuns)
{
    using namespace optiling::fag;
    static_assert(static_cast<int64_t>(GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER) == 13);
    static_assert(static_cast<int64_t>(GqaBandPhaseKind::FOLDED_LOWER_COLUMN_OWNER) == 14);
    static_assert(GQA_BAND_FOLDED_OWNER_TEMPLATE == 8);
    for (int64_t k : {32, 36}) {
        for (int64_t b : {4, 7, 37}) {
            for (int64_t h : std::vector<int64_t>{1, 3, b}) {
                for (int64_t g : {2, 4, 7}) {
                    for (bool upper : {false, true}) {
                        const int64_t m = k - 3, n = k + 5, p = upper ? 1 : 5, q = upper ? 7 : 1;
                        const int64_t offx = upper ? 0 : 3, offy = upper ? 5 : 0;
                        GqaBandPlan candidate{};
                        AppendGqaBandPhase(candidate,
                                           upper ? GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER :
                                                   GqaBandPhaseKind::FOLDED_LOWER_COLUMN_OWNER,
                                           m, n, upper ? q : offx, upper ? offy : p);
                        SetGqaBandGroups(k, b, g, h, candidate);
                        GqaBandSchedule plan{};
                        SaveGqaBandSchedule(candidate, plan);
                        EXPECT_EQ(plan.fullRounds, GqaBandFoldedColumnOwnerRounds(k, h, g, m, n, p, q));
                        EXPECT_EQ(plan.tailRounds,
                                  b % h == 0 ? 0 : GqaBandFoldedColumnOwnerRounds(k, b % h, g, m, n, p, q));
                        EXPECT_EQ(GqaBandKvRunUpperBound(candidate), b * std::min(n, m + q - 1));
                        std::vector<uint8_t> seen(b * g * m * n, 0), closed(b * n, 0);
                        std::vector<int64_t> previous(k, -1);
                        int64_t runs = 0;
                        for (int64_t round = 0; round <= plan.totalRounds + 1; ++round) {
                            std::vector<uint8_t> dq(b * g * m, 0), dkv(b * n, 0);
                            for (int64_t core = 0; core <= k + 1; ++core) {
                                int64_t w = 0, x = 0, y = 0;
                                MapGqaBandOwnerScheduleForTest(plan, core, round, w, x, y);
                                if (core < 1 || core > k || round < 1 || round > plan.totalRounds) {
                                    ASSERT_EQ(w, -1);
                                    continue;
                                }
                                if (w < 0) {
                                    continue;
                                }
                                x -= offx;
                                y -= offy;
                                ASSERT_GE(w, 1);
                                ASSERT_LE(w, b * g);
                                ASSERT_GE(x, 1);
                                ASSERT_LE(x, m);
                                ASSERT_GE(y, 1);
                                ASSERT_LE(y, n);
                                const int64_t query = (w - 1) * m + x - 1, kv = (w - 1) / g * n + y - 1;
                                ASSERT_EQ(dq[query]++, 0);
                                ASSERT_EQ(dkv[kv]++, 0);
                                ASSERT_EQ(seen[query * n + y - 1]++, 0);
                                if (previous[core - 1] != kv) {
                                    if (previous[core - 1] >= 0) {
                                        closed[previous[core - 1]] = 1;
                                    }
                                    ASSERT_EQ(closed[kv], 0);
                                    previous[core - 1] = kv;
                                    ++runs;
                                }
                            }
                        }
                        EXPECT_EQ(runs, GqaBandKvRunUpperBound(candidate));
                        for (int64_t w = 1; w <= b * g; ++w) {
                            for (int64_t x = 1; x <= m; ++x) {
                                for (int64_t y = 1; y <= n; ++y) {
                                    ASSERT_EQ(seen[((w - 1) * m + x - 1) * n + y - 1], x - y < p && y - x < q ? 1 : 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandFoldedBandChunkSafetyAndExactRuns)
{
    using namespace optiling::fag;
    static_assert(static_cast<int64_t>(GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER) == 17);
    static_assert(GQA_BAND_FOLDED_OWNER_TEMPLATE == 8);
    int64_t concurrentSameHeadRounds = 0;
    for (int64_t k : {8, 32, 36}) {
        for (int64_t m : std::vector<int64_t>{k - 1, k, k + 1}) {
            for (int64_t n : std::vector<int64_t>{m - 2, m + 3}) {
                for (int64_t b : {1, 5}) {
                    for (int64_t h : std::set<int64_t>{1, std::min<int64_t>(3, b), b}) {
                        for (int64_t g : {2, 3}) {
                            // Include narrow two-sided bands and domains wider than a row.
                            for (const auto& band : std::vector<std::pair<int64_t, int64_t>>{{2, 3}, {m, n}}) {
                                const int64_t p = band.first, q = band.second;
                                SCOPED_TRACE(::testing::Message() << k << "," << m << "," << n << "," << b << "," << h
                                                                  << "," << g << "," << p << "," << q);
                                GqaBandPlan candidate{};
                                AppendGqaBandPhase(candidate, GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER, m, n, p, q);
                                SetGqaBandGroups(k, b, g, h, candidate);
                                GqaBandSchedule plan{};
                                SaveGqaBandSchedule(candidate, plan);
                                std::vector<uint8_t> seen(b * g * m * n, 0), closed(b * n, 0);
                                std::vector<int64_t> previous(k, -1);
                                int64_t actualRuns = 0;
                                for (int64_t round = 0; round <= plan.totalRounds + 1; ++round) {
                                    std::vector<uint8_t> dq(b * g * m, 0), dkv(b * n, 0);
                                    std::vector<int64_t> sameHeadRows(b * g, 0);
                                    for (int64_t core = 0; core <= k + 1; ++core) {
                                        int64_t w = 0, x = 0, y = 0;
                                        MapGqaBandOwnerScheduleForTest(plan, core, round, w, x, y);
                                        if (core < 1 || core > k || round < 1 || round > plan.totalRounds) {
                                            ASSERT_EQ(w, -1);
                                            continue;
                                        }
                                        if (w < 0) {
                                            continue;
                                        }
                                        ASSERT_GE(w, 1);
                                        ASSERT_LE(w, b * g);
                                        ASSERT_GE(x, 1);
                                        ASSERT_LE(x, m);
                                        ASSERT_GE(y, 1);
                                        ASSERT_LE(y, n);
                                        ASSERT_LT(x - y, p);
                                        ASSERT_LT(y - x, q);
                                        const int64_t query = (w - 1) * m + x - 1;
                                        const int64_t kv = (w - 1) / g * n + y - 1;
                                        ASSERT_EQ(dq[query]++, 0);
                                        ASSERT_EQ(dkv[kv]++, 0);
                                        ASSERT_EQ(seen[query * n + y - 1]++, 0);
                                        ++sameHeadRows[w - 1];
                                        if (previous[core - 1] != kv) {
                                            if (previous[core - 1] >= 0) {
                                                closed[previous[core - 1]] = 1;
                                            }
                                            ASSERT_EQ(closed[kv], 0);
                                            previous[core - 1] = kv;
                                            ++actualRuns;
                                        }
                                    }
                                    for (int64_t rows : sameHeadRows) {
                                        if (rows > 1) {
                                            ++concurrentSameHeadRounds;
                                        }
                                    }
                                }
                                for (int64_t w = 1; w <= b * g; ++w) {
                                    for (int64_t x = 1; x <= m; ++x) {
                                        for (int64_t y = 1; y <= n; ++y) {
                                            EXPECT_EQ(seen[((w - 1) * m + x - 1) * n + y - 1],
                                                      x - y < p && y - x < q ? 1 : 0);
                                        }
                                    }
                                }
                                EXPECT_EQ(actualRuns, GqaBandKvRunUpperBound(candidate));
                                EXPECT_EQ(actualRuns, GqaBandScheduleKvRunExact(plan));
                                EXPECT_EQ(actualRuns, b * std::min(n, m + q - 1));
                            }
                        }
                    }
                }
            }
        }
    }
    // Small G does not cap core parallelism: dQ addresses are separated by physical rows.
    EXPECT_GT(concurrentSameHeadRounds, 0);
}

// Enumerate valid row/column cells explicitly in the documented lane order.
// This oracle does not use the optimized two-interval branch or prefix helpers.
static std::vector<std::pair<int64_t, int64_t>> EnumerateGqaBandOwnerLaneForTest(int64_t lanes, int64_t lane, int64_t m,
                                                                                 int64_t n, int64_t p, int64_t q)
{
    std::vector<std::pair<int64_t, int64_t>> cells;
    for (int64_t start = 0; start < n; start += 2 * lanes) {
        const int64_t columns[] = {start + lane + 1, start + 2 * lanes - lane};
        for (int64_t y : columns) {
            if (y > n) {
                continue;
            }
            for (int64_t x = 1; x <= m; ++x) {
                if (x - y < p && y - x < q) {
                    cells.emplace_back(x, y);
                }
            }
        }
    }
    return cells;
}

// Reorder an independently enumerated interval, without using the device's
// rowSlot/height arithmetic. The original list remains the owner/empty-slot oracle.
static std::vector<std::pair<int64_t, int64_t>> GqaBandOwnerRowOrderForTest(
    const std::vector<std::pair<int64_t, int64_t>>& original, int64_t lanes, int64_t m, int64_t n, int64_t p)
{
    auto ordered = original;
    if (p == m && n <= 2 * lanes && !ordered.empty()) {
        const auto firstColumnEnd = std::find_if(ordered.begin(), ordered.end(), [&ordered](const auto& cell) {
            return cell.second != ordered.front().second;
        });
        std::reverse(ordered.begin(), firstColumnEnd);
    }
    return ordered;
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandSinglePairIndependentOrder)
{
    using namespace optiling::fag;
    int64_t changedRows = 0;
    for (int64_t k : {32, 36}) {
        for (int64_t groups : {1, 3, 5, 8}) {
            for (int64_t g : {2, 4, 7, 16}) {
                const int64_t lanes = std::min(k / groups, g);
                for (int64_t m : {1, 7, 19}) {
                    for (int64_t n : std::vector<int64_t>{1, std::max<int64_t>(1, lanes - 1), lanes, lanes + 1,
                                                          2 * lanes - 1, 2 * lanes, 2 * lanes + 1}) {
                        const int64_t bands[][2] = {{1, 1}, {1, n}, {m, 1}, {m, n}};
                        for (const auto& band : bands) {
                            const int64_t p = band[0], q = band[1];
                            SCOPED_TRACE(::testing::Message() << k << "," << groups << "," << g << "," << m << "," << n
                                                              << "," << p << "," << q);
                            std::vector<std::vector<std::pair<int64_t, int64_t>>> expected, ordered;
                            int64_t maxRows = 0, validRows = 0;
                            for (int64_t lane = 0; lane < lanes; ++lane) {
                                expected.push_back(EnumerateGqaBandOwnerLaneForTest(lanes, lane, m, n, p, q));
                                ordered.push_back(GqaBandOwnerRowOrderForTest(expected.back(), lanes, m, n, p));
                                maxRows = std::max(maxRows, static_cast<int64_t>(expected.back().size()));
                                validRows += static_cast<int64_t>(expected.back().size());
                            }
                            const int64_t rounds = maxRows * g;
                            ASSERT_EQ(GqaBandColumnOwnerRounds(k, groups, g, m, n, p, q), rounds);
                            std::map<std::pair<int64_t, int64_t>, int64_t> owners;
                            std::set<std::tuple<int64_t, int64_t, int64_t>> seen;
                            for (int64_t round = 0; round <= rounds + 1; ++round) {
                                std::set<std::pair<int64_t, int64_t>> dq, dkv;
                                for (int64_t core = 0; core <= k + 1; ++core) {
                                    int64_t w = 0, x = 0, y = 0;
                                    GqaBandColumnOwnerIndex(k, groups, g, m, n, p, q, core, round, w, x, y);
                                    if (core < 1 || core > groups * lanes || core > k || round < 1) {
                                        ASSERT_EQ(w, -1);
                                        continue;
                                    }
                                    const int64_t lane = (core - 1) % lanes, group = (core - 1) / lanes;
                                    const int64_t index = (round - 1) / g;
                                    // Original interval lengths remain the independent empty-slot oracle.
                                    if (index >= static_cast<int64_t>(expected[lane].size())) {
                                        ASSERT_EQ(w, -1);
                                        continue;
                                    }
                                    ASSERT_EQ(x, ordered[lane][index].first);
                                    ASSERT_EQ(y, expected[lane][index].second);
                                    ASSERT_EQ(w, group * g + (((round - 1) % g + lane) % g) + 1);
                                    changedRows += x != expected[lane][index].first;
                                    ASSERT_TRUE(dq.emplace(w, x).second);
                                    ASSERT_TRUE(dkv.emplace(group, y).second);
                                    ASSERT_TRUE(seen.emplace(w, x, y).second);
                                    const auto owner = owners.emplace(std::make_pair(group, y), core);
                                    EXPECT_EQ(owner.first->second, core);
                                }
                            }
                            EXPECT_EQ(seen.size(), static_cast<size_t>(validRows * groups * g));
                        }
                    }
                }
            }
        }
    }
    EXPECT_GT(changedRows, 0);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandSinglePairTailAndTranslatedOrder)
{
    using namespace optiling::fag;
    int fullGeneralTailDirect = 0;
    int64_t changedTailRows = 0;
    for (int64_t k : {32, 36}) {
        for (int64_t g : {2, 4, 7, 16}) {
            const int64_t b = 12, h = 9, tail = b % h;
            const int64_t n = 2 * std::min(k / tail, g);
            for (bool translated : {false, true}) {
                const int64_t m = translated ? n : n + 7, p = translated ? m : 3,
                              q = translated ? 1 : std::min<int64_t>(n, 5);
                GqaBandPlan candidate{};
                AppendGqaBandPhase(
                    candidate,
                    translated ? GqaBandPhaseKind::LOWER_TRIANGLE_COLUMN_OWNER : GqaBandPhaseKind::BAND_COLUMN_OWNER, m,
                    n, translated ? 3 : p, translated ? 2 : q);
                SetGqaBandGroups(k, b, g, h, candidate);
                GqaBandSchedule plan{};
                SaveGqaBandSchedule(candidate, plan);
                if (n > 2 * std::min(k / h, g) && n <= 2 * std::min(k / tail, g)) {
                    ++fullGeneralTailDirect;
                }
                std::vector<std::vector<std::pair<int64_t, int64_t>>> expected[2], ordered[2];
                int64_t expectedTasks = 0;
                for (int chunk = 0; chunk < 2; ++chunk) {
                    const int64_t groups = chunk == 0 ? h : tail;
                    const int64_t lanes = std::min(k / groups, g);
                    for (int64_t lane = 0; lane < lanes; ++lane) {
                        expected[chunk].push_back(EnumerateGqaBandOwnerLaneForTest(lanes, lane, m, n, p, q));
                        ordered[chunk].push_back(GqaBandOwnerRowOrderForTest(expected[chunk].back(), lanes, m, n, p));
                        expectedTasks += static_cast<int64_t>(expected[chunk].back().size()) * groups * g;
                    }
                }
                std::map<std::pair<int64_t, int64_t>, int64_t> owners;
                std::set<std::tuple<int64_t, int64_t, int64_t>> seen;
                for (int64_t round = 0; round <= plan.totalRounds + 1; ++round) {
                    std::set<std::pair<int64_t, int64_t>> dq, dkv;
                    for (int64_t core = 0; core <= k + 1; ++core) {
                        int64_t w = 0, x = 0, y = 0;
                        MapGqaBandOwnerScheduleForTest(plan, core, round, w, x, y);
                        if (round < 1 || round > plan.totalRounds || core < 1 || core > k) {
                            ASSERT_EQ(w, -1);
                            continue;
                        }
                        const bool isTail = round > plan.fullRounds;
                        const int64_t groups = isTail ? tail : h;
                        const int64_t localRound = isTail ? round - plan.fullRounds : round;
                        const int64_t lanes = std::min(k / groups, g);
                        if (core > groups * lanes) {
                            ASSERT_EQ(w, -1);
                            continue;
                        }
                        const int64_t lane = (core - 1) % lanes, group = (core - 1) / lanes;
                        const auto& original = expected[isTail][lane];
                        const auto& rowOrder = ordered[isTail][lane];
                        const int64_t index = (localRound - 1) / g;
                        if (index >= static_cast<int64_t>(original.size())) {
                            ASSERT_EQ(w, -1);
                            continue;
                        }
                        ASSERT_EQ(x, rowOrder[index].first + (translated ? 3 : 0));
                        ASSERT_EQ(y, original[index].second + (translated ? 2 : 0));
                        ASSERT_EQ(w, (group + (isTail ? h : 0)) * g + (((localRound - 1) % g + lane) % g) + 1);
                        changedTailRows += isTail && rowOrder[index].first != original[index].first;
                        ASSERT_TRUE(dq.emplace(w, x).second);
                        ASSERT_TRUE(dkv.emplace((w - 1) / g, y).second);
                        ASSERT_TRUE(seen.emplace(w, x, y).second);
                        const auto owner = owners.emplace(std::make_pair((w - 1) / g, y), core);
                        EXPECT_EQ(owner.first->second, core);
                    }
                }
                EXPECT_EQ(seen.size(), static_cast<size_t>(expectedTasks));
            }
        }
    }
    EXPECT_GT(fullGeneralTailDirect, 0);
    EXPECT_GT(changedTailRows, 0);
}

// Core allocation must cover every logical group slot exactly once, including
// clipped budgets that intentionally leave physical cores unused.
TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandOverlapCorePartitionAndGuard)
{
    using namespace optiling::fag;
    struct Partition {
        int64_t k, groups, m, n;
        std::vector<int64_t> budgets;
    };
    const Partition examples[] = {
        {3, 1, 4, 1, {3}},  {36, 2, 16, 1, {9, 9}}, {32, 3, 8, 50, {11, 11, 10}},
        {32, 1, 8, 1, {5}}, {32, 1, 16, 17, {24}},  {32, 1, 32, 66, {32}},
    };
    for (const auto& c : examples) {
        ASSERT_TRUE(CanUseGqaBandOverlap(c.k, c.groups, 7, c.m, c.n));
        std::vector<std::vector<int>> seen;
        int64_t used = 0;
        for (int64_t group = 0; group < c.groups; ++group) {
            EXPECT_EQ(GqaBandOverlapCoreBudget(c.k, c.groups, c.m, c.n, group), c.budgets[group]);
            seen.emplace_back(c.budgets[group], 0);
            used += c.budgets[group];
        }
        int64_t active = 0;
        for (int64_t core = 0; core <= c.k + 1; ++core) {
            int64_t group = 99, local = 99, budget = 99;
            const bool valid = GqaBandOverlapCoreIndex(c.k, c.groups, c.m, c.n, core, group, local, budget);
            if (!valid) {
                EXPECT_EQ(group, -1);
                EXPECT_EQ(local, 0);
                EXPECT_EQ(budget, 0);
                continue;
            }
            ASSERT_GE(core, 1);
            ASSERT_LE(core, c.k);
            ASSERT_GE(group, 0);
            ASSERT_LT(group, c.groups);
            ASSERT_GE(local, 1);
            ASSERT_LE(local, c.budgets[group]);
            EXPECT_EQ(budget, c.budgets[group]);
            EXPECT_EQ(seen[group][local - 1]++, 0);
            ++active;
        }
        EXPECT_EQ(active, used);
        for (const auto& group : seen) {
            for (int visits : group) {
                EXPECT_EQ(visits, 1);
            }
        }
    }
    const int64_t invalid[][5] = {
        {0, 1, 2, 4, 1},  {32, 0, 2, 4, 1}, {32, 1, 1, 4, 1}, {32, 1, 2, 0, 1},
        {32, 1, 2, 1, 1}, {32, 1, 2, 4, 0}, {4, 2, 2, 4, 1},  {32, 33, 2, 2, 1},
    };
    for (const auto& c : invalid) {
        EXPECT_FALSE(CanUseGqaBandOverlap(c[0], c[1], c[2], c[3], c[4]));
        EXPECT_EQ(GqaBandOverlapChunkRounds(c[0], c[1], c[2], c[3], c[4]), 0);
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandOverlapParityAndTailInvariants)
{
    using namespace optiling::fag;
    for (int64_t k : {8, 16, 32, 36}) {
        for (int64_t m = 2; m < 2 * k; ++m) {
            for (int64_t g : {2, 3, 8}) {
                // The triangle has (m-1)*m/2 tiles. Its floor(m/2) cores
                // are exactly full for its entire active prefix, for either parity.
                const int64_t triangleCores = m / 2;
                const int64_t triangleRounds = GqaBandOverlapTriangleRounds(m, g);
                EXPECT_EQ(triangleRounds, GqaBandTriangleRounds(triangleCores, m - 1, 1, g));
                EXPECT_EQ(triangleCores * triangleRounds, (m - 1) * m / 2 * g);
                EXPECT_LE(triangleRounds, m * g);
                EXPECT_EQ(m * g - triangleRounds, m % 2 == 0 ? g : 0);
                for (int64_t h : {1, 2, 3}) {
                    for (int64_t q : std::vector<int64_t>{1, m, m + 1}) {
                        if (!CanUseGqaBandOverlap(k, h, g, m, q)) {
                            continue;
                        }
                        for (int64_t b : std::vector<int64_t>{h, 2 * h + 1}) {
                            for (bool upper : {false, true}) {
                                GqaBandPlan plan{};
                                ASSERT_TRUE(BuildGqaBandOverlapCandidate(k, m, q + m - 1, upper ? 1 : m,
                                                                         upper ? q + m - 1 : q, b, g, h, plan));
                                EXPECT_EQ(plan.phases[0].kind,
                                          upper ? GqaBandPhaseKind::OVERLAP_UPPER : GqaBandPhaseKind::OVERLAP_LOWER);
                                EXPECT_EQ(plan.phases[0].m, m);
                                EXPECT_EQ(plan.phases[0].n, q);
                                EXPECT_EQ(plan.totalRounds, b / h * plan.fullRounds + plan.tailRounds);
                                for (int64_t groups : std::vector<int64_t>{h, b % h}) {
                                    if (groups == 0) {
                                        EXPECT_EQ(plan.tailRounds, 0);
                                        continue;
                                    }
                                    int64_t used = 0, longestSuffix = 0;
                                    for (int64_t group = 0; group < groups; ++group) {
                                        const int64_t budget = GqaBandOverlapCoreBudget(k, groups, m, q, group);
                                        const int64_t stripes = budget - triangleCores;
                                        ASSERT_GE(stripes, 1);
                                        ASSERT_LE(stripes, std::min(m, q));
                                        used += budget;
                                        if (q > stripes) {
                                            longestSuffix = std::max(longestSuffix,
                                                                     GqaBandDenseRounds(budget, m, q - stripes, 1, g));
                                        }
                                    }
                                    EXPECT_LE(used, k);
                                    EXPECT_EQ(groups == h ? plan.fullRounds : plan.tailRounds, m * g + longestSuffix);
                                }
                                GqaBandSchedule serialized{};
                                SaveGqaBandSchedule(plan, serialized);
                                EXPECT_EQ(serialized.totalRounds, plan.totalRounds);
                                EXPECT_EQ(serialized.fullRounds, plan.fullRounds);
                                EXPECT_EQ(serialized.tailRounds, plan.tailRounds);
                                EXPECT_GE(GqaBandKvRunUpperBound(plan), b * (q + m - 1));
                            }
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(GqaBandOverlapTriangleRounds(1, 8), 0);
    EXPECT_EQ(GqaBandOverlapTriangleRounds(3, 1), 0);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandOverlapGeometryFullTailAndCost)
{
    using namespace optiling::fag;
    struct Cost {
        int64_t k, groups, g, m, rectangle, rounds;
    };
    const Cost examples[] = {
        {32, 1, 8, 32, 64, 640}, {32, 1, 8, 32, 66, 672}, {36, 2, 7, 16, 1, 112},
        {32, 3, 8, 8, 50, 352},  {32, 1, 8, 8, 50, 288},  {3, 1, 2, 4, 1, 8},
    };
    for (const auto& c : examples) {
        EXPECT_EQ(GqaBandOverlapChunkRounds(c.k, c.groups, c.g, c.m, c.rectangle), c.rounds);
        for (bool reflected : {false, true}) {
            const int64_t width = c.rectangle + c.m - 1;
            GqaBandPlan plan{};
            ASSERT_TRUE(BuildGqaBandOverlapCandidate(c.k, c.m, width, reflected ? 1 : c.m,
                                                     reflected ? width : c.rectangle, c.groups, c.g, c.groups, plan));
            ASSERT_EQ(plan.phaseCount, 1);
            EXPECT_EQ(plan.tailPhaseCount, 0);
            EXPECT_EQ(plan.phases[0].kind,
                      reflected ? GqaBandPhaseKind::OVERLAP_UPPER : GqaBandPhaseKind::OVERLAP_LOWER);
            EXPECT_EQ(plan.phases[0].m, c.m);
            EXPECT_EQ(plan.phases[0].n, c.rectangle);
            EXPECT_EQ(plan.phases[0].rowOffset, 0);
            EXPECT_EQ(plan.phases[0].colOffset, 0);
            EXPECT_EQ(plan.fullRounds, c.rounds);
            EXPECT_EQ(plan.totalRounds, c.rounds);
            EXPECT_EQ(plan.tailRounds, 0);
            EXPECT_GE(GqaBandKvRunUpperBound(plan), c.groups * width);
        }
    }
    // The third full-group budget is smaller; its longer work determines the
    // shared full-chunk round count. The tail receives a fresh core budget.
    GqaBandPlan tail{};
    ASSERT_TRUE(BuildGqaBandOverlapCandidate(32, 8, 57, 8, 50, 7, 8, 3, tail));
    EXPECT_EQ(tail.fullRounds, 352);
    EXPECT_EQ(tail.tailRounds, 288);
    EXPECT_EQ(tail.totalRounds, 992);
    GqaBandSchedule serialized{};
    SaveGqaBandSchedule(tail, serialized);
    EXPECT_EQ(serialized.groupsPerChunk, 3);
    EXPECT_EQ(serialized.b, 7);
    EXPECT_EQ(serialized.fullRounds, 352);
    EXPECT_EQ(serialized.tailRounds, 288);
    EXPECT_EQ(serialized.totalRounds, 992);
    // Saving rounds alone is insufficient: q=64 splits more KV runs than the
    // fixed preceding schedule, while the adjacent wider rectangle need not.
    EXPECT_EQ(GqaBandOverlapChunkKvRunUpperBound(32, 1, 8, 32, 64), 111);
    EXPECT_EQ(GqaBandOverlapChunkKvRunUpperBound(32, 1, 8, 32, 66), 124);
    for (const auto& geometry :
         std::vector<std::vector<int64_t>>{{8, 57, 7, 50}, {7, 56, 6, 50}, {8, 56, 8, 50}, {8, 57, 1, 56}}) {
        GqaBandPlan invalid{};
        EXPECT_FALSE(
            BuildGqaBandOverlapCandidate(32, geometry[0], geometry[1], geometry[2], geometry[3], 1, 8, 1, invalid));
    }
}

TEST_F(FlashAttentionScoreGradTiling, SwizzleTilingLayoutAddsExtensionOnlyForTnd)
{
    using namespace optiling::fag;
    using NonTnd = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<false, false, false, false>;
    using NonTndSwizzle = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<false, false, false, true>;
    using NonTndDeter = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, false, false, false>;
    using NonTndDeterSwizzle = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, false, false, true>;
    using Tnd = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<false, false, true, false>;
    using TndSwizzle = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<false, false, true, true>;
    using TndDeterSwizzle = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, true, true, true>;
    static_assert(std::is_same<decltype(NonTndSwizzle::tndSwizzleParam), std::nullptr_t>::value,
                  "A non-TND swizzle key must not append TND metadata");
    static_assert(std::is_same<decltype(NonTndDeterSwizzle::tndSwizzleParam), std::nullptr_t>::value,
                  "A non-TND deterministic swizzle key must not append TND metadata");
    static_assert(std::is_same<decltype(TndSwizzle::tndSwizzleParam), TndSwizzleParamRegbase>::value,
                  "TND swizzle must retain its extension");
    static_assert(std::is_same<decltype(TndDeterSwizzle::tndSwizzleParam), TndSwizzleParamRegbase>::value,
                  "Deterministic TND swizzle must retain its extension");
    EXPECT_EQ(sizeof(NonTnd), sizeof(NonTndSwizzle));
    EXPECT_EQ(sizeof(NonTndDeter), sizeof(NonTndDeterSwizzle));
    EXPECT_EQ(offsetof(NonTnd, tndParam), offsetof(NonTndSwizzle, tndParam));
    EXPECT_EQ(offsetof(NonTndDeter, deterParam), offsetof(NonTndDeterSwizzle, deterParam));
    EXPECT_GT(sizeof(TndSwizzle), sizeof(Tnd));
    static_assert(sizeof(FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase) == 200,
                  "The reserved dispatch byte must preserve the base tiling ABI");
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandPiecewisePreservesTilingLayout)
{
    using namespace optiling::fag;
    static_assert(sizeof(FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase) == 200,
                  "Base tiling size must remain unchanged");
    static_assert(offsetof(FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase, deterGqaBandScheduleMode) == 194,
                  "GQA BAND mode must occupy a former padding byte");
    static_assert(offsetof(FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase, deterGqaBandGroups) == 196,
                  "GQA BAND group size must occupy the former trailing padding");
    static_assert(sizeof(BaseDeterParamRegbase) == 608, "Deterministic tiling size must remain unchanged");
    FuzzyBaseInfoParamsRegbase params{};
    EXPECT_EQ(params.deterGqaBandScheduleMode, DeterGqaBandScheduleMode::ORIGINAL);
    EXPECT_EQ(params.deterGqaBandGroups, 0);
    FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase baseParams{};
    EXPECT_EQ(baseParams.get_deterGqaBandScheduleMode(), static_cast<uint8_t>(DeterGqaBandScheduleMode::ORIGINAL));
    EXPECT_EQ(baseParams.get_deterGqaBandGroups(), 0);
    baseParams.set_deterGqaBandScheduleMode(static_cast<uint8_t>(DeterGqaBandScheduleMode::PIECEWISE));
    baseParams.set_deterGqaBandGroups(2);
    EXPECT_EQ(baseParams.get_deterGqaBandScheduleMode(), static_cast<uint8_t>(DeterGqaBandScheduleMode::PIECEWISE));
    EXPECT_EQ(baseParams.get_deterGqaBandGroups(), 2);
}

static void ExpectGqaBandScheduleMatchesPlan(const optiling::fag::GqaBandSchedule& schedule,
                                             const optiling::fag::GqaBandPlan& plan)
{
    using namespace optiling::fag;
    EXPECT_EQ(schedule.phaseCount, plan.phaseCount);
    EXPECT_EQ(schedule.tailPhaseCount, plan.tailPhaseCount);
    EXPECT_EQ(schedule.k, plan.k);
    EXPECT_EQ(schedule.b, plan.b);
    EXPECT_EQ(schedule.g, plan.g);
    EXPECT_EQ(schedule.groupsPerChunk, plan.groupsPerChunk);
    EXPECT_EQ(schedule.fullRounds, plan.fullRounds);
    EXPECT_EQ(schedule.tailRounds, plan.tailRounds);
    EXPECT_EQ(schedule.totalRounds, plan.totalRounds);
    for (int64_t tail = 0; tail < 2; ++tail) {
        const int64_t count = tail == 0 ? plan.phaseCount : plan.tailPhaseCount;
        ASSERT_GE(count, 0);
        ASSERT_LE(count, GQA_BAND_MAX_PHASES);
        for (int64_t i = 0; i < GQA_BAND_MAX_PHASES; ++i) {
            SCOPED_TRACE(::testing::Message() << "tail=" << tail << ", phase=" << i);
            const GqaBandPhase empty{};
            const GqaBandPhase& expected = i >= count ? empty : (tail == 0 ? plan.phases[i] : plan.tailPhases[i]);
            const GqaBandPhase& actual = tail == 0 ? schedule.phases[i] : schedule.tailPhases[i];
            EXPECT_EQ(actual.kind, expected.kind);
            EXPECT_EQ(actual.m, expected.m);
            EXPECT_EQ(actual.n, expected.n);
            EXPECT_EQ(actual.rowOffset, expected.rowOffset);
            EXPECT_EQ(actual.colOffset, expected.colOffset);
            EXPECT_EQ(actual.fullRounds, expected.fullRounds);
            EXPECT_EQ(actual.tailRounds, expected.tailRounds);
            if (i >= count) {
                EXPECT_EQ(std::memcmp(&actual, &empty, sizeof(empty)), 0);
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandSerializedScheduleLayout)
{
    using namespace optiling::fag;
    static_assert(sizeof(GqaBandPhaseKind) == sizeof(int64_t), "Wire phase tags must be eight bytes");
    static_assert(sizeof(GqaBandPhase) == 56, "Wire phases must not contain padding");
    static_assert(sizeof(GqaBandSchedule) == 632, "Wire schedules must not contain padding");
    static_assert(offsetof(GqaBandSchedule, phaseCount) == 560, "Both five-phase arrays precede metadata");
    static_assert(offsetof(GqaBandSchedule, totalRounds) == 624, "The round bound is the final wire word");
    static_assert(std::is_standard_layout<GqaBandSchedule>::value, "The wire schedule must be standard layout");
    static_assert(std::is_trivially_copyable<GqaBandSchedule>::value, "The wire schedule must allow byte copies");
    using NonTnd = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<false, false, false, false>;
    using NonTndDeter = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, false, false, false>;
    using NonTndDeterSwizzle = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, false, false, true>;
    using TndDeter = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, true, true, false>;
    using TndDeterSwizzle = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<true, true, true, true>;
    static_assert(std::is_same<decltype(NonTndDeter::deterGqaBandSchedule), GqaBandSchedule>::value,
                  "Only non-TND deterministic tiling carries the schedule");
    static_assert(std::is_same<decltype(NonTndDeterSwizzle::deterGqaBandSchedule), GqaBandSchedule>::value,
                  "The existing swizzle bit must not change the schedule layout");
    static_assert(std::is_same<decltype(NonTnd::deterGqaBandSchedule), std::nullptr_t>::value,
                  "Non-deterministic tiling must not carry the GQA schedule");
    static_assert(std::is_same<decltype(TndDeter::deterGqaBandSchedule), std::nullptr_t>::value,
                  "TND must not read a non-TND schedule");
    static_assert(std::is_same<decltype(TndDeterSwizzle::deterGqaBandSchedule), std::nullptr_t>::value,
                  "TND swizzle must not read a non-TND schedule");
    // 1,904 bytes is the existing serialized TFFF prefix, including all previous placeholder fields.
    EXPECT_EQ(offsetof(NonTndDeter, deterGqaBandSchedule), 1904U);
    EXPECT_EQ(sizeof(NonTndDeter), 1904U + sizeof(GqaBandSchedule));
    EXPECT_EQ(sizeof(NonTndDeterSwizzle), sizeof(NonTndDeter));
    EXPECT_EQ(offsetof(NonTndDeter, deterGqaBandSchedule),
              offsetof(NonTndDeter, tndSwizzleParam) + sizeof(std::nullptr_t));
    EXPECT_EQ(offsetof(TndDeterSwizzle, deterGqaBandSchedule),
              offsetof(TndDeterSwizzle, tndSwizzleParam) + sizeof(TndSwizzleParamRegbase));
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandScheduleCopyClearsInactivePhases)
{
    using namespace optiling::fag;
    GqaBandPlan plan;
    GqaBandSchedule schedule;
    BuildGqaBandCandidate(32, 32, 32, 16, 17, 11, 16, 7, plan);
    ASSERT_GT(plan.tailPhaseCount, 0);
    SaveGqaBandSchedule(plan, schedule);
    ExpectGqaBandScheduleMatchesPlan(schedule, plan);
    // The reused candidate may retain inactive storage; the wire representation must not retain it.
    BuildGqaBandCandidate(32, 32, 64, 32, 33, 4, 16, 2, plan);
    ASSERT_EQ(plan.tailPhaseCount, 0);
    for (int64_t i = plan.phaseCount; i < GQA_BAND_MAX_PHASES; ++i) {
        plan.phases[i].m = 123;
    }
    for (auto& phase : plan.tailPhases) {
        phase.n = 456;
    }
    SaveGqaBandSchedule(plan, schedule);
    ExpectGqaBandScheduleMatchesPlan(schedule, plan);
    const GqaBandPlan emptyPlan{};
    const GqaBandSchedule emptySchedule{};
    SaveGqaBandSchedule(emptyPlan, schedule);
    EXPECT_EQ(std::memcmp(&schedule, &emptySchedule, sizeof(schedule)), 0);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandPiecewiseGeometryExactCoverage)
{
    using namespace optiling::fag;
    // Check both triangle directions, diagonal strips, clipped columns and one-row/one-column shapes.
    for (int64_t m = 1; m <= 8; ++m) {
        for (int64_t n = 1; n <= 8; ++n) {
            for (int64_t p = 1; p <= m; ++p) {
                for (int64_t q = 1; q <= n; ++q) {
                    SCOPED_TRACE(::testing::Message() << "m=" << m << ", n=" << n << ", p=" << p << ", q=" << q);
                    GqaBandPlan plan;
                    BuildGqaBandGeometry(m, n, p, q, plan);
                    ASSERT_GT(plan.phaseCount, 0);
                    ASSERT_LE(plan.phaseCount, GQA_BAND_MAX_PHASES);
                    std::vector<int> visits(m * n, 0);
                    for (int64_t phaseIdx = 0; phaseIdx < plan.phaseCount; ++phaseIdx) {
                        const auto& phase = plan.phases[phaseIdx];
                        for (int64_t x = 1; x <= phase.m; ++x) {
                            for (int64_t y = 1; y <= phase.n; ++y) {
                                if ((phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE && x < y) ||
                                    (phase.kind == GqaBandPhaseKind::UPPER_TRIANGLE && x > y)) {
                                    continue;
                                }
                                const int64_t row =
                                    x + phase.rowOffset + (phase.kind == GqaBandPhaseKind::DIAGONAL ? y - 1 : 0);
                                const int64_t col = y + phase.colOffset;
                                ASSERT_GE(row, 1);
                                ASSERT_LE(row, m);
                                ASSERT_GE(col, 1);
                                ASSERT_LE(col, n);
                                ++visits[(row - 1) * n + col - 1];
                            }
                        }
                    }
                    for (int64_t x = 1; x <= m; ++x) {
                        for (int64_t y = 1; y <= n; ++y) {
                            const bool valid = x >= y - q + 1 && x <= y + p - 1;
                            EXPECT_EQ(visits[(x - 1) * n + y - 1], valid ? 1 : 0);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandPiecewiseUpperGeometryCandidates)
{
    using namespace optiling::fag;
    GqaBandPlan plan;
    BuildGqaBandGeometry(32, 64, 1, 64, plan);
    SetGqaBandGroups(28, 4, 16, 4, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::UPPER_TRIANGLE);
    EXPECT_EQ(plan.totalRounds, 3632);

    BuildGqaBandCandidate(28, 32, 64, 1, 64, 4, 16, 4, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.totalRounds, 3600);
    ASSERT_EQ(plan.phaseCount, 2);
    EXPECT_EQ(plan.phases[0].kind, GqaBandPhaseKind::UPPER_TRIANGLE);
    EXPECT_EQ(plan.phases[0].m, 31);
    EXPECT_EQ(plan.phases[1].kind, GqaBandPhaseKind::RECTANGLE);
    EXPECT_EQ(plan.phases[1].n, 33);

    // Equal complete round counts must retain the previous geometry and reduction order.
    BuildGqaBandCandidate(32, 32, 64, 1, 64, 4, 16, 2, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::UPPER_TRIANGLE);
    EXPECT_EQ(plan.totalRounds, 3104);
    BuildGqaBandCandidate(32, 64, 64, 1, 64, 4, 16, 1, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::UPPER_TRIANGLE);
    EXPECT_EQ(plan.totalRounds, 4160);

    // This non-upper BAND ties its reflected candidate and keeps the previous geometry.
    BuildGqaBandCandidate(32, 32, 64, 32, 33, 4, 16, 2, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.totalRounds, 3104);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandPiecewiseUpperGeometryIncludesTail)
{
    using namespace optiling::fag;
    GqaBandPlan plan;
    BuildGqaBandCandidate(28, 32, 64, 1, 64, 5, 16, 4, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.fullRounds, 3600);
    EXPECT_EQ(plan.tailRounds, 1104);
    EXPECT_EQ(plan.totalRounds, 4704);

    // Full chunks tie, but the general geometry has a more expensive incomplete chunk.
    BuildGqaBandGeneralGeometry(32, 64, 1, 64, plan);
    SetGqaBandGroups(32, 5, 16, 2, plan);
    EXPECT_EQ(plan.fullRounds, 1552);
    EXPECT_EQ(plan.tailRounds, 1040);
    EXPECT_EQ(plan.totalRounds, 4144);
    BuildGqaBandCandidate(32, 32, 64, 1, 64, 5, 16, 2, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::UPPER_TRIANGLE);
    EXPECT_EQ(plan.fullRounds, 1552);
    EXPECT_EQ(plan.tailRounds, 1024);
    EXPECT_EQ(plan.totalRounds, 4128);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandPiecewiseReflectionGuardsAndTies)
{
    using namespace optiling::fag;
    GqaBandPlan plan;
    // A worse reflected candidate must not undo the previous upper/general selection.
    BuildGqaBandCandidate(28, 32, 64, 1, 64, 4, 16, 4, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.totalRounds, 3600);
    // Equal complete rounds retain the forward reduction order, including main-case ties.
    BuildGqaBandCandidate(32, 32, 64, 32, 33, 4, 16, 2, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.totalRounds, 3104);
    // Reflection with nonpositive normalized tokens is outside the first-version domain.
    BuildGqaBandCandidate(32, 32, 64, 1, 32, 4, 16, 4, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    BuildGqaBandCandidate(32, 64, 32, 32, 1, 4, 16, 4, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    // Symmetric geometry does not need a second candidate; plan reuse resets reflected state.
    BuildGqaBandCandidate(32, 64, 64, 1, 32, 4, 16, 2, plan);
    ASSERT_EQ(plan.geometryMode, GqaBandGeometryMode::REVERSED_GENERAL);
    BuildGqaBandCandidate(32, 64, 64, 2, 2, 4, 16, 4, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::PAIRED_EDGES);
    // A better piecewise candidate can still lose to the original folded BAND.
    EXPECT_GT(plan.totalRounds, GqaBandLegacyRounds(32, 64, 64, 2, 2, 4, 16));
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandPiecewiseReflectedGeometryExactCoverage)
{
    using namespace optiling::fag;
    for (int64_t m = 1; m <= 8; ++m) {
        for (int64_t n = 1; n <= 8; ++n) {
            for (int64_t p = 1; p <= m; ++p) {
                for (int64_t q = 1; q <= n; ++q) {
                    const int64_t reflectedP = m - n + q;
                    const int64_t reflectedQ = n - m + p;
                    if (reflectedP <= 0 || reflectedQ <= 0) {
                        continue;
                    }
                    SCOPED_TRACE(::testing::Message() << "m=" << m << ", n=" << n << ", p=" << p << ", q=" << q);
                    GqaBandPlan plan;
                    BuildGqaBandGeometry(m, n, reflectedP, reflectedQ, plan);
                    ReflectGqaBandGeometry(m, n, plan);
                    ASSERT_GT(plan.phaseCount, 0);
                    ASSERT_LE(plan.phaseCount, GQA_BAND_MAX_PHASES);
                    std::vector<int> visits(m * n, 0);
                    for (int64_t phaseIdx = 0; phaseIdx < plan.phaseCount; ++phaseIdx) {
                        const auto& phase = plan.phases[phaseIdx];
                        for (int64_t x = 1; x <= phase.m; ++x) {
                            for (int64_t y = 1; y <= phase.n; ++y) {
                                if ((phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE && x < y) ||
                                    (phase.kind == GqaBandPhaseKind::UPPER_TRIANGLE && x > y)) {
                                    continue;
                                }
                                const int64_t row =
                                    x + phase.rowOffset + (phase.kind == GqaBandPhaseKind::DIAGONAL ? y - 1 : 0);
                                const int64_t col = y + phase.colOffset;
                                ASSERT_GE(row, 1);
                                ASSERT_LE(row, m);
                                ASSERT_GE(col, 1);
                                ASSERT_LE(col, n);
                                ++visits[(row - 1) * n + col - 1];
                            }
                        }
                    }
                    for (int64_t x = 1; x <= m; ++x) {
                        for (int64_t y = 1; y <= n; ++y) {
                            const bool valid = x >= y - q + 1 && x <= y + p - 1;
                            EXPECT_EQ(visits[(x - 1) * n + y - 1], valid ? 1 : 0);
                        }
                    }
                }
            }
        }
    }
}

// Frozen pre-independent-tail construction is a test reference, not a production candidate.
static void BuildGqaBandForwardCandidateReference(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b,
                                                  int64_t g, int64_t groups, optiling::fag::GqaBandPlan& plan)
{
    using namespace optiling::fag;
    BuildGqaBandGeometry(m, n, p, q, plan);
    SetGqaBandGroups(k, b, g, groups, plan);
    if (p != 1 || q != n) {
        return;
    }
    const int64_t originalRounds = plan.totalRounds;
    BuildGqaBandGeneralGeometry(m, n, p, q, plan);
    SetGqaBandGroups(k, b, g, groups, plan);
    // Include all phases and the incomplete chunk. Ties retain the original
    // upper-triangle specialization, not a different reduction order.
    if (plan.totalRounds >= originalRounds) {
        BuildGqaBandGeometry(m, n, p, q, plan);
        SetGqaBandGroups(k, b, g, groups, plan);
    }
}

static void BuildGqaBandUniformCandidateReference(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b,
                                                  int64_t g, int64_t groups, optiling::fag::GqaBandPlan& plan)
{
    using namespace optiling::fag;
    BuildGqaBandForwardCandidateReference(k, m, n, p, q, b, g, groups, plan);
    const int64_t reflectedP = m - n + q;
    const int64_t reflectedQ = n - m + p;
    // The existing builders require positive p/q. Self-symmetric shapes have
    // identical costs, so retain their original geometry without rebuilding it.
    if (reflectedP <= 0 || reflectedQ <= 0 || (reflectedP == p && reflectedQ == q)) {
        return;
    }
    const int64_t forwardRounds = plan.totalRounds;
    BuildGqaBandForwardCandidateReference(k, m, n, reflectedP, reflectedQ, b, g, groups, plan);
    if (plan.totalRounds < forwardRounds) {
        ReflectGqaBandGeometry(m, n, plan);
    } else {
        // Include every phase and the incomplete chunk; ties keep the forward mapping.
        BuildGqaBandForwardCandidateReference(k, m, n, p, q, b, g, groups, plan);
    }
}

// Frozen pre-cache reducer checks serialized order and independent-tail ties.
static void BuildGqaBandCandidateUncachedReference(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b,
                                                   int64_t g, int64_t groups, optiling::fag::GqaBandPlan& plan)
{
    using namespace optiling::fag;
    constexpr GqaBandGeometryMode candidateModes[] = {GqaBandGeometryMode::UPPER_TRIANGLE,
                                                      GqaBandGeometryMode::GENERAL,
                                                      GqaBandGeometryMode::REVERSED_UPPER_TRIANGLE,
                                                      GqaBandGeometryMode::REVERSED_GENERAL,
                                                      GqaBandGeometryMode::PAIRED_EDGES,
                                                      GqaBandGeometryMode::GENERAL_PAIRED_EDGES};
    GqaBandGeometryMode uniformMode = GqaBandGeometryMode::GENERAL;
    GqaBandGeometryMode fullMode = GqaBandGeometryMode::GENERAL;
    GqaBandGeometryMode tailMode = GqaBandGeometryMode::GENERAL;
    int64_t uniformRounds = 0;
    int64_t fullRounds = 0;
    int64_t tailRounds = 0;
    bool firstCandidate = true;
    const int64_t fullChunks = b / groups;
    for (const GqaBandGeometryMode mode : candidateModes) {
        if (!BuildGqaBandGeometryByMode(m, n, p, q, mode, plan)) {
            continue;
        }
        SetGqaBandGroups(k, b, g, groups, plan);
        if (mode == GqaBandGeometryMode::GENERAL_PAIRED_EDGES) {
            const int64_t previousRounds = GqaBandMin(uniformRounds, fullChunks * fullRounds + tailRounds);
            const int64_t combinedRounds =
                fullChunks * GqaBandMin(fullRounds, plan.fullRounds) + GqaBandMin(tailRounds, plan.tailRounds);
            if (GqaBandMin(plan.totalRounds, combinedRounds) >= previousRounds) {
                // A new uniform geometry can tie the previous mixed full/tail plan.
                // Keep all previous modes on a total-round tie, not just per-part ties.
                continue;
            }
        }
        if (firstCandidate || plan.totalRounds < uniformRounds) {
            uniformMode = mode;
            uniformRounds = plan.totalRounds;
        }
        if (firstCandidate || plan.fullRounds < fullRounds) {
            fullMode = mode;
            fullRounds = plan.fullRounds;
        }
        if (firstCandidate || plan.tailRounds < tailRounds) {
            tailMode = mode;
            tailRounds = plan.tailRounds;
        }
        firstCandidate = false;
    }
    const int64_t tailGroups = b % groups;
    const int64_t independentRounds = fullChunks * fullRounds + tailRounds;
    if (tailGroups == 0 || independentRounds >= uniformRounds) {
        if (plan.geometryMode != uniformMode) {
            BuildGqaBandGeometryByMode(m, n, p, q, uniformMode, plan);
            SetGqaBandGroups(k, b, g, groups, plan);
        }
        ConfigureGqaBandDensePhases(plan);
        return;
    }

    BuildGqaBandGeometryByMode(m, n, p, q, tailMode, plan);
    SetGqaBandGroups(k, tailGroups, g, tailGroups, plan);
    const int64_t tailPhaseCount = plan.phaseCount;
    const GqaBandGeometryMode tailGeometryMode = plan.geometryMode;
    for (int64_t i = 0; i < tailPhaseCount; ++i) {
        plan.tailPhases[i] = plan.phases[i];
        plan.tailPhases[i].tailRounds = plan.phases[i].fullRounds;
    }
    // The builders only reset active metadata and the full-phase array. Preserve
    // the saved tail array while rebuilding the full chunks in the same object.
    BuildGqaBandGeometryByMode(m, n, p, q, fullMode, plan);
    SetGqaBandGroups(k, groups, g, groups, plan);
    plan.b = b;
    plan.tailPhaseCount = tailPhaseCount;
    plan.tailGeometryMode = tailGeometryMode;
    plan.tailRounds = tailRounds;
    plan.totalRounds = independentRounds;
    ConfigureGqaBandDensePhases(plan);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandCacheRetainsRoundFirstWireAndTailTies)
{
    using namespace optiling::fag;
    struct Domain {
        int64_t k, m, n, p, q, b, g;
    };
    const Domain domains[] = {
        {32, 32, 32, 16, 17, 11, 16}, {32, 32, 64, 1, 64, 5, 16}, {8, 4, 4, 4, 3, 5, 2},
        {32, 64, 96, 64, 65, 5, 2},   {32, 14, 14, 7, 14, 36, 5}, {8, 17, 29, 4, 7, 19, 3},
    };
    for (const auto& d : domains) {
        GqaBandCandidateCache cache(d.k, d.m, d.n, d.p, d.q, d.b, d.g);
        std::vector<int64_t> groups;
        for (int64_t h = 1; h <= std::min(d.b, d.k); ++h) {
            groups.push_back(h);
        }
        if (d.b > d.k) {
            groups.push_back(d.b);
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (const int64_t h : groups) {
                SCOPED_TRACE(::testing::Message() << d.k << "," << d.m << "," << d.n << "," << d.b << "," << h);
                GqaBandPlan cached, previous;
                cache.BuildLegacy(h, cached);
                BuildGqaBandCandidateUncachedReference(d.k, d.m, d.n, d.p, d.q, d.b, d.g, h, previous);
                GqaBandSchedule actual{}, expected{};
                SaveGqaBandSchedule(cached, actual);
                SaveGqaBandSchedule(previous, expected);
                EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(actual)), 0);
                EXPECT_EQ(cached.geometryMode, previous.geometryMode);
                EXPECT_EQ(cached.tailGeometryMode, previous.tailGeometryMode);
            }
            // Reusing the cache in the opposite order must not alias a tail
            // summary or change which uniform/mixed plan wins a tie.
            std::reverse(groups.begin(), groups.end());
        }
    }
    // The public one-group builder also supports H>K with a tail larger than K.
    // Neither summary may alias the other merely because both exceed K.
    for (int64_t h : {36, 40, 50}) {
        GqaBandPlan current, previous;
        BuildGqaBandCandidate(3, 15, 10, 13, 4, 61, 4, h, current);
        BuildGqaBandCandidateUncachedReference(3, 15, 10, 13, 4, 61, 4, h, previous);
        GqaBandSchedule actual{}, expected{};
        SaveGqaBandSchedule(current, actual);
        SaveGqaBandSchedule(previous, expected);
        EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(actual)), 0);
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandFoldedDomainPreservesWireAndPhysicalOrigin)
{
    using namespace optiling::fag;
    const GqaBandPhaseKind kinds[] = {GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER,
                                      GqaBandPhaseKind::FOLDED_LOWER_COLUMN_OWNER,
                                      GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER};
    for (const auto kind : kinds) {
        for (int64_t origin : {0, 1, 7}) {
            const bool upper = kind == GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER;
            const bool whole = kind == GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER;
            const GqaBandDomain domain{
                9, 13, upper ? 1 : 3, whole || upper ? 5 : 1, !whole && !upper ? origin : 0, upper ? origin : 0};
            GqaBandPlan encoded, reference;
            AppendGqaBandFoldedPhase(encoded, kind, domain);
            // Expected serialized fields retain the established tag-specific ABI.
            AppendGqaBandPhase(reference, kind, 9, 13, whole ? 3 : (upper ? 5 : origin),
                               whole ? 5 : (upper ? origin : 3));
            ASSERT_EQ(encoded.phaseCount, 1);
            EXPECT_EQ(std::memcmp(&encoded.phases[0], &reference.phases[0], sizeof(GqaBandPhase)), 0);
            const auto decoded = DecodeGqaBandFoldedDomain(encoded.phases[0]);
            EXPECT_EQ(decoded.m, domain.m);
            EXPECT_EQ(decoded.n, domain.n);
            EXPECT_EQ(decoded.p, domain.p);
            EXPECT_EQ(decoded.q, domain.q);
            EXPECT_EQ(decoded.rowOrigin, domain.rowOrigin);
            EXPECT_EQ(decoded.colOrigin, domain.colOrigin);
            for (int64_t h : {1, 2, 3}) {
                SetGqaBandGroups(32, 5, 3, h, encoded);
                SetGqaBandGroups(32, 5, 3, h, reference);
                GqaBandSchedule actual{}, expected{};
                SaveGqaBandSchedule(encoded, actual);
                SaveGqaBandSchedule(reference, expected);
                EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(actual)), 0);
                EXPECT_EQ(GqaBandKvRunUpperBound(encoded), 5 * std::min(domain.n, domain.m + domain.q - 1));
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandPiecewiseIndependentTailResetAndTie)
{
    using namespace optiling::fag;
    GqaBandPlan plan;
    BuildGqaBandCandidate(32, 32, 32, 16, 17, 11, 16, 7, plan);
    ASSERT_GT(plan.tailPhaseCount, 0);
    // A full-chunk-only plan must not retain the preceding mixed tail metadata.
    BuildGqaBandCandidate(32, 32, 32, 16, 17, 14, 16, 7, plan);
    EXPECT_EQ(plan.tailPhaseCount, 0);
    EXPECT_EQ(plan.tailRounds, 0);
    EXPECT_EQ(plan.totalRounds, 5544);

    BuildGqaBandCandidate(32, 32, 32, 16, 17, 11, 16, 7, plan);
    ASSERT_GT(plan.tailPhaseCount, 0);
    // Independent full/tail optima tie the existing uniform schedule here.
    BuildGqaBandCandidate(32, 32, 64, 1, 64, 5, 16, 2, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::UPPER_TRIANGLE);
    EXPECT_EQ(plan.tailPhaseCount, 0);
    EXPECT_EQ(plan.fullRounds, 1552);
    EXPECT_EQ(plan.tailRounds, 1024);
    EXPECT_EQ(plan.totalRounds, 4128);

    // Single full chunks tie; changing only their direction gives no total-round benefit.
    // Keep the complete old reflected schedule, not a mixed forward/reflected plan.
    BuildGqaBandCandidate(8, 4, 4, 4, 3, 5, 2, 3, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::REVERSED_GENERAL);
    EXPECT_EQ(plan.tailPhaseCount, 0);
    EXPECT_EQ(plan.totalRounds, 32);

    BuildGqaBandCandidate(32, 32, 32, 16, 17, 11, 16, 7, plan);
    BuildGqaBandUniformCandidateReference(32, 32, 32, 16, 17, 11, 16, 7, plan);
    EXPECT_EQ(plan.tailPhaseCount, 0);
    EXPECT_EQ(plan.totalRounds, 4360);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandPiecewiseIndependentTailCoverage)
{
    using namespace optiling::fag;
    GqaBandPlan plan;
    BuildGqaBandCandidate(32, 32, 32, 16, 17, 11, 16, 7, plan);
    ASSERT_EQ(plan.phaseCount, 4);
    ASSERT_EQ(plan.tailPhaseCount, 4);
    for (int64_t tail = 0; tail < 2; ++tail) {
        std::vector<int> visits(32 * 32, 0);
        const int64_t count = tail == 0 ? plan.phaseCount : plan.tailPhaseCount;
        for (int64_t i = 0; i < count; ++i) {
            const auto& phase = tail == 0 ? plan.phases[i] : plan.tailPhases[i];
            for (int64_t x = 1; x <= phase.m; ++x) {
                for (int64_t y = 1; y <= phase.n; ++y) {
                    if ((phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE && x < y) ||
                        (phase.kind == GqaBandPhaseKind::UPPER_TRIANGLE && x > y)) {
                        continue;
                    }
                    const int64_t row = x + phase.rowOffset + (phase.kind == GqaBandPhaseKind::DIAGONAL ? y - 1 : 0);
                    const int64_t col = y + phase.colOffset;
                    ASSERT_GE(row, 1);
                    ASSERT_LE(row, 32);
                    ASSERT_GE(col, 1);
                    ASSERT_LE(col, 32);
                    ++visits[(row - 1) * 32 + col - 1];
                }
            }
        }
        for (int64_t x = 1; x <= 32; ++x) {
            for (int64_t y = 1; y <= 32; ++y) {
                EXPECT_EQ(visits[(x - 1) * 32 + y - 1], (x >= y - 16 && x <= y + 15) ? 1 : 0);
            }
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandPairedEdgesGeometryCoverageAndGuards)
{
    using namespace optiling::fag;
    for (int64_t width = 1; width <= 8; ++width) {
        const int64_t minimum = std::max(2 * width, 4 * width - 2);
        for (int64_t m = minimum; m <= minimum + 5; ++m) {
            GqaBandPlan plan;
            ASSERT_TRUE(BuildGqaBandGeometryByMode(m, m, width, width, GqaBandGeometryMode::PAIRED_EDGES, plan));
            ASSERT_EQ(plan.geometryMode, GqaBandGeometryMode::PAIRED_EDGES);
            ASSERT_EQ(plan.phases[0].kind, GqaBandPhaseKind::EDGE_PAIR);
            // The two physical edge regions must have disjoint row and column domains.
            EXPECT_LT(2 * width - 1, m - 2 * width + 2);
            EXPECT_LT(width, m - width + 1);
            std::vector<int> visits(m * m, 0);
            for (int64_t i = 0; i < plan.phaseCount; ++i) {
                const auto& phase = plan.phases[i];
                for (int64_t x = 1; x <= phase.m; ++x) {
                    for (int64_t y = 1; y <= phase.n; ++y) {
                        int64_t row = x;
                        int64_t col = y;
                        if (phase.kind == GqaBandPhaseKind::EDGE_PAIR) {
                            if (x > phase.n - 1 + y) {
                                row += phase.rowOffset;
                                col += phase.colOffset;
                            }
                        } else {
                            ASSERT_EQ(phase.kind, GqaBandPhaseKind::DIAGONAL);
                            row += y - 1 + phase.rowOffset;
                            col += phase.colOffset;
                        }
                        ASSERT_GE(row, 1);
                        ASSERT_LE(row, m);
                        ASSERT_GE(col, 1);
                        ASSERT_LE(col, m);
                        ++visits[(row - 1) * m + col - 1];
                    }
                }
            }
            for (int64_t x = 1; x <= m; ++x) {
                for (int64_t y = 1; y <= m; ++y) {
                    EXPECT_EQ(visits[(x - 1) * m + y - 1], (x >= y - width + 1 && x <= y + width - 1) ? 1 : 0);
                }
            }
        }
        GqaBandPlan rejected;
        EXPECT_FALSE(BuildGqaBandGeometryByMode(minimum - 1, minimum - 1, width, width,
                                                GqaBandGeometryMode::PAIRED_EDGES, rejected));
        EXPECT_FALSE(BuildGqaBandGeometryByMode(minimum + 1, minimum, width, width, GqaBandGeometryMode::PAIRED_EDGES,
                                                rejected));
        EXPECT_FALSE(BuildGqaBandGeometryByMode(minimum + 1, minimum + 1, width, width + 1,
                                                GqaBandGeometryMode::PAIRED_EDGES, rejected));
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandPairedEdgesIndependentTailAndReset)
{
    using namespace optiling::fag;
    GqaBandPlan plan;
    BuildGqaBandCandidate(36, 64, 64, 16, 16, 7, 3, 4, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.tailGeometryMode, GqaBandGeometryMode::PAIRED_EDGES);
    EXPECT_EQ(plan.phaseCount, 5);
    EXPECT_EQ(plan.tailPhaseCount, 2);
    EXPECT_EQ(plan.fullRounds, 602);
    EXPECT_EQ(plan.tailRounds, 436);
    EXPECT_EQ(plan.totalRounds, 1038);
    EXPECT_EQ(plan.tailPhases[0].kind, GqaBandPhaseKind::EDGE_PAIR);

    // The reverse mixture also matters: pair complete chunks but retain the old tail.
    BuildGqaBandCandidate(36, 32, 32, 8, 8, 16, 2, 9, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::PAIRED_EDGES);
    EXPECT_EQ(plan.tailGeometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.phaseCount, 2);
    EXPECT_EQ(plan.tailPhaseCount, 5);
    EXPECT_EQ(plan.fullRounds, 212);
    EXPECT_EQ(plan.tailRounds, 186);
    EXPECT_EQ(plan.totalRounds, 398);
    EXPECT_EQ(plan.phases[0].kind, GqaBandPhaseKind::EDGE_PAIR);

    BuildGqaBandCandidate(32, 128, 128, 32, 32, 1, 16, 1, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::PAIRED_EDGES);
    EXPECT_EQ(plan.tailPhaseCount, 0);
    EXPECT_EQ(plan.tailRounds, 0);
    EXPECT_EQ(plan.totalRounds, 3536);
    // Width one is safe, but an equal-cost pairing must retain the older mapping.
    BuildGqaBandCandidate(4, 2, 2, 1, 1, 5, 2, 5, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.tailPhaseCount, 0);
    EXPECT_EQ(plan.totalRounds, 6);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandGeneralPairedEdgesCoverageAndGuards)
{
    using namespace optiling::fag;
    int64_t qBoundCases = 0;
    int64_t pBoundCases = 0;
    for (int64_t m = 1; m <= 14; ++m) {
        for (int64_t n = 1; n <= 14; ++n) {
            for (int64_t p = 1; p <= std::min(m, int64_t{5}); ++p) {
                for (int64_t q = 1; q <= std::min(n, int64_t{5}); ++q) {
                    SCOPED_TRACE(::testing::Message() << "m=" << m << ", n=" << n << ", p=" << p << ", q=" << q);
                    const int64_t validRows = std::min(m, n + p - 1);
                    const int64_t validColumns = std::min(n, m + q - 1);
                    const int64_t width = std::min(q, validColumns + p - validRows);
                    const bool oldPaired = m == n && p == q && m >= std::max(2 * p, 4 * p - 2);
                    const bool expected =
                        !oldPaired && width > 0 && validColumns >= 2 * width && validColumns >= p + q + 2 * width - 2;
                    GqaBandPlan plan;
                    ASSERT_EQ(BuildGqaBandGeometryByMode(m, n, p, q, GqaBandGeometryMode::GENERAL_PAIRED_EDGES, plan),
                              expected);
                    if (!expected) {
                        continue;
                    }
                    ASSERT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL_PAIRED_EDGES);
                    ASSERT_GE(plan.phaseCount, 1);
                    ASSERT_LE(plan.phaseCount, GQA_BAND_MAX_PHASES);
                    EXPECT_LT(p + width - 1, validColumns - width - q + 2);
                    EXPECT_LT(width, validColumns - width + 1);
                    if (width == q) {
                        ASSERT_EQ(plan.phases[0].kind, GqaBandPhaseKind::EDGE_PAIR_Q_BOUND);
                        ++qBoundCases;
                    } else {
                        ASSERT_EQ(plan.phases[0].kind, GqaBandPhaseKind::EDGE_PAIR_P_BOUND);
                        ++pBoundCases;
                    }
                    std::vector<int> visits(m * n, 0);
                    for (int64_t i = 0; i < plan.phaseCount; ++i) {
                        const auto& phase = plan.phases[i];
                        for (int64_t x = 1; x <= phase.m; ++x) {
                            for (int64_t y = 1; y <= phase.n; ++y) {
                                if ((phase.kind == GqaBandPhaseKind::LOWER_TRIANGLE && x < y) ||
                                    (phase.kind == GqaBandPhaseKind::UPPER_TRIANGLE && x > y)) {
                                    continue;
                                }
                                int64_t row = x;
                                int64_t col = y;
                                if (phase.kind == GqaBandPhaseKind::EDGE_PAIR_Q_BOUND ||
                                    phase.kind == GqaBandPhaseKind::EDGE_PAIR_P_BOUND) {
                                    const int64_t bound = phase.kind == GqaBandPhaseKind::EDGE_PAIR_Q_BOUND ?
                                                              phase.colOffset - phase.rowOffset - phase.n :
                                                              phase.m + phase.rowOffset - phase.colOffset - 1;
                                    if (x > bound + y) {
                                        row += phase.rowOffset;
                                        col += phase.colOffset;
                                    }
                                } else {
                                    row += phase.rowOffset + (phase.kind == GqaBandPhaseKind::DIAGONAL ? y - 1 : 0);
                                    col += phase.colOffset;
                                }
                                ASSERT_GE(row, 1);
                                ASSERT_LE(row, m);
                                ASSERT_GE(col, 1);
                                ASSERT_LE(col, n);
                                ++visits[(row - 1) * n + col - 1];
                            }
                        }
                    }
                    for (int64_t x = 1; x <= m; ++x) {
                        for (int64_t y = 1; y <= n; ++y) {
                            EXPECT_EQ(visits[(x - 1) * n + y - 1], (x >= y - q + 1 && x <= y + p - 1) ? 1 : 0);
                        }
                    }
                }
            }
        }
    }
    EXPECT_GT(qBoundCases, 0);
    EXPECT_GT(pBoundCases, 0);
    // Column ranges can be disjoint while the paired row ranges still overlap.
    // This shape requires 11 valid columns, but only has 9: reject it for dQ safety.
    GqaBandPlan rejected;
    EXPECT_FALSE(BuildGqaBandGeometryByMode(6, 9, 1, 4, GqaBandGeometryMode::GENERAL_PAIRED_EDGES, rejected));
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandGeneralPairedEdgesMixedTieAndReset)
{
    using namespace optiling::fag;
    GqaBandPlan plan;
    BuildGqaBandCandidate(32, 31, 32, 2, 2, 33, 3, 21, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL_PAIRED_EDGES);
    EXPECT_EQ(plan.tailGeometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.fullRounds, 187);
    EXPECT_EQ(plan.tailRounds, 111);
    EXPECT_EQ(plan.totalRounds, 298);
    EXPECT_EQ(plan.phaseCount, 3);
    EXPECT_EQ(plan.tailPhaseCount, 3);

    BuildGqaBandCandidate(7, 73, 78, 5, 7, 5, 2, 4, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::REVERSED_GENERAL);
    EXPECT_EQ(plan.tailGeometryMode, GqaBandGeometryMode::GENERAL_PAIRED_EDGES);
    EXPECT_EQ(plan.fullRounds, 910);
    EXPECT_EQ(plan.tailRounds, 246);
    EXPECT_EQ(plan.totalRounds, 1156);
    EXPECT_EQ(plan.phaseCount, 5);
    EXPECT_EQ(plan.tailPhaseCount, 4);

    // A new uniform candidate ties the old mixed plan exactly. Retain the old
    // full/tail geometries, not merely the same total number of rounds.
    GqaBandPlan uniform;
    ASSERT_TRUE(BuildGqaBandGeometryByMode(62, 67, 32, 6, GqaBandGeometryMode::GENERAL_PAIRED_EDGES, uniform));
    SetGqaBandGroups(28, 29, 2, 26, uniform);
    EXPECT_EQ(uniform.fullRounds, 3391);
    EXPECT_EQ(uniform.tailRounds, 478);
    EXPECT_EQ(uniform.totalRounds, 3869);
    BuildGqaBandCandidate(28, 62, 67, 32, 6, 29, 2, 26, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.tailGeometryMode, GqaBandGeometryMode::REVERSED_GENERAL);
    EXPECT_EQ(plan.fullRounds, uniform.fullRounds);
    EXPECT_EQ(plan.tailRounds, uniform.tailRounds);
    EXPECT_EQ(plan.totalRounds, uniform.totalRounds);
    EXPECT_EQ(plan.phaseCount, 4);
    EXPECT_EQ(plan.tailPhaseCount, 4);

    BuildGqaBandCandidate(4, 53, 54, 1, 3, 5, 2, 5, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL_PAIRED_EDGES);
    EXPECT_EQ(plan.phases[0].kind, GqaBandPhaseKind::EDGE_PAIR_P_BOUND);
    EXPECT_EQ(plan.totalRounds, 398);
    EXPECT_EQ(plan.tailPhaseCount, 0);
    EXPECT_EQ(plan.tailRounds, 0);
    BuildGqaBandCandidate(32, 128, 128, 32, 32, 1, 16, 1, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::PAIRED_EDGES);
    EXPECT_EQ(plan.phases[0].kind, GqaBandPhaseKind::EDGE_PAIR);
    EXPECT_EQ(plan.totalRounds, 3536);
    EXPECT_EQ(plan.tailPhaseCount, 0);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandDenseBatchMajorGuards)
{
    using namespace optiling::fag;
    struct GuardCase {
        int64_t rowDelta;
        int64_t colDelta;
        int64_t groups;
        GqaBandPhaseKind kind;
        bool enabled;
    };
    const GuardCase cases[] = {
        {0, 0, 2, GqaBandPhaseKind::RECTANGLE, true},   {-1, 0, 2, GqaBandPhaseKind::RECTANGLE, false},
        {0, -1, 2, GqaBandPhaseKind::RECTANGLE, false}, {0, 1, 2, GqaBandPhaseKind::RECTANGLE, false},
        {0, 0, 1, GqaBandPhaseKind::RECTANGLE, false},  {0, 0, 2, GqaBandPhaseKind::DIAGONAL, false},
        {0, 0, 2, GqaBandPhaseKind::EDGE_PAIR, false},  {0, 0, 2, GqaBandPhaseKind::UPPER_TRIANGLE, false},
    };
    for (int64_t k : {28, 32, 36}) {
        for (const auto& testCase : cases) {
            GqaBandPlan plan;
            AppendGqaBandPhase(plan, testCase.kind, k + testCase.rowDelta, 2 * k + testCase.colDelta, 3, 5);
            SetGqaBandGroups(k, testCase.groups, 2, testCase.groups, plan);
            const int64_t rounds = plan.totalRounds;
            ConfigureGqaBandDensePhases(plan);
            EXPECT_EQ(plan.phases[0].kind, testCase.enabled ? GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR : testCase.kind);
            EXPECT_EQ(plan.totalRounds, rounds);
            EXPECT_EQ(plan.phases[0].rowOffset, 3);
            EXPECT_EQ(plan.phases[0].colOffset, 5);
            EXPECT_EQ(plan.tailPhaseCount, 0);
            EXPECT_EQ(plan.tailRounds, 0);
        }
        // n == k has no repeated column wave and must retain the old mode.
        GqaBandPlan singleWave;
        AppendGqaBandPhase(singleWave, GqaBandPhaseKind::RECTANGLE, k, k, 0, 0);
        SetGqaBandGroups(k, 2, 2, 2, singleWave);
        ConfigureGqaBandDensePhases(singleWave);
        EXPECT_EQ(singleWave.phases[0].kind, GqaBandPhaseKind::RECTANGLE);
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandDenseBatchMajorFullTailAndReset)
{
    using namespace optiling::fag;
    for (int64_t b : {4, 5, 6}) {
        GqaBandPlan plan;
        BuildGqaBandGeneralGeometry(64, 96, 64, 65, plan);
        SetGqaBandGroups(32, b, 2, 4, plan);
        const int64_t rounds = plan.totalRounds;
        const int64_t fullRounds = plan.fullRounds;
        const int64_t tailRounds = plan.tailRounds;
        const int64_t tailGroups = b % 4;
        if (tailGroups > 0) {
            GqaBandPlan tail;
            BuildGqaBandGeneralGeometry(64, 96, 64, 65, tail);
            SetGqaBandGroups(32, tailGroups, 2, tailGroups, tail);
            plan.tailPhaseCount = tail.phaseCount;
            for (int64_t i = 0; i < tail.phaseCount; ++i) {
                plan.tailPhases[i] = tail.phases[i];
                plan.tailPhases[i].tailRounds = tail.phases[i].fullRounds;
            }
        }
        ConfigureGqaBandDensePhases(plan);
        ASSERT_EQ(plan.phases[0].kind, GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR);
        EXPECT_EQ(plan.totalRounds, rounds);
        EXPECT_EQ(plan.fullRounds, fullRounds);
        EXPECT_EQ(plan.tailRounds, tailRounds);
        EXPECT_EQ(plan.phases[0].fullRounds % plan.groupsPerChunk, 0);
        if (tailGroups > 0) {
            EXPECT_EQ(plan.tailPhases[0].kind,
                      tailGroups > 1 ? GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR : GqaBandPhaseKind::RECTANGLE);
        } else {
            EXPECT_EQ(plan.tailPhaseCount, 0);
        }
        // A reused plan must not keep a batch-major tag after its group count changes.
        plan.groupsPerChunk = 1;
        ConfigureGqaBandDensePhases(plan);
        EXPECT_EQ(plan.phases[0].kind, GqaBandPhaseKind::RECTANGLE);
        if (plan.tailPhaseCount > 0) {
            EXPECT_EQ(plan.tailPhases[0].kind, GqaBandPhaseKind::RECTANGLE);
        }
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandDenseBatchMajorCandidateRounds)
{
    using namespace optiling::fag;
    GqaBandPlan plan;
    GqaBandPlan previous;
    // The complete chunks share a tagged geometry with a one-group tail.
    BuildGqaBandCandidate(32, 64, 96, 64, 65, 5, 2, 2, plan);
    BuildGqaBandUniformCandidateReference(32, 64, 96, 64, 65, 5, 2, 2, previous);
    ASSERT_EQ(plan.phaseCount, 3);
    EXPECT_EQ(plan.phases[0].kind, GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR);
    EXPECT_EQ(previous.phases[0].kind, GqaBandPhaseKind::RECTANGLE);
    EXPECT_EQ(plan.geometryMode, previous.geometryMode);
    EXPECT_EQ(plan.tailPhaseCount, 0);
    EXPECT_EQ(plan.fullRounds, 706);
    EXPECT_EQ(plan.tailRounds, 384);
    EXPECT_EQ(plan.totalRounds, 1796);
    EXPECT_EQ(plan.totalRounds, previous.totalRounds);
    for (int64_t i = 0; i < plan.phaseCount; ++i) {
        EXPECT_EQ(plan.phases[i].m, previous.phases[i].m);
        EXPECT_EQ(plan.phases[i].n, previous.phases[i].n);
        EXPECT_EQ(plan.phases[i].rowOffset, previous.phases[i].rowOffset);
        EXPECT_EQ(plan.phases[i].colOffset, previous.phases[i].colOffset);
        EXPECT_EQ(plan.phases[i].fullRounds, previous.phases[i].fullRounds);
        EXPECT_EQ(plan.phases[i].tailRounds, previous.phases[i].tailRounds);
    }
    // This last rectangle shares physical columns with the preceding triangle.
    BuildGqaBandCandidate(32, 184, 215, 33, 116, 2, 4, 2, plan);
    ASSERT_EQ(plan.phaseCount, 5);
    EXPECT_EQ(plan.phases[4].kind, GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR);
    EXPECT_EQ(plan.totalRounds, 5820);
    EXPECT_EQ(plan.tailRounds, 0);
    // Independent full/tail geometries require configuring both physical phase arrays.
    BuildGqaBandCandidate(28, 69, 120, 44, 82, 9, 4, 7, plan);
    EXPECT_EQ(plan.geometryMode, GqaBandGeometryMode::GENERAL);
    EXPECT_EQ(plan.tailGeometryMode, GqaBandGeometryMode::REVERSED_GENERAL);
    ASSERT_EQ(plan.phaseCount, 5);
    ASSERT_EQ(plan.tailPhaseCount, 5);
    EXPECT_EQ(plan.phases[2].kind, GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR);
    EXPECT_EQ(plan.tailPhases[2].kind, GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR);
    EXPECT_EQ(plan.fullRounds, 7365);
    EXPECT_EQ(plan.tailRounds, 2126);
    EXPECT_EQ(plan.totalRounds, 9491);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandHeadInnerRoundFormulaAndBijection)
{
    using namespace optiling::fag;
    // Main case: H=33, G=16. All cores use the same source round, including the wraparound.
    EXPECT_EQ(GqaBandHeadInnerRound(1, 33, 16, 1), 2);
    EXPECT_EQ(GqaBandHeadInnerRound(16, 33, 16, 1), 497);
    EXPECT_EQ(GqaBandHeadInnerRound(17, 33, 16, 1), 3);
    EXPECT_EQ(GqaBandHeadInnerRound(513, 33, 16, 1), 1);
    EXPECT_EQ(GqaBandHeadInnerRound(528, 33, 16, 1), 496);
    EXPECT_EQ(GqaBandHeadInnerRound(529, 33, 16, 1), 530);

    for (const int64_t span : {1, 2, 5, 33}) {
        for (const int64_t g : {1, 2, 3, 16}) {
            for (const int64_t shift : {int64_t(0), int64_t(1), span - 1}) {
                SCOPED_TRACE(::testing::Message() << "H=" << span << ", G=" << g << ", shift=" << shift);
                const int64_t period = span * g;
                const int64_t rounds = 3 * period;
                std::vector<bool> seen(rounds, false);
                for (int64_t r = 1; r <= rounds; ++r) {
                    const int64_t block = (r - 1) / period;
                    const int64_t local = (r - 1) % period;
                    const int64_t expected = block * period + (local % g) * span + (local / g + shift) % span + 1;
                    const int64_t source = GqaBandHeadInnerRound(r, span, g, shift);
                    ASSERT_EQ(source, expected);
                    ASSERT_GE(source, 1);
                    ASSERT_LE(source, rounds);
                    EXPECT_EQ((source - 1) / period, block);
                    EXPECT_FALSE(seen[source - 1]);
                    seen[source - 1] = true;
                    const int64_t sourceLocal = (source - 1) % period;
                    const int64_t spatial = (sourceLocal % span + span - shift % span) % span;
                    EXPECT_EQ(block * period + spatial * g + sourceLocal / span + 1, r);
                }
                for (const bool visited : seen) {
                    EXPECT_TRUE(visited);
                }
            }
        }
    }
    EXPECT_EQ(GqaBandHeadInnerRound(0, 33, 16, 1), 0);
    EXPECT_EQ(GqaBandHeadInnerRound(-1, 33, 16, 1), -1);
    EXPECT_EQ(GqaBandHeadInnerRound(7, 0, 16, 1), 7);
    EXPECT_EQ(GqaBandHeadInnerRound(7, 33, 0, 1), 7);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandTriangleHeadInnerSectionBoundaries)
{
    using namespace optiling::fag;
    struct Section {
        int64_t repeats;
        int64_t span;
        int64_t shift;
        bool transpose;
    };
    struct TriangleCase {
        int64_t k;
        int64_t size;
        int64_t groups;
        int64_t g;
        std::vector<Section> sections;
    };
    // Independently enumerated by FAG_Band_GQA_0906_v2.py. Each span excludes G.
    // Cover rm1/full groups, repeated rm2/pairs, rm3/odd group and unchanged rm4/ell tail.
    const TriangleCase cases[] = {
        {8, 17, 8, 3, {{1, 153, 0, true}}},
        {8, 17, 17, 3, {{2, 153, 0, true}, {1, 19, 0, true}, {1, 1, 0, false}}},
        {8, 19, 3, 3, {{2, 23, 0, true}, {1, 23, 0, true}, {1, 3, 0, false}}},
        {8, 9, 7, 3, {{3, 11, 2, true}, {1, 9, 0, true}, {1, 1, 0, false}}},
        {8, 24, 3, 3, {{3, 25, 0, true}, {1, 41, 0, true}}},
        {8, 7, 3, 3, {{1, 14, 0, false}}},
        {32, 32, 2, 16, {{1, 33, 1, true}}},
        {28, 32, 2, 16, {{1, 37, 5, true}, {1, 5, 0, false}}},
        {36, 32, 2, 16, {{1, 33, 0, false}}},
    };
    for (const auto& testCase : cases) {
        SCOPED_TRACE(::testing::Message() << "k=" << testCase.k << ", size=" << testCase.size
                                          << ", groups=" << testCase.groups << ", G=" << testCase.g);
        const int64_t rounds = GqaBandTriangleRounds(testCase.k, testCase.size, testCase.groups, testCase.g);
        std::vector<bool> seen(rounds, false);
        int64_t offset = 0;
        for (const auto& section : testCase.sections) {
            const int64_t period = section.span * testCase.g;
            const int64_t count = section.repeats * period;
            for (int64_t r = 1; r <= count; ++r) {
                const int64_t local = (r - 1) % period;
                const int64_t expectedLocal = section.transpose ?
                                                  (r - 1) / period * period + (local % testCase.g) * section.span +
                                                      (local / testCase.g + section.shift) % section.span + 1 :
                                                  r;
                const int64_t source =
                    GqaBandTriangleSourceRound(testCase.k, testCase.size, testCase.groups, testCase.g, offset + r);
                ASSERT_EQ(source, offset + expectedLocal);
                ASSERT_GT(source, offset);
                ASSERT_LE(source, offset + count);
                ASSERT_LE(source, rounds);
                EXPECT_FALSE(seen[source - 1]);
                seen[source - 1] = true;
            }
            offset += count;
        }
        EXPECT_EQ(offset, rounds);
        for (const bool visited : seen) {
            EXPECT_TRUE(visited);
        }
        EXPECT_EQ(GqaBandTriangleSourceRound(testCase.k, testCase.size, testCase.groups, testCase.g, rounds + 1),
                  rounds + 1);
    }
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandTriangleHeadInnerBijectionAndGuards)
{
    using namespace optiling::fag;
    for (const int64_t k : {2, 3, 4}) {
        for (int64_t size = 1; size <= 2 * k + 1; ++size) {
            for (int64_t groups = 1; groups <= k + 1; ++groups) {
                for (const int64_t g : {2, 3}) {
                    SCOPED_TRACE(::testing::Message()
                                 << "k=" << k << ", size=" << size << ", groups=" << groups << ", G=" << g);
                    const int64_t rounds = GqaBandTriangleRounds(k, size, groups, g);
                    std::vector<bool> seen(rounds, false);
                    for (int64_t r = 1; r <= rounds; ++r) {
                        const int64_t source = GqaBandTriangleSourceRound(k, size, groups, g, r);
                        ASSERT_GE(source, 1);
                        ASSERT_LE(source, rounds);
                        EXPECT_FALSE(seen[source - 1]);
                        seen[source - 1] = true;
                    }
                    for (const bool visited : seen) {
                        EXPECT_TRUE(visited);
                    }
                }
            }
        }
    }
    for (int64_t r = 1; r <= GqaBandTriangleRounds(32, 32, 2, 1); ++r) {
        EXPECT_EQ(GqaBandTriangleSourceRound(32, 32, 2, 1, r), r);
    }
    EXPECT_EQ(GqaBandTriangleSourceRound(32, 32, 2, 16, 0), 0);
    EXPECT_EQ(GqaBandTriangleSourceRound(32, 32, 2, 16, -1), -1);
    EXPECT_EQ(GqaBandTriangleSourceRound(0, 32, 2, 16, 7), 7);
    EXPECT_EQ(GqaBandTriangleSourceRound(32, 0, 2, 16, 7), 7);
    EXPECT_EQ(GqaBandTriangleSourceRound(32, 32, 0, 16, 7), 7);
    EXPECT_EQ(GqaBandTriangleSourceRound(32, 32, 2, 0, 7), 7);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandEdgePairRoundBijectionAndGuards)
{
    using namespace optiling::fag;
    const GqaBandPhase phases[] = {
        {GqaBandPhaseKind::EDGE_PAIR, 23, 8, 41, 56},
        {GqaBandPhaseKind::EDGE_PAIR_Q_BOUND, 48, 16, 16, 63},
        {GqaBandPhaseKind::EDGE_PAIR_P_BOUND, 10, 1, 0, 7},
    };
    const int64_t splits[] = {7, 31, 2};
    for (size_t i = 0; i < sizeof(phases) / sizeof(phases[0]); ++i) {
        const auto& phase = phases[i];
        ASSERT_TRUE(IsGqaBandEdgePair(phase));
        EXPECT_EQ(GqaBandEdgePairSplit(phase), splits[i]);
        for (const int64_t k : {28, 32, 36}) {
            for (const int64_t groups : {int64_t(1), k / phase.n, k / phase.n + 1}) {
                for (const int64_t g : {1, 2, 16}) {
                    SCOPED_TRACE(::testing::Message()
                                 << "phase=" << i << ", k=" << k << ", groups=" << groups << ", G=" << g);
                    const bool enabled = g > 1 && groups * phase.n <= k;
                    EXPECT_EQ(CanReorderGqaBandEdgePair(phase, k, groups, g), enabled);
                    const int64_t rounds = GqaBandPhaseRounds(phase, k, groups, g);
                    std::vector<bool> seen(rounds, false);
                    for (int64_t r = 1; r <= rounds; ++r) {
                        const int64_t expected =
                            enabled ? ((r - 1) % g) * phase.m + ((r - 1) / g + splits[i] + 1) % phase.m + 1 : r;
                        const int64_t source = GqaBandEdgePairSourceRound(phase, k, groups, g, r);
                        ASSERT_EQ(source, expected);
                        ASSERT_GE(source, 1);
                        ASSERT_LE(source, rounds);
                        ASSERT_FALSE(seen[source - 1]);
                        seen[source - 1] = true;
                    }
                    for (const int64_t r : {int64_t(-1), int64_t(0), rounds + 1, rounds + 2}) {
                        EXPECT_EQ(GqaBandEdgePairSourceRound(phase, k, groups, g, r), r);
                    }
                }
            }
        }
        EXPECT_FALSE(CanReorderGqaBandEdgePair(phase, 0, 1, 2));
        EXPECT_FALSE(CanReorderGqaBandEdgePair(phase, 32, 0, 2));
        EXPECT_FALSE(CanReorderGqaBandEdgePair(phase, 32, 1, 0));
    }
    auto phase = phases[0];
    // Full chunk rejects (40 virtual columns), while its 4-group tail accepts (32).
    EXPECT_FALSE(CanReorderGqaBandEdgePair(phase, 32, 5, 16));
    EXPECT_TRUE(CanReorderGqaBandEdgePair(phase, 32, 4, 16));
    phase.kind = GqaBandPhaseKind::RECTANGLE;
    EXPECT_FALSE(CanReorderGqaBandEdgePair(phase, 32, 4, 16));
    EXPECT_EQ(GqaBandEdgePairSourceRound(phase, 32, 4, 16, 17), 17);
    phase = phases[1];
    for (const int64_t split : {int64_t(-1), int64_t(0), phase.m - phase.n - 1, phase.m - phase.n}) {
        phase.colOffset = phase.rowOffset + phase.n + split;
        EXPECT_EQ(CanReorderGqaBandEdgePair(phase, 32, 2, 16), split >= 0 && split < phase.m - phase.n);
    }
    phase.n = 0;
    EXPECT_FALSE(CanReorderGqaBandEdgePair(phase, 32, 2, 16));
    phase.n = phase.m + 1;
    EXPECT_FALSE(CanReorderGqaBandEdgePair(phase, 32, 2, 16));
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandLocalityCostBounds)
{
    using namespace optiling::fag;
    EXPECT_EQ(GqaBandGcd(128, 48), 16);
    EXPECT_EQ(GqaBandGcd(64, 0), 64);
    EXPECT_EQ(GqaBandDenseKvRunUpperBound(0, 3, 28, 16, 8, true), -1);
    EXPECT_EQ(GqaBandDenseKvRunUpperBound(32, 0, 28, 16, 8, true), -1);
    EXPECT_EQ(GqaBandDenseKvRunUpperBound(32, 3, 0, 16, 8, true), -1);
    EXPECT_EQ(GqaBandDenseKvRunUpperBound(32, 3, 28, 0, 8, true), -1);
    EXPECT_EQ(GqaBandDenseKvRunUpperBound(32, 3, 28, 16, 0, true), -1);
    EXPECT_EQ(GqaBandDenseKvRunUpperBound(1, 1, 1, 1, std::numeric_limits<uint32_t>::max(), false), 1);

    // Aligned core boundaries do not split a physical KV column; unaligned boundaries do.
    EXPECT_EQ(GqaBandDenseKvRunUpperBound(32, 3, 28, 16, 8, true), 448);
    EXPECT_EQ(GqaBandDenseKvRunUpperBound(32, 32, 97, 1, 8, false), 125);
    // With Dense ID correction, do not assume that adjacent head IDs share a physical column.
    EXPECT_EQ(GqaBandDenseKvRunUpperBound(32, 3, 32, 16, 8, false), 4096);
    EXPECT_EQ(GqaBandDenseKvRunUpperBound(32, 3, 32, 16, 8, true), 512);

    GqaBandPhase phase{};
    EXPECT_EQ(GqaBandPhaseKvRunUpperBound(phase, 32, 2, 16), -1);
    phase.kind = GqaBandPhaseKind::LOWER_TRIANGLE;
    phase.m = phase.n = 32;
    EXPECT_EQ(GqaBandPhaseKvRunUpperBound(phase, 32, 2, 16), 64);
    phase.m = phase.n = 64;
    EXPECT_EQ(GqaBandPhaseKvRunUpperBound(phase, 32, 2, 16), 190);
    phase.kind = GqaBandPhaseKind::UPPER_TRIANGLE;
    EXPECT_EQ(GqaBandPhaseKvRunUpperBound(phase, 32, 2, 16), 190);
    phase.kind = GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR;
    phase.m = 32;
    phase.n = 97;
    EXPECT_EQ(GqaBandPhaseKvRunUpperBound(phase, 32, 3, 8), 375);
    phase.m = 95;
    phase.n = 32;
    phase.colOffset = 63;
    for (auto kind :
         {GqaBandPhaseKind::EDGE_PAIR, GqaBandPhaseKind::EDGE_PAIR_Q_BOUND, GqaBandPhaseKind::EDGE_PAIR_P_BOUND}) {
        phase.kind = kind;
        EXPECT_TRUE(CanReorderGqaBandEdgePair(phase, 32, 1, 16));
        EXPECT_EQ(GqaBandPhaseKvRunUpperBound(phase, 32, 1, 16), 64);
        EXPECT_FALSE(CanReorderGqaBandEdgePair(phase, 32, 2, 16));
        EXPECT_EQ(GqaBandPhaseKvRunUpperBound(phase, 32, 2, 16), 3072);
    }

    // Only the proven ordinary-Dense legacy branch permits a stronger bound than touched columns.
    EXPECT_EQ(GqaBandLegacyKvRunLowerBound(32, 32, 97, 32, 66, 1, 8), 125);
    EXPECT_EQ(GqaBandLegacyKvRunLowerBound(32, 32, 95, 32, 64, 1, 8), 95);
    // Folded/symmetric branches still use a lower bound, not the exact legacy submission count.
    // For this fixture the separately enumerated legacy cost is 539, whereas the bound is 512.
    EXPECT_EQ(GqaBandLegacyKvRunLowerBound(32, 32, 32, 2, 2, 16, 8), 512);

    GqaBandPlan plan{};
    EXPECT_EQ(GqaBandKvRunUpperBound(plan), -1);
    BuildGqaBandCandidate(32, 32, 64, 32, 33, 4, 16, 2, plan);
    EXPECT_EQ(GqaBandChunkKvRunUpperBound(plan, false), 128);
    EXPECT_EQ(GqaBandChunkKvRunUpperBound(plan, true), 0);
    EXPECT_EQ(GqaBandKvRunUpperBound(plan), 256);
    BuildGqaBandCandidate(32, 32, 64, 32, 33, 5, 16, 2, plan);
    ASSERT_EQ(plan.tailPhaseCount, 0);
    EXPECT_EQ(GqaBandChunkKvRunUpperBound(plan, true), 64);
    EXPECT_EQ(GqaBandKvRunUpperBound(plan), 320);
    BuildGqaBandCandidate(32, 32, 32, 16, 17, 11, 16, 7, plan);
    ASSERT_GT(plan.tailPhaseCount, 0);
    EXPECT_EQ(GqaBandChunkKvRunUpperBound(plan, false), 502);
    EXPECT_EQ(GqaBandChunkKvRunUpperBound(plan, true), 1216);
    EXPECT_EQ(GqaBandKvRunUpperBound(plan), 1718);
}

TEST_F(FlashAttentionScoreGradTiling, DeterGqaBandQueryLocalityChecksFullAndTail)
{
    using namespace optiling::fag;
    GqaBandPlan plan{};
    EXPECT_FALSE(HasGqaBandQueryLocality(plan));
    BuildGqaBandCandidate(32, 32, 32, 2, 2, 16, 8, 16, plan);
    // 0354 passes KV admission exactly, but the thin DIAGONAL has H*G % R != 0.
    EXPECT_EQ(GqaBandKvRunUpperBound(plan), GqaBandLegacyKvRunLowerBound(32, 32, 32, 2, 2, 16, 8));
    EXPECT_FALSE(HasGqaBandQueryLocality(plan));

    plan = GqaBandPlan{};
    plan.k = 32;
    plan.b = 4;
    plan.g = 2;
    plan.groupsPerChunk = 2;
    plan.phaseCount = 1;
    plan.phases[0].kind = GqaBandPhaseKind::DIAGONAL;
    plan.phases[0].m = 2;
    plan.phases[0].n = 7;
    EXPECT_TRUE(HasGqaBandQueryLocality(plan)); // Full H=2: H*G=4, R=4.
    plan.b = 3;
    EXPECT_FALSE(HasGqaBandQueryLocality(plan)); // Shared tail H=1: H*G=2, R=4.
    plan.tailPhaseCount = 1;
    plan.tailPhases[0] = plan.phases[0];
    plan.tailPhases[0].n = 2;
    EXPECT_TRUE(HasGqaBandQueryLocality(plan)); // Independent tail has R=2 instead.
    plan.phases[0].n = 2;
    plan.tailPhases[0].n = 7;
    EXPECT_FALSE(HasGqaBandQueryLocality(plan)); // Check independent tail, not full phases twice.

    plan.b = 2; // An unused tail must not reject a complete full chunk.
    EXPECT_TRUE(HasGqaBandQueryLocality(plan));
    plan.phases[0].n = 7;
    plan.groupsPerChunk = plan.b = 1;
    EXPECT_FALSE(HasGqaBandQueryLocality(plan));
    plan.phases[0].m = 1;
    EXPECT_TRUE(HasGqaBandQueryLocality(plan)); // No cross-column reuse of a Q row.
    plan.phases[0].m = plan.k;
    EXPECT_TRUE(HasGqaBandQueryLocality(plan)); // This guard is limited to thin strips.
    plan.phases[0].m = 2;
    plan.phases[0].n = 1;
    EXPECT_TRUE(HasGqaBandQueryLocality(plan));
    plan.phases[0].n = 7;
    plan.phases[0].kind = GqaBandPhaseKind::RECTANGLE;
    EXPECT_TRUE(HasGqaBandQueryLocality(plan));
}

static Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo MakeA5CompileInfo()
{
    return {64,       // aivNum
            32,       // aicNum
            196608,   // ubSize
            524288,   // l1Size
            65536,    // l0aSize
            65536,    // l0bSize
            131072,   // l0cSize
            33554432, // l2CacheSize
            32,       // coreNum
            platform_ascendc::SocVersion::ASCEND950};
}

static gert::TilingContextPara MakeMlaRopeTilingContext(
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo* compileInfo, int64_t qkD, int64_t valueD)
{
    return gert::TilingContextPara("FlashAttentionScoreGrad",
                                   {
                                       {{{1, 128, 1, qkD}, {1, 128, 1, qkD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{1, 128, 1, qkD}, {1, 128, 1, qkD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{1, 128, 1, valueD}, {1, 128, 1, valueD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{1, 128, 1, valueD}, {1, 128, 1, valueD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                       {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{1, 128, 1, valueD}, {1, 128, 1, valueD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                   },
                                   {
                                       {{{1, 128, 1, qkD}, {1, 128, 1, qkD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{1, 128, 1, qkD}, {1, 128, 1, qkD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{1, 128, 1, valueD}, {1, 128, 1, valueD}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                   },
                                   {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.072168f)},
                                    {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                    {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                                    {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                                    {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                    {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                    {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                    {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                    {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                    {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                    {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                    {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                    {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
                                   compileInfo, "Ascend950", A5SocInfo, 4096);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_0)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q: S1=256, B=1, H1=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask: S1=256, S2=256
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope and keyRope are omitted to cover nullptr optional inputs.
        },
        {
            // 输出Tensor
            // dq
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577349537174544;
    std::string expectTilingData =
        "32 1 1 1 256 256 128 128 4575657222443697349 255 65536 65536 0 0 0 2 1 0 0 2 1099511627776 0 0 0 0 2 "
        "549755813952 549755813952 2 549755814016 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16384 8192 4 8192 "
        "8192 4 8192 8192 4 8192 0 0 0 4 61440 30720 114688 30720 1 16384 1 16384 1 0 0 0 0 0 0 0 0 0 0 1 32768 16384 "
        "16384 1 32768 16384 16384 1 32768 16384 16384 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_1)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dv
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(45)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146416;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21562880};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_2)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{1, 1}, {1, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dv
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724962;
    std::string expectTilingData =
        "32 1 1 1 1 1 1 1 4575657222473777152 255 65536 0 2 0 0 2 1 0 0 2 4294967296 281474976710656 0 0 0 1 "
        "4294967360 4294967297 1 4294967424 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 1 1 1 1 1 1 1 1 1 0 0 0 "
        "1 61440 30720 114688 30720 1 32 1 32 1 0 0 0 0 0 0 0 0 0 0 1 1 16384 1 1 1 16384 1 1 1 16384 1 65536 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_3)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08804509063256238f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577350074044432;
    std::string expectTilingData =
        "32 1 1 1 64 128 129 129 4575657222443651324 255 2147483647 2147483647 2 0 0 3 1 0 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 549755814016 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8192 8256 1 8256 "
        "16512 1 16512 16512 1 16512 0 0 0 1 61440 30720 114688 30720 1 8192 1 8192 1 0 0 0 0 0 0 0 0 0 0 1 8256 16384 "
        "8256 1 16512 16384 128 1 16512 16384 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_4)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{2, 8, 2000, 8}, {2, 8, 2000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{2, 8, 2000, 8}, {2, 8, 2000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dv
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dq_rope
            {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk_rope
            {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 29132661700694064;
    std::string expectTilingData =
        "32 2 8 1 2000 2000 192 128 4575657222441520442 255 2147483647 2147483647 2 0 256 1 1 0 0 0 0 "
        "72057594037927936 0 0 0 16 549755813952 343597383696 16 343597383808 549755813920 0 0 128 256 384 512 640 768 "
        "896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 "
        "3712 3840 3968 0 0 0 0 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 "
        "2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096 0 0 0 0 2000000 96000 64 96000 96000 64 "
        "96000 64000 64 64000 0 0 0 64 61440 30720 114688 30720 66 3200 600479950314048 27776 1 242442313924672 "
        "20203526216832 147 4 4 59 59 500 500 32000 6 6144000 18432 6144 6 6144000 18432 6144 4 4096000 18432 4096 "
        "65536 24678912 49292288 0 65701376 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {87698944};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_5)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{64, 16, 123, 8}, {64, 16, 123, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{64, 16, 123, 8}, {64, 16, 123, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk
            {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // dv
            {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.05070201265633938f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140301101207586;
    std::string expectTilingData =
        "32 64 16 1 123 64 389 389 4575657222437055722 255 2147483647 2147483647 2 0 0 1 1 0 0 0 0 0 0 0 0 1 "
        "528280977472 528280977467 1 274877907072 137438953504 0 0 32 64 96 128 160 192 224 256 288 320 352 384 416 "
        "448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 64 96 128 160 192 224 256 "
        "288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024 0 0 0 0 "
        "251904 1531104 32 1531104 796672 32 796672 796672 32 796672 0 0 0 32 61440 30720 114688 30720 9 6144 9 6144 1 "
        "0 0 0 0 0 0 0 0 0 0 47 48995328 16384 7168 25 25493504 16384 16384 25 25493504 16384 16384 65536 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {49349120};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_6)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{3, 1, 128, 8}, {3, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{3, 1, 128, 8}, {3, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dv
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140299490594850;
    std::string expectTilingData =
        "32 3 1 1 128 121 128 128 4575657222443697395 255 2147483647 2147483647 2 0 0 1 1 0 0 0 0 0 0 0 0 1 "
        "549755813952 549755813952 1 519691042944 4294967299 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 15488 16384 3 16384 "
        "15488 3 15488 15488 3 15488 0 0 0 3 61440 30720 114688 30720 1 15488 1 15488 1 0 0 0 0 0 0 0 0 0 0 1 49152 "
        "16384 16384 1 46464 16384 13696 1 46464 16384 13696 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {25756160};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_7)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 73, 8}, {1, 1, 73, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 73, 8}, {1, 1, 73, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk
            {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // dv
            {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07071067811865475f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140300564337952;
    std::string expectTilingData =
        "32 1 1 1 73 4096 200 200 4561245704492732611 204 2147483647 0 2 0 0 1 1 0 0 8589934594 8796093022209 "
        "72339069014638592 0 0 0 1 313532612672 313532612617 32 549755814016 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 299008 7300 2 7300 409600 2 409600 409600 2 409600 0 0 0 2 61440 30720 114688 30720 10 22528 0 30720 "
        "1099511627777 244091581366274 15255723892224 111 1 1 37 36 37 36 73 1 14600 16384 14600 25 819200 16384 16384 "
        "25 819200 16384 16384 65536 168448 3445760 0 6723072 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {27699200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_8)
{
    int64_t actual_seq_qlist[3] = {13, 14, 81};
    int64_t actual_seq_kvlist[3] = {13, 14, 81};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0625f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577350610917264;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_9)
{
    int64_t actual_seq_qlist[3] = {13, 14, 81};
    int64_t actual_seq_kvlist[3] = {13, 14, 81};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dv
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(70)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703249444018098;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {25756160};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_10)
{
    int64_t actual_seq_qlist[3] = {13, 14, 81};
    int64_t actual_seq_kvlist[3] = {13, 14, 81};
    int64_t prefix_list[3] = {10, 12, 60};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0625f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577350610917264;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_11_tnd_same_as_input)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(45)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146416;
    std::string expectTilingData =
        "32 4 1 2 384 384 32 32 4575657222452086003 255 45 2 2 0 0 4294967300 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 3 549755813952 549755813952 3 549755814016 4294967328 0 0 1 2 3 4 5 6 7 8 9 10 11 13 "
        "14 15 17 18 19 20 22 23 24 26 27 28 29 30 31 32 33 34 35 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 13 14 15 17 18 19 "
        "20 22 23 24 26 27 28 29 30 31 32 33 34 35 36 0 0 0 0 16992 1948 32 1948 974 32 974 974 32 974 0 0 0 32 "
        "61440 30720 114688 30720 1 16992 1 16896 1 0 0 0 0 0 0 0 0 0 0 1 62336 16384 13184 1 31168 16384 14784 1 "
        "31168 16384 14784 65536 328192 459776 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 "
        "3 3 3 3 3 3 0 0 0 0 0 0 16384 16384 16384 16384 16384 16384 16384 16384 81920 81920 81920 81920 81920 81920 "
        "81920 81920 81920 81920 81920 81920 81920 81920 229376 229376 229376 229376 229376 229376 229376 229376 0 0 "
        "0 0 0 0 16384 16384 16384 16384 16384 16384 16384 16384 81920 81920 81920 81920 81920 81920 81920 81920 "
        "81920 81920 81920 81920 81920 81920 229376 229376 229376 229376 229376 229376 229376 229376 0 0 0 0 0 0 1 1 "
        "1 1 1 1 1 1 5 5 5 5 5 5 5 5 5 5 5 5 5 5 14 14 14 14 14 14 14 14 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21562880};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_12_pad_causal_deter)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19182080395580448;
    std::string expectTilingData =
        "32 1 1 1 256 256 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 2 549755813952 549755813952 2 549755814016 4294967299 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 21856 5462 3 5460 5462 3 5460 5462 3 5460 0 0 0 3 61440 30720 114688 30720 1 21856 1 21824 1 0 0 0 0 "
        "0 0 0 0 0 0 1 16384 16384 16384 1 16384 16384 16384 1 16384 16384 16384 65536 131584 197632 0 0 0 0 0 0 0 1 "
        "3 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21235200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_13_pad_band_deter)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19747229372257328;
    std::string expectTilingData =
        "32 1 1 1 256 256 64 64 4575657222448611328 255 64 64 0 0 0 1 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 2 549755813952 549755813952 2 549755814016 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 16384 4096 4 4096 4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 16384 1 16384 1 0 0 0 0 "
        "0 0 0 0 0 0 1 16384 16384 16384 1 16384 16384 16384 1 16384 16384 16384 65536 131584 197632 0 0 0 0 0 0 0 1 "
        "2 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21235200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_15_pad_prefix5)
{
    int64_t prefix_list[2] = {16, 16};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146288;
    std::string expectTilingData =
        "32 2 2 1 256 256 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 21474836482 1099511627776 "
        "844424930131968 0 0 0 2 549755813952 549755813952 2 549755814016 4294967308 0 0 1 2 4 5 6 8 9 10 12 13 14 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 4 5 6 8 9 10 12 13 14 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 21856 5462 12 5454 5462 12 5454 5462 12 5454 0 0 0 12 61440 30720 114688 30720 1 21856 1 "
        "21728 1 0 0 0 0 0 0 0 0 0 0 1 65536 16384 16384 1 65536 16384 16384 1 65536 16384 16384 65536 328192 590848 "
        "0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_16_pad_prefix6)
{
    int64_t prefix_list[2] = {16, 24};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724960;
    std::string expectTilingData =
        "32 2 2 1 256 256 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 25769803778 8796093022212 "
        "844424930131968 0 0 0 2 549755813952 549755813952 2 549755814016 4294967308 0 0 1 2 4 5 6 8 9 10 12 13 14 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 4 5 6 8 9 10 12 13 14 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 21856 5462 12 5454 5462 12 5454 5462 12 5454 0 0 0 12 61440 30720 114688 30720 1 21856 1 "
        "21728 1 0 0 0 0 0 0 0 0 0 0 1 65536 16384 16384 1 65536 16384 16384 1 65536 16384 16384 65536 328192 590848 "
        "0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_17_tnd_sparse7)
{
    int64_t actual_seq_qlist[3] = {0, 32, 64};
    int64_t actual_seq_kvlist[3] = {0, 32, 64};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{64, 2, 8}, {64, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 2, 8}, {64, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146416;
    std::string expectTilingData =
        "32 3 2 1 32 32 64 64 4575657222448611328 255 16 16 0 0 0 4 1 0 0 30064771074 8796093022213 844424930131968 "
        "0 0 0 1 137438953536 137438953504 1 137438953600 4294967300 2 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 2048 "
        "4 2048 2048 4 2048 2048 4 2048 0 0 0 4 61440 30720 114688 30720 1 1024 1 1024 4294967297 0 0 0 0 0 0 0 0 0 "
        "0 1 8192 16384 8192 1 8192 16384 8192 1 8192 16384 8192 65536 98816 132096 0 0 0 0 0 0 0 0 0 0 1 2 2 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 1024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 1024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21136896};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_18_tnd_sparse8_inner_pse)
{
    int64_t actual_seq_qlist[3] = {32, 64, 96};
    int64_t actual_seq_kvlist[3] = {32, 64, 96};
    int64_t q_start_list[1] = {0};
    int64_t kv_start_list[1] = {0};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{96, 2, 8}, {96, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 8}, {96, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, q_start_list},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, kv_start_list},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577349000304272;
    std::string expectTilingData =
        "32 3 2 1 32 32 64 64 4575657222448611328 255 16 16 0 0 0 4 25769803778 3 0 34359738370 8796093022214 "
        "844424930131968 0 0 0 1 137438953536 137438953504 1 137438953600 4294967302 0 0 1 2 3 4 5 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1024 2048 6 2048 2048 6 2048 2048 6 2048 0 0 0 6 61440 30720 114688 30720 1 1024 1 1024 1 0 0 0 0 0 0 "
        "0 0 0 0 1 12288 16384 12288 1 12288 16384 12288 1 12288 16384 12288 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 2 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 1024 2048 2048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 1024 2048 2048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 1 1 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_19_outer_pse_bn1s)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145778;
    std::string expectTilingData =
        "32 1 1 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 3 4294967297 1 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 1 4096 "
        "4096 1 4096 4096 1 4096 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 4096 16384 "
        "4096 1 4096 16384 4096 1 4096 16384 4096 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_20_inner_pse_bn)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1}, {2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 1, 64, 8}, {2, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 64, 8}, {2, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724450;
    std::string expectTilingData =
        "32 2 1 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 3 21474836482 2 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967298 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 2 4096 "
        "4096 2 4096 4096 2 4096 0 0 0 2 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 8192 16384 "
        "8192 1 8192 16384 8192 1 8192 16384 8192 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_21_dropout_dropmask)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 200, 1, 200}, {1, 200, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 200, 1, 200}, {1, 200, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1825}, {1825}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 73, 8}, {1, 1, 73, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 73, 8}, {1, 1, 73, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 200, 1, 200}, {1, 200, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 200, 1, 200}, {1, 200, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07071067811865475f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140300564336928;
    std::string expectTilingData =
        "32 1 1 1 73 200 200 200 4561245704492732611 204 2147483647 2147483647 2 0 0 1 1 0 0 0 0 72058693549555712 0 "
        "0 0 1 313532612672 313532612617 2 309237645440 4294967298 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7328 3650 4 "
        "3650 10000 4 10000 10000 4 10000 0 0 0 4 61440 30720 114688 30720 1 7328 0 23360 1 244091581366276 "
        "15255723892224 111 1 1 19 16 19 16 73 1 14600 16384 14600 1 40000 16384 7232 1 40000 16384 7232 65536 "
        "168448 373760 0 579072 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21555200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_22_bn2)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 4, 64, 8}, {1, 4, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 4, 64, 8}, {1, 4, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953723938;
    std::string expectTilingData =
        "32 1 4 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 274877907008 "
        "274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 4096 4 4096 "
        "4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 16384 1 "
        "16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_23_bn2s2)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953723936;
    std::string expectTilingData =
        "32 1 1 1 64 256 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 2 549755814016 4294967298 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8192 2048 2 2048 "
        "8192 2 8192 8192 2 8192 0 0 0 2 61440 30720 114688 30720 1 8192 1 8192 1 0 0 0 0 0 0 0 0 0 0 1 4096 16384 "
        "4096 1 16384 16384 16384 1 16384 16384 16384 65536 98816 164864 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21202432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_24_bn2_multiblk)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{16, 8, 256, 8}, {16, 8, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8, 256, 8}, {16, 8, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19773617651322930;
    std::string expectTilingData =
        "32 16 8 1 256 256 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 2 "
        "549755813952 549755813952 2 549755814016 68719476768 0 0 16 32 48 64 80 96 112 128 144 160 176 192 208 224 "
        "240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 0 0 0 0 16 32 48 64 80 96 112 128 144 "
        "160 176 192 208 224 240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512 0 0 0 0 262144 "
        "65536 32 65536 65536 32 65536 65536 32 65536 0 0 0 32 61440 30720 114688 30720 9 16384 9 16384 1 0 0 0 0 0 "
        "0 0 0 0 0 2 2097152 16384 16384 2 2097152 16384 16384 2 2097152 16384 16384 65536 2424832 3604480 0 0 0 0 0 "
        "0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {25755648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_25_tnd_all_same)
{
    int64_t actual_seq_qlist[3] = {64, 128, 192};
    int64_t actual_seq_kvlist[3] = {64, 128, 192};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{192, 2, 8}, {192, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{192, 2, 8}, {192, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145266;
    std::string expectTilingData =
        "32 3 2 1 64 64 32 32 4575657222452086003 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 274877907008 "
        "274877907008 1 274877907072 4294967302 0 0 1 2 3 4 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 2 3 4 5 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 2048 6 2048 2048 6 2048 "
        "2048 6 2048 0 0 0 6 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 12288 16384 12288 1 "
        "12288 16384 12288 1 12288 16384 12288 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_27_fp32_large_d)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.055901699437f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577351147784208;
    std::string expectTilingData =
        "32 1 1 1 64 64 320 320 4575657222438451502 255 2147483647 2147483647 0 0 0 3 1 0 0 0 0 0 0 0 0 1 "
        "274877906976 274877906976 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 20480 1 "
        "20480 20480 1 20480 20480 1 20480 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 "
        "20480 16384 4096 1 20480 16384 4096 1 20480 16384 4096 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_28_atten_mask_4d)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 64, 64}, {1, 2, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 2, 64, 8}, {1, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 64, 8}, {1, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 1 2 1 64 64 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 0 274877906944 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967298 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 2 4096 "
        "4096 2 4096 4096 2 4096 0 0 0 2 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 8192 16384 "
        "8192 1 8192 16384 8192 1 8192 16384 8192 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_29_tnd_outer_pse_1d)
{
    int64_t actual_seq_qlist[3] = {32, 64, 96};
    int64_t actual_seq_kvlist[3] = {32, 64, 96};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{96, 2, 8}, {96, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 8}, {96, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145778;
    std::string expectTilingData =
        "32 3 2 1 32 32 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 4294967297 1 0 0 0 0 0 0 0 1 "
        "137438953536 137438953504 1 137438953600 4294967302 0 0 1 2 3 4 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 2048 6 2048 "
        "2048 6 2048 2048 6 2048 0 0 0 6 61440 30720 114688 30720 1 1024 1 1024 1 0 0 0 0 0 0 0 0 0 0 1 12288 16384 "
        "12288 1 12288 16384 12288 1 12288 16384 12288 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_30_alibi_pse)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 1024}, {1, 1, 1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146800;
    std::string expectTilingData =
        "32 1 1 1 1024 1024 64 64 4575657222448611328 255 2147483647 0 0 0 0 3 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 8 549755813952 549755813952 8 549755814016 8589934610 0 0 2 4 6 9 11 13 15 19 21 23 "
        "28 30 36 38 45 47 55 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 6 9 11 13 15 19 21 23 28 30 36 38 45 47 55 64 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 58272 3641 18 3639 3641 18 3639 3641 18 3639 0 0 0 18 61440 30720 "
        "114688 30720 2 27552 2 27232 1 0 0 0 0 0 0 0 0 0 0 1 65536 16384 16384 1 65536 16384 16384 1 65536 16384 "
        "16384 65536 328192 590848 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_softmax_dim)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 2, 974, 8}, {1, 2, 974, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 974, 8}, {1, 2, 974, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(45)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_atten_mask_shape)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_dropmask_keepprob)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_fp8)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_pad_tokens)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_inner_pse_dim)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_tokens)
{
    // Prefix-sum must stay unequal so layout remains TND (all-same would rewrite to pad).
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_atten_mask_rank)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 64, 64}, {1, 1, 1, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_sparse5)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    int64_t prefix_list[4] = {8, 8, 8, 8};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_pad_sparse7)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_prefix6_empty_prefix)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_pse_type)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_softmax_max_dim)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 4}, {1, 1, 64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_dscale)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_atten_mask_4d_bn)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 5, 64, 64}, {3, 5, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 2, 64, 8}, {1, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 64, 8}, {1, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_31_all_mask)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 1 1 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 4294967298 274877906944 0 "
        "0 0 0 1 274877907008 274877907008 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 "
        "1 4096 4096 1 4096 4096 1 4096 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 4096 "
        "16384 4096 1 4096 16384 4096 1 4096 16384 4096 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_32_pad_right_down_causal)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 1 1 1 64 64 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 12884901890 8796093022210 "
        "281474976710656 0 0 0 1 274877907008 274877907008 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 4096 4096 1 4096 4096 1 4096 4096 1 4096 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 "
        "0 0 0 0 1 4096 16384 4096 1 4096 16384 4096 1 4096 16384 4096 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_33_prefix5_empty)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724960;
    std::string expectTilingData =
        "32 2 2 1 256 256 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 4294967298 1099511627776 0 0 0 0 2 "
        "549755813952 549755813952 2 549755814016 4294967312 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "16384 4096 16 4096 4096 16 4096 4096 16 4096 0 0 0 16 61440 30720 114688 30720 1 16384 1 16384 1 0 0 0 0 0 "
        "0 0 0 0 0 1 65536 16384 16384 1 65536 16384 16384 1 65536 16384 16384 65536 328192 590848 0 0 0 0 0 0 0 0 0 "
        "0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_34_outer_pse_bnss)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 64, 64}, {2, 2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145778;
    std::string expectTilingData =
        "32 2 2 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 274877907008 "
        "274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 4096 4 4096 "
        "4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 16384 1 "
        "16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_35_outer_pse_1nss)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 64, 64}, {1, 2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145778;
    std::string expectTilingData =
        "32 2 2 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 8589934593 0 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 "
        "4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 "
        "16384 1 16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_36_atten_mask_4d_b1)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 2 2 1 64 64 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 1 274877906944 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 "
        "4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 "
        "16384 1 16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_37_atten_mask_4d_11)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 2 2 1 64 64 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 2 274877906944 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 "
        "4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 "
        "16384 1 16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_38_qkv_start_idx)
{
    int64_t q_start_list[1] = {8};
    int64_t kv_start_list[1] = {4};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, q_start_list},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, kv_start_list},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145266;
    std::string expectTilingData =
        "32 1 1 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 17179869192 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 1 4096 "
        "4096 1 4096 4096 1 4096 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 4096 16384 "
        "4096 1 4096 16384 4096 1 4096 16384 4096 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_39_dropout_s2_not_div8)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 65, 8}, {1, 1, 65, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 65, 8}, {1, 1, 65, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145522;
    std::string expectTilingData =
        "32 1 1 1 65 65 64 64 4568451461326831616 229 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 279172874304 "
        "279172874241 1 279172874368 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4256 4160 1 4160 4160 1 4160 "
        "4160 1 4160 0 0 0 1 61440 30720 114688 30720 1 4256 1 4256 0 0 0 0 0 0 0 0 0 0 0 1 4160 16384 4160 1 4160 "
        "16384 4160 1 4160 16384 4160 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_40_pad_gqa)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 4, 64, 8}, {1, 4, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 4, 64, 8}, {1, 4, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145264;
    std::string expectTilingData =
        "32 1 2 2 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 274877907008 "
        "274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 2048 4 2048 "
        "2048 4 2048 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 16384 1 8192 "
        "16384 8192 1 8192 16384 8192 65536 147968 197632 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21218816};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_41_atten_mask_bool)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 1 1 1 64 64 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 2 274877906944 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 1 4096 "
        "4096 1 4096 4096 1 4096 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 4096 16384 "
        "4096 1 4096 16384 4096 1 4096 16384 4096 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_42_inner_pse_2d)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724450;
    std::string expectTilingData =
        "32 2 2 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 21474836482 2 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 "
        "4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 "
        "16384 1 16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_compress_mask_not_square)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 2048}, {1024, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_softmax_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_attention_in_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_attention_in_rank)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_softmax_max_rank)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_dropmask_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_dropmask_size)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_sparse7_inner_pse)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_sparse8_inner_pse_s1s2)
{
    int64_t actual_seq_qlist[3] = {32, 80, 160};
    int64_t actual_seq_kvlist[3] = {32, 64, 96};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{160, 2, 64}, {160, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{160, 2, 64}, {160, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{160, 2, 8}, {160, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{160, 2, 8}, {160, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{160, 2, 64}, {160, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{160, 2, 64}, {160, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_prefix_len)
{
    int64_t prefix_list[3] = {8, 8, 8};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_43_prefix5_deter)
{
    int64_t prefix_list[2] = {16, 16};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 512}, {512, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19705447930401840;
    std::string expectTilingData =
        "32 2 2 1 512 512 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 21474836482 2199023255552 "
        "844424930131968 0 0 0 4 549755813952 549755813952 4 549755814016 8589934612 0 0 2 4 7 11 16 18 20 23 27 32 "
        "34 36 39 43 48 50 52 55 59 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 7 11 16 18 20 23 27 32 34 36 39 43 48 50 52 "
        "55 59 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 52448 6554 20 6546 6554 20 6546 6554 20 6546 0 0 0 20 61440 30720 "
        "114688 30720 2 21728 2 21344 1 0 0 0 0 0 0 0 0 0 0 1 131072 16384 16384 1 131072 16384 16384 1 131072 16384 "
        "16384 65536 590336 1115136 0 0 0 0 0 0 0 0 0 1639936 7931392 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {28909056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_44_sbh_prefix5_deter)
{
    int64_t prefix_list[2] = {16, 16};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 512}, {512, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19705447930401840;
    std::string expectTilingData =
        "32 2 2 1 512 512 64 64 4575657222448611328 255 65536 65536 0 0 0 2 1 0 0 21474836482 2199023255552 "
        "844424930131968 0 0 0 4 549755813952 549755813952 4 549755814016 8589934612 0 0 2 4 7 11 16 18 20 23 27 32 "
        "34 36 39 43 48 50 52 55 59 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 7 11 16 18 20 23 27 32 34 36 39 43 48 50 52 "
        "55 59 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 52448 6554 20 6546 6554 20 6546 6554 20 6546 0 0 0 20 61440 30720 "
        "114688 30720 2 21728 2 21344 1 0 0 0 0 0 0 0 0 0 0 1 131072 16384 16384 1 131072 16384 16384 1 131072 16384 "
        "16384 65536 590336 1115136 0 0 0 0 0 0 0 0 0 1639936 7931392 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {28909056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_45_tnd_prefix6_deter)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    int64_t prefix_list[4] = {8, 8, 8, 8};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19705447930401968;
    std::string expectTilingData =
        "32 4 2 1 384 384 32 32 4575657222448611328 255 65536 65536 0 0 0 4 1 0 0 25769803778 8796093022212 "
        "844424930131968 0 0 0 3 549755813952 549755813952 3 549755814016 8589934610 0 0 2 4 6 8 10 12 14 16 18 20 "
        "22 24 26 28 30 32 34 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 30208 3464 18 3448 3464 18 3448 3464 18 3448 0 0 0 18 61440 30720 "
        "114688 30720 1 30208 1 30112 1 0 0 0 0 0 0 0 0 0 0 1 62336 16384 13184 1 62336 16384 13184 1 62336 16384 "
        "13184 65536 328192 590848 0 0 0 0 0 0 0 0 0 853504 7144960 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 16384 16384 16384 16384 81920 81920 81920 81920 81920 81920 81920 81920 81920 "
        "229376 229376 229376 229376 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16384 16384 16384 16384 81920 81920 81920 "
        "81920 81920 81920 81920 81920 81920 229376 229376 229376 229376 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 "
        "1 5 5 5 5 5 5 5 5 5 14 14 14 14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {28122624};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_46_bnsd_prefix5_deter)
{
    int64_t prefix_list[2] = {16, 16};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 512}, {512, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19705447930401840;
    std::string expectTilingData =
        "32 2 2 1 512 512 64 64 4575657222448611328 255 65536 65536 0 0 0 3 1 0 0 21474836482 2199023255552 "
        "844424930131968 0 0 0 4 549755813952 549755813952 4 549755814016 8589934612 0 0 2 4 7 11 16 18 20 23 27 32 "
        "34 36 39 43 48 50 52 55 59 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 7 11 16 18 20 23 27 32 34 36 39 43 48 50 52 "
        "55 59 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 52448 6554 20 6546 6554 20 6546 6554 20 6546 0 0 0 20 61440 30720 "
        "114688 30720 2 21728 2 21344 1 0 0 0 0 0 0 0 0 0 0 1 131072 16384 16384 1 131072 16384 16384 1 131072 16384 "
        "16384 65536 590336 1115136 0 0 0 0 0 0 0 0 0 1639936 7931392 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {28909056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_47_causal_s1_lt_s2)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 4096, 8}, {1, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 4096, 8}, {1, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 28710448161887280;
    std::string expectTilingData =
        "32 1 1 1 4096 16384 64 64 4575657222448611328 255 2147483647 0 0 0 528 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 32 549755813952 549755813952 128 549755814016 73014444064 0 0 17 35 52 71 88 108 125 "
        "146 168 185 208 232 249 274 300 317 344 372 401 431 462 479 511 561 596 632 669 729 793 861 1023 0 0 0 0 17 "
        "35 52 71 88 108 125 146 168 185 208 232 249 274 300 317 344 372 401 431 462 479 511 561 596 632 669 729 793 "
        "861 1023 4096 0 0 0 0 2097152 8192 32 8192 32768 32 32768 32768 32 32768 0 0 0 32 61440 30720 114688 30720 "
        "69 8192 69 8192 1099511627777 0 0 0 0 0 0 0 0 0 0 1 262144 16384 16384 1 1048576 16384 16384 1 1048576 "
        "16384 16384 65536 1114624 5309440 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {30475776};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_48_rd_causal_s1_lt_s2)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 4096, 8}, {1, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 4096, 8}, {1, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146288;
    std::string expectTilingData =
        "32 1 1 1 4096 16384 64 64 4575657222448611328 255 2147471359 12288 0 0 0 1 1 0 0 12884901890 8796093022210 "
        "281474976710656 0 0 0 32 549755813952 549755813952 128 549755814016 485331304480 0 0 113 226 339 452 565 "
        "678 791 904 1017 1130 1243 1356 1469 1582 1695 1808 1921 2034 2147 2260 2373 2486 2599 2712 2825 2938 3051 "
        "3167 3305 3468 3674 0 0 0 0 113 226 339 452 565 678 791 904 1017 1130 1243 1356 1469 1582 1695 1808 1921 "
        "2034 2147 2260 2373 2486 2599 2712 2825 2938 3051 3167 3305 3468 3674 4096 0 0 0 0 2097152 8192 32 8192 "
        "32768 32 32768 32768 32 32768 0 0 0 32 61440 30720 114688 30720 69 8192 69 8192 1 0 0 0 0 0 0 0 0 0 0 1 "
        "262144 16384 16384 1 1048576 16384 16384 1 1048576 16384 16384 65536 1114624 5309440 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {30475776};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_49_band_split_block)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 8, 4096, 8}, {2, 8, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 4096, 8}, {2, 8, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 27584548791915536;
    std::string expectTilingData =
        "32 2 8 1 4096 4096 128 128 4575657222443697349 255 256 256 0 0 154 1 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 32 549755813952 549755813952 32 549755814016 330712481824 0 0 526 1024 1550 2048 2574 "
        "3072 3598 4096 4622 5120 5646 6144 6670 7168 7694 8192 8718 9216 9742 10240 10766 11264 11790 12288 12814 "
        "13312 13838 14336 14862 15360 15886 0 0 0 0 526 1024 1550 2048 2574 3072 3598 4096 4622 5120 5646 6144 6670 "
        "7168 7694 8192 8718 9216 9742 10240 10766 11264 11790 12288 12814 13312 13838 14336 14862 15360 15886 16384 "
        "0 0 0 0 8388608 262144 32 262144 262144 32 262144 262144 32 262144 0 0 0 32 61440 30720 114688 30720 274 "
        "2048 274 2048 1 0 0 0 0 0 0 0 0 0 0 8 8388608 16384 16384 8 8388608 16384 16384 8 8388608 16384 16384 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_50_alibi_1nhs)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 1024}, {1, 1, 1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146800;
    std::string expectTilingData =
        "32 1 1 1 1024 1024 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 8 549755813952 549755813952 8 549755814016 8589934610 0 0 2 4 6 9 11 13 15 19 21 23 "
        "28 30 36 38 45 47 55 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 6 9 11 13 15 19 21 23 28 30 36 38 45 47 55 64 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 58272 3641 18 3639 3641 18 3639 3641 18 3639 0 0 0 18 61440 30720 "
        "114688 30720 2 27552 2 27232 1 0 0 0 0 0 0 0 0 0 0 1 65536 16384 16384 1 65536 16384 16384 1 65536 16384 "
        "16384 65536 328192 590848 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_51_alibi_bnhs)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 1024, 1024}, {2, 2, 1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 1024, 8}, {2, 2, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 1024, 8}, {2, 2, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146800;
    std::string expectTilingData =
        "32 2 2 1 1024 1024 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 8 549755813952 549755813952 8 549755814016 21474836509 0 0 5 11 18 23 31 45 63 68 74 "
        "79 86 94 103 119 131 137 142 149 157 166 182 194 199 205 212 220 229 239 0 0 0 0 0 0 0 5 11 18 23 31 45 63 "
        "68 74 79 86 94 103 119 131 137 142 149 157 166 182 194 199 205 212 220 229 239 256 0 0 0 0 0 0 0 144640 "
        "9040 29 9024 9040 29 9024 9040 29 9024 0 0 0 29 61440 30720 114688 30720 5 21760 5 21504 1 0 0 0 0 0 0 0 0 "
        "0 0 1 262144 16384 16384 1 262144 16384 16384 1 262144 16384 16384 65536 1114624 2163712 0 0 0 0 0 0 0 0 0 "
        "0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {24184320};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_sparse4_tokens)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_sparse7_tokens)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_pse_alibi_shape)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_pse_alibi_shape)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 64, 64}, {1, 2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_pse_not_causal)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 8, 8}, {1, 2, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_pse_pad_not_alibi)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_regbase_h_helpers)
{
    using optiling::fag::AbsCeil;
    using optiling::fag::CeilDivideBy;
    using optiling::fag::Gcd;
    using optiling::fag::SliceVector;

    EXPECT_EQ(AbsCeil(5, 3), 2);
    EXPECT_EQ(AbsCeil(-5, 3), -2);
    EXPECT_EQ(AbsCeil(5, -3), -2);
    EXPECT_EQ(Gcd(48, 18), 6);
    EXPECT_EQ(Gcd(7, 0), 7);
    EXPECT_EQ(CeilDivideBy(static_cast<int64_t>(10), static_cast<int64_t>(3)), 4);
    EXPECT_EQ(CeilDivideBy(static_cast<int64_t>(10), static_cast<int64_t>(0)), 0);
    const std::vector<int64_t> arr{1, 2, 3, 4, 5, 6};
    const std::vector<int64_t> sliced = SliceVector(arr, static_cast<int64_t>(2));
    ASSERT_EQ(sliced.size(), 3U);
    EXPECT_EQ(sliced[0], 1);
    EXPECT_EQ(sliced[1], 3);
    EXPECT_EQ(sliced[2], 5);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_52_tnd_deter_b129)
{
    constexpr int64_t kBatch = 129;
    int64_t actual_seq_qlist[kBatch];
    int64_t actual_seq_kvlist[kBatch];
    int64_t acc = 0;
    for (int64_t i = 0; i < kBatch; ++i) {
        acc += (i + 1 == kBatch) ? 256 : 32;
        actual_seq_qlist[i] = acc;
        actual_seq_kvlist[i] = acc;
    }
    const int64_t tLen = acc;
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{tLen, 2, 8}, {tLen, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tLen, 2, 8}, {tLen, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{kBatch}, {kBatch}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{kBatch}, {kBatch}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19742831325745328;
    std::string expectTilingData =
        "32 129 2 1 256 256 32 32 4575657222452086003 255 2147483647 2147483647 0 0 0 4 1 0 0 0 0 0 0 0 0 2 "
        "549755813952 549755813952 2 549755814016 38654705694 0 0 9 18 27 36 45 54 63 72 81 90 99 108 117 126 135 "
        "144 153 162 171 180 189 198 207 216 225 234 243 252 261 0 0 0 0 0 0 9 18 27 36 45 54 63 72 81 90 99 108 117 "
        "126 135 144 153 162 171 180 189 198 207 216 225 234 243 252 261 264 0 0 0 0 0 0 13120 9285 30 9263 9285 30 "
        "9263 9285 30 9263 0 0 0 30 61440 30720 114688 30720 1 13120 1 12736 1 0 0 0 0 0 0 0 0 0 0 1 278528 16384 "
        "16384 1 278528 16384 16384 1 278528 16384 16384 65536 1180160 2294784 0 0 0 0 0 0 0 1 9 0 0 9 1 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 2 0 2048 4096 6144 8192 10240 12288 14336 16384 18432 20480 22528 24576 26624 28672 "
        "30720 32768 34816 36864 38912 40960 43008 45056 47104 49152 51200 53248 55296 57344 59392 61440 63488 65536 "
        "67584 69632 71680 73728 75776 77824 79872 81920 83968 86016 88064 90112 92160 94208 96256 98304 100352 "
        "102400 104448 106496 108544 110592 112640 114688 116736 118784 120832 122880 124928 126976 129024 131072 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 2048 4096 6144 8192 10240 12288 14336 16384 18432 20480 22528 24576 26624 28672 "
        "30720 32768 34816 36864 38912 40960 43008 45056 47104 49152 51200 53248 55296 57344 59392 61440 63488 65536 "
        "67584 69632 71680 73728 75776 77824 79872 81920 83968 86016 88064 90112 92160 94208 96256 98304 100352 "
        "102400 104448 106496 108544 110592 112640 114688 116736 118784 120832 122880 124928 126976 129024 131072 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 "
        "58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100 102 104 106 108 110 112 114 116 118 120 "
        "122 124 126 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 262176 -1 -1 -1 -1 -1 -1 -1 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {24380928};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_query_rope_xor)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_key_rope_without_query_rope)
{
    constexpr size_t queryRopeIndex = 22;
    auto compileInfo = MakeA5CompileInfo();
    auto tilingContextPara = MakeMlaRopeTilingContext(&compileInfo, 128, 128);
    tilingContextPara.inputTensorDesc_.erase(tilingContextPara.inputTensorDesc_.begin() + queryRopeIndex);
    tilingContextPara.inputInstanceNum_ = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    tilingContextPara.outputInstanceNum_ = {1, 1, 1, 0, 1, 1};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_eod_nonzero)
{
    int64_t actual_seq_qlist[3] = {32, 16, 8};
    int64_t actual_seq_kvlist[3] = {32, 16, 8};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{32, 2, 8}, {32, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 2, 8}, {32, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_53_gqa_dense_deter)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{5, 1024, 2, 64}, {5, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 1, 64}, {5, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 1, 64}, {5, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 2, 64}, {5, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 2, 1024, 8}, {5, 2, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 2, 1024, 8}, {5, 2, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 2, 64}, {5, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{5, 1024, 2, 64}, {5, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 1, 64}, {5, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 1, 64}, {5, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19707646953656368;
    std::string expectTilingData =
        "32 5 1 2 1024 1024 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 8 "
        "549755813952 549755813952 8 549755814016 85899345952 0 0 20 40 60 80 100 120 140 160 180 200 220 240 260 "
        "280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 0 0 0 0 20 40 60 80 100 120 140 160 "
        "180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 640 0 0 0 0 "
        "327680 20480 32 20480 10240 32 10240 10240 32 10240 0 0 0 32 61440 30720 114688 30720 11 20480 11 20480 1 0 "
        "0 0 0 0 0 0 0 0 0 1 655360 16384 16384 1 327680 16384 16384 1 327680 16384 16384 65536 2687488 3998720 0 0 "
        "0 0 0 0 0 1 24 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {26281472};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_55_deter_left_up)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19745030349001776;
    std::string expectTilingData =
        "32 1 1 1 256 256 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 2 549755813952 549755813952 2 549755814016 4294967299 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 21856 5462 3 5460 5462 3 5460 5462 3 5460 0 0 0 3 61440 30720 114688 30720 1 21856 1 21824 1 0 0 0 0 "
        "0 0 0 0 0 0 1 16384 16384 16384 1 16384 16384 16384 1 16384 16384 16384 65536 131584 197632 0 0 0 0 0 0 0 1 "
        "3 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21235200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_56_deter_nomask_neg_next)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-64)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19745030349001776;
    std::string expectTilingData =
        "32 1 1 1 256 256 64 64 4575657222448611328 255 65536 -64 0 0 0 1 1 0 0 2 1099511627776 562949953421312 0 0 "
        "0 2 549755813952 549755813952 2 549755814016 4294967299 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 1 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 21856 5462 3 "
        "5460 5462 3 5460 5462 3 5460 0 0 0 3 61440 30720 114688 30720 1 21856 1 21824 1 0 0 0 0 0 0 0 0 0 0 1 16384 "
        "16384 16384 1 16384 16384 16384 1 16384 16384 16384 65536 131584 197632 0 0 0 0 0 0 0 1 3 0 0 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21235200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_57_bn2_multiblk_sparse)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{16, 16, 256, 8}, {16, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16, 256, 8}, {16, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19773617651323954;
    std::string expectTilingData =
        "32 16 16 1 256 256 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 2 549755813952 549755813952 2 549755814016 103079215136 0 0 32 64 96 128 160 192 224 "
        "256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 "
        "64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 "
        "928 960 992 1024 0 0 0 0 524288 131072 32 131072 131072 32 131072 131072 32 131072 0 0 0 32 61440 30720 "
        "114688 30720 18 2048 18 2048 1 0 0 0 0 0 0 0 0 0 0 4 4194304 16384 16384 4 4194304 16384 16384 4 4194304 "
        "16384 16384 65536 2424832 3604480 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {25755648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_58_bn2_multiblk_invalid)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{16, 16, 256, 8}, {16, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16, 256, 8}, {16, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19773617651323954;
    std::string expectTilingData =
        "32 16 16 1 256 256 64 64 4575657222448611328 255 16 16 0 0 0 1 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 2 549755813952 549755813952 2 549755814016 137438953504 0 0 32 64 96 128 160 192 224 "
        "256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 "
        "64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 "
        "928 960 992 1024 0 0 0 0 524288 131072 32 131072 131072 32 131072 131072 32 131072 0 0 0 32 61440 30720 "
        "114688 30720 18 2048 18 2048 1 0 0 0 0 0 0 0 0 0 0 4 4194304 16384 16384 4 4194304 16384 16384 4 4194304 "
        "16384 16384 65536 2424832 3604480 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {25755648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_59_nz_out)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 4, 2048, 8}, {1, 4, 2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 4, 2048, 8}, {1, 4, 2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.117851f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703249444016176;
    std::string expectTilingData =
        "32 1 4 1 2048 2048 72 72 4575657222447651805 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 72057594037927936 "
        "0 0 0 16 549755813952 549755813952 16 549755814016 137438953504 0 0 32 64 96 128 160 192 224 256 288 320 "
        "352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 64 96 128 "
        "160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 "
        "1024 0 0 0 0 524288 9216 64 9216 9216 64 9216 9216 64 9216 0 0 0 64 61440 30720 114688 30720 18 2048 "
        "600479950315538 2048 1 237494511599680 29686814005248 216 1 1 128 128 128 128 8192 1 589824 16384 16384 1 "
        "589824 16384 16384 1 589824 16384 16384 65536 2425344 4785152 0 7144960 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {28379136};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_60_tnd_swizzle)
{
    int64_t actual_seq_qlist[1] = {2048};
    int64_t actual_seq_kvlist[1] = {2048};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2048, 4, 8}, {2048, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 4, 8}, {2048, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);
    int64_t expectTilingKey = 19703248907145396;
    std::string expectTilingData =
        "32 1 4 1 2048 2048 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 4 1 0 0 0 0 0 0 0 0 16 "
        "549755813952 549755813952 16 549755814016 154618822688 0 0 32 64 96 128 160 192 224 256 288 320 352 384 416 "
        "448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 64 96 128 160 192 224 "
        "256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024 0 0 0 "
        "0 524288 16384 32 16384 16384 32 16384 16384 32 16384 0 0 0 32 61440 30720 114688 30720 18 2048 18 2048 1 0 "
        "0 0 0 0 0 0 0 0 0 1 524288 16384 16384 1 524288 16384 16384 1 524288 16384 16384 65536 2163200 3343360 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {25495040};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_62_tnd_gqa_nz)
{
    int64_t actual_seq_qlist[1] = {2048};
    int64_t actual_seq_kvlist[1] = {2048};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2048, 4, 72}, {2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1, 72}, {2048, 1, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1, 72}, {2048, 1, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 72}, {2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2048, 4, 8}, {2048, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 4, 8}, {2048, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 72}, {2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2048, 4, 72}, {2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1, 72}, {2048, 1, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1, 72}, {2048, 1, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.117851f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);
    int64_t expectTilingKey = 19703249444016176;
    std::string expectTilingData =
        "32 1 1 4 2048 2048 72 72 4575657222447651805 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 72057594037927936 "
        "0 0 0 16 549755813952 549755813952 16 549755814016 137438953504 0 0 32 64 96 128 160 192 224 256 288 320 "
        "352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 64 96 128 "
        "160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 "
        "1024 0 0 0 0 524288 9216 64 9216 2304 64 2304 2304 64 2304 0 0 0 64 61440 30720 114688 30720 18 2048 "
        "600479950315538 2048 1 237494511599680 29686814005248 216 1 1 128 128 128 128 8192 1 589824 16384 16384 1 "
        "147456 16384 16384 1 147456 16384 16384 65536 2425344 3015680 0 3606016 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {24840192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_63_mla_rope)
{
    auto compileInfo = MakeA5CompileInfo();
    auto tilingContextPara = MakeMlaRopeTilingContext(&compileInfo, 128, 128);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_mla_rope_query_key_d)
{
    auto compileInfo = MakeA5CompileInfo();
    auto tilingContextPara = MakeMlaRopeTilingContext(&compileInfo, 192, 128);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_mla_rope_value_d)
{
    auto compileInfo = MakeA5CompileInfo();
    auto tilingContextPara = MakeMlaRopeTilingContext(&compileInfo, 128, 192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_64_sink)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // faker drops dimNum==0 outputs; keep placeholders so dsink stays at IR index 6
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145266;
    std::string expectTilingData =
        "32 1 1 1 128 128 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 1 2 1 0 1 "
        "549755813952 549755813952 1 549755814016 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16384 8192 1 8192 "
        "8192 1 8192 8192 1 8192 512 1 512 1 61440 30720 114688 30720 1 16384 1 16384 1 0 0 0 0 0 0 0 0 0 0 1 8192 "
        "16384 8192 1 8192 16384 8192 1 8192 16384 8192 65536 0 0 0 0 2425344 2 1 1 2 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23397376};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_65_rd_causal_deter)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19745030349001776;
    std::string expectTilingData =
        "32 1 1 1 256 256 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 12884901890 8796093022210 "
        "281474976710656 0 0 0 2 549755813952 549755813952 2 549755814016 4294967299 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 21856 5462 3 5460 5462 3 5460 5462 3 5460 0 0 0 3 61440 30720 114688 30720 1 21856 1 21824 1 0 0 0 0 "
        "0 0 0 0 0 0 1 16384 16384 16384 1 16384 16384 16384 1 16384 16384 16384 65536 131584 197632 0 0 0 0 0 0 0 1 "
        "3 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21235200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_66_tnd_deter_bn2s2_swizzle)
{
    // Prefix-sum TND, MHA, dense+deter. Many unique (B,N,S1) tiles so CheckExceedL2Cache
    // trips the real 128MB L2; BN2S2 is kept so ConfigureTndDeterBn2S2Swizzle can run.
    int64_t actual_seq_qlist[8] = {2048, 4096, 6144, 8192, 10240, 12288, 14336, 16640};
    int64_t actual_seq_kvlist[8] = {2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{16640, 16, 64}, {16640, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 16, 64}, {16384, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 16, 64}, {16384, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16640, 16, 64}, {16640, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{16640, 16, 8}, {16640, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16640, 16, 8}, {16640, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16640, 16, 64}, {16640, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16640, 16, 64}, {16640, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 16, 64}, {16384, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 16, 64}, {16384, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 28750030580486324;
    std::string expectTilingData =
        "32 8 16 1 2304 2048 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 4 1 0 0 0 0 0 0 0 0 18 "
        "549755813952 549755813952 16 549755814016 4754528796704 0 0 1040 2080 3120 4160 5200 6240 7280 8320 9360 "
        "10400 11440 12480 13520 14560 15600 16640 17680 18720 19760 20800 21840 22880 23920 24960 26000 27040 28080 "
        "29122 30166 31210 32254 0 0 0 0 1040 2080 3120 4160 5200 6240 7280 8320 9360 10400 11440 12480 13520 14560 "
        "15600 16640 17680 18720 19760 20800 21840 22880 23920 24960 26000 27040 28080 29122 30166 31210 32254 33280 "
        "0 0 0 0 17039360 532480 32 532480 524288 32 524288 524288 32 524288 0 0 0 32 61440 30720 114688 30720 555 "
        "20480 555 20480 1 0 0 0 0 0 0 0 0 0 0 17 17039360 16384 16384 16 16777216 16384 16384 16 16777216 16384 "
        "16384 65536 68223488 69403648 0 0 0 0 0 0 0 1 1040 70583808 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 "
        "4194304 8388608 12582912 16777216 20971520 25165824 29360128 34078720 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "4194304 8388608 12582912 16777216 20971520 25165824 29360128 34078720 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 256 "
        "512 768 1024 1280 1536 1792 2080 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4194304 8388608 12582912 "
        "16777216 20971520 25165824 29360128 34078720 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4194304 8388608 12582912 16777216 "
        "20971520 25165824 29360128 34078720 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 256 384 512 640 768 896 1040 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {93915648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_67_band_deter_pq_le_m)
{
    // Pad BAND + deter. B*N is a multiple of AIC so blockOuter==aicNum; unique tiles
    // exceed 128MB L2; Hybrid soc reports AIV=2*AIC so SelectDeterBandSchedule runs
    // the p+q<=m branch.
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{32, 1, 4096, 8}, {32, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 1, 4096, 8}, {32, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfoHybrid, 4096, 1);
    int64_t expectTilingKey = 28754429163869232;
    std::string expectTilingData =
        "64 32 1 1 4096 4096 128 128 4575657222443697349 255 256 256 0 0 154 1 1 0 0 17179869186 8796093022211 "
        "72620543991349248 0 0 3 32 549755813952 549755813952 32 549755814016 661424963616 0 0 1024 2048 3072 4096 "
        "5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 "
        "23552 24576 25600 26624 27648 28672 29696 30720 31744 0 0 0 0 1024 2048 3072 4096 5120 6144 7168 8192 9216 "
        "10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 23552 24576 25600 26624 27648 "
        "28672 29696 30720 31744 32768 0 0 0 0 16777216 262144 64 262144 262144 64 262144 262144 64 262144 0 0 0 64 "
        "61440 30720 114688 30720 547 4096 600479950299136 30720 1 237494511599680 29686814005248 216 10 10 104 104 "
        "2048 2048 131072 16 16777216 16384 16384 16 16777216 16384 16384 16 16777216 16384 16384 65536 67174912 "
        "134284288 0 201393664 0 0 0 0 0 1 160 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {226560000};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_68_band_deter_causal_odd_b)
{
    // Near-causal BAND (next=0), odd B so SelectDeterBandSchedule can take the
    // lower-causal path and CalcCausalSingleBatchRound.
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{33, 1, 4096, 8}, {33, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{33, 1, 4096, 8}, {33, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfoHybrid, 4096, 1);
    int64_t expectTilingKey = 28754429163869232;
    std::string expectTilingData =
        "64 33 1 1 4096 4096 128 128 4575657222443697349 255 65536 0 0 0 528 1 1 0 0 17179869186 8796093022211 "
        "72339069014638592 0 0 1 32 549755813952 549755813952 32 549755814016 2340757176352 0 0 1041 2083 3124 4167 "
        "5208 6252 7293 8338 9384 10425 11472 12520 13561 14610 15660 16701 17752 18804 19857 20911 21966 23007 "
        "24063 25137 26196 27256 28317 29401 30489 31581 32767 0 0 0 0 1041 2083 3124 4167 5208 6252 7293 8338 9384 "
        "10425 11472 12520 13561 14610 15660 16701 17752 18804 19857 20911 21966 23007 24063 25137 26196 27256 28317 "
        "29401 30489 31581 32767 33792 0 0 0 0 17301504 270336 64 270336 270336 64 270336 270336 64 270336 0 0 0 64 "
        "61440 30720 114688 30720 564 6144 600479950298607 28672 1 237494511599680 29686814005248 216 10 10 168 168 "
        "2112 2112 135168 17 17301504 16384 16384 17 17301504 16384 16384 17 17301504 16384 16384 65536 69272064 "
        "138478592 0 207685120 0 0 0 0 0 1 576 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {232982528};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_69_tnd_line_left_up)
{
    int64_t actual_seq_qlist[2] = {4096, 10240};
    int64_t actual_seq_kvlist[2] = {4096, 8192};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{10240, 5, 64}, {10240, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 5, 64}, {8192, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 5, 64}, {8192, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{10240, 5, 64}, {10240, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{10240, 5, 8}, {10240, 5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10240, 5, 8}, {10240, 5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{10240, 5, 64}, {10240, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{10240, 5, 64}, {10240, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 5, 64}, {8192, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 5, 64}, {8192, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19745030349001904;
    std::string expectTilingData =
        "32 2 5 1 6144 4096 64 64 4575657222448611328 255 2147483647 0 0 0 0 4 1 0 0 8589934594 8796093022209 "
        "844424930131968 0 0 0 48 549755813952 549755813952 32 549755814016 1052266987552 0 0 281 766 1259 1686 2232 "
        "2615 3213 3568 4195 4525 5176 5441 5743 6120 6651 6911 7207 7557 8037 8378 8662 8998 9432 9851 10129 10443 "
        "10859 11325 11590 11905 12269 0 0 0 0 281 766 1259 1686 2232 2615 3213 3568 4195 4525 5176 5441 5743 6120 "
        "6651 6911 7207 7557 8037 8378 8662 8998 9432 9851 10129 10443 10859 11325 11590 11905 12269 12800 0 0 0 0 "
        "6553600 102400 32 102400 81920 32 81920 81920 32 81920 0 0 0 32 61440 30720 114688 30720 214 10240 214 "
        "10240 1 0 0 0 0 0 0 0 0 0 0 4 3276800 16384 16384 3 2621440 16384 16384 3 2621440 16384 16384 65536 "
        "13173248 23659520 0 0 0 0 0 0 0 1 276 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 256 1 0 16777216 41943040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 16777216 41943040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 98 276 276 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 "
        "0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {55117312};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_73_tnd_causal_gqa)
{
    int64_t actual_seq_qlist[2] = {512, 1280};
    int64_t actual_seq_kvlist[2] = {512, 1024};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1280, 6, 64}, {1280, 6, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 3, 64}, {1024, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 3, 64}, {1024, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 6, 64}, {1280, 6, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1280, 6, 8}, {1280, 6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1280, 6, 8}, {1280, 6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 6, 64}, {1280, 6, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1280, 6, 64}, {1280, 6, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 3, 64}, {1024, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 3, 64}, {1024, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19709845976913072;
    std::string expectTilingData =
        "32 2 3 2 768 512 64 64 4575657222448611328 255 2147483647 0 0 0 0 4 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 6 549755813952 549755813952 4 549755814016 25769803804 0 0 7 18 27 37 48 55 66 75 85 "
        "96 103 111 120 127 135 144 151 159 168 175 183 192 199 207 216 223 231 0 0 0 0 0 0 0 0 7 18 27 37 48 55 66 "
        "75 85 96 103 111 120 127 135 144 151 159 168 175 183 192 199 207 216 223 231 240 0 0 0 0 0 0 0 0 140448 "
        "17555 28 17535 7022 28 7014 7022 28 7014 0 0 0 28 61440 30720 114688 30720 5 17568 5 17184 1 0 0 0 0 0 0 0 "
        "0 0 0 1 491520 16384 16384 1 196608 16384 16384 1 196608 16384 16384 65536 2032128 2819072 0 0 0 0 0 0 0 1 "
        "18 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 1 1 0 262144 655360 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 262144 655360 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 20 56 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 16 10 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 4 12 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {24577536};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_74_tnd_causal_n1)
{
    int64_t actual_seq_qlist[2] = {512, 1536};
    int64_t actual_seq_kvlist[2] = {768, 1280};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1536, 1, 64}, {1536, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 1, 64}, {1280, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 1, 64}, {1280, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 1, 64}, {1536, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1536, 1, 8}, {1536, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1536, 1, 8}, {1536, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 1, 64}, {1536, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1536, 1, 64}, {1536, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 1, 64}, {1280, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 1, 64}, {1280, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19745030349001904;
    std::string expectTilingData =
        "32 2 1 1 1024 768 64 64 4575657222448611328 255 2147483647 0 0 0 0 4 1 0 0 8589934594 8796093022209 "
        "844424930131968 0 0 0 8 549755813952 549755813952 6 549755814016 8589934610 0 0 2 5 7 11 24 26 28 30 33 35 "
        "37 39 43 45 47 52 54 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 5 7 11 24 26 28 30 33 35 37 39 43 45 47 52 54 56 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 50976 5462 18 5450 4552 18 4536 4552 18 4536 0 0 0 18 61440 30720 "
        "114688 30720 2 20256 2 20192 1 0 0 0 0 0 0 0 0 0 0 1 98304 16384 16384 1 81920 16384 16384 1 81920 16384 "
        "16384 65536 459264 787456 0 0 0 0 0 0 0 1 13 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 1 0 393216 917504 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 393216 917504 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 72 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 6 20 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 16 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 "
        "0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {22087168};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_75_tnd_band_bns2)
{
    int64_t actual_seq_qlist[2] = {1024, 2304};
    int64_t actual_seq_kvlist[2] = {256, 640};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2304, 1, 64}, {2304, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2304, 1, 64}, {2304, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2304, 1, 8}, {2304, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2304, 1, 8}, {2304, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2304, 1, 64}, {2304, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2304, 1, 64}, {2304, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19747229372257456;
    std::string expectTilingData =

        "32 2 1 1 1280 384 64 64 4575657222448611328 255 256 256 0 0 0 4 1 0 0 17179869186 8796093022211 "
        "844424930131968 0 0 0 10 549755813952 549755813952 3 "
        "549755814016 4294967315 0 0 5 6 7 13 14 15 21 22 23 24 25 32 33 34 35 43 44 45 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 5 6 7 13 14 15 21 22 23 24 25 32 33 34 35 43 "
        "44 45 46 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 39680 7761 19 7758 2156 19 2152 2156 19 2152 0 0 0 19 61440 30720 "
        "114688 30720 2 8960 2 8704 1 0 0 0 0 0 0 0 0 0 0 "
        "1 147456 16384 16384 1 40960 16384 8192 1 40960 16384 8192 65536 655872 820224 0 0 0 0 0 0 0 1 7 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 262144 753664 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 262144 753664 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 12 33 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21956096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_79_tnd_line_band_unpad)
{
    // Stay TND (not isAllSame). n2>=2 + deter BAND so PreferTNDLineDeter actually calls GetTNDBandMN.
    int64_t actual_seq_qlist[2] = {4096, 10240};
    int64_t actual_seq_kvlist[2] = {2048, 5120};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{10240, 4, 64}, {10240, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 4, 64}, {5120, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 4, 64}, {5120, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{10240, 4, 64}, {10240, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{10240, 4, 8}, {10240, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10240, 4, 8}, {10240, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{10240, 4, 64}, {10240, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{10240, 4, 64}, {10240, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 4, 64}, {5120, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 4, 64}, {5120, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19747229372257456;
    std::string expectTilingData =

        "32 2 4 1 6144 3072 64 64 4575657222448611328 255 256 256 0 0 0 4 1 0 0 17179869186 8796093022211 "
        "844424930131968 0 0 0 48 549755813952 549755813952 24 "
        "549755814016 107374182432 0 0 179 344 510 661 826 991 1171 1336 1501 1653 1818 1983 2219 2464 2709 2954 3222 "
        "3467 3712 3957 4202 4426 4671 4916 5161 5406 5674 "
        "5919 6164 6409 6655 0 0 0 0 179 344 510 661 826 991 1171 1336 1501 1653 1818 1983 2219 2464 2709 2954 3222 "
        "3467 3712 3957 4202 4426 4671 4916 5161 5406 5674 "
        "5919 6164 6409 6655 6656 0 0 0 0 3407872 81920 32 81920 40960 32 40960 40960 32 40960 0 0 0 32 61440 30720 "
        "114688 30720 111 28672 111 28672 1 0 0 0 0 0 0 0 0 "
        "0 0 3 2621440 16384 16384 2 1310720 16384 16384 2 1310720 16384 16384 65536 10551808 15795200 0 0 0 0 0 0 0 1 "
        "35 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 256 1 0 "
        "8388608 27262976 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8388608 27262976 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 14 35 35 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {42010112};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_80_tnd_band_bns2_n1)
{
    // n2==1 skips line deter; per-batch square but different lengths keep TND BN2S2 for split-dk.
    int64_t actual_seq_qlist[2] = {256, 768};
    int64_t actual_seq_kvlist[2] = {256, 768};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{768, 1, 8}, {768, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{768, 1, 8}, {768, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19747229372257456;
    std::string expectTilingData =

        "32 2 1 1 512 512 64 64 4575657222448611328 255 256 256 0 0 0 4 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 4 549755813952 549755813952 4 "
        "549755814016 4294967314 0 0 1 2 3 4 5 6 8 9 10 11 12 13 14 15 17 18 19 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 "
        "2 3 4 5 6 8 9 10 11 12 13 14 15 17 18 19 20 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 18208 2731 18 2725 2731 18 2725 2731 18 2725 0 0 0 18 61440 30720 114688 30720 "
        "1 18208 1 18144 1 0 0 0 0 0 0 0 0 0 0 1 49152 "
        "16384 16384 1 49152 16384 16384 1 49152 16384 16384 65536 262656 459776 0 0 0 0 0 0 0 1 4 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 65536 327680 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 65536 327680 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 20 "
        "4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21628416};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_81_tnd_bn2s2_lu_ge)
{
    // TND CASUAL needs n1 even and S1>=S2 every batch; stay unpadded via different batch sizes.
    int64_t actual_seq_qlist[2] = {512, 1280};
    int64_t actual_seq_kvlist[2] = {256, 640};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 4, 64}, {640, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 4, 64}, {640, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1280, 4, 8}, {1280, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1280, 4, 8}, {1280, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 4, 64}, {640, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 4, 64}, {640, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);
    int64_t expectTilingKey = 19703248907146416;
    std::string expectTilingData =
        "32 2 4 1 768 384 64 64 4575657222448611328 255 2147483647 0 0 0 0 4 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 6 549755813952 549755813952 3 549755814016 12884901918 0 0 3 7 10 14 17 21 24 27 31 "
        "34 37 41 46 49 52 55 59 64 67 70 73 77 82 85 88 91 95 100 103 0 0 0 0 0 0 3 7 10 14 17 21 24 27 31 34 37 41 "
        "46 49 52 55 59 64 67 70 73 77 82 85 88 91 95 100 103 104 0 0 0 0 0 0 56800 10923 30 10913 5462 30 5442 5462 "
        "30 5442 0 0 0 30 61440 30720 114688 30720 2 26080 2 26016 1 0 0 0 0 0 0 0 0 0 0 1 327680 16384 16384 1 "
        "163840 16384 16384 1 163840 16384 16384 65536 1376768 2032640 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 "
        "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 131072 131072 131072 131072 131072 131072 "
        "131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 "
        "131072 131072 131072 131072 131072 131072 131072 131072 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 8 8 8 8 8 8 8 8 8 "
        "8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {23660032};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_82_tnd_bn2s2_rd_le)
{
    int64_t actual_seq_qlist[2] = {256, 768};
    int64_t actual_seq_kvlist[2] = {512, 1280};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{768, 4, 64}, {768, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 4, 64}, {768, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{768, 4, 8}, {768, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{768, 4, 8}, {768, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 4, 64}, {768, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{768, 4, 64}, {768, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);
    int64_t expectTilingKey = 19703248907146420;
    std::string expectTilingData =
        "32 2 4 1 512 768 64 64 4575657222448611328 255 2147483647 0 0 0 0 4 1 0 0 12884901890 8796093022210 "
        "281474976710656 0 0 0 4 549755813952 549755813952 6 549755814016 25769803808 0 0 2 4 8 10 12 16 18 20 24 26 "
        "28 32 36 40 44 48 56 60 64 68 72 80 84 88 92 96 104 108 112 116 120 0 0 0 0 2 4 8 10 12 16 18 20 24 26 28 "
        "32 36 40 44 48 56 60 64 68 72 80 84 88 92 96 104 108 112 116 120 128 0 0 0 0 65536 6144 32 6144 10240 32 "
        "10240 10240 32 10240 0 0 0 32 61440 30720 114688 30720 3 4096 3 4096 1 0 0 0 0 0 0 0 0 0 0 1 196608 16384 "
        "16384 1 327680 16384 16384 1 327680 16384 16384 65536 852480 2032640 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 131072 131072 131072 131072 "
        "131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 "
        "131072 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 131072 131072 131072 131072 131072 131072 131072 131072 131072 "
        "131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {24184320};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_83_tnd_bn2s2_band_sq)
{
    // BAND sparseType needs isS1S2Same; different per-batch lengths keep TND (not isAllSame).
    int64_t actual_seq_qlist[2] = {512, 1536};
    int64_t actual_seq_kvlist[2] = {512, 1536};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1536, 4, 8}, {1536, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1536, 4, 8}, {1536, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);
    int64_t expectTilingKey = 19703248907146420;
    std::string expectTilingData =
        "32 2 4 1 1024 1024 64 64 4575657222448611328 255 256 256 0 0 0 4 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 8 549755813952 549755813952 8 549755814016 42949672992 0 0 8 16 24 32 40 48 56 64 80 "
        "88 96 104 112 128 144 152 160 168 176 192 208 216 224 232 240 256 272 280 288 296 304 0 0 0 0 8 16 24 32 40 "
        "48 56 64 80 88 96 104 112 128 144 152 160 168 176 192 208 216 224 232 240 256 272 280 288 296 304 320 0 0 0 "
        "0 163840 12288 32 12288 12288 32 12288 12288 32 12288 0 0 0 32 61440 30720 114688 30720 6 10240 6 10240 1 0 "
        "0 0 0 0 0 0 0 0 0 1 393216 16384 16384 1 393216 16384 16384 1 393216 16384 16384 65536 1638912 2819072 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 "
        "262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 "
        "262144 262144 262144 262144 262144 262144 262144 262144 262144 0 0 0 0 0 0 0 0 0 0 0 0 262144 262144 262144 "
        "262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 "
        "262144 262144 262144 262144 262144 262144 0 0 0 0 0 0 0 0 0 0 0 0 16 16 16 16 16 16 16 16 16 16 16 16 16 16 "
        "16 16 16 16 16 16 16 16 16 16 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {24970752};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_84_tnd_band_deter_bns2_small_m)
{
    // Stay TND: different per-batch S1 so isAllSame is false (equal lengths are rewritten to pad
    // in ProcessSparseModeInfo and CalcleTNDDeterParam never runs). S2=128 keeps n=1 for
    // SupportTNDBns2. S1>128 / token<S / n2=1 as before. p+q>m -> SmallM.
    int64_t actual_seq_qlist[2] = {256, 640};
    int64_t actual_seq_kvlist[2] = {128, 256};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 1, 64}, {256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 1, 64}, {256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{640, 1, 8}, {640, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{640, 1, 8}, {640, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 1, 64}, {256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 1, 64}, {256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19747229372257456;
    std::string expectTilingData =
        "32 2 1 1 384 128 64 64 4575657222448611328 255 64 64 0 0 0 4 1 0 0 17179869186 8796093022211 844424930131968 "
        "0 0 0 3 549755813952 549755813952 1 549755814016 4294967300 0 0 1 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 1 3 4 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20480 10240 4 "
        "10240 4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 20480 1 20480 1 0 0 0 0 0 0 0 0 0 0 1 40960 "
        "16384 8192 1 16384 16384 16384 1 16384 16384 16384 65536 229888 295936 0 0 0 0 0 0 0 1 3 0 0 3 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 1 0 32768 81920 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32768 81920 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 5 3 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 8192 8192 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 "
        "0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21333504};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_85_tnd_band_deter_bns2_large_m)
{
    // Same TND/BN2S2/SupportTNDBns2 gates as tiling_84 (unequal S1, S2=128). Larger S1 so p+q<=m.
    int64_t actual_seq_qlist[2] = {512, 1152};
    int64_t actual_seq_kvlist[2] = {128, 256};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1152, 1, 64}, {1152, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 1, 64}, {256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 1, 64}, {256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1152, 1, 64}, {1152, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1152, 1, 8}, {1152, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1152, 1, 8}, {1152, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1152, 1, 64}, {1152, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1152, 1, 64}, {1152, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 1, 64}, {256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 1, 64}, {256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19747229372257456;
    std::string expectTilingData =
        "32 2 1 1 640 128 64 64 4575657222448611328 255 64 64 0 0 0 4 1 0 0 17179869186 8796093022211 844424930131968 "
        "0 0 0 5 549755813952 549755813952 1 549755814016 4294967300 0 0 3 7 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 3 7 8 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 36864 18432 4 "
        "18432 4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 2 6144 2 6144 1 0 0 0 0 0 0 0 0 0 0 1 73728 "
        "16384 8192 1 16384 16384 16384 1 16384 16384 16384 65536 360960 427008 0 0 0 0 0 0 0 1 4 0 0 4 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 1 0 65536 147456 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 65536 147456 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 7 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "8192 8192 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21464576};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_86_tnd_bn2s2_short_seq_swizzle)
{
    int64_t actualSeqQlen[36] = {71,   159,  242,  323,  436,  608,  740,  824,  944,  1012, 1098, 1183,
                                 1259, 1419, 1486, 1569, 1725, 1941, 2042, 2166, 2242, 2369, 2456, 2594,
                                 2674, 2832, 2916, 3083, 3204, 3306, 3503, 3622, 3709, 3853, 4023, 4096};
    int64_t actualSeqKvlen[36] = {71,   159,  242,  323,  436,  608,  740,  824,  944,  1012, 1098, 1183,
                                  1259, 1419, 1486, 1569, 1725, 1941, 2042, 2166, 2242, 2369, 2456, 2594,
                                  2674, 2832, 2916, 3083, 3204, 3306, 3503, 3622, 3709, 3853, 4023, 4096};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{4096, 128, 192}, {4096, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 128, 192}, {4096, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 128, 128}, {4096, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 128, 128}, {4096, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{4096, 128, 8}, {4096, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4096, 128, 8}, {4096, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 128, 128}, {4096, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{36}, {36}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQlen},
            {{{36}, {36}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKvlen},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 128, 192}, {4096, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 128, 192}, {4096, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 128, 128}, {4096, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    constexpr uint64_t swizzleTilingKeyMask = 1ULL << 53;
    ASSERT_EQ(tilingInfo.tilingKey & swizzleTilingKeyMask, swizzleTilingKeyMask);
    using TilingData = optiling::fag::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<false, false, true, true>;
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(TilingData));
    TilingData tilingData;
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(TilingData));

    EXPECT_EQ(tilingData.s1s2BNGS1S2SplitCoreParams.get_blockOuter(), 32);
    EXPECT_EQ(tilingData.s1s2BNGS1S2SplitCoreParams.get_maxValidBBLen(), 271);
}

// ---------------------------------------------------------------------------
// 覆盖补充用例（合并自 test_flash_attention_score_grad_tiling_coverage.cpp；该文件名不
// 匹配 CMake 的 test_*_tiling.cpp 收集规则，此前未参与编译，合并后纳入 tiling UT 目标）。
// ---------------------------------------------------------------------------
// 覆盖补充（函数覆盖率优先）：静态分析发现的 0 覆盖/低覆盖函数族。
//   1. GQA BAND deter（非 TND）：SelectGqaBandSchedule + 10 个 static 辅助函数
//   2. GQA BAND deter + 负 token：SelectGqaBandNegativeSchedule + grouped 搜索
//   3. GQA CAUSAL deter（非 TND）：CalcleCausalDeterParam grouped 主体 + 10 函数
//   4. GQA CAUSAL deter + RIGHT_DOWN：RD 分支
//   5. 空输出 tiling：RunEmptyTilingRegbase + CalculateTschBlockDim
//   6. HIFLOAT8 量化成功路径：ProcessQuantInfo 接受分支等
//   7. E4M3FN dtype：DetermineMode/GetS1S2TemplateType 判定分支
// 断言采用 ExecuteTiling 的结果校验（成功用例不比对 tiling data 全串，
// 未运行环境下避免硬编码错误期望；上板跑通后可升级为精确断言）。
// ---------------------------------------------------------------------------

// 27 输入 / 6 输出的 BSND 参数构造（GQA deter / 量化覆盖用）。
static gert::TilingContextPara MakeGqaDeterPara(
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo* compileInfo, int64_t b, int64_t s1, int64_t s2,
    int64_t n1, int64_t n2, int64_t d, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    ge::DataType dtype = ge::DT_FLOAT16, int64_t deter = 1)
{
    return gert::TilingContextPara("FlashAttentionScoreGrad",
                                   {
                                       // q: [B, S1, N1, D]
                                       {{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND},
                                       // k: [B, S2, N2, D]
                                       {{{b, s2, n2, d}, {b, s2, n2, d}}, dtype, ge::FORMAT_ND},
                                       // v: [B, S2, N2, D]
                                       {{{b, s2, n2, d}, {b, s2, n2, d}}, dtype, ge::FORMAT_ND},
                                       // dy
                                       {{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND},
                                       // pse_shift
                                       {{{}, {}}, dtype, ge::FORMAT_ND},
                                       // drop_mask
                                       {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                       // padding_mask
                                       {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                       // atten_mask
                                       {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
                                       // softmax_max: [B, N1, S1, 8]，dim0 必须等于 b
                                       {{{b, n1, s1, 8}, {b, n1, s1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // softmax_sum
                                       {{{b, n1, s1, 8}, {b, n1, s1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // softmax_in
                                       {{{}, {}}, dtype, ge::FORMAT_ND},
                                       // attention_in
                                       {{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND},
                                       // prefix
                                       {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       // actual_seq_qlen
                                       {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       // actual_seq_kvlen
                                       {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       // q_start_idx
                                       {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       // kv_start_idx
                                       {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       // dScaleQ
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // dScaleK
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // dScaleV
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // dScaledy
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // dScaleo
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       // queryRope
                                       {{{}, {}}, dtype, ge::FORMAT_ND},
                                       // keyRope
                                       {{{}, {}}, dtype, ge::FORMAT_ND},
                                       // sink
                                       {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                   },
                                   {
                                       // dq
                                       {{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND},
                                       // dk
                                       {{{b, s2, n2, d}, {b, s2, n2, d}}, dtype, ge::FORMAT_ND},
                                       // dv
                                       {{{b, s2, n2, d}, {b, s2, n2, d}}, dtype, ge::FORMAT_ND},
                                       // dpse
                                       {{{}, {}}, dtype, ge::FORMAT_ND},
                                       // dq_rope
                                       {{{}, {}}, dtype, ge::FORMAT_ND},
                                       // dk_rope
                                       {{{}, {}}, dtype, ge::FORMAT_ND},
                                   },
                                   {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
                                    {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                    {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(preTokens)},
                                    {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(nextTokens)},
                                    {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n1)},
                                    {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                    {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                    {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
                                    {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                    {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                    {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                    {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                    {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
                                   compileInfo, "Ascend950", A5SocInfoHybrid, 262144, deter);
}

// ---------------------------------------------------------------------------
// P0-1: GQA BAND deter（非 TND）—— SelectGqaBandSchedule 及 10 个 static 辅助。
// 条件：BN2GS1S2 + g>1 + DETER_BAND + isSwizzle（Hybrid soc 使 coreNum==2*aicNum）。
// ---------------------------------------------------------------------------
TEST_F(FlashAttentionScoreGradTiling, Coverage_GqaBandDeter_Bsnd_G2)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 131072, 134217728, 32, platform_ascendc::SocVersion::ASCEND950};
    auto para = MakeGqaDeterPara(&compileInfo, 4, 4096, 4096, 2, 1, 64, 4, 65536, 65536);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
}

// P0-2: GQA BAND deter + 负 token——SelectGqaBandNegativeSchedule + grouped 搜索。
// s1、s2 均 128 对齐；pre_tockens 为负触发负 token 变换。
TEST_F(FlashAttentionScoreGradTiling, Coverage_GqaBandDeter_NegativeTokens)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 131072, 134217728, 32, platform_ascendc::SocVersion::ASCEND950};
    auto para = MakeGqaDeterPara(&compileInfo, 4, 4096, 4096, 2, 1, 64, 4, -256, 1024);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
}

// P0-3: GQA CAUSAL deter（非 TND）—— CalcleCausalDeterParam grouped 主体 + 10 函数。
TEST_F(FlashAttentionScoreGradTiling, Coverage_GqaCausalDeter_Bsnd_G2)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 131072, 134217728, 32, platform_ascendc::SocVersion::ASCEND950};
    auto para = MakeGqaDeterPara(&compileInfo, 5, 2048, 2048, 2, 1, 64, 2, 65536, 65536);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
}

// P0-4: GQA CAUSAL deter + RIGHT_DOWN（sparse_mode=3、s1==s2）——RD 分支。
TEST_F(FlashAttentionScoreGradTiling, Coverage_GqaCausalDeter_RightDown)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 131072, 134217728, 32, platform_ascendc::SocVersion::ASCEND950};
    auto para = MakeGqaDeterPara(&compileInfo, 5, 2048, 2048, 2, 1, 64, 3, 65536, 65536);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
}

// P0-5: 空输出 tiling——RunEmptyTilingRegbase + CalculateTschBlockDim。
// dq 输出 shape 含 0 维（shapeSize==0），其余输入合法；3 维输入按 SBH 声明，
// compileInfo 带 NpuArch::DAV_3510 走 regbase 空输出分支。
TEST_F(FlashAttentionScoreGradTiling, Coverage_EmptyOutput_DqDimZero)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,
        32,
        196608,
        524288,
        65536,
        65536,
        131072,
        134217728,
        32,
        platform_ascendc::SocVersion::ASCEND950,
        NpuArch::DAV_3510};
    gert::TilingContextPara para("FlashAttentionScoreGrad",
                                 {
                                     // q: SBH [S, B, H*D]
                                     {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                     {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     // dq: shape 含 0 维 -> 空输出 tiling 路径
                                     {{{1, 0, 1, 64}, {1, 0, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     // dk / dv 合法
                                     {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{128, 1, 128}, {128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
                                  {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                  {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                                  {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                                  {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                  {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
                                  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                  {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
                                 &compileInfo, "Ascend950", A5SocInfoHybrid, 4096);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
}

// P0-6: HIFLOAT8 输入——当前 arch35 ProcessQuantInfo（common_regbase 拒绝列表）对
// DT_HIFLOAT8 返回失败；本用例覆盖 DetermineMode HIFLOAT8 档与拒绝分支，
// 产品侧恢复量化支持后可升级为成功断言（isQuant/GetS1S2TemplateType 档等）。
TEST_F(FlashAttentionScoreGradTiling, Coverage_Hifloat8Rejected)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 131072, 134217728, 32, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara para("FlashAttentionScoreGrad",
                                 {
                                     // q/k/dy: HIFLOAT8 [B, S1, N, D]
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                     // v: HIFLOAT8 [B, S2, N, D]
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                     // pse_shift
                                     {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                     // drop_mask
                                     {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                     // padding_mask
                                     {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                     // atten_mask
                                     {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                     // softmax_max: isQuant 形状 [B, N1, S1, 1]
                                     {{{1, 1, 256, 1}, {1, 1, 256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     // softmax_sum
                                     {{{1, 1, 256, 1}, {1, 1, 256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     // softmax_in
                                     {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                     // attention_in
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                     // prefix
                                     {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     // dScaleQ/K/V/Dy/O: [1] FP32
                                     {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
                                  {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                  {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                                  {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                                  {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                  {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                  {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                  {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
                                 &compileInfo, "Ascend950", A5SocInfoHybrid, 4096);
    TilingInfo tilingInfo;
    // HIFLOAT8 当前在 ProcessQuantInfo 被拒绝（common_regbase.cpp:2203），预期失败返回。
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// P0-7: E4M3FN dtype——同构参数下 FP16 完整 tiling 成功作为对照；E4M3FN 输入在
// ProcessQuantInfo 的拒绝列表中（common_regbase.cpp 拒绝分支），断言失败返回。
// s1=s2=2048 与 MakeGqaDeterPara 固定的 [2048, 2048] atten_mask 匹配（sparse_mode=0）。
TEST_F(FlashAttentionScoreGradTiling, Coverage_E4m3fn_Dtype)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 131072, 134217728, 32, platform_ascendc::SocVersion::ASCEND950};
    // 合法 dtype 对照：同构夹具 FP16 走完整 tiling，验证参数构造可通过全部前置检查。
    auto fp16Para = MakeGqaDeterPara(&compileInfo, 1, 2048, 2048, 1, 1, 128, 0, 65536, 65536, ge::DT_FLOAT16, 0);
    TilingInfo fp16TilingInfo;
    ASSERT_TRUE(ExecuteTiling(fp16Para, fp16TilingInfo));

    auto para = MakeGqaDeterPara(&compileInfo, 1, 2048, 2048, 1, 1, 128, 0, 65536, 65536, ge::DT_FLOAT8_E4M3FN, 0);
    TilingInfo tilingInfo;
    // E4M3FN 输入在 ProcessQuantInfo 被拒绝，预期失败返回。
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ---------------------------------------------------------------------------
// 第二批覆盖补充（P1 缺口）：
//   8.  Sink 错误族（4 条 fail：shape≠[n1]、dsinkOut 形状、dtype≠FP32、query=FP32）
//   9.  CheckBaseInput 错误族（head_num=0、BSH b 不匹配、非法 layout）
//   10. pse_type=0 成功路径
//   11. TND sparse_mode=9 fail
//   12. TND RIGHT_DOWN causal deter（GetTNDRightDownVirt/RowShift）
//   13. TND deter 含零长度批次
//   14. TND GQA deter 变长 batch（Exclusive 深层分支）
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// P1-8: Sink 错误族——ProcessSinkInfo 的 4 个失败分支。
// ---------------------------------------------------------------------------

// sink shape 不为 [n1]（n1=1 时传 [2]）
TEST_F(FlashAttentionScoreGradTiling, Coverage_SinkWrongShape)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 131072, 134217728, 32, platform_ascendc::SocVersion::ASCEND950};
    // 直接构造完整 para：sink shape 为 [2]（应为 [n1]=[1]）
    gert::TilingContextPara sinkPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // sink: shape=[2] 而不是 [n1]=[1]
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // faker drops dimNum==0 outputs; keep placeholders so dsink stays at IR index 6
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dsink: [n1]=[1]
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfoHybrid, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(sinkPara, tilingInfo));
}

// sink dtype 不为 FP32（传 FP16）
TEST_F(FlashAttentionScoreGradTiling, Coverage_SinkWrongDtype)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 131072, 134217728, 32, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara sinkPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // sink: shape 正确 [1] 但 dtype=FP16（应为 FP32）
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // faker drops dimNum==0 outputs; keep placeholders so dsink stays at IR index 6
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfoHybrid, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(sinkPara, tilingInfo));
}

// ---------------------------------------------------------------------------
// P1-9: CheckBaseInput 错误族。
// ---------------------------------------------------------------------------

// head_num=0 —— CheckBaseInput 的 headNum 非法值分支（headNum 必须大于 0）。
TEST_F(FlashAttentionScoreGradTiling, Coverage_HeadNumZero)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64, 32, 196608, 524288, 65536, 65536, 131072, 134217728, 32, platform_ascendc::SocVersion::ASCEND950};
    // attrs 构造后不可修改，head_num=0 需直接构造完整 para。
    gert::TilingContextPara headNumZeroPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfoHybrid, 4096);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(headNumZeroPara, tilingInfo));
}

// ---------------------------------------------------------------------------
// P1-11: TND sparse_mode=9 fail（CheckVarLenSparseModeValue > 8 分支）。
// ---------------------------------------------------------------------------

// TND 布局参数构造（与 tiling_25_tnd_all_same 同构，仅 sparse_mode 可变）：
// 3 维 Q/K/V/DY + 合法累计序列长度，保证夹具本身可通过 TND 前置检查。
// 注意：const tensor 的 constValue_ 是裸指针，数组必须用 static 存储期，
// 避免 helper 返回后栈帧失效导致 use-after-return。
static gert::TilingContextPara MakeTndSparseModePara(
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo* compileInfo, int64_t sparseMode)
{
    static int64_t actualSeqQlist[3] = {64, 128, 192};
    static int64_t actualSeqKvlist[3] = {64, 128, 192};
    return gert::TilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{192, 2, 8}, {192, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{192, 2, 8}, {192, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        compileInfo, "Ascend950", A5SocInfo, 4096);
}

TEST_F(FlashAttentionScoreGradTiling, Coverage_TndSparseMode9)
{
    auto compileInfo = MakeA5CompileInfo();
    // 合法模式对照：sparse_mode=0 走完整 TND tiling 成功，验证夹具本身有效。
    auto validPara = MakeTndSparseModePara(&compileInfo, 0);
    TilingInfo validTilingInfo;
    ASSERT_TRUE(ExecuteTiling(validPara, validTilingInfo));

    // TND 布局 + sparse_mode=9（超过合法范围 0-8），CheckVarLenSparseModeValue 拒绝。
    auto para = MakeTndSparseModePara(&compileInfo, 9);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}
