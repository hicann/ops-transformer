/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FIXED_CAUSAL_GROUPED_H_
#define FLASH_ATTN_GRAD_DETER_FIXED_CAUSAL_GROUPED_H_

// Exact useful-work accounting and cache cost for contiguous causal BN2 groups.
// These pure helpers depend on normalized candidate inputs, never on Host APIs.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include "candidate_types.h"
#include "fag_deter_utils.h"

namespace optiling {
namespace fag_pto_arch35 {
inline uint64_t SaturatingCausalMultiply(uint64_t lhs, uint64_t rhs)
{
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    return rhs != 0 && lhs > limit / rhs ? limit : lhs * rhs;
}

inline uint64_t SaturatingCausalAdd(uint64_t lhs, uint64_t rhs)
{
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    return lhs > limit - rhs ? limit : lhs + rhs;
}

// Capacity proxy for the Q/dY/dQ and K/V/dK/dV data belonging to one (B, N2).
// This is not measured traffic: softmax, mask and pipeline temporaries are excluded.
inline uint64_t EstimateCausalBn2Bytes(const FixedCandidateState& params)
{
    const int64_t inputBytes = params.inputBytes;
    if (inputBytes <= 0 || params.d <= 0 || params.dv <= 0 || params.s1 <= 0 || params.s2 <= 0) {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t gradAlign = 16;
    const uint64_t d = static_cast<uint64_t>(params.d);
    const uint64_t dv = static_cast<uint64_t>(params.dv);
    const uint64_t gradD = SaturatingCausalMultiply((d + gradAlign - 1) / gradAlign, gradAlign);
    const uint64_t gradDv = SaturatingCausalMultiply((dv + gradAlign - 1) / gradAlign, gradAlign);
    const uint64_t inputRow = SaturatingCausalMultiply(SaturatingCausalAdd(d, dv), inputBytes);
    const uint64_t queryRow = SaturatingCausalAdd(inputRow, SaturatingCausalMultiply(gradD, sizeof(float)));
    const uint64_t kvRow =
        SaturatingCausalAdd(inputRow, SaturatingCausalMultiply(SaturatingCausalAdd(gradD, gradDv), sizeof(float)));
    const uint64_t queryRows = SaturatingCausalMultiply(params.g, params.s1);
    return SaturatingCausalAdd(SaturatingCausalMultiply(queryRows, queryRow),
                               SaturatingCausalMultiply(params.s2, kvRow));
}

struct CausalRoundInterval {
    int64_t begin;
    int64_t end;
};

struct CausalWork {
    int64_t activeRounds = 0;
    std::vector<int64_t> coreTasks;
};

// A column traversal has x = direction * round + rowOffset. Intersect its
// branch interval with the same coordinate bounds used by the kernel mapper.
inline void AddCausalColumnWork(int64_t core, int64_t begin, int64_t end, int64_t column, int64_t direction,
                                int64_t rowOffset, int64_t m, int64_t n, CausalWork& work,
                                std::vector<CausalRoundInterval>& intervals)
{
    if (column < 1 || column > n) {
        return;
    }
    begin = std::max<int64_t>(begin, 1);
    if (direction > 0) {
        begin = std::max(begin, column - rowOffset);
        end = std::min(end, m - rowOffset);
    } else {
        begin = std::max(begin, rowOffset - m);
        end = std::min(end, rowOffset - column);
    }
    if (begin <= end) {
        work.coreTasks[core] += end - begin + 1;
        intervals.push_back({begin, end});
    }
}

inline int64_t CountCausalActiveRounds(std::vector<CausalRoundInterval>& intervals)
{
    if (intervals.empty()) {
        return 0;
    }
    std::sort(intervals.begin(), intervals.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.begin < rhs.begin; });
    int64_t total = 0;
    int64_t begin = intervals.front().begin;
    int64_t end = intervals.front().end;
    for (const auto& interval : intervals) {
        if (interval.begin > end + 1) {
            total += end - begin + 1;
            begin = interval.begin;
            end = interval.end;
        } else {
            end = std::max(end, interval.end);
        }
    }
    return total + end - begin + 1;
}

// Count the final single-BN2 part of CalCausalSingleBatchDeterIndex. Its local
// mapper can use a square larger than the valid columns, so keep both limits.
inline void AddCausalSingleRemainderWork(int64_t k, int64_t m, int64_t mapperN, int64_t validN, int64_t rounds,
                                         CausalWork& work)
{
    std::vector<CausalRoundInterval> intervals;
    intervals.reserve(NUM_TWO * k);
    const bool doubleCoreColumns = NUM_TWO * k < m + 1 && k < mapperN;
    for (int64_t core = 0; core < k; ++core) {
        const int64_t j = core + 1;
        if (doubleCoreColumns) {
            const int64_t pairedColumn = NUM_TWO * k + 1 - j;
            if (j % NUM_TWO != 0) {
                const int64_t split = m - j + 1;
                AddCausalColumnWork(core, 1, std::min(rounds, split), j, 1, j - 1, m, validN, work, intervals);
                AddCausalColumnWork(core, split + 1, rounds, pairedColumn, -1, NUM_TWO * m + 2 - j, m, validN, work,
                                    intervals);
            } else {
                const int64_t split = m - NUM_TWO * k + 1 + j;
                AddCausalColumnWork(core, split, rounds, j, -1, j + NUM_TWO * m - NUM_TWO * k + 1, m, validN, work,
                                    intervals);
                AddCausalColumnWork(core, 1, std::min(rounds, split - 1), pairedColumn, 1, pairedColumn - 1, m, validN,
                                    work, intervals);
            }
        } else if (j <= m / NUM_TWO + 1) {
            const int64_t parity = m % NUM_TWO;
            const int64_t pairedColumn = m + 3 - j - parity;
            if (j % NUM_TWO != 0) {
                const int64_t split = m + 1 - j;
                AddCausalColumnWork(core, 1, std::min(rounds, split), j, 1, j - 1, m, validN, work, intervals);
                AddCausalColumnWork(core, split + 1, rounds, pairedColumn, -1, NUM_TWO * m + 2 - j, m, validN, work,
                                    intervals);
            } else {
                const int64_t split = j - 1 + parity;
                AddCausalColumnWork(core, split, rounds, j, -1, m + j - 1 + parity, m, validN, work, intervals);
                AddCausalColumnWork(core, 1, std::min(rounds, split - 1), pairedColumn, 1, m + NUM_TWO - j - parity, m,
                                    validN, work, intervals);
            }
        }
    }
    work.activeRounds += CountCausalActiveRounds(intervals);
}

// Exact useful-work count for CalGQACausalIndex, without enumerating its rounds.
// rm1/rm2 and complete 2K-column single-BN2 parts occupy every core. Only the
// remaining ell=n%K columns need interval counting: at most K/2 windows, each
// with at most two intervals per core. G repeats the same ownership pattern.
inline CausalWork CalcCausalDeterWork(int64_t k, int64_t m, int64_t n, int64_t b, int64_t g)
{
    CausalWork work;
    if (k <= 0) {
        return work;
    }
    work.coreTasks.resize(k, 0);
    if (m < n || n <= 0 || b <= 0 || g <= 0) {
        return work;
    }
    int64_t bTail = b % k;
    const int64_t t = n / k;
    const int64_t ell = n % k;
    const int64_t n1 = t * k;
    int64_t uniformRounds = b / k * (n * m - n * (n - 1) / NUM_TWO);
    uniformRounds += (NUM_TWO * m - n1 + 1) * t * (bTail / NUM_TWO);
    if (bTail % NUM_TWO != 0) {
        const int64_t pairedGroups = n / (NUM_TWO * k);
        const int64_t completeRounds = (NUM_TWO * m + 1) * pairedGroups - NUM_TWO * k * pairedGroups * pairedGroups;
        uniformRounds += completeRounds;
        if (t % NUM_TWO != 0) {
            const int64_t remainingRows = m - pairedGroups * NUM_TWO * k;
            const int64_t remainingColumns = n - pairedGroups * NUM_TWO * k;
            const int64_t remainingRounds =
                ell == 0 ? remainingRows : std::max(remainingRows, NUM_TWO * remainingRows - NUM_TWO * k + 1);
            const int64_t mapperN = pairedGroups == 0 ? n : remainingRows;
            AddCausalSingleRemainderWork(k, remainingRows, mapperN, remainingColumns, remainingRounds, work);
            --bTail;
        }
    }
    work.activeRounds += uniformRounds;
    for (auto& tasks : work.coreTasks) {
        tasks += uniformRounds;
    }
    if (ell > 0 && bTail > 0) {
        const int64_t delta = m - n;
        const int64_t parity = ell % NUM_TWO;
        const int64_t pairCount = (ell + 1) / NUM_TWO;
        const int64_t pairHeight = ell + 1 - parity;
        const int64_t period = pairHeight + NUM_TWO * delta;
        const int64_t fullWindows = bTail * pairCount / k;
        const int64_t residual = bTail * pairCount % k;
        const int64_t windows = fullWindows + (residual != 0);
        for (int64_t window = 0; window < windows; ++window) {
            std::vector<CausalRoundInterval> intervals;
            intervals.reserve(NUM_TWO * k);
            for (int64_t core = 0; core < k; ++core) {
                const int64_t j = core + 1;
                if (window == fullWindows && residual <= k / NUM_TWO) {
                    if (j <= residual) {
                        const int64_t column = pairCount - (residual - j) / bTail;
                        const int64_t limit = (residual - j) / bTail + (pairHeight + 1) / NUM_TWO + delta;
                        AddCausalColumnWork(core, 1, limit, column, 1, column - 1, ell + delta, ell, work, intervals);
                    } else if (j >= k - residual + 1) {
                        const int64_t offset = j - (k - residual + 1);
                        const int64_t column = pairCount + 1 + offset / bTail;
                        const int64_t limit = (pairHeight - 1) / NUM_TWO - offset / bTail + delta;
                        AddCausalColumnWork(core, 1, limit, column, 1, column - 1, ell + delta, ell, work, intervals);
                    }
                } else {
                    const int64_t pair = CeilDivideBy(k * window + j, bTail);
                    if (pair > pairCount) {
                        continue;
                    }
                    const int64_t pairedColumn = ell + 1 + parity - pair;
                    if (pair % NUM_TWO != 0) {
                        const int64_t split = ell - pair + 1 + delta;
                        AddCausalColumnWork(core, 1, std::min(period, split), pair, 1, pair - 1, ell + delta, ell, work,
                                            intervals);
                        AddCausalColumnWork(core, split + 1, period, pairedColumn, -1,
                                            NUM_TWO * ell + NUM_TWO * delta + 2 - pair, ell + delta, ell, work,
                                            intervals);
                    } else {
                        const int64_t split = pair + 1 + delta - parity;
                        AddCausalColumnWork(core, split, period, pair, -1, pair + ell + NUM_TWO * delta + 1 - parity,
                                            ell + delta, ell, work, intervals);
                        AddCausalColumnWork(core, 1, std::min(period, split - 1), pairedColumn, 1, ell - pair + parity,
                                            ell + delta, ell, work, intervals);
                    }
                }
            }
            work.activeRounds += CountCausalActiveRounds(intervals);
        }
    }
    work.activeRounds *= g;
    for (auto& tasks : work.coreTasks) {
        tasks *= g;
    }
    return work;
}

// Match the mapper's task count to the complete causal region, not to another
// potentially incomplete schedule. Split the even factor before multiplication.
inline bool HasCompleteCausalWork(const CausalWork& work, int64_t m, int64_t n, int64_t b, int64_t g)
{
    uint64_t columns = static_cast<uint64_t>(n);
    uint64_t heightSum = static_cast<uint64_t>(m) + static_cast<uint64_t>(m - n + 1);
    if (columns % NUM_TWO == 0) {
        columns /= NUM_TWO;
    } else {
        heightSum /= NUM_TWO;
    }
    const uint64_t regionTasks = SaturatingCausalMultiply(columns, heightSum);
    const uint64_t expectedTasks = SaturatingCausalMultiply(SaturatingCausalMultiply(regionTasks, b), g);
    if (expectedTasks == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    uint64_t actualTasks = 0;
    for (const auto tasks : work.coreTasks) {
        if (tasks < 0) {
            return false;
        }
        actualTasks = SaturatingCausalAdd(actualTasks, static_cast<uint64_t>(tasks));
    }
    return actualTasks == expectedTasks;
}

struct CausalGroupCost {
    int64_t activeRounds;
    int64_t maxCoreTasks;
    int64_t rounds;
    uint64_t spillBytes;
    int64_t groupCount;
    uint64_t peakBytes;
};

inline CausalGroupCost EstimateCausalGroupCost(int64_t b, int64_t groups, int64_t rounds, uint64_t bn2Bytes,
                                               uint64_t l2Bytes, const CausalWork& fullWork, const CausalWork& tailWork)
{
    const int64_t fullGroups = b / groups;
    const int64_t tailGroups = b % groups;
    const uint64_t fullBytes = SaturatingCausalMultiply(groups, bn2Bytes);
    const uint64_t tailBytes = SaturatingCausalMultiply(tailGroups, bn2Bytes);
    const uint64_t fullSpill = fullBytes > l2Bytes ? fullBytes - l2Bytes : 0;
    const uint64_t tailSpill = tailBytes > l2Bytes ? tailBytes - l2Bytes : 0;
    int64_t maxCoreTasks = 0;
    for (size_t core = 0; core < fullWork.coreTasks.size(); ++core) {
        maxCoreTasks = std::max(maxCoreTasks, fullGroups * fullWork.coreTasks[core] + tailWork.coreTasks[core]);
    }
    return {fullGroups * fullWork.activeRounds + tailWork.activeRounds,
            maxCoreTasks,
            rounds,
            SaturatingCausalAdd(SaturatingCausalMultiply(fullGroups, fullSpill), tailSpill),
            fullGroups + (tailGroups != 0),
            fullBytes};
}

inline bool IsBetterCausalGroupCost(const CausalGroupCost& candidate, const CausalGroupCost& incumbent)
{
    if (candidate.activeRounds != incumbent.activeRounds) {
        return candidate.activeRounds < incumbent.activeRounds;
    }
    if (candidate.maxCoreTasks != incumbent.maxCoreTasks) {
        return candidate.maxCoreTasks < incumbent.maxCoreTasks;
    }
    if (candidate.rounds != incumbent.rounds) {
        return candidate.rounds < incumbent.rounds;
    }
    if (candidate.spillBytes != incumbent.spillBytes) {
        return candidate.spillBytes < incumbent.spillBytes;
    }
    // With useful work, rounds and spill tied, avoid unnecessary changes of the
    // active BN2 group, then prefer the smaller peak working set.
    if (candidate.groupCount != incumbent.groupCount) {
        return candidate.groupCount < incumbent.groupCount;
    }
    return candidate.peakBytes < incumbent.peakBytes;
}
} // namespace fag_pto_arch35
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FIXED_CAUSAL_GROUPED_H_
