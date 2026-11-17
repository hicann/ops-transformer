/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_SCHEDULE_INT8_H_
#define QUANT_FLASH_ATTN_SCHEDULE_INT8_H_
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "../../op_kernel/arch22/quant_flash_attn_tiling_data_int8.h"
#define QFA_WORKLOAD_INLINE inline
#include "../../op_kernel/arch22/quant_flash_attn_workload_int8.h"
#undef QFA_WORKLOAD_INLINE
namespace BaseApi::Int8GqaTnd {
constexpr uint32_t QFA_MIN_CORES_FOR_TASK_MAP = 2U;
struct QfaInt8ScheduleInput {
    int64_t batch, headsQ, headsK, dim, outputDim, maxQ, maxK, totalQ, totalK;
    int64_t layoutQ, layoutK, layoutOut, mask;
    uint32_t aicNum, metadataCapacity;
    uint64_t l2Bytes, reservedWorkspaceBytes, tilingCapacity;
    bool hasMetadata;
    float softmaxScale;
    int32_t winLeft = QFA_UNBOUNDED_WINDOW, winRight = QFA_UNBOUNDED_WINDOW;
};
struct QfaInt8Schedule {
    QuantFlashAttnTilingQBlockPayloadInt8 payload{};
    uint64_t workspaceBytes = 0;
    uint32_t config = 0;
    bool hasTaskBuffer = false;
};
struct QBlockCoreWork {
    uint64_t cost;
    uint32_t core;
};

class QfaTaskHeap {
    std::vector<QBlockCoreWork> heap_;
    struct Compare {
        bool operator()(const QBlockCoreWork& a, const QBlockCoreWork& b) const
        {
            return a.cost != b.cost ? a.cost > b.cost : a.core > b.core;
        }
    };

public:
    QfaTaskHeap(uint32_t first, uint32_t count)
    {
        for (uint32_t core = first; core < first + count; ++core)
            heap_.push_back({0U, core});
        std::make_heap(heap_.begin(), heap_.end(), Compare{});
    }
    uint32_t Assign(uint64_t cost)
    {
        std::pop_heap(heap_.begin(), heap_.end(), Compare{});
        auto& work = heap_.back();
        const uint32_t core = work.core;
        work.cost += cost;
        std::push_heap(heap_.begin(), heap_.end(), Compare{});
        return core;
    }
};

template <class TaskMap>
static bool StoreQBlockTaskMap(const QuantFlashAttnTilingPayloadInt8& payload,
                               QuantFlashAttnTilingQBlockPayloadInt8& mapped,
                               const std::vector<std::vector<uint32_t>>& tasks, TaskMap mapTask,
                               uint32_t* metadataIds = nullptr)
{
    const auto& t = payload.config;
    mapped = {};
    mapped.base = payload;
    mapped.base.useCoreRanges = QFA_CORE_RANGES_TASK_MAP;
    uint32_t index = 0U;
    uint32_t* ids = metadataIds == nullptr ? mapped.taskIds : metadataIds;
    for (uint32_t core = 0U; core < t.usedCoreNum; ++core) {
        mapped.base.coreTaskBegin[core] = index;
        for (uint32_t task : tasks[core])
            ids[index++] = mapTask(task);
    }
    mapped.base.coreTaskBegin[t.usedCoreNum] = index;
    return index == t.totalTasks;
}

template <class Payload>
static bool QBlockMapImproves(const Payload& payload, const std::vector<std::vector<uint32_t>>& tasks,
                              const std::vector<uint64_t>& kvWork, uint32_t costGroup)
{
    const auto& t = payload.config;
    uint64_t oldMaxCost = 0U, oldMaxKv = 0U, newMaxCost = 0U, newMaxKv = 0U;
    uint32_t oldMaxTasks = 0U, newMaxTasks = 0U;
    for (uint32_t core = 0U; core < t.usedCoreNum; ++core) {
        const uint32_t begin = payload.coreTaskBegin[core], end = payload.coreTaskBegin[core + 1U];
        if (begin > end || end > t.totalTasks || tasks[core].empty())
            return false;
        const uint32_t count = end - begin;
        const uint64_t cost =
            BaseApi::Int8GqaTnd::QfaSequenceWork(end, t.seqLenQ, t.seqLenKv, costGroup, t.blockM, t.blockN, true) -
            BaseApi::Int8GqaTnd::QfaSequenceWork(begin, t.seqLenQ, t.seqLenKv, costGroup, t.blockM, t.blockN, true);
        oldMaxCost = std::max(oldMaxCost, cost - count);
        oldMaxKv = std::max(oldMaxKv, cost - QFA_CORE_WORK_SETUP_STAGES * count);
        oldMaxTasks = std::max(oldMaxTasks, count);
        newMaxCost = std::max(newMaxCost, kvWork[core] + tasks[core].size());
        newMaxKv = std::max(newMaxKv, kvWork[core]);
        newMaxTasks = std::max(newMaxTasks, static_cast<uint32_t>(tasks[core].size()));
    }
    // Require at least 3% lower total cost and 5% lower KV cost, with no extra tasks on the busiest core.
    constexpr uint32_t percentScale = 100U;
    constexpr uint32_t maxTotalCostPercent = 97U;
    constexpr uint32_t maxKvCostPercent = 95U;
    if (percentScale * newMaxCost > maxTotalCostPercent * oldMaxCost ||
        percentScale * newMaxKv > maxKvCostPercent * oldMaxKv || newMaxTasks > oldMaxTasks)
        return false;
    return true;
}

static bool BuildD64HeadLocalQBlockMap(const QuantFlashAttnTilingPayloadInt8& payload,
                                       QuantFlashAttnTilingQBlockPayloadInt8& mapped)
{
    const auto& t = payload.config;
    constexpr uint32_t capacity = sizeof(mapped.taskIds) / sizeof(mapped.taskIds[0]);
    if (t.batchSize != 1U || (t.layoutQ != QFA_LAYOUT_BSND && t.layoutQ != QFA_LAYOUT_TND) || t.layoutKv != t.layoutQ ||
        t.layoutOut != t.layoutQ || t.maskMode != QFA_MASK_CAUSAL_TILING || t.headDim != QFA_HEAD_DIM_64 ||
        t.blockM != QFA_LARGE_QUERY_TILE_ROWS || t.blockN != QFA_DEFAULT_KV_TILE_TOKENS || t.seqLenQ != t.seqLenKv ||
        t.seqLenQ <= QFA_LARGE_QUERY_TILE_ROWS || t.seqLenQ % QFA_LARGE_QUERY_TILE_ROWS != 0U ||
        payload.useCoreRanges != QFA_CORE_RANGES_CONTIGUOUS || t.groupSize == 0U || t.numHeadsKv == 0U ||
        t.usedCoreNum < QFA_MIN_CORES_FOR_TASK_MAP || t.usedCoreNum > QFA_MAX_AIC_CORES ||
        t.usedCoreNum % t.numHeadsKv != 0U || t.totalTasks < t.usedCoreNum || t.totalTasks > capacity ||
        // The existing packed scheduler tests mirror-pair mapping before host ranges. Do not send original task IDs
        // into that Decode path.
        (t.layoutQ == QFA_LAYOUT_TND && t.totalTilesM % QFA_MIRRORED_TASK_PAIR_SIZE == 0U &&
         (t.totalTasks / QFA_MIRRORED_TASK_PAIR_SIZE) % t.usedCoreNum == 0U) ||
        (t.layoutQ == QFA_LAYOUT_TND && (t.totalTokensQ != t.seqLenQ || t.totalTokensKv != t.seqLenKv)))
        return false;
    const uint32_t qTiles = t.seqLenQ / t.blockM;
    const uint64_t folded = static_cast<uint64_t>(qTiles) * t.groupSize;
    if (folded != t.totalTilesM || folded * t.numHeadsKv != t.totalTasks || payload.coreTaskBegin[0] != 0U ||
        payload.coreTaskBegin[t.usedCoreNum] != t.totalTasks)
        return false;

    const uint32_t coresPerHead = t.usedCoreNum / t.numHeadsKv;
    std::vector<std::vector<uint32_t>> tasks(t.usedCoreNum);
    std::vector<uint64_t> kvWork(t.usedCoreNum, 0U);
    for (uint32_t head = 0U; head < t.numHeadsKv; ++head) {
        QfaTaskHeap heap(head * coresPerHead, coresPerHead);
        // Entire M512 Q tasks remain intact. A KV tile contains G phases in S1G order. Allocate long tasks first,
        // then execute each core's map in increasing original ID so its one K-scale cache is never evicted.
        for (uint32_t tile = qTiles; tile > 0U; --tile) {
            for (uint32_t phase = 0U; phase < t.groupSize; ++phase) {
                const uint32_t core = heap.Assign(tile + 1U);
                const uint32_t task = head * t.totalTilesM + (tile - 1U) * t.groupSize + phase;
                tasks[core].push_back(task);
                kvWork[core] += tile;
            }
        }
    }

    if (!QBlockMapImproves(payload, tasks, kvWork, t.groupSize))
        return false;
    for (auto& coreTasks : tasks)
        std::sort(coreTasks.begin(), coreTasks.end());
    return StoreQBlockTaskMap(payload, mapped, tasks, [](uint32_t task) { return task; });
}

// Minimize the largest contiguous core cost for causal prefill. Count visible KV tiles plus two pipeline stages per
// Q task, as in the existing BNSD cost model. Prefix sums keep host work independent of S^2.
static void BuildDenseCoreRanges(QuantFlashAttnTilingPayloadInt8& payload)
{
    const auto& t = payload.config;
    // BNSD causal tasks never cross a query-head boundary. This also makes the group=1 prefix exact when S1 has a
    // partial final M tile.
    const bool headAligned = QfaUseHeadAlignedTiles(t.seqLenQ, t.seqLenKv, t.headDim, t.blockM);
    const bool bnsd = t.layoutQ == QFA_LAYOUT_BNSD && t.layoutKv == QFA_LAYOUT_BNSD && t.layoutOut == QFA_LAYOUT_BNSD &&
                      (headAligned || t.seqLenQ % t.blockM == 0U);
    const bool bsnd = t.layoutQ == QFA_LAYOUT_BSND && t.layoutKv == QFA_LAYOUT_BSND;
    // Dense INT8 metadata is bound to the full shape lengths for every batch. Static scheduling does not depend on
    // whether that descriptor is supplied. QfaSequenceWork forms signed G * (Sk - Sq) and adds at most one M tile
    // of padding; bound that arithmetic before enabling unequal lengths.
    const bool unequalBsnd = bsnd && t.layoutOut == QFA_LAYOUT_BSND && t.seqLenQ != t.seqLenKv &&
                             static_cast<uint64_t>(t.groupSize) * t.seqLenKv <=
                                 static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - t.blockM;
    const bool singleTnd = t.layoutQ == QFA_LAYOUT_TND && t.layoutKv == QFA_LAYOUT_TND && t.batchSize == 1U &&
                           t.seqLenQ == t.totalTokensQ && t.seqLenKv == t.totalTokensKv;
    if ((!bnsd && !bsnd && !singleTnd) || t.maskMode != QFA_MASK_CAUSAL_TILING ||
        (t.seqLenQ != t.seqLenKv && !unequalBsnd) || t.usedCoreNum > QFA_MAX_AIC_CORES)
        return;
    // Include partial Q/KV tails in the cost partition.
    auto prefix = [&](uint64_t n) {
        return BaseApi::Int8GqaTnd::QfaSequenceWork(n, t.seqLenQ, t.seqLenKv, bnsd ? 1U : t.groupSize, t.blockM,
                                                    t.blockN, true);
    };
    auto partition = [&](uint64_t limit, bool save) {
        uint32_t start = 0;
        for (uint32_t core = 0; core < t.usedCoreNum; ++core) {
            if (save)
                payload.coreTaskBegin[core] = start;
            uint32_t lo = start, hi = t.totalTasks;
            const uint64_t base = prefix(start);
            while (lo < hi) {
                const uint32_t mid = lo + (hi - lo + 1U) / 2U;
                if (prefix(mid) - base <= limit)
                    lo = mid;
                else
                    hi = mid - 1U;
            }
            start = lo;
        }
        if (save)
            payload.coreTaskBegin[t.usedCoreNum] = start;
        return start == t.totalTasks;
    };
    uint64_t lo = (prefix(t.totalTasks) + t.usedCoreNum - 1U) / t.usedCoreNum;
    uint64_t hi = prefix(t.totalTasks);
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2U;
        if (partition(mid, false))
            hi = mid;
        else
            lo = mid + 1U;
    }
    partition(lo, true);
    payload.useCoreRanges = QFA_CORE_RANGES_CONTIGUOUS;
}

static void RefineQBlockTaskMap(std::vector<std::vector<uint32_t>>& tasks, std::vector<uint64_t>& kvWork,
                                uint32_t tilesPerHead, uint32_t m = QFA_LARGE_QUERY_TILE_ROWS,
                                uint32_t n = QFA_DEFAULT_KV_TILE_TOKENS)
{
    uint32_t taskLimit = 0U;
    for (const auto& core : tasks)
        taskLimit = std::max(taskLimit, static_cast<uint32_t>(core.size()));
    constexpr uint32_t maxTasksPerCoreForRefinement = 8U;
    constexpr uint32_t maxRefinementPasses = 8U;
    constexpr uint32_t maxCandidatesPerPass = 65536U;
    if (taskLimit > maxTasksPerCoreForRefinement)
        return;
    const uint64_t kvLimit = *std::max_element(kvWork.begin(), kvWork.end());
    std::vector<uint64_t> cost(tasks.size());
    auto tileCount = [&](uint32_t task) { return ((task % tilesPerHead + 1ULL) * m + n - 1U) / n; };
    for (uint32_t pass = 0U; pass < maxRefinementPasses; ++pass) {
        for (uint32_t core = 0U; core < tasks.size(); ++core)
            cost[core] = kvWork[core] + tasks[core].size();
        const uint64_t high = *std::max_element(cost.begin(), cost.end());
        const uint64_t low = *std::min_element(cost.begin(), cost.end());
        if (high - low <= 1U)
            break;
        uint64_t bestGain = 0U, bestKvDelta = 0U;
        uint32_t bestHigh = 0U, bestLow = 0U, bestFirst = 0U, bestSecond = 0U, bestLowTask = 0U;
        bool bestTwo = false;
        uint32_t budget = maxCandidatesPerPass;
        for (uint32_t h = 0U; h < tasks.size() && budget != 0U; ++h) {
            if (cost[h] != high)
                continue;
            for (uint32_t l = 0U; l < tasks.size() && budget != 0U; ++l) {
                if (cost[l] != low)
                    continue;
                for (uint32_t i = 0U; i < tasks[h].size() && budget != 0U; ++i) {
                    for (uint32_t j = i; j < tasks[h].size() && budget != 0U; ++j) {
                        const bool two = j != i;
                        if (two && tasks[l].size() >= taskLimit)
                            continue;
                        const uint64_t give = tileCount(tasks[h][i]) + 1U + (two ? tileCount(tasks[h][j]) + 1U : 0U);
                        for (uint32_t k = 0U; k < tasks[l].size() && budget != 0U; ++k) {
                            --budget;
                            const int64_t delta = static_cast<int64_t>(give) - tileCount(tasks[l][k]) - 1;
                            if (delta <= 0 || static_cast<uint64_t>(delta) >= high - low)
                                continue;
                            const uint64_t kvDelta = static_cast<uint64_t>(delta) - static_cast<uint32_t>(two);
                            if (kvWork[l] + kvDelta > kvLimit)
                                continue;
                            // Reduction in squared load when moving delta work from the heavy core to the light core.
                            const uint64_t gain = 2U * delta * (high - low - delta);
                            if (gain > bestGain || (gain == bestGain && bestTwo && !two)) {
                                bestGain = gain;
                                bestKvDelta = kvDelta;
                                bestHigh = h;
                                bestLow = l;
                                bestFirst = i;
                                bestSecond = j;
                                bestLowTask = k;
                                bestTwo = two;
                            }
                        }
                    }
                }
            }
        }
        if (bestGain == 0U)
            break;
        const uint32_t first = tasks[bestHigh][bestFirst], second = tasks[bestHigh][bestSecond];
        tasks[bestHigh][bestFirst] = tasks[bestLow][bestLowTask];
        tasks[bestLow][bestLowTask] = first;
        if (bestTwo) {
            tasks[bestLow].push_back(second);
            tasks[bestHigh].erase(tasks[bestHigh].begin() + bestSecond);
        }
        kvWork[bestHigh] -= bestKvDelta;
        kvWork[bestLow] += bestKvDelta;
    }
}

static bool BuildQBlockTaskMap(const QuantFlashAttnTilingPayloadInt8& payload,
                               QuantFlashAttnTilingQBlockPayloadInt8& mapped)
{
    const auto& t = payload.config;
    const bool bnsd = t.layoutQ == QFA_LAYOUT_BNSD && t.layoutKv == QFA_LAYOUT_BNSD && t.layoutOut == QFA_LAYOUT_BNSD &&
                      t.headDim == QFA_HEAD_DIM_96;
    const bool bsnd = t.layoutQ == QFA_LAYOUT_BSND && t.layoutKv == QFA_LAYOUT_BSND && t.layoutOut == QFA_LAYOUT_BSND &&
                      t.headDim == QFA_HEAD_DIM_96;
    const bool singleTnd = t.layoutQ == QFA_LAYOUT_TND && t.layoutKv == QFA_LAYOUT_TND &&
                           t.layoutOut == QFA_LAYOUT_TND && t.headDim == QFA_HEAD_DIM_128 && t.batchSize == 1U &&
                           t.totalTokensQ == t.seqLenQ && t.totalTokensKv == t.seqLenKv;
    if ((!bnsd && !bsnd && !singleTnd) || t.maskMode != QFA_MASK_CAUSAL_TILING ||
        (t.blockM != QFA_LARGE_QUERY_TILE_ROWS &&
         !(bsnd && t.blockM == QFA_PV_OVERLAY_MIN_QUERY_TILE && t.groupSize == 1U)) ||
        t.blockN != QFA_DEFAULT_KV_TILE_TOKENS || t.seqLenQ != t.seqLenKv || t.seqLenQ <= QFA_LARGE_QUERY_TILE_ROWS ||
        t.seqLenQ % QFA_LARGE_QUERY_TILE_ROWS != 0U || payload.useCoreRanges != QFA_CORE_RANGES_CONTIGUOUS ||
        t.usedCoreNum < QFA_MIN_CORES_FOR_TASK_MAP || t.usedCoreNum > QFA_MAX_AIC_CORES ||
        t.totalTasks < t.usedCoreNum || t.totalTasks > QFA_Q_BLOCK_TASK_CAPACITY || t.groupSize == 0U ||
        t.batchSize == 0U || t.numHeadsKv == 0U || payload.coreTaskBegin[0] != 0U ||
        payload.coreTaskBegin[t.usedCoreNum] != t.totalTasks)
        return false;
    // The packed scheduler already has an optimal constant-work pairing when every core receives the same number of
    // mirror pairs.
    if (singleTnd && t.totalTilesM % QFA_MIRRORED_TASK_PAIR_SIZE == 0U &&
        (t.totalTasks / QFA_MIRRORED_TASK_PAIR_SIZE) % t.usedCoreNum == 0U)
        return false;
    const uint64_t foldedTiles = static_cast<uint64_t>(t.seqLenQ / t.blockM) * t.groupSize;
    if (foldedTiles != t.totalTilesM || static_cast<uint64_t>(t.batchSize) * t.numHeadsKv * foldedTiles != t.totalTasks)
        return false;

    const uint32_t costGroup = bnsd ? 1U : t.groupSize;
    const uint32_t tilesPerHead = t.seqLenQ / t.blockM;
    const uint32_t heads = t.totalTasks / tilesPerHead;
    QfaTaskHeap heap(0U, t.usedCoreNum);
    std::vector<std::vector<uint32_t>> tasks(t.usedCoreNum);
    std::vector<uint64_t> kvWork(t.usedCoreNum, 0U);
    // A folded BSND task stays intact. Fake IDs arrange its phase within a token tile as a head axis, so the
    // existing N+1 refinement can be reused. Mapping back below restores the original S1G task before device
    // decode.
    for (uint32_t tile = tilesPerHead; tile > 0U; --tile) {
        for (uint32_t head = 0U; head < heads; ++head) {
            const uint64_t visible = (static_cast<uint64_t>(tile) * t.blockM + t.blockN - 1U) / t.blockN;
            const uint32_t core = heap.Assign(visible + 1U);
            tasks[core].push_back(head * tilesPerHead + tile - 1U);
            kvWork[core] += visible;
        }
    }
    RefineQBlockTaskMap(tasks, kvWork, tilesPerHead, t.blockM, t.blockN);

    if (!QBlockMapImproves(payload, tasks, kvWork, costGroup))
        return false;
    return StoreQBlockTaskMap(payload, mapped, tasks, [&](uint32_t task) {
        if (bnsd)
            return task;
        const uint32_t headPhase = task / tilesPerHead;
        const uint32_t tokenTile = task % tilesPerHead;
        const uint32_t batchKvHead = headPhase / t.groupSize;
        const uint32_t phase = headPhase % t.groupSize;
        return batchKvHead * t.totalTilesM + tokenTile * t.groupSize + phase;
    });
}

inline bool BuildInt8Schedule(const QfaInt8ScheduleInput& input, QfaInt8Schedule& result)
{
    const int64_t b = input.batch, hq = input.headsQ, hkv = input.headsK, d = input.dim;
    const int64_t sq = input.maxQ, sk = input.maxK, outputDim = input.outputDim;
    const int64_t lq = input.layoutQ, lk = input.layoutK, lo = input.layoutOut;
    const int64_t totalQ = input.totalQ, totalK = input.totalK;
    if (b <= 0 || hq <= 0 || hkv <= 0 || hq % hkv != 0 || sq <= 0 || sk <= 0 || d <= 0 || d > QFA_MAX_HEAD_DIM)
        return false;
    const uint32_t metadataCapacity = input.metadataCapacity;
    const uint32_t aicNum = input.hasMetadata ? std::min(input.aicNum, metadataCapacity) : input.aicNum;
    const uint64_t l2Bytes = input.l2Bytes, reservedWorkspaceBytes = input.reservedWorkspaceBytes;
    const bool hasMetadata = input.hasMetadata;
    const bool windowed = input.winLeft >= 0 || input.winRight >= 0;
    // Packed total tokens bound core coverage without reading cumulative lengths on the host.
    const uint64_t groupedM = static_cast<uint64_t>(hq / hkv) * sq;
    if (groupedM > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint64_t group = static_cast<uint64_t>(hq / hkv);
    const bool bnsdCausal =
        lq == QFA_LAYOUT_BNSD && lk == QFA_LAYOUT_BNSD && lo == QFA_LAYOUT_BNSD && input.mask == QFA_MASK_CAUSAL_ATTR;
    auto headAlignedForM = [&](uint32_t m) {
        return bnsdCausal && QfaUseHeadAlignedTiles(static_cast<uint64_t>(sq), static_cast<uint64_t>(sk),
                                                    static_cast<uint32_t>(d), m);
    };
    auto tilesForM = [&](uint32_t m) {
        return headAlignedForM(m) ? group * ((static_cast<uint64_t>(sq) + m - 1U) / m) : (groupedM + m - 1U) / m;
    };
    auto coversCores = [&](uint32_t m) {
        const uint64_t tiles = tilesForM(m);
        if (tiles == 0U || static_cast<uint64_t>(b) * hkv < (aicNum + tiles - 1U) / tiles)
            return false;
        if (lq == QFA_LAYOUT_TND) {
            // Hkv * ceil(totalTokensQ * G / M) >= AIC. Rearranging the small right-hand threshold avoids
            // multiplying large token counts.
            const uint64_t neededTiles = (static_cast<uint64_t>(aicNum) + hkv - 1U) / hkv;
            return neededTiles == 0U || static_cast<uint64_t>(totalQ) > (neededTiles - 1U) * m / group;
        }
        return true;
    };
    auto saturatingMul = [](uint64_t left, uint64_t right) {
        const uint64_t limit = std::numeric_limits<uint64_t>::max();
        return right != 0U && left > limit / right ? limit : left * right;
    };
    auto saturatingAdd = [](uint64_t left, uint64_t right) {
        const uint64_t limit = std::numeric_limits<uint64_t>::max();
        return left > limit - right ? limit : left + right;
    };
    const uint64_t totalQTokens = lq == QFA_LAYOUT_TND ?
                                      static_cast<uint64_t>(totalQ) :
                                      saturatingMul(static_cast<uint64_t>(b), static_cast<uint64_t>(sq));
    const uint64_t totalKvTokens = lk == QFA_LAYOUT_TND ?
                                       static_cast<uint64_t>(totalK) :
                                       saturatingMul(static_cast<uint64_t>(b), static_cast<uint64_t>(sk));
    // Each K/V token contributes two INT8 vectors and one FP32 K scale per head.
    const uint64_t inputWorkingSet = saturatingAdd(
        saturatingMul(saturatingMul(totalQTokens, static_cast<uint64_t>(hq)), static_cast<uint64_t>(d) + sizeof(float)),
        saturatingMul(saturatingMul(totalKvTokens, static_cast<uint64_t>(hkv)),
                      (sizeof(int8_t) + sizeof(int8_t)) * static_cast<uint64_t>(d) + sizeof(float)));
    // Reserve the library workspace and the M256 score/P working tile before deciding whether the INT8 inputs can
    // be retained in L2. Cache-resident workloads keep M128's larger vector chunks; unknown L2 does too.
    constexpr uint32_t m256ScoreAndPBytes =
        QFA_PV_OVERLAY_MIN_QUERY_TILE * QFA_DEFAULT_KV_TILE_TOKENS * (sizeof(uint16_t) + sizeof(int8_t));
    const uint64_t m256TileWorkingSet = saturatingMul(aicNum, m256ScoreAndPBytes);
    const uint64_t workspaceWorkingSet = saturatingAdd(reservedWorkspaceBytes, m256TileWorkingSet);
    const uint64_t inputCacheBytes = l2Bytes > workspaceWorkingSet ? l2Bytes - workspaceWorkingSet : 0U;
    const bool inputExceedsL2 = l2Bytes != 0U && inputWorkingSet > inputCacheBytes;
    // A 16-row vector chunk leaves room for 128xD FP32 accumulation on each AIV. M256 then shares every KV panel
    // across twice as many Q rows. Preserve enough independent tasks and the existing Q L1 capacity.
    const uint64_t largeDAlign =
        (static_cast<uint64_t>(d) + QFA_HEAD_DIM_ALIGNMENT - 1U) / QFA_HEAD_DIM_ALIGNMENT * QFA_HEAD_DIM_ALIGNMENT;
    const bool largeFp32M256 =
        !windowed && group > 1U && d > QFA_LARGE_FP32_HEAD_DIM_THRESHOLD && sk > QFA_DEFAULT_KV_TILE_TOKENS &&
        inputExceedsL2 && QFA_PV_OVERLAY_MIN_QUERY_TILE * largeDAlign <= QFA_Q_L1_TILE_BYTES &&
        (QFA_PV_OVERLAY_MIN_QUERY_TILE / QFA_VECTOR_CORES_PER_CUBE) * largeDAlign * sizeof(float) <=
            QFA_LARGE_FP32_ACCUMULATOR_BYTES &&
        tilesForM(QFA_PV_OVERLAY_MIN_QUERY_TILE) < tilesForM(QFA_DEFAULT_QUERY_TILE_ROWS) &&
        coversCores(QFA_PV_OVERLAY_MIN_QUERY_TILE);
    uint32_t blockM = QFA_DEFAULT_QUERY_TILE_ROWS;
    constexpr uint32_t maxHeadDimForShortQueryM512 = 32U;
    constexpr uint32_t minQueryLengthForLargerTile = 16U;
    bool useDynamicMhaTile = false;
    bool useCausalMhaMap = false;
    if (!windowed) {
        if (d <= QFA_HEAD_DIM_128) {
            if (sq > QFA_LARGE_QUERY_TILE_ROWS) {
                blockM = QFA_LARGE_QUERY_TILE_ROWS;
            } else if (d <= maxHeadDimForShortQueryM512 && sq > minQueryLengthForLargerTile &&
                       coversCores(QFA_LARGE_QUERY_TILE_ROWS)) {
                blockM = QFA_LARGE_QUERY_TILE_ROWS;
            }
        } else {
            if (sq <= QFA_LARGE_QUERY_TILE_ROWS && sk <= QFA_DEFAULT_KV_TILE_TOKENS &&
                coversCores(QFA_PV_OVERLAY_MIN_QUERY_TILE)) {
                // One PV tile needs no running FP32 accumulation; M256 fits the existing 64 KiB accumulator as FP16
                // for D<=256.
                blockM = QFA_PV_OVERLAY_MIN_QUERY_TILE;
            } else if (largeFp32M256) {
                blockM = QFA_PV_OVERLAY_MIN_QUERY_TILE;
            } else {
                blockM = sq > QFA_LARGE_QUERY_TILE_ROWS ||
                                 (sq > minQueryLengthForLargerTile && coversCores(QFA_DEFAULT_QUERY_TILE_ROWS)) ?
                             QFA_DEFAULT_QUERY_TILE_ROWS :
                             QFA_MIN_QUERY_TILE_ROWS;
            }
        }

        // Fit sparse long-MHA grids to the existing number of core waves. A 32-row granularity keeps both AIV
        // halves aligned without adding more Q tasks per core. Mixed packed lengths retain their device scheduler.
        constexpr uint32_t maxMhaQueryTile = QFA_LARGE_QUERY_TILE_ROWS;
        constexpr uint32_t mhaQueryTileAlignment = 32U;
        // Require a modeled critical-path improvement of at least 10%.
        constexpr uint32_t mhaCostDenominator = 10U, mhaMaxCostNumerator = 9U;
        constexpr uint32_t taskSetupRows = 32U;
        // The alternate M256 map doubles the M512 tasks. Restrict the search to at most two M512 waves.
        constexpr uint64_t maxM512WavesForMap = 2U;
        constexpr uint32_t minMhaQueryTilesForRetiling = 2U;
        if (lq == QFA_LAYOUT_BSND && lk == QFA_LAYOUT_BSND && lo == QFA_LAYOUT_BSND && group == 1U &&
            d == QFA_HEAD_DIM_96 && input.mask == QFA_MASK_CAUSAL_ATTR && sq == sk &&
            sq > minMhaQueryTilesForRetiling * maxMhaQueryTile && sq % QFA_LARGE_QUERY_TILE_ROWS == 0 &&
            blockM == QFA_LARGE_QUERY_TILE_ROWS && aicNum != 0U && aicNum <= QFA_MAX_AIC_CORES) {
            const uint64_t batchHeads = static_cast<uint64_t>(b) * hkv;
            const uint64_t oldTasks = batchHeads * (sq / QFA_LARGE_QUERY_TILE_ROWS);
            constexpr uint32_t m256TasksPerM512Task = QFA_LARGE_QUERY_TILE_ROWS / QFA_PV_OVERLAY_MIN_QUERY_TILE;
            const uint64_t nextTasks = m256TasksPerM512Task * oldTasks;
            if (oldTasks <= maxM512WavesForMap * aicNum && nextTasks <= std::numeric_limits<uint32_t>::max()) {
                // Compare M256's critical range against a lower bound for even an optimally paired M512 schedule.
                // Charge fixed setup per task and require 15% headroom before giving up the tuned M512 path.
                constexpr uint64_t setupCost = 64U;
                const uint64_t oldCores = std::min<uint64_t>(aicNum, oldTasks);
                const uint64_t oldWork = BaseApi::Int8GqaTnd::QfaSequenceWork(
                    oldTasks, sq, sk, 1U, QFA_LARGE_QUERY_TILE_ROWS, QFA_DEFAULT_KV_TILE_TOKENS, true);
                const uint64_t oldFloor = std::max<uint64_t>(
                    (oldWork * QFA_LARGE_QUERY_TILE_ROWS + setupCost * oldTasks + oldCores - 1U) / oldCores,
                    (sk / QFA_DEFAULT_KV_TILE_TOKENS + QFA_CORE_WORK_SETUP_STAGES) * QFA_LARGE_QUERY_TILE_ROWS +
                        setupCost);
                QuantFlashAttnTilingPayloadInt8 model{};
                auto& mt = model.config;
                mt.batchSize = b;
                mt.numHeadsQ = hq;
                mt.numHeadsKv = hkv;
                mt.seqLenQ = sq;
                mt.seqLenKv = sk;
                mt.headDim = d;
                mt.groupSize = 1U;
                mt.blockM = QFA_PV_OVERLAY_MIN_QUERY_TILE;
                mt.blockN = QFA_DEFAULT_KV_TILE_TOKENS;
                mt.maskMode = QFA_MASK_CAUSAL_TILING;
                mt.layoutQ = lq;
                mt.layoutKv = lk;
                mt.layoutOut = lo;
                mt.totalTilesM = sq / QFA_PV_OVERLAY_MIN_QUERY_TILE;
                mt.totalTasks = nextTasks;
                mt.usedCoreNum = std::min<uint64_t>(aicNum, nextTasks);
                BuildDenseCoreRanges(model);
                QuantFlashAttnTilingQBlockPayloadInt8 taskMap{};
                const bool mapped = BuildQBlockTaskMap(model, taskMap);
                uint64_t critical = 0U;
                for (uint32_t core = 0U; core < mt.usedCoreNum; ++core) {
                    const auto& ranges = mapped ? taskMap.base : model;
                    const uint64_t begin = ranges.coreTaskBegin[core], end = ranges.coreTaskBegin[core + 1U];
                    uint64_t work = 0U;
                    for (uint32_t index = begin; index < end; ++index) {
                        const uint32_t task = mapped ? taskMap.taskIds[index] : index;
                        work +=
                            BaseApi::Int8GqaTnd::QfaSequenceWork(task + 1U, sq, sk, 1U, QFA_PV_OVERLAY_MIN_QUERY_TILE,
                                                                 QFA_DEFAULT_KV_TILE_TOKENS, true) -
                            BaseApi::Int8GqaTnd::QfaSequenceWork(task, sq, sk, 1U, QFA_PV_OVERLAY_MIN_QUERY_TILE,
                                                                 QFA_DEFAULT_KV_TILE_TOKENS, true);
                    }
                    critical = std::max(critical, work * QFA_PV_OVERLAY_MIN_QUERY_TILE + setupCost * (end - begin));
                }
                constexpr uint32_t percentScale = 100U, causalMhaMaxCostPercent = 85U;
                const uint64_t threshold = oldFloor / percentScale * causalMhaMaxCostPercent +
                                           oldFloor % percentScale * causalMhaMaxCostPercent / percentScale;
                if (mapped && critical <= threshold) {
                    blockM = QFA_PV_OVERLAY_MIN_QUERY_TILE;
                    useDynamicMhaTile = true;
                    useCausalMhaMap = true;
                }
            }
        }
        if (group == 1U && input.mask == QFA_MASK_NONE && blockM >= QFA_DEFAULT_QUERY_TILE_ROWS && aicNum != 0U) {
            const uint64_t batchKvHeads = static_cast<uint64_t>(b) * hkv;
            if (batchKvHeads <= std::numeric_limits<uint32_t>::max()) {
                const uint64_t initialTasks = batchKvHeads * tilesForM(blockM);
                const uint64_t initialCores = std::min<uint64_t>(aicNum, initialTasks);
                const uint64_t initialWaves = (initialTasks + initialCores - 1U) / initialCores;
                const bool uniformPacked = lq != QFA_LAYOUT_TND || (totalQTokens == static_cast<uint64_t>(b) * sq &&
                                                                    totalKvTokens == static_cast<uint64_t>(b) * sk);
                constexpr uint32_t maxInitialWavesForRetiling = 2U;
                if (sq > minMhaQueryTilesForRetiling * maxMhaQueryTile && d > QFA_HEAD_DIM_64 &&
                    d <= QFA_HEAD_DIM_128 && sk > QFA_DEFAULT_KV_TILE_TOKENS && blockM == QFA_LARGE_QUERY_TILE_ROWS &&
                    (initialWaves <= maxInitialWavesForRetiling || (lq == QFA_LAYOUT_BSND && lk == QFA_LAYOUT_BSND &&
                                                                    lo == QFA_LAYOUT_BSND && d == QFA_HEAD_DIM_96)) &&
                    uniformPacked) {
                    const uint64_t targetTiles = initialWaves * aicNum / batchKvHeads;
                    if (targetTiles != 0U) {
                        const uint64_t rows = (static_cast<uint64_t>(sq) + targetTiles - 1U) / targetTiles;
                        const uint64_t candidate =
                            (rows + mhaQueryTileAlignment - 1U) / mhaQueryTileAlignment * mhaQueryTileAlignment;
                        if (candidate > QFA_PV_OVERLAY_MIN_QUERY_TILE && candidate < maxMhaQueryTile) {
                            const uint64_t tasks = batchKvHeads * tilesForM(static_cast<uint32_t>(candidate));
                            const uint64_t waves = (tasks + aicNum - 1U) / aicNum;
                            const uint64_t oldRows = initialWaves * blockM;
                            const uint64_t criticalRows = waves * candidate;
                            if (tasks <= std::numeric_limits<uint32_t>::max() && waves <= initialWaves &&
                                mhaCostDenominator * criticalRows <= mhaMaxCostNumerator * oldRows) {
                                blockM = static_cast<uint32_t>(candidate);
                                useDynamicMhaTile = true;
                            }
                        }
                    }
                } else if (sq <= minMhaQueryTilesForRetiling * maxMhaQueryTile) {
                    auto criticalCost = [&](uint32_t m) {
                        const uint64_t tasks = batchKvHeads * tilesForM(m);
                        const uint64_t cores = std::min<uint64_t>(aicNum, tasks);
                        return ((tasks + cores - 1U) / cores) * (m + taskSetupRows);
                    };
                    uint64_t bestCost = criticalCost(blockM);
                    for (uint32_t candidateM :
                         {QFA_PV_OVERLAY_MIN_QUERY_TILE, QFA_DEFAULT_QUERY_TILE_ROWS, QFA_MIN_QUERY_TILE_ROWS}) {
                        if (candidateM >= blockM ||
                            batchKvHeads * tilesForM(candidateM) > std::numeric_limits<uint32_t>::max())
                            continue;
                        const uint64_t cost = criticalCost(candidateM);
                        if (mhaCostDenominator * cost <= mhaMaxCostNumerator * bestCost) {
                            blockM = candidateM;
                            bestCost = cost;
                        }
                    }
                }
            }
        }
    }
    const uint32_t dAlign =
        (static_cast<uint32_t>(d) + QFA_HEAD_DIM_ALIGNMENT - 1U) / QFA_HEAD_DIM_ALIGNMENT * QFA_HEAD_DIM_ALIGNMENT;
    const bool overlayPv = blockM == QFA_PV_OVERLAY_MIN_QUERY_TILE || blockM == QFA_LARGE_QUERY_TILE_ROWS;
    uint32_t blockN = QFA_DEFAULT_KV_TILE_TOKENS;
    if (!windowed && sk <= QFA_MEDIUM_KV_TILE_TOKENS) {
        // A single KV block keeps the original static-P quantization domain. Shared P/PV slots need enough bytes
        // for their FP16 output rows.
        blockN = sk <= QFA_SMALL_KV_TILE_TOKENS ? QFA_SMALL_KV_TILE_TOKENS : QFA_MEDIUM_KV_TILE_TOKENS;
        const uint32_t pvRowBytes = dAlign * sizeof(uint16_t); // FP16 PV overwrites the INT8 P row.
        if (overlayPv && blockN < pvRowBytes)
            blockN = pvRowBytes <= QFA_MEDIUM_KV_TILE_TOKENS ? QFA_MEDIUM_KV_TILE_TOKENS : QFA_DEFAULT_KV_TILE_TOKENS;
    }
    const uint32_t slotCount = blockM >= QFA_PV_OVERLAY_MIN_QUERY_TILE ? QFA_PV_OVERLAY_SLOTS : QFA_PIPELINE_SLOTS;
    const uint64_t tilesM = tilesForM(blockM);
    uint64_t tasks = static_cast<uint64_t>(b);
    for (uint64_t factor : {static_cast<uint64_t>(hkv), tilesM}) {
        if (tasks > std::numeric_limits<uint32_t>::max() / factor) {
            return false;
        }
        tasks *= factor;
    }
    if (tilesM > std::numeric_limits<uint32_t>::max() || tasks == 0 || tasks > std::numeric_limits<uint32_t>::max() ||
        aicNum == 0) {
        return false;
    }
    QuantFlashAttnTilingDataInt8 td{};
    td.batchSize = static_cast<uint32_t>(b);
    td.numHeadsQ = static_cast<uint32_t>(hq);
    td.numHeadsKv = static_cast<uint32_t>(hkv);
    td.seqLenQ = static_cast<uint32_t>(sq);
    td.seqLenKv = static_cast<uint32_t>(sk);
    td.headDim = static_cast<uint32_t>(d);
    td.headDimAlign = dAlign;
    td.outputHeadDim = static_cast<uint32_t>(outputDim);
    td.groupSize = static_cast<uint32_t>(hq / hkv);
    td.blockM = blockM;
    td.blockN = blockN;
    td.usedCoreNum = std::min<uint32_t>(aicNum, static_cast<uint32_t>(tasks));
    td.totalTilesM = static_cast<uint32_t>(tilesM);
    td.totalTasks = static_cast<uint32_t>(tasks);
    td.scoreSlotElems = blockM * blockN;
    td.pSlotElems = blockM * blockN;
    td.pvSlotElems = blockM * td.headDimAlign;
    {
        // For M>=256, each AIV owns half of a P/PV slot. MM2 consumes its P partition before Fixpipe overwrites
        // 128-row regions with FP16 PV.
        td.coreWorkspaceBytes =
            overlayPv ? slotCount * (td.scoreSlotElems * sizeof(uint16_t) + td.pSlotElems * sizeof(int8_t)) :
                        slotCount * (td.scoreSlotElems * sizeof(uint16_t) + td.pSlotElems * sizeof(int8_t) +
                                     td.pvSlotElems * sizeof(uint16_t));
        // A fixed per-core cache removes repeated K-scale conversion across Q tasks of the same head. Its maximum
        // size is independent of sequence.
        const bool cacheKScale = lq != QFA_LAYOUT_BNSD && lk != QFA_LAYOUT_BNSD &&
                                 QfaUseKScaleCache(static_cast<uint64_t>(sq), static_cast<uint64_t>(sk),
                                                   static_cast<uint32_t>(d), groupedM, blockM, blockN);
        const bool kScalePrefixCache = !useDynamicMhaTile && lq == QFA_LAYOUT_TND && d == QFA_HEAD_DIM_256 &&
                                       sk > QFA_K_SCALE_CACHE_TOKENS && blockM < QFA_PV_OVERLAY_MIN_QUERY_TILE;
        td.coreWorkspaceBytes +=
            BaseApi::Int8GqaTnd::QfaKScaleWorkspaceElements(cacheKScale, kScalePrefixCache, slotCount, blockN) *
            sizeof(uint64_t);
    }
    td.softmaxScale = input.softmaxScale;
    td.layoutQ = lq;
    td.layoutKv = lk;
    td.layoutOut = lo;
    td.winLeft = input.winLeft;
    td.winRight = input.winRight;
    td.maskMode = input.mask == QFA_MASK_NONE ? QFA_MASK_NONE : QFA_MASK_CAUSAL_TILING;
    td.hasMetadata = hasMetadata ? QfaPackMetadataCapacity(metadataCapacity) : 0U;
    td.totalTokensQ = lq == QFA_LAYOUT_TND ? totalQ : 0;
    td.totalTokensKv = lk == QFA_LAYOUT_TND ? totalK : 0;

    if (l2Bytes != 0U) {
        const uint64_t available = l2Bytes > reservedWorkspaceBytes ? l2Bytes - reservedWorkspaceBytes : 0U;
        if (available < td.coreWorkspaceBytes) {
            return false;
        }
        const uint64_t maxCoresByL2 = available / td.coreWorkspaceBytes;
        td.usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(td.usedCoreNum, maxCoresByL2));
    }

    QuantFlashAttnTilingPayloadInt8 payload{};
    payload.config = td;
    if (!windowed)
        BuildDenseCoreRanges(payload);
    const uint64_t tilingCapacity = input.tilingCapacity;
    if (tilingCapacity < sizeof(payload))
        return false;
    QuantFlashAttnTilingQBlockPayloadInt8 mappedPayload;
    // D96 uses the bounded Fixpipe scale window with optional task maps.
    const bool d96ScaleFb = lq == lk && lq == lo && lq <= QFA_LAYOUT_BSND && d == QFA_HEAD_DIM_96 && sq == sk &&
                            sq > QFA_LARGE_QUERY_TILE_ROWS && sq % QFA_LARGE_QUERY_TILE_ROWS == 0 &&
                            blockM == QFA_LARGE_QUERY_TILE_ROWS && blockN == QFA_DEFAULT_KV_TILE_TOKENS &&
                            tilingCapacity >= sizeof(mappedPayload);
    if (d96ScaleFb) {
        mappedPayload = {};
        mappedPayload.base = payload;
        BuildQBlockTaskMap(payload, mappedPayload);
    }
    const bool singleTaskGeneric = lq == QFA_LAYOUT_BNSD && lk == QFA_LAYOUT_BNSD && lo == QFA_LAYOUT_BNSD &&
                                   td.totalTasks <= td.usedCoreNum && blockM <= QFA_PV_OVERLAY_MIN_QUERY_TILE &&
                                   !(d == QFA_HEAD_DIM_256 && sk > QFA_K_SCALE_CACHE_TOKENS);
    // CFG9 shares CFG0's buffers and arithmetic, while keeping BNSD's cross-Q scheduler off for this new two-slot
    // overlaid FP32 layout.
    const bool isolatedLargeFp32M256 = largeFp32M256 && blockM == QFA_PV_OVERLAY_MIN_QUERY_TILE &&
                                       lq == QFA_LAYOUT_BNSD && lk == QFA_LAYOUT_BNSD && lo == QFA_LAYOUT_BNSD;
    const uint32_t baseConfig =
        windowed                                     ? QFA_INT8_CFG_SHARED_QP :
        useDynamicMhaTile                            ? QFA_INT8_CFG_DYNAMIC_MHA :
        d96ScaleFb                                   ? QFA_INT8_CFG_D96_M512_FIXPIPE :
        (singleTaskGeneric || isolatedLargeFp32M256) ? QFA_INT8_CFG_ISOLATED_SHARED_QP :

        blockM == QFA_LARGE_QUERY_TILE_ROWS && d == QFA_HEAD_DIM_64  ? QFA_INT8_CFG_D64_M512 :
        blockM == QFA_LARGE_QUERY_TILE_ROWS && d == QFA_HEAD_DIM_128 ? QFA_INT8_CFG_D128_M512 :
        blockM == QFA_LARGE_QUERY_TILE_ROWS                          ? QFA_INT8_CFG_GENERIC_M512 :
        (d == QFA_HEAD_DIM_256 && sk > QFA_K_SCALE_CACHE_TOKENS && blockM < QFA_PV_OVERLAY_MIN_QUERY_TILE) ?
                                              QFA_INT8_CFG_RESIDENT_Q :
                                              QFA_INT8_CFG_SHARED_QP;
    const bool useD64Opt = lq == lk && lq == lo && lq <= QFA_LAYOUT_TND && d == QFA_HEAD_DIM_64 &&
                           blockM == QFA_LARGE_QUERY_TILE_ROWS && blockN == QFA_DEFAULT_KV_TILE_TOKENS &&
                           baseConfig == QFA_INT8_CFG_D64_M512 && tilingCapacity >= sizeof(mappedPayload);
    if (useD64Opt) {
        mappedPayload = {};
        mappedPayload.base = payload;
        const bool useD64Map = BuildD64HeadLocalQBlockMap(payload, mappedPayload);
        if (!useD64Map) {
            mappedPayload = {};
            mappedPayload.base = payload;
        }
    }
    const bool tndD128Overlap = lq == QFA_LAYOUT_TND && lk == QFA_LAYOUT_TND && lo == QFA_LAYOUT_TND &&
                                d == QFA_HEAD_DIM_128 && blockM == QFA_LARGE_QUERY_TILE_ROWS &&
                                blockN == QFA_DEFAULT_KV_TILE_TOKENS && baseConfig == QFA_INT8_CFG_D128_M512;
    const bool tndQBlockMap =
        tndD128Overlap && tilingCapacity >= sizeof(mappedPayload) && BuildQBlockTaskMap(payload, mappedPayload);
    const bool causalMhaMap =
        useCausalMhaMap && tilingCapacity >= sizeof(mappedPayload) && BuildQBlockTaskMap(payload, mappedPayload);
    const uint32_t config = causalMhaMap ? QFA_INT8_CFG_D96_M256_CAUSAL_MHA :
                            useD64Opt    ? QFA_INT8_CFG_D64_M512_TASK_MAP :
                                           baseConfig;
    result.hasTaskBuffer = d96ScaleFb || useD64Opt || tndQBlockMap || causalMhaMap;
    if (result.hasTaskBuffer)
        result.payload = mappedPayload;
    else
        result.payload.base = payload;
    result.config = config | (windowed ? QFA_INT8_CFG_WINDOW_FLAG : 0U);
    result.workspaceBytes = reservedWorkspaceBytes + static_cast<uint64_t>(td.usedCoreNum) * td.coreWorkspaceBytes;
    return true;
}
} // namespace BaseApi::Int8GqaTnd
#endif
