/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_HOST_TILING_A8C8_H_
#define QUANT_FLASH_ATTN_HOST_TILING_A8C8_H_

#include "../../op_kernel/arch22/quant_flash_attn_tiling_data_int8.h"
#include <register/op_impl_registry.h>
#include "../qfa_tiling_info.h"
#include "../../../common/op_host/split_core.h"
#include <limits>
#include <cmath>
#include <tuple>
#include <optional>

namespace optiling {

namespace quant_flash_attn {
namespace a8c8_plan {
struct QfaPlanChoice {
    uint32_t m = 16U, unit = 128U, fdRows = 16U, r1 = 8U, r2 = 8U, columns = 1U;
    uint64_t score = std::numeric_limits<uint64_t>::max();
    uint32_t kind = 0U, budget = 0U;
};

inline uint32_t QfaVecRows(uint32_t rows)
{
    const uint32_t first = rows <= 8U ? rows : (((rows + 1U) / 2U + 7U) & ~7U);
    return std::min(first, rows);
}

inline uint64_t QfaUbBytes(uint32_t m, uint32_t d, uint32_t r1, uint32_t r2, uint32_t scratch = 32768U,
                           bool compactD256 = false)
{
    const uint32_t alignedD = (d + 31U) & ~31U;
    if (d != 128U && d != 256U) {
        return 2ULL * r1 * 512U * 5U + 2ULL * r2 * alignedD * 4U + (std::max(64U, m) / 2U) * alignedD * 4ULL +
               r2 * alignedD * 2ULL + scratch + 24576U;
    }
    const uint32_t rows = (QfaVecRows(m) + 7U) & ~7U;
    const uint64_t score = std::max(2ULL * r1 * 512U * 4U, 2ULL * 16U * alignedD * 4U);
    const bool wideScale = d == 128U && r1 == 16U;
    const uint64_t rowScale = compactD256 ? 2048ULL : std::max(4096ULL, rows * (wideScale ? 256ULL : 32ULL));
    // Match HpInit; GM workspace pitch and FD padding remain independent of UB rows.
    return score + 2ULL * r1 * 512U + 2ULL * r2 * alignedD * 4U + rows * alignedD * 4ULL + r2 * alignedD * 2ULL +
           scratch + (compactD256 ? 3072U : 4096U) + 512U + rowScale + 4096U + (wideScale ? 256U : 0U) + rows * 32ULL;
}

// Keep whole tasks in the same ordered coordinate system and compact empty intervals.
inline SplitResult QfaWholeTaskPlan(const SplitResult& source, const BaseInfo& base, uint32_t m, uint32_t unit)
{
    SplitResult result = source;
    uint32_t count = 0U, lastBN = 0U, lastGS = 0U;
    const uint32_t heads = base.bSize * base.n2Size;
    for (uint32_t c = 0U; c < source.usedCoreNum; ++c) {
        uint32_t bn = source.bN2End[c], gs = source.gS1End[c];
        if (c + 1U == source.usedCoreNum) {
            bn = heads;
            gs = 0U;
        } else if (bn < heads) {
            const uint32_t panels = (base.actualSeqS2Size[bn / base.n2Size] + unit - 1U) / unit;
            if (source.s2End[c] != 0U && panels != 0U)
                ++gs;
            while (bn < heads) {
                const uint32_t blocks = (base.actualSeqS1Size[bn / base.n2Size] * base.gSize + m - 1U) / m;
                if (gs < blocks)
                    break;
                ++bn;
                gs = 0U;
            }
        }
        if (bn == lastBN && gs == lastGS)
            continue;
        result.bN2End[count] = bn;
        result.gS1End[count] = gs;
        result.s2End[count] = 0U;
        ++count;
        lastBN = bn;
        lastGS = gs;
        if (bn == heads)
            break;
    }
    result.usedCoreNum = count;
    result.numOfFdHead = 0U;
    result.maxS2SplitNum = 0U;
    result.usedVecNumOfFd = 0U;
    return result;
}

inline uint64_t QfaVectorService(uint32_t rows, uint32_t d, uint32_t r1, uint32_t r2)
{
    if (rows == 0U)
        return 0U;
    // Relative units constrained by existing R8/R16 FP32 PMU, not predicted kernel ns.
    const uint32_t softmaxRow = rows >= 8U ? 90U : 100U;
    return rows * (softmaxRow + 15ULL * d / 128U) + 150ULL * ((rows + r1 - 1U) / r1) + 100ULL * ((rows + r2 - 1U) / r2);
}

// Relative service units; keep production and whole-task partition weights identical.
inline uint64_t QfaTaskService(uint32_t rows, uint64_t length, uint32_t d, uint32_t layout, uint32_t r1, uint32_t r2,
                               uint32_t heads, uint64_t fullKvLength = 0U)
{
    if (rows == 0U || length == 0U)
        return 0U;
    const uint32_t alignedD = (d + 31U) & ~31U;
    const uint64_t count = (length + 511U) / 512U;
    const uint64_t n = (length + 31U) & ~31ULL;
    const uint32_t v0 = heads == 2U ? rows : QfaVecRows(rows);
    const uint32_t v1 = heads == 2U ? rows : rows - v0;
    const uint64_t vec = std::max(QfaVectorService(v0, d, r1, r2), QfaVectorService(v1, d, r1, r2));
    // Conservative calibration for small rows and at most 16 full-sequence panels.
    // Use the unsplit length: changing the FD budget must not change this service regime.
    const uint64_t fullPanels = ((fullKvLength == 0U ? length : fullKvLength) + 511U) / 512U;
    const bool shortPair = heads == 2U && rows <= 16U && fullPanels <= 16U;
    const uint32_t headServices = shortPair ? heads : 1U;
    // Short pairs have not demonstrated the long-KV supply gain; both heads still issue QK/PV.
    const uint64_t kvRate = layout == 0U ? (heads == 2U && !shortPair ? 98304U : 45056U) : 65536U;
    const uint64_t kvSupply = 2ULL * heads * length * d * 1000U / kvRate;
    const uint64_t pSupply = heads * rows * n * 1000U / 65536U;
    const uint64_t mac = 4ULL * heads * ((rows + 15U) & ~15U) * n * alignedD * 1000U / 32000000U;
    const uint64_t fix = 4ULL * heads * rows * (n + count * alignedD) * 1000U / 131072U;
    const uint64_t cube = std::max(kvSupply + pSupply, std::max(mac, fix)) / count + 300U * headServices;
    const uint64_t prepare = 1500U * headServices + std::max(v0, v1) * 4U;
    const uint64_t publish = 200U * headServices + std::max(v0, v1) * alignedD * 4ULL * 1000U / 65536U;
    return prepare + count * (std::max(cube, vec) + 2000U * headServices) + std::min(cube, vec) + publish;
}

// Build contiguous whole-task ranges independently of the KV split boundaries.
inline SplitResult QfaBalancedWholeTasks(const BaseInfo& base, uint32_t m, uint32_t cores, uint32_t d = 0U,
                                         uint32_t layout = 0U, uint32_t r1 = 8U, uint32_t r2 = 8U, uint32_t heads = 1U)
{
    SplitResult result(cores, 2U);
    // Equal-length heads in a batch share the same full/tail costs. Store one prefix per batch.
    std::vector<uint64_t> tasks{0U}, weights{0U}, fullCosts, headCosts, blockCounts;
    tasks.reserve(base.bSize + 1U);
    weights.reserve(base.bSize + 1U);
    fullCosts.reserve(base.bSize);
    headCosts.reserve(base.bSize);
    blockCounts.reserve(base.bSize);
    for (uint32_t b = 0; b < base.bSize; ++b) {
        const uint64_t rows = base.actualSeqS1Size[b] * base.gSize;
        const uint64_t panels = (base.actualSeqS2Size[b] + 511U) / 512U;
        const uint64_t blocks = (rows + m - 1U) / m;
        const uint64_t full = d == 128U ? QfaTaskService(m, base.actualSeqS2Size[b], d, layout, r1, r2, heads) :
                                          1U + panels * CalcCost(m, 512U);
        const uint64_t tail = rows % m == 0U ? full :
                              d == 128U ? QfaTaskService(rows % m, base.actualSeqS2Size[b], d, layout, r1, r2, heads) :
                                          1U + panels * CalcCost(rows % m, 512U);
        const uint64_t head = blocks == 0U ? 0U : (blocks - 1U) * full + tail;
        fullCosts.push_back(full);
        headCosts.push_back(head);
        blockCounts.push_back(blocks);
        tasks.push_back(tasks.back() + blocks * base.n2Size);
        weights.push_back(weights.back() + head * base.n2Size);
    }
    const auto batchAt = [&](uint64_t index) {
        return std::upper_bound(tasks.begin(), tasks.end(), index) - tasks.begin() - 1;
    };
    const auto weightAt = [&](uint64_t index) -> uint64_t {
        if (index == tasks.back())
            return weights.back();
        const auto b = batchAt(index);
        const uint64_t off = index - tasks[b];
        return weights[b] + off / blockCounts[b] * headCosts[b] + off % blockCounts[b] * fullCosts[b];
    };
    result.usedCoreNum = std::min<uint64_t>(cores, tasks.back());
    uint64_t last = 0U;
    for (uint32_t c = 0; c < result.usedCoreNum; ++c) {
        const uint32_t remaining = result.usedCoreNum - c;
        uint64_t end = tasks.back();
        if (remaining > 1U) {
            const uint64_t start = weightAt(last);
            const uint64_t target = start + (weights.back() - start) / remaining;
            uint64_t lo = last + 1U;
            end -= remaining - 1U;
            while (lo < end) {
                const uint64_t mid = lo + (end - lo) / 2U;
                if (weightAt(mid) < target)
                    lo = mid + 1U;
                else
                    end = mid;
            }
            if (end > last + 1U && weightAt(end) > target && target - weightAt(end - 1U) <= weightAt(end) - target)
                --end;
        }
        const auto b = batchAt(end);
        const uint64_t off = end - tasks[b];
        result.bN2End[c] = end == tasks.back() ? base.bSize * base.n2Size : b * base.n2Size + off / blockCounts[b];
        result.gS1End[c] = end == tasks.back() ? 0U : off % blockCounts[b];
        result.s2End[c] = 0U;
        last = end;
    }
    return result;
}

// Keep single-row-block heads independent while retaining the shared FIA planners and coordinates.
inline SplitResult QfaHeadPlan(const BaseInfo& base, const SplitParam& param, uint32_t budget)
{
    SplitResult result(budget, 2U);
    const uint64_t heads = static_cast<uint64_t>(base.bSize) * base.n2Size;
    if (heads < 2U || heads > budget)
        return result;
    for (uint32_t b = 0U; b < base.bSize; ++b) {
        if (base.actualSeqS1Size[b] <= 0 || base.actualSeqS2Size[b] <= 0 ||
            static_cast<uint64_t>(base.actualSeqS1Size[b]) * base.gSize > param.mBaseSize)
            return result;
    }
    BaseInfo one = base;
    one.bSize = one.n2Size = one.actualLenQDims = one.actualLenKvDims = 1U;
    one.actualSeqS1Size.resize(1U);
    one.actualSeqS2Size.resize(1U);
    uint32_t previousCores = 0U;
    SplitResult part(budget, 2U);
    for (uint32_t bn = 0U; bn < heads; ++bn) {
        const uint32_t b = bn / base.n2Size;
        const uint32_t cores = budget / heads + (bn < budget % heads ? 1U : 0U);
        if (cores != previousCores || one.actualSeqS1Size[0] != base.actualSeqS1Size[b] ||
            one.actualSeqS2Size[0] != base.actualSeqS2Size[b]) {
            one.actualSeqS1Size[0] = base.actualSeqS1Size[b];
            one.actualSeqS2Size[0] = base.actualSeqS2Size[b];
            part = SplitResult(cores, 2U);
            SplitCore(cores, one, param, part);
            previousCores = cores;
        }
        if (part.usedCoreNum == 0U)
            return SplitResult(budget, 2U);
        for (uint32_t c = 0U; c < part.usedCoreNum; ++c) {
            const uint32_t dst = result.usedCoreNum + c;
            const bool last = c + 1U == part.usedCoreNum;
            result.bN2End[dst] = last ? bn + 1U : bn + part.bN2End[c];
            result.gS1End[dst] = last ? 0U : part.gS1End[c];
            result.s2End[dst] = last ? 0U : part.s2End[c];
            result.fdRes.s2SplitStartIdxOfCore[dst] = part.fdRes.s2SplitStartIdxOfCore[c];
        }
        for (uint32_t g = 0U; g < part.numOfFdHead; ++g) {
            const uint32_t dst = result.numOfFdHead + g;
            result.fdRes.bN2IdxOfFdHead[dst] = bn + part.fdRes.bN2IdxOfFdHead[g];
            result.fdRes.gS1IdxOfFdHead[dst] = part.fdRes.gS1IdxOfFdHead[g];
            result.fdRes.s2SplitNumOfFdHead[dst] = part.fdRes.s2SplitNumOfFdHead[g];
            result.fdRes.gS1SplitNumOfFdHead[dst] = part.fdRes.gS1SplitNumOfFdHead[g];
            result.fdRes.gS1LastPartSizeOfFdHead[dst] = part.fdRes.gS1LastPartSizeOfFdHead[g];
        }
        result.usedCoreNum += part.usedCoreNum;
        result.numOfFdHead += part.numOfFdHead;
        result.maxS2SplitNum = std::max(result.maxS2SplitNum, part.maxS2SplitNum);
        result.maxCost = std::max(result.maxCost, part.maxCost);
    }
    return result;
}

inline uint32_t QfaLongestChain(const SplitResult& result, const BaseInfo& base, uint32_t m, uint32_t unit,
                                uint64_t* rowPanels = nullptr, uint32_t* maxPieces = nullptr, bool uniform = false)
{
    uint32_t longest = 0U, mostPieces = 0U;
    uint64_t mostRowPanels = 0U;
    for (uint32_t c = 0U; c < result.usedCoreNum; ++c) {
        const uint32_t sb = c == 0U ? 0U : result.bN2End[c - 1U];
        const uint32_t sg = c == 0U ? 0U : result.gS1End[c - 1U];
        const uint32_t sk = c == 0U ? 0U : result.s2End[c - 1U];
        uint32_t chain = 0U, pieces = 0U;
        uint64_t rowChain = 0U;
        for (uint32_t bn = sb; bn <= result.bN2End[c] && bn < base.bSize * base.n2Size;) {
            const uint32_t b = bn / base.n2Size;
            const uint32_t fullEnd = std::min(result.bN2End[c], base.bSize * base.n2Size);
            const uint32_t headRepeats =
                uniform && bn < fullEnd && (bn != sb || (sg == 0U && sk == 0U)) ? fullEnd - bn : 1U;
            const uint32_t blocks = (base.actualSeqS1Size[b] * base.gSize + m - 1U) / m;
            const uint32_t panels = (base.actualSeqS2Size[b] + unit - 1U) / unit;
            uint32_t lo = bn == sb ? sg : 0U;
            if (bn == sb && sk >= panels)
                ++lo;
            const uint32_t hi =
                std::min(blocks, bn == result.bN2End[c] ? result.gS1End[c] + (result.s2End[c] > 0U) : blocks);
            for (uint32_t gs = lo; gs < hi; ++gs) {
                const uint32_t kl = bn == sb && gs == sg ? sk : 0U;
                const uint32_t kh = bn == result.bN2End[c] && gs == result.gS1End[c] ? result.s2End[c] : panels;
                const uint64_t last = std::min<uint64_t>(kh * static_cast<uint64_t>(unit), base.actualSeqS2Size[b]);
                const uint64_t first = std::min<uint64_t>(kl * static_cast<uint64_t>(unit), last);
                const uint32_t count = (last - first + 511U) / 512U;
                chain += count * headRepeats;
                rowChain += count * std::min<uint64_t>(m, base.actualSeqS1Size[b] * base.gSize - gs * m) * headRepeats;
                pieces += (count != 0U) * headRepeats;
            }
            bn += headRepeats;
        }
        longest = std::max(longest, chain);
        mostRowPanels = std::max(mostRowPanels, rowChain);
        mostPieces = std::max(mostPieces, pieces);
    }
    if (rowPanels != nullptr)
        *rowPanels = mostRowPanels;
    if (maxPieces != nullptr)
        *maxPieces = mostPieces;
    return longest;
}

struct QfaProductionCost {
    uint64_t maxService, traffic, sharedInputs;
};

inline QfaProductionCost QfaProductionScore(const SplitResult& result, const BaseInfo& base, uint32_t d,
                                            uint32_t layout, uint32_t m, uint32_t unit, uint32_t r1, uint32_t r2,
                                            bool optimized, uint32_t headGroupSize, bool unified, bool uniform = false)
{
    uint64_t maxService = 0U, traffic = 0U, sharedInputs = 0U;
    const uint32_t alignedD = (d + 31U) & ~31U;
    // Per-core supply rates are model parameters from the existing native/ATK profiles.
    const uint64_t rate = layout == 0U ? (d <= 64U ? 16384U : (d <= 128U ? 45056U : 65536U)) :
                                         (d <= 64U ? 22528U : (d <= 128U ? 65536U : 81920U));
    uint32_t segmentCount = 0U;
    uint32_t segmentRows[4]{};
    uint64_t segmentLengths[4]{};
    QfaProductionCost segmentCosts[4]{};
    for (uint32_t c = 0U; c < result.usedCoreNum; ++c) {
        const uint32_t sb = c == 0U ? 0U : result.bN2End[c - 1U];
        const uint32_t sg = c == 0U ? 0U : result.gS1End[c - 1U];
        const uint32_t sk = c == 0U ? 0U : result.s2End[c - 1U];
        uint64_t service = 0U;
        for (uint32_t bn = sb; bn <= result.bN2End[c] && bn < base.bSize * base.n2Size;) {
            const uint32_t b = bn / base.n2Size;
            const uint32_t fullEnd = std::min(result.bN2End[c], base.bSize * base.n2Size);
            const uint32_t headRepeats =
                uniform && bn < fullEnd && (bn != sb || (sg == 0U && sk == 0U)) ? fullEnd - bn : 1U;
            const uint32_t blocks = (base.actualSeqS1Size[b] * base.gSize + m - 1U) / m;
            const uint32_t panels = (base.actualSeqS2Size[b] + unit - 1U) / unit;
            uint32_t lo = bn == sb ? sg : 0U;
            if (bn == sb && sk >= panels)
                ++lo;
            const uint32_t hi =
                std::min(blocks, bn == result.bN2End[c] ? result.gS1End[c] + (result.s2End[c] > 0U) : blocks);
            auto charge = [&](uint32_t rows, uint64_t first, uint64_t last, uint32_t repeats) {
                if (last == first || repeats == 0U)
                    return;
                repeats *= headRepeats;
                // Equal-length heads reuse the same full/tail segment arithmetic within this score.
                for (uint32_t i = 0U; uniform && i < segmentCount; ++i) {
                    if (segmentRows[i] == rows && segmentLengths[i] == last - first) {
                        service += segmentCosts[i].maxService * repeats;
                        traffic += segmentCosts[i].traffic * repeats;
                        sharedInputs += segmentCosts[i].sharedInputs * repeats;
                        return;
                    }
                }
                uint64_t localService = 0U, localTraffic = 0U, localShared = 0U;
                const uint64_t count = (last - first + 511U) / 512U;
                if (!optimized) {
                    const uint64_t bytes = count * (1024ULL * d + rows * 512ULL);
                    localService += count * (CalcCost(rows, 512U) + bytes / count / 4096U) + CalcCost(16U, 512U);
                    localTraffic += bytes;
                    service += localService * repeats;
                    traffic += localTraffic * repeats;
                    return;
                }
                const uint64_t n = (last - first + 31U) & ~31ULL;
                const uint32_t v0 = headGroupSize == 2U ? rows : QfaVecRows(rows);
                const uint32_t v1 = headGroupSize == 2U ? rows : rows - v0;
                const uint64_t vec = std::max(QfaVectorService(v0, d, r1, r2), QfaVectorService(v1, d, r1, r2));
                const uint64_t inputs = headGroupSize * (2ULL * (last - first) * d + rows * n);
                const uint64_t supply = inputs * 1000U / rate / count;
                const uint64_t mac =
                    4ULL * headGroupSize * ((rows + 15U) & ~15U) * n * alignedD * 1000U / 32000000U / count;
                const uint64_t fix = 4ULL * headGroupSize * rows * (n + count * alignedD) * 1000U / 131072U / count;
                const uint64_t cube = std::max(supply, std::max(mac, fix)) + 300U;
                // First panel has a complete chain; steady overlap retains per-panel handoff cost.
                if (unified) {
                    localService +=
                        QfaTaskService(rows, last - first, d, layout, r1, r2, headGroupSize, base.actualSeqS2Size[b]);
                } else {
                    localService +=
                        1500U + std::max(v0, v1) * 4U + count * (std::max(cube, vec) + 2000U) + std::min(cube, vec);
                    localService += 200U + std::max(v0, v1) * alignedD * 4ULL * 1000U / 65536U;
                }
                localTraffic += inputs + 4ULL * rows * (n + count * alignedD);
                localShared += 2ULL * headGroupSize * (last - first) * d;
                if (uniform && segmentCount < 4U) {
                    segmentRows[segmentCount] = rows;
                    segmentLengths[segmentCount] = last - first;
                    segmentCosts[segmentCount++] = {localService, localTraffic, localShared};
                }
                service += localService * repeats;
                traffic += localTraffic * repeats;
                sharedInputs += localShared * repeats;
            };
            auto chargeRow = [&](uint32_t gs) {
                const uint32_t kl = bn == sb && gs == sg ? sk : 0U;
                const uint32_t kh = bn == result.bN2End[c] && gs == result.gS1End[c] ? result.s2End[c] : panels;
                const uint64_t last = std::min<uint64_t>(kh * static_cast<uint64_t>(unit), base.actualSeqS2Size[b]);
                const uint64_t first = std::min<uint64_t>(kl * static_cast<uint64_t>(unit), last);
                const uint32_t rows = std::min<uint64_t>(m, base.actualSeqS1Size[b] * base.gSize - gs * m);
                charge(rows, first, last, 1U);
            };
            uint32_t end = hi;
            if (lo < end && bn == sb && lo == sg && sk != 0U)
                chargeRow(lo++);
            if (lo < end && bn == result.bN2End[c] && end - 1U == result.gS1End[c])
                chargeRow(--end);
            if (lo < end && end == blocks && (base.actualSeqS1Size[b] * base.gSize) % m != 0U)
                chargeRow(--end);
            if (lo < end) {
                const uint64_t last = std::min<uint64_t>(panels * static_cast<uint64_t>(unit), base.actualSeqS2Size[b]);
                charge(m, 0U, last, end - lo);
            }
            bn += headRepeats;
        }
        maxService = std::max(maxService, service);
    }
    return {maxService, traffic, sharedInputs};
}

inline uint64_t QfaMergeItemCost(uint32_t rows, uint32_t splits, uint32_t d, uint32_t width, uint32_t r1, uint32_t r2,
                                 const BaseInfo& base)
{
    const uint64_t perSplit = 96U + rows * 6U + rows * width * 4ULL * 1000U / 65536U;
    uint64_t cost = 512U + splits * perSplit;
    if (d == 128U && r1 == 8U && r2 == 8U && base.bSize > 1U &&
        static_cast<uint64_t>(base.s1Size) * base.gSize <= 16U) {
        const uint32_t capacity = r1 * 512U / (rows * width);
        const uint32_t refills = (splits - 1U) / capacity;
        cost += refills * (1500U + 2000U + 2U * QfaVectorService(rows, d, r1, r2));
    }
    return cost;
}

// Equal-weight FD items are balanced into floor/ceil item counts by FIA SplitFD.
inline bool QfaUniformFdCost(const SplitResult& result, const BaseInfo& base, uint32_t d, uint32_t fdRows, uint32_t r1,
                             uint32_t r2, uint32_t columns, uint32_t heads, uint64_t& cost)
{
    cost = 0U;
    if (result.numOfFdHead == 0U)
        return true;
    const uint32_t splits = result.fdRes.s2SplitNumOfFdHead[0];
    uint32_t rows = 0U;
    uint64_t items = 0U;
    for (uint32_t g = 0U; g < result.numOfFdHead; ++g) {
        const uint32_t count = result.fdRes.gS1SplitNumOfFdHead[g];
        const uint32_t blocks = count / columns / heads;
        const uint32_t last = result.fdRes.gS1LastPartSizeOfFdHead[g];
        const uint32_t live = blocks == 1U ? last : fdRows;
        if (blocks == 0U || (blocks > 1U && last != fdRows) || result.fdRes.s2SplitNumOfFdHead[g] != splits ||
            (g != 0U && live != rows))
            return false;
        rows = live;
        items += count;
    }
    const uint64_t vectors = std::min<uint64_t>(items, result.usedCoreNum * result.vecCubeRatio);
    if (vectors == 0U)
        return false;
    const uint32_t width = (((d + 31U) & ~31U) + columns - 1U) / columns;
    cost = ((items + vectors - 1U) / vectors) * QfaMergeItemCost(rows, splits, d, width, r1, r2, base);
    return true;
}

inline uint64_t QfaPlanScore(const SplitResult& result, const BaseInfo& base, uint32_t d, uint32_t layout, uint32_t m,
                             uint32_t unit, uint32_t fdRows, uint32_t r1, uint32_t r2, uint32_t columns, bool optimized,
                             uint32_t headGroupSize = 1U, bool unified = false,
                             const QfaProductionCost* production = nullptr, const uint64_t* mergeCost = nullptr)
{
    const auto cost = production == nullptr ? QfaProductionScore(result, base, d, layout, m, unit, r1, r2, optimized,
                                                                 headGroupSize, unified) :
                                              *production;
    const auto maxService = cost.maxService, traffic = cost.traffic, sharedInputs = cost.sharedInputs;
    const uint32_t alignedD = (d + 31U) & ~31U;
    uint64_t worstMerge = mergeCost == nullptr ? 0U : *mergeCost;
    if (mergeCost == nullptr && optimized && result.usedVecNumOfFd != 0U) {
        // Preserve FIA intervals; score the full/tail row pattern by exact prefix sums.
        uint64_t groupPrefix[QFA_PLAN_MAX_CORES + 1U];
        uint64_t fullCost[QFA_PLAN_MAX_CORES], tailCost[QFA_PLAN_MAX_CORES];
        uint32_t period[QFA_PLAN_MAX_CORES];
        groupPrefix[0] = 0U;
        const uint32_t width = (alignedD + columns - 1U) / columns;
        const auto prefix = [&](uint32_t group, uint32_t count) {
            const uint32_t tailItems =
                count / period[group] * columns + (count % period[group] > period[group] - columns ?
                                                       count % period[group] - (period[group] - columns) :
                                                       0U);
            return groupPrefix[group] + (count - tailItems) * fullCost[group] + tailItems * tailCost[group];
        };
        for (uint32_t g = 0U; g < result.numOfFdHead; ++g) {
            period[g] = result.fdRes.gS1SplitNumOfFdHead[g] / headGroupSize;
            const uint32_t lastRows = result.fdRes.gS1LastPartSizeOfFdHead[g];
            const uint32_t splits = result.fdRes.s2SplitNumOfFdHead[g];
            fullCost[g] = QfaMergeItemCost(fdRows, splits, d, width, r1, r2, base);
            tailCost[g] = lastRows == fdRows ? fullCost[g] : QfaMergeItemCost(lastRows, splits, d, width, r1, r2, base);
            groupPrefix[g + 1U] = prefix(g, result.fdRes.gS1SplitNumOfFdHead[g]);
        }
        uint64_t previous = 0U;
        for (uint32_t v = 0U; v < result.usedVecNumOfFd; ++v) {
            const uint64_t end = prefix(result.fdRes.gS1IdxEndOfFdHead[v], result.fdRes.gS1IdxEndOfFdHeadSplit[v] + 1U);
            worstMerge = std::max(worstMerge, end - previous);
            previous = end;
        }
    } else if (mergeCost == nullptr && !optimized) {
        uint32_t prevGroup = 0U, prevItem = 0U;
        for (uint32_t v = 0U; v < result.usedVecNumOfFd; ++v) {
            uint64_t load = 0U;
            const uint32_t endGroup = result.fdRes.gS1IdxEndOfFdHead[v];
            const uint32_t endItem = result.fdRes.gS1IdxEndOfFdHeadSplit[v];
            for (uint32_t g = prevGroup; g <= endGroup; ++g) {
                const uint32_t lo = g == prevGroup ? prevItem : 0U;
                const uint32_t hi = g == endGroup ? endItem + 1U : result.fdRes.gS1SplitNumOfFdHead[g];
                load += (hi - lo) *
                        (16ULL + result.fdRes.s2SplitNumOfFdHead[g] * 16ULL * ((d + columns - 1U) / columns) / 256U);
            }
            worstMerge = std::max(worstMerge, load);
            prevGroup = endGroup;
            prevItem = endItem + 1U;
        }
    }
    if (!optimized)
        return maxService + (result.numOfFdHead == 0U ? 0U : 3U * CalcCost(16U, 512U)) + worstMerge +
               traffic / (std::max(1U, result.usedCoreNum) * 4096ULL);
    // Drain/publication and the all-core rendezvous do not disappear when FD rows get smaller.
    const uint64_t fdStartup = result.numOfFdHead == 0U ? 0U : 8000U + result.usedCoreNum * 64ULL;
    return std::max(maxService, sharedInputs / 1024U) + fdStartup + worstMerge;
}

// Reuse FIA preprocessing and the best prefix while budgets increase within one geometry.
class QfaSplitSweep {
public:
    QfaSplitSweep(const BaseInfo& base, uint32_t m, uint32_t unit, uint32_t cores)
        : param_{m, unit, 16U},
          context_(base, param_),
          best_(cores, 2U),
          temporary_(cores, 2U)
    {}

    const SplitResult& Get(uint32_t budget, bool withFd = true)
    {
        if (!initialized_) {
            CalcSplitInfo(context_);
            CalcCostInfo(context_);
            firstCore_ =
                static_cast<uint32_t>(std::sqrt(static_cast<float>(context_.costInfo.totalBlockNum) + 0.25f) + 0.5f);
            const auto& base = context_.baseInfo;
            if (firstCore_ < best_.bN2End.size() && !base.attenMaskFlag && !base.isAccumSeqS1 && !base.isAccumSeqS2 &&
                base.bSize != 0U && !base.actualSeqS1Size.empty() && !base.actualSeqS2Size.empty() &&
                std::all_of(base.actualSeqS1Size.begin(), base.actualSeqS1Size.end(),
                            [&](int64_t q) { return q == base.actualSeqS1Size[0]; }) &&
                std::all_of(base.actualSeqS2Size.begin(), base.actualSeqS2Size.end(),
                            [&](int64_t k) { return k == base.actualSeqS2Size[0]; })) {
                const auto& split = context_.splitInfo;
                const uint32_t tailM = split.s1GTailSize[0], tailN = split.s2TailSize[0];
                const uint64_t mCounts[2] = {split.s1GBaseNum[0] - (tailM != 0U), tailM != 0U};
                const uint64_t nCounts[2] = {split.s2BaseNum[0] - (tailN != 0U), tailN != 0U};
                const uint32_t rows[2] = {param_.mBaseSize, tailM};
                const uint32_t cols[2] = {param_.s2BaseSize, tailN};
                uint64_t counts[4]{};
                for (uint32_t i = 0U; i < 4U; ++i) {
                    counts[i] = mCounts[i / 2U] * nCounts[i % 2U] * base.bSize * base.n2Size;
                    blockCosts_[i] = counts[i] == 0U ? 0U : CalcCost(rows[i / 2U], cols[i % 2U]);
                }
                for (uint32_t i = 0U; i < 4U; ++i)
                    for (uint32_t j = 0U; j < 4U; ++j)
                        if (blockCosts_[j] >= blockCosts_[i])
                            blockCounts_[i] += counts[j];
            }
            nextCore_ = firstCore_;
            best_.usedCoreNum = 1U;
            best_.maxCost = std::numeric_limits<int64_t>::max();
            initialized_ = true;
        }
        if (context_.splitInfo.isKvSeqAllZero) {
            best_.usedCoreNum = 1U;
            best_.maxCost = 0;
            best_.bN2End[0] = context_.baseInfo.bSize * context_.baseInfo.n2Size;
            return best_;
        }
        if (temporaryLive_) {
            ClearTmpResult(temporary_);
            temporaryLive_ = false;
        }
        const uint32_t limit = std::min(budget, context_.costInfo.totalBlockNum);
        SplitResult* result = &best_;
        if (limit < firstCore_) {
            CalcSplitPlan(limit, std::numeric_limits<int64_t>::max(), context_, temporary_);
            temporaryLive_ = true;
            result = &temporary_;
        } else {
            for (; nextCore_ <= limit; ++nextCore_) {
                if (best_.maxCost != std::numeric_limits<int64_t>::max()) {
                    const uint64_t total = context_.costInfo.totalCost;
                    uint64_t lowerBound = total / nextCore_ + (total % nextCore_ != 0U);
                    for (uint32_t i = 0U; i < 4U; ++i)
                        lowerBound = std::max(lowerBound, blockCosts_[i] * (blockCounts_[i] / nextCore_ +
                                                                            (blockCounts_[i] % nextCore_ != 0U)));
                    // A tied cost cannot replace the earlier plan. No boundaries or FD state are needed.
                    if (lowerBound >= static_cast<uint64_t>(best_.maxCost))
                        continue;
                }
                CalcSplitPlan(nextCore_, best_.maxCost, context_, temporary_);
                if (temporary_.maxCost < best_.maxCost)
                    std::swap(temporary_, best_);
                ClearTmpResult(temporary_);
            }
        }
        result->usedCoreNum = std::max(result->usedCoreNum, 1U);
        // The read-only view is consumed before the next Get; FD output uses its own storage.
        if (!withFd)
            return *result;
        if (!current_.has_value())
            current_.emplace(best_.bN2End.size(), 2U);
        *current_ = *result;
        if (current_->numOfFdHead != 0U)
            SplitFD(*current_);
        return *current_;
    }

private:
    SplitParam param_;
    SplitContext context_;
    SplitResult best_, temporary_;
    std::optional<SplitResult> current_;
    bool temporaryLive_ = false;
    uint64_t blockCosts_[4]{}, blockCounts_[4]{};
    uint32_t firstCore_ = 0U, nextCore_ = 0U;
    bool initialized_ = false;
};

template <bool Materialize = true>
inline QfaPlanChoice QfaSelectUnifiedPlan(BaseInfo& base, uint32_t d, uint32_t layout, uint32_t blockSize,
                                          uint32_t aicNum, bool allowPairedHeads, SplitResult& best, uint32_t& heads)
{
    const BaseInfo physical = base;
    uint64_t maxRows = 0U;
    for (auto q : base.actualSeqS1Size)
        maxRows = std::max<uint64_t>(maxRows, q * base.gSize);
    const bool optimized = maxRows != 0U && (d == 64U || d == 128U || d == 256U);
    const uint32_t maxM = d <= 128U ? 256U : 128U;
    // Keep the former search neighborhood so existing M <= 256 choices remain available.
    const uint32_t scanMaxM = d <= 64U ? 512U : maxM;
    uint32_t primaryM = 16U;
    while (primaryM < maxRows && primaryM < scanMaxM)
        primaryM *= 2U;
    const uint64_t pairRows = static_cast<uint64_t>(base.s1Size) * base.gSize;
    const bool pairEligible =
        allowPairedHeads && d == 128U && layout == 0U && blockSize == 128U &&
        (base.s1Size == 1U || (pairRows > 32U && pairRows <= 64U)) && base.gSize <= 32U &&
        (base.n2Size == 2U || base.n2Size == 4U) &&
        std::all_of(base.actualSeqS1Size.begin(), base.actualSeqS1Size.end(),
                    [&](int64_t q) { return q > 0 && static_cast<uint64_t>(q) == base.s1Size; }) &&
        std::all_of(base.actualSeqS2Size.begin(), base.actualSeqS2Size.end(),
                    [&](int64_t k) { return k > 0 && k == base.actualSeqS2Size[0]; });
    // Short KV chains and wide partial row blocks retain the general planner.
    const bool regular = optimized && (d == 128U || d == 256U) && blockSize == 128U && maxRows <= maxM &&
                         (maxRows <= 64U || maxRows == primaryM) && !base.actualSeqS1Size.empty() &&
                         !base.actualSeqS2Size.empty() &&
                         std::all_of(base.actualSeqS1Size.begin(), base.actualSeqS1Size.end(),
                                     [&](int64_t q) { return q > 0 && q == base.actualSeqS1Size[0]; }) &&
                         std::all_of(base.actualSeqS2Size.begin(), base.actualSeqS2Size.end(),
                                     [&](int64_t k) { return k > 1024 && k == base.actualSeqS2Size[0]; });
    QfaPlanChoice chosen;
    uint32_t retainedChain = 0U;
    uint64_t retainedParts = 0U;
    heads = 1U;
    for (uint32_t group = 1U; group <= (pairEligible ? 2U : 1U); ++group) {
        BaseInfo view = physical;
        view.n2Size /= group;
        // The primary block covers the live rows, subject to the existing UB contract.
        uint32_t topM = group == 2U ? (maxRows <= 16U ? 16U : maxRows <= 32U ? 32U : 64U) : primaryM;
        auto legalRows = [&](uint32_t m) {
            std::pair<uint32_t, uint32_t> selected{0U, 0U};
            uint32_t v = group == 2U ? std::min<uint64_t>(m, maxRows) : QfaVecRows(std::min<uint64_t>(m, maxRows));
            const uint32_t oldR1 = m <= 64U || m * ((d + 31U) & ~31U) > 16384U ? 8U : 16U;
            const uint32_t oldR2 = ((d + 31U) & ~31U) > 128U || m <= 64U ? 8U : 16U;
            uint64_t score = std::numeric_limits<uint64_t>::max();
            for (uint32_t r1 : {16U, 8U})
                for (uint32_t r2 : {16U, 8U}) {
                    if ((group == 2U && (r1 != (maxRows <= 8U ? 8U : 16U) || r2 != r1)) ||
                        (!optimized && (r1 != oldR1 || r2 != oldR2)) ||
                        (optimized && v <= 8U && (r1 != 8U || r2 != 8U)) || (d == 256U && r2 != 8U) ||
                        QfaUbBytes(m * group, d, r1, r2) > 192U * 1024U)
                        continue;
                    const auto cost = QfaVectorService(v, d, r1, r2);
                    if (cost < score) {
                        score = cost;
                        selected = {r1, r2};
                    }
                }
            return selected;
        };
        while (topM > 16U && legalRows(topM).first == 0U)
            topM /= 2U;
        uint64_t primaryTasks = 0U;
        if (regular) {
            primaryTasks = static_cast<uint64_t>(view.bSize) * view.n2Size *
                           ((view.actualSeqS1Size[0] * view.gSize + topM - 1U) / topM);
        } else {
            for (auto q : view.actualSeqS1Size)
                primaryTasks += view.n2Size * ((q * view.gSize + topM - 1U) / topM);
        }
        std::vector<uint32_t> blocks{topM};
        if (group == 1U && topM > 16U && (!regular || (!pairEligible && primaryTasks * 2U <= aicNum && topM >= 64U))) {
            blocks.push_back(topM / 2U);
            if (!regular && primaryTasks < aicNum) {
                uint64_t rows = 0U;
                for (auto q : view.actualSeqS1Size)
                    rows += q * view.gSize * view.n2Size;
                const uint64_t perCore = (rows + aicNum - 1U) / aicNum;
                uint32_t parallelM = 16U;
                while (parallelM < topM && parallelM * 2U <= perCore)
                    parallelM *= 2U;
                for (uint32_t candidate : {parallelM, std::min(topM, parallelM * 2U)}) {
                    if (std::find(blocks.begin(), blocks.end(), candidate) == blocks.end())
                        blocks.push_back(candidate);
                }
            }
        }
        std::sort(blocks.begin(), blocks.end());
        for (uint32_t m : blocks) {
            if (m > maxM)
                continue;
            const auto rowBatch = legalRows(m);
            if (rowBatch.first == 0U)
                continue;
            const uint32_t r1 = rowBatch.first, r2 = rowBatch.second;
            uint64_t tasks = 0U, totalPanels = 0U, longest = 0U;
            for (uint32_t b = 0U; b < (regular ? 1U : view.bSize); ++b) {
                const uint64_t blocks = (view.actualSeqS1Size[b] * view.gSize + m - 1U) / m;
                const uint64_t panels = (view.actualSeqS2Size[b] + 511U) / 512U;
                tasks += view.n2Size * blocks;
                totalPanels += view.n2Size * blocks * panels;
                longest = std::max(longest, panels);
            }
            if (regular) {
                tasks *= view.bSize;
                totalPanels *= view.bSize;
            }
            std::vector<uint32_t> budgets;
            auto addBudget = [&](uint64_t c) {
                c = std::min<uint64_t>(aicNum, std::max<uint64_t>(1U, c));
                if (std::find(budgets.begin(), budgets.end(), c) == budgets.end())
                    budgets.push_back(c);
            };
            addBudget(aicNum);
            const bool variable = !regular && (!std::all_of(view.actualSeqS1Size.begin(), view.actualSeqS1Size.end(),
                                                            [&](int64_t q) { return q == view.actualSeqS1Size[0]; }) ||
                                               !std::all_of(view.actualSeqS2Size.begin(), view.actualSeqS2Size.end(),
                                                            [&](int64_t k) { return k == view.actualSeqS2Size[0]; }));
            if (variable && aicNum > 1U)
                addBudget(aicNum - 1U);
            if (totalPanels != 0U) {
                const uint64_t chain = (totalPanels + aicNum - 1U) / aicNum;
                addBudget((totalPanels + chain - 1U) / chain);
            }
            if (tasks != 0U) {
                if (tasks <= aicNum) {
                    addBudget(tasks);
                    const uint64_t splits = aicNum / tasks;
                    addBudget(tasks * splits);
                    // A further partial-input refill is a real cost discontinuity, not a case boundary.
                    for (uint32_t rows : {8U, 16U}) {
                        const uint64_t liveRows =
                            std::max<uint64_t>(1U, std::min<uint64_t>(rows, std::min<uint64_t>(m, maxRows)));
                        const uint64_t capacity = r1 * 512U / (liveRows * ((d + 31U) & ~31U));
                        if (capacity != 0U && splits >= capacity)
                            addBudget(tasks * (splits / capacity * capacity));
                    }
                    if (longest != 0U) {
                        const uint64_t chain = (longest + splits - 1U) / splits;
                        addBudget(tasks * ((longest + chain - 1U) / chain));
                        addBudget(tasks * ((longest + chain) / (chain + 1U)));
                    }
                } else {
                    const uint64_t waves = (tasks + aicNum - 1U) / aicNum;
                    addBudget((tasks + waves - 1U) / waves);
                }
            }
            std::sort(budgets.begin(), budgets.end());
            QfaSplitSweep alignedSweep(view, m, 512U, aicNum);
            std::optional<QfaSplitSweep> pageSweep;
            SplitResult candidate(aicNum, 2U);
            uint64_t headLowerBound = 0U;
            const uint64_t headTasks = static_cast<uint64_t>(view.bSize) * view.n2Size;
            // In this range at least one whole head remains on one core; other heads introduce FD.
            if (regular && !view.attenMaskFlag && !view.isAccumSeqS1 && !view.isAccumSeqS2 && maxRows <= m &&
                headTasks >= 2U && headTasks < aicNum && 2U * headTasks > aicNum) {
                SplitResult whole(1U, 2U);
                whole.usedCoreNum = 1U;
                whole.bN2End[0] = 1U;
                const auto cost =
                    QfaProductionScore(whole, view, d, layout, m, 512U, r1, r2, optimized, group, d == 128U, true);
                headLowerBound =
                    std::max(cost.maxService, cost.sharedInputs * headTasks / 1024U) + 8000U + headTasks * 64U;
            }
            for (uint32_t budget : budgets) {
                const auto& aligned = alignedSweep.Get(budget, false);
                for (uint32_t unit : {512U, 128U}) {
                    if ((regular && unit != 512U) || (!optimized && unit != 128U) || (group == 2U && unit != 512U))
                        continue;
                    SplitParam param{m, unit, 16U};
                    if (unit != 512U && !pageSweep.has_value())
                        pageSweep.emplace(view, m, 128U, aicNum);
                    const auto& split = unit == 512U ? aligned : pageSweep->Get(budget, false);
                    if (unit != 512U) {
                        if (optimized && d >= 128U &&
                            QfaLongestChain(split, view, m, unit) >= QfaLongestChain(aligned, view, m, 512U))
                            continue;
                    }
                    for (uint32_t kind = 0U; kind < (optimized ? 4U : 1U); ++kind) {
                        if ((regular && (kind == 1U || kind == 3U)) || (kind == 3U && unit != 512U))
                            continue;
                        if (kind == 2U &&
                            (view.bSize * view.n2Size < 2U || view.bSize * view.n2Size > budget || maxRows > m))
                            continue;
                        if (kind == 2U && budget > headTasks && budget < 2U * headTasks && headLowerBound != 0U &&
                            headLowerBound >= chosen.score)
                            continue;
                        std::optional<SplitResult> alternate;
                        if (kind == 1U)
                            alternate.emplace(QfaWholeTaskPlan(split, view, m, unit));
                        else if (kind == 2U)
                            alternate.emplace(QfaHeadPlan(view, param, budget));
                        else if (kind == 3U)
                            alternate.emplace(QfaBalancedWholeTasks(view, m, budget, d, layout, r1, r2, group));
                        const auto& topology = kind == 0U ? split : *alternate;
                        if (topology.usedCoreNum == 0U || topology.maxS2SplitNum > 32U)
                            continue;
                        const auto production = QfaProductionScore(topology, view, d, layout, m, unit, r1, r2,
                                                                   optimized, group, d == 128U, regular);
                        const uint64_t lowerBound =
                            std::max(production.maxService, production.sharedInputs / 1024U) +
                            (topology.numOfFdHead == 0U ? 0U : 8000U + topology.usedCoreNum * 64ULL);
                        if (optimized && lowerBound >= chosen.score)
                            continue;
                        const auto chain = QfaLongestChain(topology, view, m, unit, nullptr, nullptr, regular);
                        uint64_t parts = 0U;
                        for (uint32_t g = 0U; g < topology.numOfFdHead; ++g)
                            parts += topology.fdRes.s2SplitNumOfFdHead[g] * group;
                        candidate = topology;
                        for (uint32_t fdRows : {16U, 8U}) {
                            if (fdRows == 8U && (!optimized || group == 2U || topology.numOfFdHead == 0U))
                                continue;
                            for (uint32_t columns = 1U; columns <= (d == 256U ? 2U : 1U); ++columns) {
                                bool complete = fdRows == 16U;
                                for (uint32_t g = 0U; g < candidate.numOfFdHead; ++g) {
                                    const uint32_t rows = (topology.fdRes.gS1SplitNumOfFdHead[g] - 1U) * 16U +
                                                          topology.fdRes.gS1LastPartSizeOfFdHead[g];
                                    complete &= topology.fdRes.gS1LastPartSizeOfFdHead[g] == 16U;
                                    candidate.fdRes.gS1SplitNumOfFdHead[g] =
                                        (rows + fdRows - 1U) / fdRows * columns * group;
                                    candidate.fdRes.gS1LastPartSizeOfFdHead[g] = (rows - 1U) % fdRows + 1U;
                                }
                                if (columns > 1U && (!complete || candidate.numOfFdHead == 0U))
                                    continue;
                                uint64_t mergeCost = 0U;
                                const bool summarized = optimized && QfaUniformFdCost(candidate, view, d, fdRows, r1,
                                                                                      r2, columns, group, mergeCost);
                                if (!summarized && candidate.numOfFdHead != 0U)
                                    SplitFD(candidate);
                                const auto cost = QfaPlanScore(candidate, view, d, layout, m, unit, fdRows, r1, r2,
                                                               columns, optimized, group, d == 128U, &production,
                                                               summarized ? &mergeCost : nullptr);
                                const bool extraPublication =
                                    parts > retainedParts || candidate.usedCoreNum > best.usedCoreNum || group != heads;
                                const bool sameChain = group == heads && m == chosen.m && chain >= retainedChain;
                                const bool marginalExpansion = chosen.score != std::numeric_limits<uint64_t>::max() &&
                                                               extraPublication && (sameChain || group != heads) &&
                                                               cost < chosen.score && chosen.score - cost <= 2000U;
                                if (cost < chosen.score && !marginalExpansion) {
                                    chosen = {m, unit, fdRows, r1, r2, columns, cost, kind, budget};
                                    if constexpr (Materialize) {
                                        if (summarized && candidate.numOfFdHead != 0U)
                                            SplitFD(candidate);
                                        best = candidate;
                                    } else {
                                        best.usedCoreNum = candidate.usedCoreNum;
                                    }
                                    heads = group;
                                    retainedChain = chain;
                                    retainedParts = parts;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    base = physical;
    base.n2Size /= heads;
    return chosen;
}

template <typename ScratchBytes>
inline bool QfaBuildPlan(BaseInfo base, uint32_t d, uint32_t layout, uint32_t pageSize, uint32_t aicNum,
                         QfaHostPlanData& plan, bool& useOptimizedVectorService, bool allowPairedHeads,
                         ScratchBytes scratchBytes)
{
    // Device-only lengths remain conservative here and are checked by the iterator.
    SplitResult best(aicNum, 2U);
    const auto chosen =
        QfaSelectUnifiedPlan(base, d, layout, pageSize, aicNum, allowPairedHeads, best, plan.headGroupSize);
    if (chosen.score == std::numeric_limits<uint64_t>::max())
        return false;
    const uint32_t chosenM = chosen.m, chosenColumns = chosen.columns;
    uint32_t maxVectorRows = 0U;
    for (uint32_t b = 0U; b < base.bSize; ++b)
        maxVectorRows =
            std::max(maxVectorRows,
                     plan.headGroupSize == 2U ?
                         static_cast<uint32_t>(std::min<uint64_t>(chosenM, base.actualSeqS1Size[b] * base.gSize)) :
                         QfaVecRows(std::min<uint64_t>(chosenM, base.actualSeqS1Size[b] * base.gSize)));
    useOptimizedVectorService = maxVectorRows >= 4U && (d == 64U || d == 72U || d == 128U || d == 256U);
    plan.returnSoftmaxLse = 1U;
    plan.m = chosenM;
    plan.kvUnit = chosen.unit;
    plan.fdRows = chosen.fdRows;
    plan.hpRows = chosen.r1;
    plan.hpVec2Rows = chosen.r2;
    plan.hpScratchBytes = scratchBytes(plan.hpRows, 32768U);
    const uint64_t ub = QfaUbBytes(chosenM * plan.headGroupSize, d, plan.hpRows, plan.hpVec2Rows, plan.hpScratchBytes);
    if (ub > 192U * 1024U)
        return false;
    plan.cores = std::max(1U, best.usedCoreNum);
    plan.fdGroups = best.numOfFdHead;
    plan.fdVectors = best.usedVecNumOfFd;
    plan.fdColumns = chosenColumns;
    for (uint32_t b = 0; b < base.bSize; ++b) {
        plan.qBlocks[b] = (base.actualSeqS1Size[b] * base.gSize + chosenM - 1U) / chosenM;
        plan.kvPanels[b] = (base.actualSeqS2Size[b] + plan.kvUnit - 1U) / plan.kvUnit;
    }
    // With uniform batch geometry, a head-major traversal preserves every core's panel count.
    // Separate cores can read adjacent heads of the same PA_BBND page in the same wave.
    plan.headMajorOrder = d == 128U && layout == 0U && base.n2Size == 4U && base.bSize >= plan.cores &&
                          plan.cores % base.n2Size == 0U &&
                          std::all_of(base.actualSeqS1Size.begin(), base.actualSeqS1Size.end(),
                                      [&](int64_t q) { return q == base.actualSeqS1Size[0]; }) &&
                          std::all_of(base.actualSeqS2Size.begin(), base.actualSeqS2Size.end(),
                                      [&](int64_t k) { return k == base.actualSeqS2Size[0]; });
    if (plan.headGroupSize == 2U)
        plan.headMajorOrder = false;
    std::fill(plan.group0, plan.group0 + 48U, ~0U);
    std::fill(plan.group1, plan.group1 + 48U, ~0U);
    const auto coord = [](uint32_t bn, uint32_t gs, uint32_t k) { return std::make_tuple(bn, gs, k); };
    for (uint32_t c = 0; c < best.usedCoreNum; ++c) {
        plan.bnEnd[c] = best.bN2End[c];
        plan.gsEnd[c] = best.gS1End[c];
        plan.kvEnd[c] = best.s2End[c];
    }
    uint32_t prefix = 0;
    for (uint32_t j = 0; j < best.numOfFdHead; ++j) {
        plan.fdBN[j] = best.fdRes.bN2IdxOfFdHead[j];
        plan.fdGS[j] = best.fdRes.gS1IdxOfFdHead[j];
        plan.fdSplits[j] = best.fdRes.s2SplitNumOfFdHead[j];
        plan.fdPrefix[j] = prefix;
        plan.fdItems[j] = best.fdRes.gS1SplitNumOfFdHead[j];
        plan.fdLastRows[j] = best.fdRes.gS1LastPartSizeOfFdHead[j];
        const auto begin = coord(plan.fdBN[j], plan.fdGS[j], 0U);
        const auto end = coord(plan.fdBN[j], plan.fdGS[j], plan.kvPanels[plan.fdBN[j] / base.n2Size]);
        uint32_t index = 0;
        for (uint32_t c = 0; c < best.usedCoreNum; ++c) {
            const auto coreBegin =
                c == 0 ? coord(0, 0, 0) : coord(best.bN2End[c - 1], best.gS1End[c - 1], best.s2End[c - 1]);
            const auto coreEnd = coord(best.bN2End[c], best.gS1End[c], best.s2End[c]);
            if (coreBegin < end && coreEnd > begin) {
                if (plan.group0[c] == ~0U) {
                    plan.group0[c] = j;
                    plan.slot0[c] = prefix + index;
                } else {
                    if (plan.group1[c] != ~0U)
                        return false;
                    plan.group1[c] = j;
                    plan.slot1[c] = prefix + index;
                }
                ++index;
            }
        }
        if (index != plan.fdSplits[j])
            return false;
        prefix += plan.fdSplits[j];
    }
    plan.partials = prefix * plan.headGroupSize;
    for (uint32_t v = 0; v < best.usedVecNumOfFd; ++v) {
        plan.fdEndGroup[v] = best.fdRes.gS1IdxEndOfFdHead[v];
        plan.fdEndItem[v] = best.fdRes.gS1IdxEndOfFdHeadSplit[v];
    }
    // Preserve production and FD ranges; specialize only the D256 Vector layout.
    const bool compactD256 =
        useOptimizedVectorService && d == 256U && chosenM == 128U && plan.headGroupSize == 1U && plan.hpRows == 8U &&
        plan.hpVec2Rows == 8U &&
        std::all_of(plan.fdSplits, plan.fdSplits + plan.fdGroups, [](uint32_t splits) { return splits <= 16U; });
    if (compactD256) {
        const uint32_t scratch = scratchBytes(16U, 16384U);
        if (scratch == 16384U && QfaUbBytes(chosenM, d, 16U, 8U, scratch, true) <= 192U * 1024U) {
            plan.hpRows = 16U;
            plan.hpScratchBytes = scratch;
        }
    }
    return true;
}

} // namespace a8c8_plan

ge::graphStatus TilingQuantFlashAttnA8C8(gert::TilingContext* context, const QfaTilingInfo& qfaInfo);

} // namespace quant_flash_attn
} // namespace optiling

#endif
