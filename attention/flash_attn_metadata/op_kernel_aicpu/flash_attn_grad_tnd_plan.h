/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// PTO-owned TND planning, migrated from the same repository AscendC production policies.
#ifndef FLASH_ATTN_GRAD_TND_PLAN_H
#define FLASH_ATTN_GRAD_TND_PLAN_H
#include "flash_attn_grad_tnd_normalized.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
namespace aicpu {
namespace fag_pto_tnd {
constexpr int32_t MAGIC = 0x46545031;
constexpr int32_t VERSION = 2;
constexpr int64_t LEGACY_WORDS = 412;
constexpr int64_t HEADER_WORDS = 16;
constexpr int64_t SCALAR_COUNT = 16;
constexpr int64_t CUT_COUNT = 37;
constexpr int64_t BATCH_FIELDS = 16;
constexpr int64_t NUM_TWO = 2, NUM_THREE = 3;
constexpr int64_t CORE_LIST_NUM = 36, DETER_PREFIX_ARRAY_SIZE = 132;
constexpr int64_t INPUT_FORMAT_TND = 1;
constexpr int FAG_GATE114_VARIANT = 2, FAG_CAUSAL38_VARIANT = 4, FAG_CAUSAL38_DISABLE_MODE1 = 0;
enum class SparseMode : uint32_t {
    NO_MASK = 0,
    LEFT_UP_CAUSAL = 2,
    RIGHT_DOWN_CAUSAL = 3,
    BAND = 4
};
enum class DeterSparseType : uint32_t {
    DETER_DENSE = 0,
    DETER_CAUSAL = 1,
    DETER_BAND = 2
};
enum class SplitAxisEnum {
    BN2GS1S2,
    BN2S2
};
enum class DtypeEnum {
    BFLOAT16
};
enum class ConstAxisTemplateNum {
    NUM16 = 16
};
struct DeterParamRegbase {
    static constexpr int64_t SCHEDULE_DEFAULT = 0, SCHEDULE_TND_GQA_COLUMN_EXCLUSIVE = 1;
    static constexpr int64_t SCHEDULE_TND_GQA_BN2_EXCLUSIVE = 2, SCHEDULE_TND_GQA_CAUSAL_MIRROR_COLUMN_EXCLUSIVE = 3;
};
inline int64_t CeilDivideBy(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}
inline int64_t AlignTo(int64_t x, int64_t y)
{
    return CeilDivideBy(x, y) * y;
}
inline std::vector<int64_t> SliceVector(const std::vector<int64_t>& x, int64_t step)
{
    std::vector<int64_t> result;
    for (size_t i = 0; i < x.size(); i += step)
        result.push_back(x[i]);
    return result;
}
struct FuzzyBaseInfoParamsRegbase {
    int64_t b = 0, n1 = 0, n2 = 0, g = 0, aicNum = 0, d = 0, d1 = 0;
    int64_t s1Inner = 64, s1CvRatio = 2, s2Inner = 128, s2CvRatio = 1;
    int64_t s1Token = 0, s2Token = 0, deterMaxRound = 0, deterPrefixStep = 1;
    int64_t layoutType = INPUT_FORMAT_TND, tndLineDeter = 0, tndGqaScheduleMode = 0, tndGqaHeadPairGroup = 0;
    uint32_t sparseMode = 0, deterSparseType = 0;
    bool isS1S2Same = true, isDeterministic = true, tndGqaColumnOwnerAllowed = true, coreDivide = false;
    SplitAxisEnum splitAxis = SplitAxisEnum::BN2GS1S2;
    DtypeEnum inputDtype = DtypeEnum::BFLOAT16;
    std::vector<int64_t> actualSeqQlen, actualSeqKvlen;
    int64_t deterPrefix0[132] = {}, deterPrefix1[132] = {}, deterPrefix2[132] = {};
    int64_t deterPrefix[132] = {}, deterPrefixAlign[132] = {}, deterColStart[37] = {};
};
inline void CalcleActualToken(const FuzzyBaseInfoParamsRegbase& p, int64_t b, int64_t& a, int64_t& z)
{
    a = p.s1Token;
    z = p.s2Token;
    if (p.sparseMode == 3 || p.sparseMode == 4) {
        a += p.actualSeqQlen[b] - p.actualSeqKvlen[b];
        z -= p.actualSeqQlen[b] - p.actualSeqKvlen[b];
    }
}
inline void GetTNDOuterMN(const FuzzyBaseInfoParamsRegbase& p, int64_t bIdx, int64_t& m, int64_t& n)
{
    const int64_t cubeM = p.s1Inner * p.s1CvRatio;
    const int64_t cubeN = p.s2Inner * p.s2CvRatio;
    if (p.actualSeqQlen[bIdx] <= 0 || p.actualSeqKvlen[bIdx] <= 0) {
        m = 0;
        n = 0;
        return;
    }
    m = CeilDivideBy(p.actualSeqQlen[bIdx], cubeM);
    n = CeilDivideBy(p.actualSeqKvlen[bIdx], cubeN);
    if (m < 1) {
        m = 1;
    }
    if (n < 1) {
        n = 1;
    }
}

inline int64_t CalTNDCase0RecRounds(int64_t k, int64_t m, int64_t n)
{
    if (n <= 0) {
        return 0;
    }
    if (NUM_TWO * k < m + 1 && k < n) {
        return std::max(static_cast<int64_t>(0), NUM_TWO * m - NUM_TWO * k + 1);
    }
    return m;
}

inline int64_t CalTNDSingleCausalRounds(int64_t k, int64_t m, int64_t n)
{
    const int64_t t = n / (NUM_TWO * k);
    const int64_t ell = n % (NUM_TWO * k);
    const int64_t rounds2k = (NUM_TWO * m + 1) * t - NUM_TWO * k * t * t;
    if (ell == 0) {
        return rounds2k;
    }
    if (t == 0) {
        return CalTNDCase0RecRounds(k, m, n);
    }
    const int64_t l = m - NUM_TWO * k * t;
    const int64_t phaseCount = CeilDivideBy(ell, k);
    int64_t roundsTail = 0;
    for (int64_t s = 0; s < phaseCount; ++s) {
        roundsTail += std::max(static_cast<int64_t>(0), l - s * k);
    }
    return rounds2k + roundsTail;
}

inline int64_t GetTNDRightDownRowShift(int64_t m, int64_t n)
{
    return std::max(static_cast<int64_t>(0), m - n - 1);
}

inline void GetTNDLeftUpVirt(int64_t m, int64_t n, int64_t& virtM, int64_t& virtN)
{
    if (m < n) {
        n = m;
    }
    virtM = NUM_TWO * m - n + 1;
    virtN = n;
}

inline void GetTNDRightDownVirt(int64_t m, int64_t n, int64_t& virtM, int64_t& virtN)
{
    m -= GetTNDRightDownRowShift(m, n);
    if (m < 1 || n < 1) {
        virtM = 0;
        virtN = 0;
        return;
    }
    virtM = m;
    virtN = NUM_TWO * n - m + NUM_THREE;
}

constexpr int64_t TND_LINE_TINY_AREA = 8;

inline bool TNDRunIsLineWorthy(int64_t k, int64_t virtM, int64_t virtN, int64_t pairCount)
{
    if (virtM <= 0 || virtN <= 0 || virtM * virtN <= TND_LINE_TINY_AREA) {
        return false;
    }
    return pairCount > 0 && virtN * pairCount >= k && virtM >= std::min(k, virtN);
}

enum class TNDRunShapeKind : uint8_t {
    INVALID = 0,
    CAUSAL_LEFT_UP,
    CAUSAL_RIGHT_DOWN,
    BAND,
    BAND_DENSE,
};

struct TNDRunShape {
    int64_t m = 1;
    int64_t n = 1;
    int64_t p = 0;
    int64_t q = 0;
    TNDRunShapeKind kind = TNDRunShapeKind::INVALID;
};

inline bool SameTNDRunShape(const TNDRunShape& a, const TNDRunShape& b)
{
    return a.m == b.m && a.n == b.n && a.p == b.p && a.q == b.q && a.kind == b.kind;
}

inline TNDRunShape MakeTNDCausalRunShape(const FuzzyBaseInfoParamsRegbase& p, int64_t bIdx, bool leftUp)
{
    TNDRunShape s;
    GetTNDOuterMN(p, bIdx, s.m, s.n);
    if (leftUp) {
        if (s.m < s.n) {
            s.n = s.m;
        }
        s.kind = TNDRunShapeKind::CAUSAL_LEFT_UP;
    } else {
        s.m -= GetTNDRightDownRowShift(s.m, s.n);
        s.kind = TNDRunShapeKind::CAUSAL_RIGHT_DOWN;
    }
    return s;
}

inline void GetTNDBandMN(FuzzyBaseInfoParamsRegbase& p, int64_t bIdx, int64_t& m, int64_t& n, int64_t& bandP,
                         int64_t& bandQ)
{
    GetTNDOuterMN(p, bIdx, m, n);
    if (m < 1 || n < 1) {
        bandP = 0;
        bandQ = 0;
        return;
    }
    int64_t actualS1Token = 0;
    int64_t actualS2Token = 0;
    CalcleActualToken(p, bIdx, actualS1Token, actualS2Token);
    const int64_t cubeM = p.s1Inner * p.s1CvRatio;
    const int64_t cubeN = p.s2Inner * p.s2CvRatio;
    bandP = CeilDivideBy(actualS1Token, cubeM) + 1;
    bandQ = CeilDivideBy(actualS2Token, cubeN) + 1;
    bandP = bandP > m ? m : bandP;
    bandQ = bandQ > n ? n : bandQ;
    if (bandP < 0) {
        n = n + bandP;
        bandQ = bandP + bandQ;
        bandP = 1;
    } else if (bandQ < 0) {
        m = m + bandQ;
        bandP = bandP + bandQ;
        bandQ = 1;
    }
    int64_t actualM = m;
    int64_t actualN = n;
    if (bandP + bandQ <= m) {
        const int64_t l1 = bandQ - 1;
        const int64_t l2 = std::min(n - bandQ + 1, m + NUM_TWO - bandP - bandQ);
        const int64_t l3 = std::max(static_cast<int64_t>(0), std::min(bandP + n - m - 1, bandP + bandQ - NUM_TWO));
        actualM = (l3 == 0) ? (bandP + bandQ + l2 - NUM_TWO) : m;
        actualN = l1 + l2 + l3;
    } else {
        actualM = m;
        actualN = std::min(m - 1 + bandQ, n);
    }
    m = actualM;
    n = actualN;
}

inline TNDRunShape MakeTNDBandRunShape(FuzzyBaseInfoParamsRegbase& p, int64_t bIdx)
{
    TNDRunShape s;
    GetTNDBandMN(p, bIdx, s.m, s.n, s.p, s.q);
    s.kind = (s.p >= s.m) ? TNDRunShapeKind::BAND_DENSE : TNDRunShapeKind::BAND;
    return s;
}

inline int64_t CountTNDSameShapeRun(FuzzyBaseInfoParamsRegbase& p, int64_t start, bool leftUp, bool isBand,
                                    TNDRunShape& shape)
{
    shape = isBand ? MakeTNDBandRunShape(p, start) : MakeTNDCausalRunShape(p, start, leftUp);
    int64_t end = start;
    while (end + 1 < p.b) {
        const TNDRunShape next = isBand ? MakeTNDBandRunShape(p, end + 1) : MakeTNDCausalRunShape(p, end + 1, leftUp);
        if (!SameTNDRunShape(next, shape)) {
            break;
        }
        end += 1;
    }
    return end - start + 1;
}

inline void CopyTNDPrefix(const std::vector<int64_t>& src, int64_t* dst, size_t dstLen)
{
    std::fill(dst, dst + dstLen, static_cast<int64_t>(0));
    const size_t n = std::min(src.size(), dstLen);
    std::copy(src.begin(), src.begin() + static_cast<long>(n), dst);
}

inline void StoreTNDPrefix(const std::vector<int64_t>& raw, int64_t step, int64_t tailValue, int64_t* dst)
{
    std::vector<int64_t> sliced = SliceVector(raw, step);
    sliced.push_back(tailValue);
    CopyTNDPrefix(sliced, dst, DETER_PREFIX_ARRAY_SIZE);
}

inline int64_t CalTNDBandWideCols(int64_t m, int64_t n, int64_t p, int64_t q)
{
    const int64_t nNew = std::min(m - 1 + q, n);
    const int64_t l1 = m - p;
    const int64_t l2 = p + q - m;
    const int64_t l3 = nNew - l1 - l2;
    const int64_t foldCols = std::max(static_cast<int64_t>(0), l3 - p + 1);
    return nNew - foldCols;
}

inline int64_t CalTNDBandNarrowRounds(int64_t k, int64_t n1, int64_t m, int64_t n, int64_t p, int64_t q)
{
    const int64_t l1 = q - 1;
    const int64_t l2 = std::min(n - q + 1, m + NUM_TWO - p - q);
    const int64_t l3 = std::max(static_cast<int64_t>(0), std::min(p + n - m - 1, p + q - NUM_TWO));
    int64_t overlap = 0;
    int64_t nNew = l1 + l2 + l3;
    if (l3 != 0 && p <= l3) {
        overlap = std::min(l3 - p + 1, l1);
    }
    const int64_t seg = p + q - 1;
    return seg * CeilDivideBy((nNew - overlap) * n1 + overlap, k);
}

inline bool TNDRunEmpty(const TNDRunShape& shape, bool isBand)
{
    return shape.m < 1 || shape.n < 1 || (isBand && shape.p + shape.q < NUM_TWO);
}

inline bool PreferTNDLineDeter(FuzzyBaseInfoParamsRegbase& p)
{
    if (p.g != 1) {
        return false;
    }
    if (p.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) && !p.isS1S2Same) {
        return false;
    }
    if (p.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL) &&
        p.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
        return false;
    }
    if (p.n2 < NUM_TWO) {
        return false;
    }

    const int64_t k = static_cast<int64_t>(p.aicNum);
    const int64_t n1 = p.n2;
    const bool isBand = (p.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) ||
                        (p.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                         p.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND));
    const bool leftUp = !isBand && (p.sparseMode != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL));
    bool hasWork = false;
    int64_t i = 0;
    while (i < p.b) {
        TNDRunShape shape;
        const int64_t groupSize = CountTNDSameShapeRun(p, i, leftUp, isBand, shape);
        const int64_t total = n1 * groupSize;
        const int64_t runPairs = total / NUM_TWO;
        const bool runHasSingle = (total % NUM_TWO == 1);
        if (TNDRunEmpty(shape, isBand)) {
            i += groupSize;
            continue;
        }
        hasWork = true;
        if (!isBand) {
            int64_t virtM = 0;
            int64_t virtN = 0;
            if (leftUp) {
                GetTNDLeftUpVirt(shape.m, shape.n, virtM, virtN);
            } else {
                GetTNDRightDownVirt(shape.m, shape.n, virtM, virtN);
            }
            const bool foldedCausalPairsSafe = !leftUp && k > 0 && runPairs > 0 && p.isS1S2Same && shape.m == shape.n;
            const bool genericLineWorthy = TNDRunIsLineWorthy(k, virtM, virtN, runPairs);
            if (!foldedCausalPairsSafe && !genericLineWorthy) {
                return false;
            }
            if (!leftUp && runHasSingle && shape.m < std::min(k, shape.n)) {
                return false;
            }
        } else if (shape.kind == TNDRunShapeKind::BAND_DENSE) {
            if (!TNDRunIsLineWorthy(k, shape.m, shape.n, total)) {
                return false;
            }
        } else if (shape.p + shape.q <= shape.m) {
            if (shape.m * shape.n <= TND_LINE_TINY_AREA) {
                return false;
            }
        } else {
            const int64_t denseN = CalTNDBandWideCols(shape.m, shape.n, shape.p, shape.q);
            if (!TNDRunIsLineWorthy(k, shape.m, denseN, total)) {
                return false;
            }
        }
        i += groupSize;
    }
    return hasWork;
}

struct TNDLineDeterPack {
    std::vector<int64_t> prefixSolo = {0};
    std::vector<int64_t> prefixSingle = {0};
    int64_t rLine = 0;
};

inline TNDLineDeterPack BuildTNDLineDeter(FuzzyBaseInfoParamsRegbase& p)
{
    TNDLineDeterPack pack;
    const int64_t k = static_cast<int64_t>(p.aicNum);
    const int64_t n1 = p.n2;
    const bool isBand = (p.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) ||
                        (p.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                         p.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND));
    const bool leftUp = !isBand && (p.sparseMode != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL));
    for (int64_t batch = 0; batch < p.b; ++batch) {
        int64_t m = 0;
        int64_t n = 0;
        GetTNDOuterMN(p, batch, m, n);
        if (!isBand) {
            const TNDRunShape shape = MakeTNDCausalRunShape(p, batch, leftUp);
            m = shape.m;
            n = shape.n;
        }
        const int64_t tasks = !isBand && m == 1 && n == 1 ? n1 : 0;
        pack.prefixSingle.push_back(pack.prefixSingle.back() + tasks);
    }
    const int64_t singleRounds = CeilDivideBy(pack.prefixSingle.back(), k);
    std::vector<int64_t> prefixSolo = {singleRounds};

    int64_t i = 0;
    while (i < p.b) {
        TNDRunShape shape;
        const int64_t groupSize = CountTNDSameShapeRun(p, i, leftUp, isBand, shape);
        const int64_t total = n1 * groupSize;
        const int64_t runPairs = total / NUM_TWO;
        const bool runHasSingle = (total % NUM_TWO == 1);
        int64_t runSolo = 0;
        if (!TNDRunEmpty(shape, isBand)) {
            if (!isBand && shape.m == 1 && shape.n == 1) {
                runSolo = 0;
            } else if (!isBand) {
                int64_t virtM = 0;
                int64_t virtN = 0;
                if (leftUp) {
                    GetTNDLeftUpVirt(shape.m, shape.n, virtM, virtN);
                } else {
                    GetTNDRightDownVirt(shape.m, shape.n, virtM, virtN);
                }
                const int64_t pairRounds =
                    (runPairs > 0 && virtN > 0) ? (CeilDivideBy(virtN * runPairs, k) * virtM) : 0;
                const int64_t singleTailRounds = runHasSingle ?
                                                     (leftUp ? CalTNDSingleCausalRounds(k, shape.m, shape.n) :
                                                               (CeilDivideBy(shape.n, k) * shape.m)) :
                                                     0;
                runSolo = pairRounds + singleTailRounds;
            } else if (shape.kind == TNDRunShapeKind::BAND_DENSE) {
                runSolo = CeilDivideBy(shape.n * total, k) * shape.m;
            } else if (shape.p + shape.q <= shape.m) {
                runSolo = CalTNDBandNarrowRounds(k, total, shape.m, shape.n, shape.p, shape.q);
            } else {
                const int64_t denseN = CalTNDBandWideCols(shape.m, shape.n, shape.p, shape.q);
                runSolo = CeilDivideBy(denseN * total, k) * shape.m;
            }
        }
        const int64_t soloEnd = prefixSolo.back() + runSolo;
        for (int64_t runIdx = 0; runIdx < groupSize; ++runIdx) {
            prefixSolo.push_back(soloEnd);
        }
        i += groupSize;
    }

    pack.prefixSolo = std::move(prefixSolo);
    pack.rLine = pack.prefixSolo.back();
    return pack;
}

struct Plan {
    int32_t family = 0, strategy = 0, status = 0;
    std::array<int64_t, SCALAR_COUNT> scalar{};
    std::array<int64_t, CUT_COUNT> cuts{};
    std::vector<std::array<int64_t, BATCH_FIELDS>> batch;
    fag_pto_normalized::Wire normalized{};
};
class Planner {
public:
    FuzzyBaseInfoParamsRegbase fBaseParams;
    struct TndExclusiveRun {
        int64_t count;
        int64_t rounds;
        int64_t area;
    };

    struct TndExclusiveCandidate {
        std::vector<int64_t> cuts;
        int64_t maxRounds = 0;
        int64_t totalTasks = 0;
        int64_t activeCores = 0;
    };

    static int64_t ExclusiveCeilDivide(int64_t value, int64_t divisor)
    {
        return value / divisor + static_cast<int64_t>(value % divisor != 0);
    }

    bool CanApplyTndGqaExclusive() const
    {
        constexpr int64_t kCubeSize = 128;
        const bool supported =
            FAG_GATE114_VARIANT != 0 && fBaseParams.isDeterministic && fBaseParams.layoutType == INPUT_FORMAT_TND &&
            fBaseParams.splitAxis == SplitAxisEnum::BN2GS1S2 && fBaseParams.tndGqaColumnOwnerAllowed &&
            !fBaseParams.coreDivide && fBaseParams.g > 1 && fBaseParams.b > 0 && fBaseParams.n2 > 0 &&
            fBaseParams.aicNum > 0 && fBaseParams.aicNum <= CORE_LIST_NUM && fBaseParams.s1Inner > 0 &&
            kCubeSize % fBaseParams.s1Inner == 0 && fBaseParams.s1CvRatio == kCubeSize / fBaseParams.s1Inner &&
            fBaseParams.s2Inner > 0 && kCubeSize % fBaseParams.s2Inner == 0 &&
            fBaseParams.s2CvRatio == kCubeSize / fBaseParams.s2Inner &&
            static_cast<int64_t>(fBaseParams.actualSeqQlen.size()) >= fBaseParams.b &&
            static_cast<int64_t>(fBaseParams.actualSeqKvlen.size()) >= fBaseParams.b;
        if (!supported) {
            return false;
        }
        for (int64_t batch = 0; batch < fBaseParams.b; ++batch) {
            // Token normalization uses INT32_MAX. Beyond that window, nominal
            // rectangle/triangle counts do not prove the number of useful tasks.
            const int64_t qLen = fBaseParams.actualSeqQlen[batch];
            const int64_t kvLen = fBaseParams.actualSeqKvlen[batch];
            if (qLen <= 0 || kvLen <= 0 || qLen > std::numeric_limits<int32_t>::max() ||
                kvLen > std::numeric_limits<int32_t>::max()) {
                return false;
            }
        }
        return true;
    }

    // Keep the Host search proportional to batches, not to the number of BN2 units or S2 columns.
    bool AppendExclusiveRun(std::vector<TndExclusiveRun>& runs, int64_t count, int64_t rounds, int64_t area,
                            int64_t& totalRounds, int64_t& totalArea) const
    {
        if (count == 0) {
            return true;
        }
        constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
        if (count < 0 || rounds <= 0 || area <= 0 || rounds > (kMax - totalRounds) / count ||
            area > (kMax - totalArea) / count) {
            return false;
        }
        runs.push_back({count, rounds, area});
        totalRounds += count * rounds;
        totalArea += count * area;
        return true;
    }

    // Positive weights make maximal-prefix greedy an exact feasibility test for fixed-order partitions,
    // including two simultaneous upper bounds. It does not assert globally optimal device time.
    bool ExclusiveFits(const std::vector<TndExclusiveRun>& runs, int64_t roundCap, int64_t areaCap) const
    {
        int64_t coreCount = 1;
        int64_t rounds = 0;
        int64_t area = 0;
        for (const auto& run : runs) {
            if (run.rounds > roundCap || run.area > areaCap) {
                return false;
            }
            int64_t left = run.count;
            while (left > 0) {
                int64_t take = std::min((roundCap - rounds) / run.rounds, (areaCap - area) / run.area);
                if (take == 0) {
                    if (++coreCount > static_cast<int64_t>(fBaseParams.aicNum)) {
                        return false;
                    }
                    rounds = 0;
                    area = 0;
                    take = std::min(roundCap / run.rounds, areaCap / run.area);
                }
                take = std::min(take, left);
                rounds += take * run.rounds;
                area += take * run.area;
                left -= take;
            }
        }
        return true;
    }

    int64_t MinExclusiveLimit(const std::vector<TndExclusiveRun>& runs, int64_t total, int64_t fixedCap,
                              bool minimizeRounds) const
    {
        int64_t lo = ExclusiveCeilDivide(total, static_cast<int64_t>(fBaseParams.aicNum));
        for (const auto& run : runs) {
            lo = std::max(lo, minimizeRounds ? run.rounds : run.area);
        }
        int64_t hi = total;
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo) / NUM_TWO;
            const bool fits = minimizeRounds ? ExclusiveFits(runs, mid, fixedCap) : ExclusiveFits(runs, fixedCap, mid);
            if (fits) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    bool BuildExclusiveCuts(const std::vector<TndExclusiveRun>& runs, int64_t roundCap, int64_t areaCap, int64_t total,
                            int64_t taskScale, TndExclusiveCandidate& candidate) const
    {
        if (taskScale <= 0 || total <= 0 || total > std::numeric_limits<int64_t>::max() / taskScale || runs.empty()) {
            return false;
        }
        const int64_t aicNum = static_cast<int64_t>(fBaseParams.aicNum);
        std::vector<int64_t> cuts(aicNum + 1, 0);
        int64_t coreIdx = 0;
        int64_t rounds = 0;
        int64_t area = 0;
        int64_t cursor = 0;
        int64_t maxRounds = 0;
        for (const auto& run : runs) {
            if (run.count <= 0 || run.rounds <= 0 || run.area <= 0 || run.rounds > roundCap || run.area > areaCap ||
                run.rounds > (total - cursor) / run.count) {
                return false;
            }
            int64_t left = run.count;
            while (left > 0) {
                int64_t take = std::min((roundCap - rounds) / run.rounds, (areaCap - area) / run.area);
                if (take == 0) {
                    maxRounds = std::max(maxRounds, rounds);
                    if (++coreIdx >= aicNum) {
                        return false;
                    }
                    cuts[coreIdx] = cursor * taskScale;
                    rounds = 0;
                    area = 0;
                    take = std::min(roundCap / run.rounds, areaCap / run.area);
                }
                take = std::min(take, left);
                rounds += take * run.rounds;
                area += take * run.area;
                cursor += take * run.rounds;
                left -= take;
            }
        }
        maxRounds = std::max(maxRounds, rounds);
        if (cursor != total || maxRounds <= 0 || maxRounds > roundCap) {
            return false;
        }
        for (int64_t c = coreIdx + 1; c <= aicNum; c++) {
            cuts[c] = cursor * taskScale;
        }
        candidate.cuts = std::move(cuts);
        candidate.maxRounds = maxRounds * taskScale;
        candidate.totalTasks = total * taskScale;
        candidate.activeCores = coreIdx + 1;
        return true;
    }

    bool AdmitExclusiveCandidate(const std::vector<TndExclusiveRun>& runs, int64_t taskScale,
                                 const TndExclusiveCandidate& candidate, const char* /* mode */,
                                 bool columnMode = false) const
    {
        const int64_t cores = static_cast<int64_t>(fBaseParams.aicNum);
        const int64_t oldRounds = fBaseParams.deterMaxRound;
        if (cores <= 0 || cores > CORE_LIST_NUM || taskScale <= 0 || candidate.totalTasks <= 0 || oldRounds <= 0 ||
            candidate.cuts.size() != static_cast<size_t>(cores + 1)) {
            (void)0;
            return false;
        }
        const int64_t lowerBound = ExclusiveCeilDivide(candidate.totalTasks, cores);
        const char* reason = nullptr;
        int64_t maxRounds = 0;
        int64_t activeCores = 0;
        int64_t runBegin = 0;
        int64_t runEnd = 0;
        size_t runIdx = 0;
        for (int64_t core = 0; core <= cores; ++core) {
            const int64_t cut = candidate.cuts[core];
            if (cut < 0 || cut > candidate.totalTasks || cut % taskScale != 0 ||
                (core > 0 && cut < candidate.cuts[core - 1])) {
                reason = "invalid or non-monotonic cut";
                break;
            }
            const int64_t normalizedCut = cut / taskScale;
            while (normalizedCut > runEnd && runIdx < runs.size()) {
                const auto& run = runs[runIdx++];
                if (run.count <= 0 || run.rounds <= 0 ||
                    run.rounds > (candidate.totalTasks / taskScale - runEnd) / run.count) {
                    reason = "invalid unit extent";
                    break;
                }
                runBegin = runEnd;
                runEnd += run.count * run.rounds;
            }
            if (reason != nullptr || normalizedCut > runEnd ||
                (runIdx > 0 && (normalizedCut - runBegin) % runs[runIdx - 1].rounds != 0)) {
                reason = "cut splits an owned unit";
                break;
            }
            if (core > 0) {
                const int64_t load = cut - candidate.cuts[core - 1];
                maxRounds = std::max(maxRounds, load);
                activeCores += static_cast<int64_t>(load > 0);
            }
        }
        if (reason == nullptr && (candidate.cuts.front() != 0 || candidate.cuts.back() != candidate.totalTasks ||
                                  maxRounds != candidate.maxRounds || activeCores != candidate.activeCores ||
                                  runIdx != runs.size() || runEnd != candidate.totalTasks / taskScale)) {
            reason = "incomplete replay";
        }
        // CalDeterIndex uses uint32_t round arguments. The final drain asks for R + 1;
        // allow R <= UINT32_MAX - 1 so neither the prefetch argument nor its search loop wraps.
        if (reason == nullptr && maxRounds >= std::numeric_limits<uint32_t>::max()) {
            reason = "rounds exceed decoder uint32 drain boundary";
        }
        // Positive excess comparisons avoid overflowing a cross multiplication or reference + allowance.
        // Modes 2/3 retain 5% / 5%; mode 1 retains its 15% fallback budget and adds a 10% lower-bound budget.
        const int64_t oldAllowance = columnMode ? (oldRounds / 20) * 3 + ((oldRounds % 20) * 3) / 20 : oldRounds / 20;
        const int64_t lowerAllowance = lowerBound / (columnMode ? 10 : 20);
        if (reason == nullptr && maxRounds > oldRounds && maxRounds - oldRounds > oldAllowance) {
            reason = "rounds exceed fallback budget";
        }
        if (reason == nullptr && maxRounds > lowerBound && maxRounds - lowerBound > lowerAllowance) {
            reason = "rounds exceed task lower-bound budget";
        }
        if (reason == nullptr && activeCores != std::min(cores, candidate.totalTasks)) {
            reason = "insufficient active cores";
        }
        (void)0;
        return reason == nullptr;
    }

    void CommitExclusiveCuts(const TndExclusiveCandidate& candidate)
    {
        std::fill(std::begin(fBaseParams.deterColStart), std::end(fBaseParams.deterColStart), static_cast<int64_t>(0));
        std::copy(candidate.cuts.begin(), candidate.cuts.end(), fBaseParams.deterColStart);
        fBaseParams.deterMaxRound = candidate.maxRounds;
    }

    bool TryApplyTndGqaBn2Exclusive(int64_t prefixArea)
    {
        if (!CanApplyTndGqaExclusive()) {
            return false;
        }
        constexpr int64_t kCubeSize = 128;
        constexpr int64_t kAlignSize = 16;
        constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
        std::vector<TndExclusiveRun> runs;
        runs.reserve(fBaseParams.b);
        int64_t totalRounds = 0;
        int64_t totalArea = 0;
        for (int64_t bIdx = 0; bIdx < fBaseParams.b; bIdx++) {
            const int64_t qLen = fBaseParams.actualSeqQlen[bIdx];
            const int64_t kvLen = fBaseParams.actualSeqKvlen[bIdx];
            if (qLen <= 0 || kvLen <= 0) {
                return false;
            }
            const int64_t m = ExclusiveCeilDivide(qLen, kCubeSize);
            const int64_t n = ExclusiveCeilDivide(kvLen, kCubeSize);
            const int64_t mq = ExclusiveCeilDivide(qLen, kAlignSize);
            const int64_t nkv = ExclusiveCeilDivide(kvLen, kAlignSize);
            if (m > kMax / n || mq > kMax / nkv ||
                !AppendExclusiveRun(runs, fBaseParams.n2, m * n, mq * nkv, totalRounds, totalArea)) {
                return false;
            }
        }
        if (prefixArea <= 0 || prefixArea > kMax / fBaseParams.n2 || totalRounds != prefixArea * fBaseParams.n2) {
            return false;
        }
        int64_t roundCap = MinExclusiveLimit(runs, totalRounds, totalArea, true);
        if (FAG_GATE114_VARIANT == 1 && totalRounds <= kMax / 11) {
            const int64_t candidateCap = 11 * totalRounds / (10 * static_cast<int64_t>(fBaseParams.aicNum));
            if (ExclusiveFits(runs, candidateCap, totalArea)) {
                roundCap = candidateCap;
            }
        }
        const int64_t areaCap = MinExclusiveLimit(runs, totalArea, roundCap, false);
        roundCap = MinExclusiveLimit(runs, totalRounds, areaCap, true);
        TndExclusiveCandidate candidate;
        if (!BuildExclusiveCuts(runs, roundCap, areaCap, totalRounds, fBaseParams.g, candidate)) {
            (void)0;
            return false;
        }
        if (!AdmitExclusiveCandidate(runs, fBaseParams.g, candidate, "dense_bn2")) {
            return false;
        }
        CommitExclusiveCuts(candidate);
        fBaseParams.tndGqaScheduleMode = DeterParamRegbase::SCHEDULE_TND_GQA_BN2_EXCLUSIVE;
        (void)0;
        return true;
    }

    bool TryApplyTndGqaCausalMirrorColumnExclusive()
    {
        constexpr int64_t kCubeSize = 128;
        constexpr int64_t kHeadPairGroup = FAG_GATE114_VARIANT == 2 ? 2 : 1;
        // A token-window NO_MASK input may also be classified DETER_CAUSAL,
        // but its skipped blocks invalidate the full-triangle load proof.
        if (fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) &&
            fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
            return false;
        }
        if (!CanApplyTndGqaExclusive() || fBaseParams.n2 != 1 || fBaseParams.g % NUM_TWO != 0 ||
            (fBaseParams.g / NUM_TWO) % kHeadPairGroup != 0) {
            return false;
        }
        constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
        std::vector<TndExclusiveRun> runs;
        std::vector<int64_t> prefix = {0};
        runs.reserve(fBaseParams.b * NUM_TWO);
        prefix.reserve(fBaseParams.b + 1);
        int64_t totalRounds = 0;
        int64_t totalArea = 0;
        for (int64_t bIdx = 0; bIdx < fBaseParams.b; bIdx++) {
            const int64_t qLen = fBaseParams.actualSeqQlen[bIdx];
            if (qLen <= 0 || qLen != fBaseParams.actualSeqKvlen[bIdx]) {
                return false;
            }
            const int64_t m = ExclusiveCeilDivide(qLen, kCubeSize);
            if (m >= kMax / NUM_TWO) {
                return false;
            }
            const int64_t virtualRows = m + 1;
            if (!AppendExclusiveRun(runs, m / NUM_TWO, NUM_TWO * virtualRows, NUM_TWO * virtualRows, totalRounds,
                                    totalArea) ||
                !AppendExclusiveRun(runs, m % NUM_TWO, virtualRows, virtualRows, totalRounds, totalArea)) {
                return false;
            }
            // Each batch's normalized full-rectangle area is twice its physical causal triangle.
            prefix.push_back(totalRounds / NUM_TWO);
        }
        if (totalRounds <= 0) {
            return false;
        }
        const int64_t roundCap = MinExclusiveLimit(runs, totalRounds, totalArea, true);
        TndExclusiveCandidate candidate;
        if (!BuildExclusiveCuts(runs, roundCap, totalArea, totalRounds, fBaseParams.g / NUM_TWO, candidate)) {
            (void)0;
            return false;
        }
        if (fBaseParams.deterPrefixStep <= 0) {
            (void)0;
            return false;
        }
        auto serializedPrefix = SliceVector(prefix, fBaseParams.deterPrefixStep);
        serializedPrefix.push_back(candidate.maxRounds);
        if (serializedPrefix.size() > DETER_PREFIX_ARRAY_SIZE) {
            (void)0;
            return false;
        }
        if (!AdmitExclusiveCandidate(runs, fBaseParams.g / NUM_TWO, candidate, "causal_mirror")) {
            return false;
        }
        // All validation and allocation are complete. Rejection above leaves every fallback field untouched.
        CommitExclusiveCuts(candidate);
        CopyTNDPrefix(serializedPrefix, fBaseParams.deterPrefix0, DETER_PREFIX_ARRAY_SIZE);
        std::fill(std::begin(fBaseParams.deterPrefix1), std::end(fBaseParams.deterPrefix1), static_cast<int64_t>(0));
        std::fill(std::begin(fBaseParams.deterPrefix2), std::end(fBaseParams.deterPrefix2), static_cast<int64_t>(0));
        fBaseParams.tndGqaScheduleMode = DeterParamRegbase::SCHEDULE_TND_GQA_CAUSAL_MIRROR_COLUMN_EXCLUSIVE;
        fBaseParams.tndGqaHeadPairGroup = kHeadPairGroup;
        (void)0;
        return true;
    }

    // Validate dimensions before multiplying them; this method must not mutate fallback state.
    bool CanApplyTndGqaColumnExclusive(int64_t N11, int64_t headGroup) const
    {
        // DETER_CAUSAL can also describe a negative-token window in NO_MASK mode.
        // Its folded columns may contain skipped blocks, so nominal task counts
        // do not prove useful-core occupancy or load balance for that window.
        // Explicit causal modes normalize their tokens before this scheduler.
        if (fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) &&
            fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
            return false;
        }
        constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
        if (!fBaseParams.isDeterministic || fBaseParams.layoutType != INPUT_FORMAT_TND ||
            fBaseParams.splitAxis != SplitAxisEnum::BN2GS1S2 || !fBaseParams.tndGqaColumnOwnerAllowed ||
            fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL) ||
            fBaseParams.coreDivide || fBaseParams.g <= 1 || fBaseParams.b <= 0 || fBaseParams.n2 < 2 ||
            fBaseParams.n2 % NUM_TWO != 0 || N11 != fBaseParams.n2 / NUM_TWO || fBaseParams.aicNum == 0 ||
            fBaseParams.aicNum > CORE_LIST_NUM || fBaseParams.deterMaxRound <= 0 || fBaseParams.s1Inner <= 0 ||
            fBaseParams.s2Inner <= 0 || fBaseParams.s1CvRatio <= 0 || fBaseParams.s2CvRatio <= 0 ||
            fBaseParams.s1Inner > kMax / fBaseParams.s1CvRatio || fBaseParams.s2Inner > kMax / fBaseParams.s2CvRatio ||
            static_cast<int64_t>(fBaseParams.actualSeqQlen.size()) < fBaseParams.b ||
            static_cast<int64_t>(fBaseParams.actualSeqKvlen.size()) < fBaseParams.b) {
            return false;
        }
        const int64_t cubeM = fBaseParams.s1Inner * fBaseParams.s1CvRatio;
        const int64_t cubeN = fBaseParams.s2Inner * fBaseParams.s2CvRatio;
        if (cubeM != cubeN) {
            return false;
        }
        if (headGroup != 0 && ((headGroup != 2 && headGroup != 4 && headGroup != 8) || fBaseParams.g % headGroup != 0 ||
                               cubeM != 128 || fBaseParams.inputDtype != DtypeEnum::BFLOAT16 ||
                               fBaseParams.d != fBaseParams.d1 || (fBaseParams.d != 128 && fBaseParams.d != 256))) {
            return false;
        }
        for (int64_t batch = 0; batch < fBaseParams.b; ++batch) {
            // Causal preTokens is INT32_MAX, not an unbounded left window.
            // Longer queries could therefore skip nominally assigned blocks.
            if (fBaseParams.actualSeqQlen[batch] > std::numeric_limits<int32_t>::max() ||
                (headGroup != 0 && fBaseParams.actualSeqQlen[batch] != fBaseParams.actualSeqKvlen[batch])) {
                return false;
            }
        }
        return true;
    }

    bool CanUseCausalRowWindow(const TndExclusiveCandidate& candidate, int64_t stride) const
    {
        // Called only after the complete C4 admission. Adjacent owners of one
        // KV-head pair have phases stride apart. C*stride <= M makes their
        // stride-row circular windows disjoint for every global token group.
        if ((stride != 2 && stride != 4) || fBaseParams.g % 8 != 0 || candidate.maxRounds % (8 * stride) != 0) {
            return false;
        }
        constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
        const int64_t pairs = fBaseParams.n2 / 2;
        const int64_t g = fBaseParams.g;
        if (pairs <= 0 || g <= 0 || candidate.cuts.size() != static_cast<size_t>(fBaseParams.aicNum) + 1) {
            return false;
        }
        int64_t start = 0;
        for (int64_t batch = 0; batch < fBaseParams.b; ++batch) {
            const int64_t m = ExclusiveCeilDivide(fBaseParams.actualSeqQlen[batch], int64_t(128));
            if (m <= 0 || m == kMax) {
                return false;
            }
            const int64_t height = m + 1;
            if (height < stride || m > kMax / height || m * height > kMax / g) {
                return false;
            }
            const int64_t pairTasks = m * height * g;
            if (pairTasks > (kMax - start) / pairs) {
                return false;
            }
            const int64_t end = start + pairTasks * pairs;
            if (end > candidate.totalTasks) {
                return false;
            }
            int64_t lastPair = -1;
            int64_t owners = 1;
            // Count cuts strictly inside a pair, without expanding its columns
            // or all N2 pairs. Exact pair boundaries introduce no shared writer.
            for (size_t core = 1; core + 1 < candidate.cuts.size(); ++core) {
                const int64_t cut = candidate.cuts[core];
                if (cut <= start || cut >= end || (cut - start) % pairTasks == 0) {
                    continue;
                }
                const int64_t pair = (cut - start) / pairTasks;
                owners = pair == lastPair ? owners + 1 : 2;
                lastPair = pair;
                if (owners > height / stride) {
                    return false;
                }
            }
            start = end;
        }
        return start == candidate.totalTasks;
    }

    void TryApplyTndGqaColumnExclusive(int64_t prefix0Area, int64_t N11)
    {
        if (FAG_CAUSAL38_DISABLE_MODE1 != 0) {
            return; // Diagnostic same-source reference: preserve all mode-0 state and all other modes.
        }
        if (FAG_CAUSAL38_VARIANT == 0) {
            return;
            return;
        }
        // Candidate 8 extends only the C4 domain: G=4 stays H=4 and G=6
        // still falls back. In particular, do not introduce an H=2 subcase.
        const int64_t kHeadGroup = FAG_CAUSAL38_VARIANT >= 16 ? 4 :
                                   FAG_CAUSAL38_VARIANT == 8  ? (fBaseParams.g % 8 == 0 ? 8 : 4) :
                                                                (FAG_CAUSAL38_VARIANT == 1 ? 0 : FAG_CAUSAL38_VARIANT);
        if (!CanApplyTndGqaColumnExclusive(N11, kHeadGroup) || prefix0Area <= 0) {
            (void)0;
            return;
        }
        constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
        const int64_t cubeM = fBaseParams.s1Inner * fBaseParams.s1CvRatio;
        const int64_t cubeN = fBaseParams.s2Inner * fBaseParams.s2CvRatio;
        std::vector<TndExclusiveRun> runs;
        runs.reserve(fBaseParams.b);
        int64_t area = 0;
        int64_t totalRounds = 0;
        int64_t totalArea = 0;
        for (int64_t batch = 0; batch < fBaseParams.b; ++batch) {
            const int64_t q = fBaseParams.actualSeqQlen[batch];
            const int64_t kv = fBaseParams.actualSeqKvlen[batch];
            if (q <= 0 || kv <= 0) {
                return;
            }
            const int64_t m = ExclusiveCeilDivide(q, cubeM);
            const int64_t n = std::min(m, ExclusiveCeilDivide(kv, cubeN));
            if (m > (kMax - 1) / NUM_TWO || n > kMax / N11) {
                return;
            }
            const int64_t virtualHeight = m + (m - n) + 1;
            if (virtualHeight > (kMax - area) / n) {
                return;
            }
            area += virtualHeight * n;
            // Normalize out G: accepted ownership remains identical when only the query-head count changes.
            if (!AppendExclusiveRun(runs, n * N11, virtualHeight, virtualHeight, totalRounds, totalArea)) {
                return;
            }
        }
        if (area != prefix0Area || totalRounds <= 0 || totalRounds > kMax / fBaseParams.g) {
            (void)0;
            return;
        }
        const int64_t limit = MinExclusiveLimit(runs, totalRounds, totalArea, true);
        TndExclusiveCandidate candidate;
        if (!BuildExclusiveCuts(runs, limit, totalArea, totalRounds, fBaseParams.g, candidate) ||
            !AdmitExclusiveCandidate(runs, fBaseParams.g, candidate, "Causal column", true)) {
            return;
        }
        if (kHeadGroup != 0) {
            // All column, batch and core boundaries are multiples of G, hence of H. Check serialized cuts too.
            const int64_t kAlignment = kHeadGroup;
            if (candidate.maxRounds % kAlignment != 0) {
                return;
            }
            for (const int64_t cut : candidate.cuts) {
                if (cut % kAlignment != 0) {
                    return;
                }
            }
        }
        int64_t scheduleGroup = kHeadGroup;
        if ((FAG_CAUSAL38_VARIANT == 16 || FAG_CAUSAL38_VARIANT == 32) &&
            CanUseCausalRowWindow(candidate, FAG_CAUSAL38_VARIANT / 8)) {
            scheduleGroup = FAG_CAUSAL38_VARIANT;
        }
        if (FAG_CAUSAL38_VARIANT == 64 || FAG_CAUSAL38_VARIANT == 128) {
            // T4 changes traversal; F4 only simplifies the C4 decoder. Both
            // retain the C4 domain, cuts and four-round dQ synchronization.
            scheduleGroup = FAG_CAUSAL38_VARIANT;
        }
        // Encodings 16/32 mean H=8 and row stride 2/4. Internal boundaries
        // remain H-aligned; global groups may cross them. Idle paths keep the
        // same group clock and final R is group-aligned. Otherwise preserve C4.
        // Commit only after final cuts, max load, active cores and decoder/synchronization premises agree.
        CommitExclusiveCuts(candidate);
        fBaseParams.tndGqaScheduleMode = DeterParamRegbase::SCHEDULE_TND_GQA_COLUMN_EXCLUSIVE;
        fBaseParams.tndGqaHeadPairGroup = static_cast<uint8_t>(scheduleGroup);
    }
};

// batch fields: Q,KV,M,N,P,Qband,areaPrefix,mode2RoundPrefix,
// runBegin,runEnd,lineRoundEnd,singleBegin,singleEnd,normalM,normalN,rowOffset-minus-columnOffset.
inline Plan Build(const std::vector<int64_t>& q, const std::vector<int64_t>& kv, int64_t n2, int64_t g, int64_t d,
                  int64_t dv, int64_t cores, int64_t mask, int64_t left, int64_t right, int64_t referenceUsedCube)
{
    Plan plan;
    plan.batch.resize(q.size());
    if (q.size() != kv.size() || q.empty() || q.size() > 65536 || n2 <= 0 || g <= 0 || n2 > INT32_MAX / g || d <= 0 ||
        dv <= 0 || d > INT32_MAX || dv > INT32_MAX || cores <= 0 || cores > 36) {
        plan.status = 1;
        return plan;
    }
    Planner builder;
    auto& p = builder.fBaseParams;
    p.b = q.size();
    p.n2 = n2;
    p.g = g;
    p.n1 = n2 * g;
    p.aicNum = cores;
    p.d = d;
    p.d1 = dv;
    p.actualSeqQlen = q;
    p.actualSeqKvlen = kv;
    p.deterPrefixStep = CeilDivideBy(p.b, 128);
    if (mask == 4 && left == -1 && right == -1)
        mask = 0;
    p.sparseMode = static_cast<uint32_t>(mask);
    p.s1Token = mask == 3 ? INT32_MAX : (left < 0 ? INT32_MAX : left);
    p.s2Token = mask == 3 ? 0 : (right < 0 ? INT32_MAX : right);
    int64_t area = 0, maxM = 0, maxN = 0, maxQ = 0, maxKv = 0, mode2Rounds = 0, workingBytes = 0;
    int64_t sum0 = 0, sum1 = 0, sum2 = 0, max0 = 0, max1 = 0, max2 = 0;
    bool mode2Safe = true;
    for (size_t i = 0; i < q.size(); ++i) {
        if (q[i] < 0 || kv[i] < 0 || q[i] > INT32_MAX || kv[i] > INT32_MAX) {
            plan.status = 2;
            return plan;
        }
        auto& row = plan.batch[i];
        int64_t m = CeilDivideBy(q[i], 128), n = CeilDivideBy(kv[i], 128);
        row[0] = q[i];
        row[1] = kv[i];
        row[2] = m;
        row[3] = n;
        row[6] = area;
        row[7] = mode2Rounds;
        row[8] = i;
        row[9] = i + 1;
        row[13] = m;
        row[14] = n;
        if ((mask == 4 || mask == 3) && q[i] > 0 && kv[i] > 0) {
            const int64_t pre = std::min(m, CeilDivideBy(p.s1Token + q[i] - kv[i], 128) + 1);
            const int64_t next = std::min(n, CeilDivideBy(p.s2Token - q[i] + kv[i], 128) + 1);
            row[15] = pre < 0 ? pre : (next < 0 ? -next : 0);
        }
        // Bound all products before the planner's signed arithmetic. Tensor element
        // counts are already bounded by the public input contract; reject oversized plans.
        constexpr int64_t safe = std::numeric_limits<int64_t>::max() / 1024;
        const __int128 nextArea = (static_cast<__int128>(area) + m * n) * p.n1;
        const __int128 rowBytes =
            static_cast<__int128>(m) * p.n1 * 128 * d * 10 + static_cast<__int128>(n) * n2 * 128 * d * 12;
        if (nextArea > safe || rowBytes + workingBytes > std::numeric_limits<int64_t>::max()) {
            plan.status = 3;
            return plan;
        }
        area += m * n;
        mode2Rounds += CeilDivideBy(n * p.n1, cores) * m;
        maxQ = std::max(maxQ, q[i]);
        maxKv = std::max(maxKv, kv[i]);
        maxM = std::max(maxM, m);
        maxN = std::max(maxN, n);
        workingBytes += static_cast<int64_t>(rowBytes);
        mode2Safe = mode2Safe && q[i] > 0 && kv[i] > 0 && m >= std::min(cores, n);
        p.isS1S2Same = p.isS1S2Same && q[i] == kv[i];
        n = std::min(m, n);
        if (m > 0 && n > 0) {
            sum0 += (2 * m - n + 1) * n;
            sum1 += (m - (n + 1) / 2 + 1) * (n / 2);
            sum2 += (m - n / 2) * ((n + 1) / 2);
            max0 = std::max(max0, g * (2 * m - n + 1));
            if (g == 1 || n >= 2)
                max1 = std::max(max1, g * (m - (n + 1) / 2 + 1));
            max2 = std::max(max2, g * (m - n / 2));
        }
    }
    // SetSparseParams/GetDeterSparseTilingKey use actual TND maxima before
    // right-down per-batch token correction, not public max_seqlen capacity.
    const bool isDense = mask == 0 || (mask == 4 && p.s1Token >= maxQ && p.s2Token >= maxKv);
    p.deterSparseType = isDense ? 0 : ((mask == 3 && p.isS1S2Same) ? 1 : 2);
    p.deterMaxRound = area ? std::max({CeilDivideBy(area * p.n1, cores), maxM * g, maxN}) : 0;
    plan.family = mask == 0 ? (g == 1 ? 12 : 14) : (g == 1 ? 13 : (mask == 4 ? 14 : (n2 == 1 ? 15 : 16)));
    // The reference adds its compressed mask even when mask4 normalizes Dense.
    if (mask == 4 && workingBytes <= std::numeric_limits<int64_t>::max() - 2048 * 2048) {
        workingBytes += 2048 * 2048;
    }
    plan.scalar[7] = workingBytes;
    plan.scalar[8] = referenceUsedCube;
    if (isDense && g == 1) {
        const int64_t slots = mode2Rounds * cores, tasks = area * p.n1;
        if (p.b < 129 && mode2Safe && slots > 0 && tasks * 100 >= slots * 90 &&
            mode2Rounds * 100 <= p.deterMaxRound * 103) {
            plan.strategy = 1;
            p.deterMaxRound = mode2Rounds;
        }
    } else if (isDense) {
        if (builder.TryApplyTndGqaBn2Exclusive(area))
            plan.strategy = 3;
    } else if (g == 1 && PreferTNDLineDeter(p)) {
        const auto line = BuildTNDLineDeter(p);
        plan.strategy = 2;
        p.deterMaxRound = line.rLine;
        plan.scalar[1] = CeilDivideBy(line.prefixSingle.back(), cores);
        for (int64_t i = 0; i < p.b;) {
            TNDRunShape shape;
            const bool band = p.deterSparseType == 2;
            const int64_t count = CountTNDSameShapeRun(p, i, false, band, shape);
            for (int64_t j = i; j < i + count; ++j) {
                auto& row = plan.batch[j];
                row[4] = shape.p;
                row[5] = shape.q;
                row[8] = i;
                row[9] = i + count;
                row[10] = line.prefixSolo[j + 1];
                row[11] = line.prefixSingle[j];
                row[12] = line.prefixSingle[j + 1];
                row[13] = shape.m;
                row[14] = shape.n;
            }
            i += count;
        }
    } else if (g > 1 && p.deterSparseType == 1) {
        const int64_t pairs = n2 / 2, odd = n2 % 2;
        int64_t r0 = pairs ? std::max(CeilDivideBy(sum0 * g * pairs, cores), max0) : 0;
        int64_t r1 = odd ? std::max(CeilDivideBy(sum1 * g, cores), max1) : 0;
        int64_t r2 = odd ? std::max(CeilDivideBy(sum2 * g, cores), max2) : 0;
        p.deterMaxRound = r0 + r1 + r2;
        if (n2 == 1) {
            if (builder.TryApplyTndGqaCausalMirrorColumnExclusive())
                plan.strategy = 4;
        } else if (odd == 0) {
            builder.TryApplyTndGqaColumnExclusive(sum0, pairs);
            if (p.tndGqaScheduleMode == DeterParamRegbase::SCHEDULE_TND_GQA_COLUMN_EXCLUSIVE)
                plan.strategy = 5;
        }
        int64_t prefix = 0;
        for (auto& row : plan.batch) {
            const int64_t m = row[2], n = std::min(m, row[3]);
            row[6] = prefix;
            prefix += plan.strategy == 4 ? m * (m + 1) / 2 : (2 * m - n + 1) * n;
        }
    }
    plan.scalar[0] = p.deterMaxRound;
    plan.scalar[6] = p.tndGqaHeadPairGroup;
    plan.scalar[9] = p.deterSparseType;
    plan.scalar[10] = mask;
    plan.scalar[11] = n2;
    plan.scalar[12] = g;
    std::copy(std::begin(p.deterColStart), std::end(p.deterColStart), plan.cuts.begin());
    plan.normalized = fag_pto_normalized::Build(q, kv, n2, g, d, dv, cores, mask, left, right);
    return plan;
}

inline int64_t WordCount(int64_t batches)
{
    return HEADER_WORDS + 2 * (SCALAR_COUNT + CUT_COUNT + BATCH_FIELDS * batches) + fag_pto_normalized::WORDS;
}
inline void Write64(uint32_t* p, int64_t value)
{
    const uint64_t bits = static_cast<uint64_t>(value);
    p[0] = static_cast<uint32_t>(bits);
    p[1] = static_cast<uint32_t>(bits >> 32);
}
inline bool Serialize(const Plan& p, int64_t cores, uint32_t* tail, int64_t available)
{
    const int64_t words = WordCount(p.batch.size());
    if (available < words)
        return false;
    std::fill_n(tail, words, uint32_t(0));
    tail[0] = MAGIC;
    tail[1] = VERSION;
    tail[2] = static_cast<int32_t>(words);
    tail[3] = HEADER_WORDS;
    tail[4] = static_cast<int32_t>(cores);
    tail[5] = static_cast<int32_t>(p.batch.size());
    tail[6] = p.family;
    tail[7] = p.strategy;
    tail[8] = p.status;
    tail[9] = BATCH_FIELDS;
    tail[10] = SCALAR_COUNT;
    tail[11] = CUT_COUNT;
    int64_t off = HEADER_WORDS;
    for (auto value : p.scalar) {
        Write64(tail + off, value);
        off += 2;
    }
    for (auto value : p.cuts) {
        Write64(tail + off, value);
        off += 2;
    }
    for (const auto& batch : p.batch)
        for (auto value : batch) {
            Write64(tail + off, value);
            off += 2;
        }
    for (auto value : p.normalized) {
        Write64(tail + off, value);
        off += 2;
    }
    return true;
}
} // namespace fag_pto_tnd
} // namespace aicpu
#endif
