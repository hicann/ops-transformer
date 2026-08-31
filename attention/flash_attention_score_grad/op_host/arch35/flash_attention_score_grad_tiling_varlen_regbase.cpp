/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_tiling_varlen_regbase.cpp
 * \brief
 */
#include "flash_attention_score_grad_tiling_normal_regbase.h"

using namespace Ops::Transformer::OpTiling;

using namespace optiling::fag;
namespace optiling {
namespace fag {
inline void GetTNDOuterMN(const FuzzyBaseInfoParamsRegbase &p, int64_t bIdx, int64_t &m, int64_t &n)
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

inline void GetTNDLeftUpVirt(int64_t m, int64_t n, int64_t &virtM, int64_t &virtN)
{
    if (m < n) {
        n = m;
    }
    virtM = NUM_TWO * m - n + 1;
    virtN = n;
}

inline void GetTNDRightDownVirt(int64_t m, int64_t n, int64_t &virtM, int64_t &virtN)
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

struct TNDRunShape {
    int64_t m = 1;
    int64_t n = 1;
    int64_t p = 0;
    int64_t q = 0;
    int64_t kind = 0;
};

inline bool SameTNDRunShape(const TNDRunShape &a, const TNDRunShape &b)
{
    return a.m == b.m && a.n == b.n && a.p == b.p && a.q == b.q && a.kind == b.kind;
}

inline TNDRunShape MakeTNDCausalRunShape(const FuzzyBaseInfoParamsRegbase &p, int64_t bIdx, bool leftUp)
{
    TNDRunShape s;
    GetTNDOuterMN(p, bIdx, s.m, s.n);
    if (leftUp) {
        if (s.m < s.n) {
            s.n = s.m;
        }
        s.kind = 2;
    } else {
        s.m -= GetTNDRightDownRowShift(s.m, s.n);
        s.kind = 3;
    }
    return s;
}

inline void GetTNDBandMN(FuzzyBaseInfoParamsRegbase &p, int64_t bIdx, int64_t &m, int64_t &n, int64_t &bandP,
                         int64_t &bandQ)
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

inline TNDRunShape MakeTNDBandRunShape(FuzzyBaseInfoParamsRegbase &p, int64_t bIdx)
{
    TNDRunShape s;
    GetTNDBandMN(p, bIdx, s.m, s.n, s.p, s.q);
    s.kind = (s.p >= s.m) ? 43 : 4;
    return s;
}

inline int64_t CountTNDSameShapeRun(FuzzyBaseInfoParamsRegbase &p, int64_t start, bool leftUp, bool isBand,
                                    TNDRunShape &shape)
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

inline void CopyTNDPrefix(const std::vector<int64_t> &src, int64_t *dst, size_t dstLen)
{
    std::fill(dst, dst + dstLen, static_cast<int64_t>(0));
    const size_t n = std::min(src.size(), dstLen);
    std::copy(src.begin(), src.begin() + static_cast<long>(n), dst);
}

inline void StoreTNDPrefix(const std::vector<int64_t> &raw, int64_t step, int64_t tailValue, int64_t *dst)
{
    std::vector<int64_t> sliced = SliceVector(raw, step);
    sliced.push_back(tailValue);
    CopyTNDPrefix(sliced, dst, 132);
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

inline bool TNDRunEmpty(const TNDRunShape &shape, bool isBand)
{
    return shape.m < 1 || shape.n < 1 || (isBand && shape.p + shape.q < NUM_TWO);
}

inline bool PreferTNDLineDeter(FuzzyBaseInfoParamsRegbase &p)
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
            if (!TNDRunIsLineWorthy(k, virtM, virtN, runPairs)) {
                return false;
            }
            if (!leftUp && runHasSingle && shape.m < std::min(k, shape.n)) {
                return false;
            }
        } else if (shape.kind == 43) {
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
    int64_t rLine = 0;
};

inline TNDLineDeterPack BuildTNDLineDeter(FuzzyBaseInfoParamsRegbase &p)
{
    TNDLineDeterPack pack;
    const int64_t k = static_cast<int64_t>(p.aicNum);
    const int64_t n1 = p.n2;
    const bool isBand = (p.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) ||
                        (p.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                         p.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND));
    const bool leftUp = !isBand && (p.sparseMode != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL));
    std::vector<int64_t> prefixSolo = {0};

    int64_t i = 0;
    while (i < p.b) {
        TNDRunShape shape;
        const int64_t groupSize = CountTNDSameShapeRun(p, i, leftUp, isBand, shape);
        const int64_t total = n1 * groupSize;
        const int64_t runPairs = total / NUM_TWO;
        const bool runHasSingle = (total % NUM_TWO == 1);
        int64_t runSolo = 0;
        if (!TNDRunEmpty(shape, isBand)) {
            if (!isBand) {
                int64_t virtM = 0;
                int64_t virtN = 0;
                if (leftUp) {
                    GetTNDLeftUpVirt(shape.m, shape.n, virtM, virtN);
                } else {
                    GetTNDRightDownVirt(shape.m, shape.n, virtM, virtN);
                }
                const int64_t pairRounds =
                    (runPairs > 0 && virtN > 0) ? (CeilDivideBy(virtN * runPairs, k) * virtM) : 0;
                const int64_t singleRounds = runHasSingle ? (leftUp ? CalTNDSingleCausalRounds(k, shape.m, shape.n) :
                                                                      (CeilDivideBy(shape.n, k) * shape.m)) :
                                                            0;
                runSolo = pairRounds + singleRounds;
            } else if (shape.kind == 43) {
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

inline void ApplyTNDLineDeterParam(FuzzyBaseInfoParamsRegbase &p, const TNDLineDeterPack &pack)
{
    p.splitAxis = SplitAxisEnum::BN2GS1S2;
    p.tndLineDeter = 1;
    p.deterMaxRound = pack.rLine;
    StoreTNDPrefix(pack.prefixSolo, p.deterPrefixStep, pack.rLine, p.deterPrefix0);

    std::vector<int64_t> deterPrefix = {0};
    std::vector<int64_t> deterPrefixAlign = {0};
    for (int64_t bIdx = 0; bIdx < p.b; bIdx++) {
        deterPrefix.push_back(deterPrefix.back() + p.actualSeqQlen[bIdx] * p.actualSeqKvlen[bIdx]);
        deterPrefixAlign.push_back(
            deterPrefixAlign.back() +
            p.actualSeqQlen[bIdx] * AlignTo(p.actualSeqKvlen[bIdx], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
    }
    deterPrefix = SliceVector(deterPrefix, p.deterPrefixStep);
    deterPrefixAlign = SliceVector(deterPrefixAlign, p.deterPrefixStep);
    std::copy(deterPrefix.begin(), deterPrefix.end(), p.deterPrefix);
    std::copy(deterPrefixAlign.begin(), deterPrefixAlign.end(), p.deterPrefixAlign);
}

class FlashAttentionScoreGradTilingVarlenRegbase : public FlashAttentionScoreGradTilingNormalRegbase {
public:
    explicit FlashAttentionScoreGradTilingVarlenRegbase(gert::TilingContext *curContext_)
        : FlashAttentionScoreGradTilingNormalRegbase(curContext_)
    {}
    ~FlashAttentionScoreGradTilingVarlenRegbase() override = default;

protected:
    bool IsCapable() override
    {
        auto actualSeqQLenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_Q_LEN));
        OP_LOGD(context_, "coreNum is %lu", fBaseParams.coreNum);
        if (npuArch == NpuArch::DAV_3510 && actualSeqQLenTensor != nullptr &&
            actualSeqQLenTensor->GetShapeSize() != 0) {
            OP_LOGD(context_, "FlashAttentionScoreGradTilingUnpaddedAttensionRegbase hit");
            return true;
        }
        return false;
    }

    void CalcleTNDDeterParam() override
    {
        if (fBaseParams.layoutType != INPUT_FORMAT_TND) {
            return;
        }

        // 如果b小于deterPrefixThreshold，传完整的前缀和，否则按步长切分传部分
        if (fBaseParams.b > fBaseParams.deterPrefixThreshold) {
            fBaseParams.deterPrefixStep = CeilDivideBy(fBaseParams.b, fBaseParams.deterPrefixThreshold);
        }

        std::fill(std::begin(fBaseParams.deterPrefix0), std::end(fBaseParams.deterPrefix0), static_cast<int64_t>(0));
        std::fill(std::begin(fBaseParams.deterPrefix1), std::end(fBaseParams.deterPrefix1), static_cast<int64_t>(0));
        std::fill(std::begin(fBaseParams.deterPrefix2), std::end(fBaseParams.deterPrefix2), static_cast<int64_t>(0));
        fBaseParams.tndLineDeter = 0;

        if (PreferTNDLineDeter(fBaseParams)) {
            ApplyTNDLineDeterParam(fBaseParams, BuildTNDLineDeter(fBaseParams));
            OP_LOGD("CalcleTNDDeterParam", "TND deterMaxRound is %ld.", fBaseParams.deterMaxRound);
            return;
        }

        CalcleTNDDenseDeterParam();
        CalcleTNDCausalDeterParam();
        CalcleTNDBandDeterParam();
        OP_LOGD("CalcleTNDDeterParam", "TND deterMaxRound is %ld.", fBaseParams.deterMaxRound);
    }

    void CalcTNDSwizzleParam() override
    {
        if (fBaseParams.b >= TND_SWIZZLE_PREFIX_NUM) {
            return;
        }
        for (int64_t bIdx = 0; bIdx < fBaseParams.b; bIdx++) {
            int64_t s1OuterTmp = (fBaseParams.actualSeqQlen[bIdx] + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
            int64_t s2OuterTmp = (fBaseParams.actualSeqKvlen[bIdx] + fBaseParams.s2Inner * S2CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s2Inner * S2CV_RATIO_DEFAULT);
            tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
                tndBaseInfo.tndS2BlockPrefixSum[bIdx] +
                CeilDivideBy(fBaseParams.n2 * fBaseParams.g * s2OuterTmp, static_cast<int64_t>(fBaseParams.aicNum)) *
                    s1OuterTmp;
            OP_LOGD("CalcTNDSwizzleParam", " bIdx = %ld: tndS2BlockPrefixSum = %ld", bIdx + 1,
                    tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1]);
        }
    }

    void CalcleTNDDenseDeterParam()
    {
        if (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_DENSE)) {
            return;
        }
        DeterPrefixData deterPrefixData;
        int64_t s1Max = 0;
        int64_t s2Max = 0;
        for (int64_t i = 0; i < fBaseParams.b; i++) {
            int64_t actualS1Outer =
                CeilDivideBy(fBaseParams.actualSeqQlen[i], fBaseParams.s1Inner * fBaseParams.s1CvRatio);
            int64_t actualS2Outer =
                CeilDivideBy(fBaseParams.actualSeqKvlen[i], fBaseParams.s2Inner * fBaseParams.s2CvRatio);
            deterPrefixData.deterPrefix.push_back(deterPrefixData.deterPrefix.back() +
                                                  fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i]);
            deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + actualS1Outer * actualS2Outer);
            deterPrefixData.deterPrefixAlign.push_back(
                deterPrefixData.deterPrefixAlign.back() +
                fBaseParams.actualSeqQlen[i] *
                    AlignTo(fBaseParams.actualSeqKvlen[i], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
            s1Max = actualS1Outer > s1Max ? actualS1Outer : s1Max;
            s2Max = actualS2Outer > s2Max ? actualS2Outer : s2Max;
            deterPrefixData.mNewList.push_back(actualS1Outer);
            deterPrefixData.nNewList.push_back(actualS2Outer);
        }
        int64_t totalArea = deterPrefixData.prefix1.back() * fBaseParams.n1;
        fBaseParams.deterMaxRound =
            std::max({CeilDivideBy(totalArea, static_cast<int64_t>(fBaseParams.aicNum)), s1Max * fBaseParams.g, s2Max});

        deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
        deterPrefixData.deterPrefix = SliceVector(deterPrefixData.deterPrefix, fBaseParams.deterPrefixStep);
        deterPrefixData.deterPrefixAlign = SliceVector(deterPrefixData.deterPrefixAlign, fBaseParams.deterPrefixStep);
        std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);
        std::copy(deterPrefixData.deterPrefix.begin(), deterPrefixData.deterPrefix.end(), fBaseParams.deterPrefix);
        std::copy(deterPrefixData.deterPrefixAlign.begin(), deterPrefixData.deterPrefixAlign.end(),
                  fBaseParams.deterPrefixAlign);
        deterPrefixData.prefix1.push_back(fBaseParams.deterMaxRound);
        CalcleTNDDenseBns2DeterParam(deterPrefixData);
        return;
    }

    void CalcleTNDCausalDeterParam()
    {
        if (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL)) {
            return;
        }
        fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
        int64_t m0Max{0}, m1Max{0}, m2Max{0};
        DeterPrefixData deterPrefixData;
        CalcleTNDCausalDeterPrefix(deterPrefixData, m0Max, m1Max, m2Max);

        deterPrefixData.deterPrefix = SliceVector(deterPrefixData.deterPrefix, fBaseParams.deterPrefixStep);
        deterPrefixData.deterPrefixAlign = SliceVector(deterPrefixData.deterPrefixAlign, fBaseParams.deterPrefixStep);
        std::copy(deterPrefixData.deterPrefix.begin(), deterPrefixData.deterPrefix.end(), fBaseParams.deterPrefix);
        std::copy(deterPrefixData.deterPrefixAlign.begin(), deterPrefixData.deterPrefixAlign.end(),
                  fBaseParams.deterPrefixAlign);

        if (fBaseParams.g == 1) {
            CalcleTNDCausalDeterParamNormal(deterPrefixData, m0Max, m1Max, m2Max);
        } else {
            CalcleTNDCausalDeterParamGQA(deterPrefixData, m0Max, m1Max, m2Max);
        }
    }

    void CalcleTNDBandDeterParam()
    {
        if (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
            return;
        }
        int64_t N11 = fBaseParams.n1 / fBaseParams.aicNum;
        int64_t N12 = fBaseParams.n1 % fBaseParams.aicNum;
        int64_t mnMax = 0;

        DeterPrefixData deterPrefixData;
        CalcleTNDBandDeterPrefix(deterPrefixData, N11, mnMax);

        if (N11 > 0 && fBaseParams.g == 1) {
            int64_t R0 = deterPrefixData.prefix0.back();
            deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix0, fBaseParams.deterPrefixStep);
            deterPrefixData.prefix0.push_back(R0);
        }

        // 将最大轮次append在了prefix的最后，需要的时候可以直接取用，形式更简洁
        // prefix0的是乘了N1的结果，也可以不乘，乘了后二分查找不用额外申请空间
        int64_t R1 = fBaseParams.g == 1 ? (N12 > 0 ? std::max(CeilDivideBy(deterPrefixData.prefix1.back() * N12,
                                                                           static_cast<int64_t>(fBaseParams.aicNum)),
                                                              mnMax) :
                                                     0) :
                                          std::max(CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.n1,
                                                                static_cast<int64_t>(fBaseParams.aicNum)),
                                                   mnMax);
        std::vector<int64_t> slicePrefix1 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix1.push_back(R1);
        slicePrefix1.push_back(R1);

        fBaseParams.deterMaxRound += deterPrefixData.prefix0.back();
        fBaseParams.deterMaxRound += slicePrefix1.back();
        deterPrefixData.deterPrefix = SliceVector(deterPrefixData.deterPrefix, fBaseParams.deterPrefixStep);
        deterPrefixData.deterPrefixAlign = SliceVector(deterPrefixData.deterPrefixAlign, fBaseParams.deterPrefixStep);
        std::copy(deterPrefixData.deterPrefix.begin(), deterPrefixData.deterPrefix.end(), fBaseParams.deterPrefix);
        std::copy(deterPrefixData.deterPrefixAlign.begin(), deterPrefixData.deterPrefixAlign.end(),
                  fBaseParams.deterPrefixAlign);
        std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);
        std::copy(slicePrefix1.begin(), slicePrefix1.end(), fBaseParams.deterPrefix1);

        CalcleTNDBandBns2DeterParam(deterPrefixData);
    }

    void CalcleTNDDenseBns2DeterParam(DeterPrefixData &deterPrefixData)
    {
        if (fBaseParams.splitAxis != SplitAxisEnum::BN2S2) {
            return;
        }

        // 最多允许coreNum列分给不同的核fBaseParams.deterMaxRound
        if (!SupportTNDBns2(deterPrefixData, fBaseParams.deterMaxRound)) {
            fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
            return;
        }

        // BNS2分核按顺序分核，存在前后两核收尾分同一列的情况，计算可能分开的列
        std::vector<std::pair<uint64_t, uint64_t>> syncRounds, syncRoundRanges;
        std::fill(std::begin(fBaseParams.startNeedSyncRound), std::end(fBaseParams.startNeedSyncRound),
                  static_cast<uint64_t>(0));
        std::fill(std::begin(fBaseParams.endNeedSyncRound), std::end(fBaseParams.endNeedSyncRound),
                  static_cast<uint64_t>(0));
        std::fill(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
                  static_cast<int64_t>(-1));
        CalcleTNDDenseDeterSplitDkOffset(deterPrefixData, syncRounds, syncRoundRanges);
        std::copy(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
                  std::begin(fBaseParams.deterPrefix2));

        CalcleTNDDeterSyncRounds(syncRounds, syncRoundRanges);
    }

    void CalcleTNDCausalDeterPrefix(DeterPrefixData &deterPrefixData, int64_t &m0Max, int64_t &m1Max, int64_t &m2Max)
    {
        int64_t N12 = fBaseParams.g == 1 ? fBaseParams.n2 % fBaseParams.aicNum % NUM_TWO : fBaseParams.n2 % NUM_TWO;
        for (int64_t i = 0; i < fBaseParams.b; i++) {
            int64_t actualS1Outer =
                CeilDivideBy(fBaseParams.actualSeqQlen[i], fBaseParams.s1Inner * fBaseParams.s1CvRatio);
            int64_t actualS2Outer =
                CeilDivideBy(fBaseParams.actualSeqKvlen[i], fBaseParams.s2Inner * fBaseParams.s2CvRatio);
            deterPrefixData.deterPrefix.push_back(deterPrefixData.deterPrefix.back() +
                                                  fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i]);
            deterPrefixData.deterPrefixAlign.push_back(
                deterPrefixData.deterPrefixAlign.back() +
                fBaseParams.actualSeqQlen[i] *
                    AlignTo(fBaseParams.actualSeqKvlen[i], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));

            // left_up_causal场景下，如果m<n, 需要将n裁剪为m
            if (actualS1Outer < actualS2Outer) {
                actualS2Outer = actualS1Outer;
            }

            m0Max = std::max(m0Max, fBaseParams.g * (NUM_TWO * actualS1Outer - actualS2Outer + 1));
            deterPrefixData.prefix0.push_back(deterPrefixData.prefix0.back() +
                                              (NUM_TWO * actualS1Outer - actualS2Outer + 1) * actualS2Outer);

            if (N12 > 0) {
                deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() +
                                                  (actualS1Outer - (actualS2Outer + 1) / NUM_TWO + 1) *
                                                      (actualS2Outer / NUM_TWO));
                if (fBaseParams.g == 1 || (actualS2Outer >= NUM_TWO && fBaseParams.g != 1)) {
                    m1Max = std::max(m1Max, fBaseParams.g * (actualS1Outer - (actualS2Outer + 1) / NUM_TWO + 1));
                }

                deterPrefixData.prefix2.push_back(deterPrefixData.prefix2.back() +
                                                  (actualS1Outer - actualS2Outer / NUM_TWO) *
                                                      ((actualS2Outer + 1) / NUM_TWO));
                m2Max = std::max(m2Max, fBaseParams.g * (actualS1Outer - actualS2Outer / NUM_TWO));
            }
        }
    }

    void CalcleTNDCausalDeterParamNormal(DeterPrefixData &deterPrefixData, const int64_t m0Max, const int64_t m1Max,
                                         const int64_t m2Max)
    {
        int64_t N11 = fBaseParams.n2 % fBaseParams.aicNum / NUM_TWO;
        int64_t N12 = fBaseParams.n2 % fBaseParams.aicNum % NUM_TWO;
        int64_t prefix0Max1 = deterPrefixData.prefix0.back() / NUM_TWO * (fBaseParams.n2 / fBaseParams.aicNum);
        int64_t prefix0Max2 = std::max(CeilDivideBy(deterPrefixData.prefix0.back() * N11 * fBaseParams.g,
                                                    static_cast<int64_t>(fBaseParams.aicNum)),
                                       m0Max);
        deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix0, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix0.push_back(prefix0Max1);
        fBaseParams.deterMaxRound += prefix0Max1;
        if (N11 > 0) {
            deterPrefixData.prefix0.push_back(prefix0Max2);
            fBaseParams.deterMaxRound += prefix0Max2;
        } else {
            deterPrefixData.prefix0.push_back(0);
        }
        std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);

        if (N12 > 0) {
            int64_t r1 = std::max(
                CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
                m1Max);
            int64_t r2 = std::max(
                CeilDivideBy(deterPrefixData.prefix2.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
                m2Max);
            deterPrefixData.prefix1 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
            deterPrefixData.prefix2 = SliceVector(deterPrefixData.prefix2, fBaseParams.deterPrefixStep);
            deterPrefixData.prefix1.push_back(r1);
            deterPrefixData.prefix2.push_back(r2);
            fBaseParams.deterMaxRound += deterPrefixData.prefix1.back();
            fBaseParams.deterMaxRound += deterPrefixData.prefix2.back();
            std::copy(deterPrefixData.prefix1.begin(), deterPrefixData.prefix1.end(), fBaseParams.deterPrefix1);
            std::copy(deterPrefixData.prefix2.begin(), deterPrefixData.prefix2.end(), fBaseParams.deterPrefix2);
        }
    }

    void CalcleTNDCausalDeterParamGQA(DeterPrefixData &deterPrefixData, const int64_t m0Max, const int64_t m1Max,
                                      const int64_t m2Max)
    {
        int64_t N11 = fBaseParams.n2 / NUM_TWO;
        int64_t N12 = fBaseParams.n2 % NUM_TWO;

        int64_t prefix0Max{0}, prefix1Max{0}, prefix2Max{0};
        if (fBaseParams.n2 == 1) {
            prefix0Max = 0;
            prefix1Max = std::max(
                CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
                m1Max);
            prefix2Max = std::max(
                CeilDivideBy(deterPrefixData.prefix2.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
                m2Max);
            fBaseParams.deterMaxRound = prefix1Max + prefix2Max;
        } else if (N12 == 0) {
            prefix0Max = std::max(CeilDivideBy(deterPrefixData.prefix0.back() * fBaseParams.g * N11,
                                               static_cast<int64_t>(fBaseParams.aicNum)),
                                  m0Max);
            prefix1Max = 0;
            prefix2Max = 0;
            fBaseParams.deterMaxRound = prefix0Max;
        } else {
            prefix0Max = std::max(CeilDivideBy(deterPrefixData.prefix0.back() * fBaseParams.g * N11,
                                               static_cast<int64_t>(fBaseParams.aicNum)),
                                  m0Max);
            prefix1Max = std::max(
                CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
                m1Max);
            prefix2Max = std::max(
                CeilDivideBy(deterPrefixData.prefix2.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
                m2Max);
            int64_t totalRound = prefix0Max + prefix1Max + prefix2Max;

            int64_t k2 = CeilDivideBy(static_cast<int64_t>(fBaseParams.aicNum), fBaseParams.n2);
            int64_t k1 = static_cast<int64_t>(fBaseParams.aicNum) - k2;

            int64_t prefix0MaxNew =
                std::max(CeilDivideBy(deterPrefixData.prefix0.back() * fBaseParams.g * N11, k1), m0Max);
            int64_t prefix1MaxNew = std::max(CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.g, k2), m1Max);
            int64_t prefix2MaxNew = std::max(CeilDivideBy(deterPrefixData.prefix2.back() * fBaseParams.g, k2), m2Max);
            int64_t totalRoundNew = std::max(prefix0MaxNew, prefix1MaxNew + prefix2MaxNew);
            if (totalRoundNew < totalRound) {
                fBaseParams.coreDivide = true;
                prefix0Max = prefix0MaxNew;
                prefix1Max = prefix1MaxNew;
                prefix2Max = prefix2MaxNew;
                totalRound = totalRoundNew;
            }
            fBaseParams.deterMaxRound = totalRound;
        }

        deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix0, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix1 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix2 = SliceVector(deterPrefixData.prefix2, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix0.push_back(prefix0Max);
        deterPrefixData.prefix1.push_back(prefix1Max);
        deterPrefixData.prefix2.push_back(prefix2Max);
        std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);
        std::copy(deterPrefixData.prefix1.begin(), deterPrefixData.prefix1.end(), fBaseParams.deterPrefix1);
        std::copy(deterPrefixData.prefix2.begin(), deterPrefixData.prefix2.end(), fBaseParams.deterPrefix2);
    }

    void CalcleTNDBandDeterPrefix(DeterPrefixData &deterPrefixData, int64_t N11, int64_t &mnMax)
    {
        for (int64_t i = 0; i < fBaseParams.b; i++) {
            int64_t m, n, p, q, mNew, nNew;
            m = CeilDivideBy(fBaseParams.actualSeqQlen[i], fBaseParams.s1Inner * fBaseParams.s1CvRatio);
            n = CeilDivideBy(fBaseParams.actualSeqKvlen[i], fBaseParams.s2Inner * fBaseParams.s2CvRatio);
            deterPrefixData.deterPrefix.push_back(deterPrefixData.deterPrefix.back() +
                                                  fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i]);
            deterPrefixData.deterPrefixAlign.push_back(
                deterPrefixData.deterPrefixAlign.back() +
                fBaseParams.actualSeqQlen[i] *
                    AlignTo(fBaseParams.actualSeqKvlen[i], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));

            int64_t actualCalcS1Token, actualCalcS2Token;
            CalcleActualToken(fBaseParams, i, actualCalcS1Token, actualCalcS2Token);
            p = CeilDivideBy(actualCalcS1Token, fBaseParams.s1Inner * fBaseParams.s1CvRatio) + 1;
            q = CeilDivideBy(actualCalcS2Token, fBaseParams.s2Inner * fBaseParams.s2CvRatio) + 1;
            p = p > m ? m : p;
            q = q > n ? n : q;

            // 负数场景变换
            if (p < 0) {
                n = n + p;
                q = p + q;
                p = 1;
            } else if (q < 0) {
                m = m + q;
                p = p + q;
                q = 1;
            }
            if (p + q <= m) {
                int64_t L1{q - 1}, L2{std::min(n - q + 1, m + NUM_TWO - p - q)},
                    L3{std::max(static_cast<int64_t>(0), std::min(p + n - m - 1, p + q - NUM_TWO))};
                mNew = L3 == 0 ? p + q + L2 - NUM_TWO : m;
                nNew = L1 + L2 + L3;
                mnMax = n <= m || fBaseParams.g == 1 ? std::max({mnMax, (p + q - 1) * fBaseParams.g}) :
                                                       std::max({mnMax, mNew * fBaseParams.g, p + q - 1});
                deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + std::min(mNew, nNew) * (p + q - 1));
            } else {
                mNew = m;
                nNew = std::min(m - 1 + q, n);
                if (p + q <= n) {
                    mnMax = std::max({mnMax, mNew * fBaseParams.g, p + q - 1});
                    deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + mNew * (p + q - 1));
                } else {
                    mnMax = std::max({mnMax, mNew * fBaseParams.g, nNew});
                    deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + mNew * nNew);
                }
            }

            // 注意，这里更新后的mNewList和nNewList是可以替代list_m和list_n的，它们的引入是避免空白行/列
            deterPrefixData.mNewList.push_back(mNew);
            deterPrefixData.nNewList.push_back(nNew);
            deterPrefixData.pNewList.push_back(p);
            deterPrefixData.qNewList.push_back(q);
            if (N11 > 0 && fBaseParams.g == 1) {
                int64_t R0 =
                    deterPrefixData.prefix0.back() +
                    (mNew * nNew - (mNew - p) * (mNew - p + 1) / NUM_TWO - (nNew - q) * (nNew - q + 1) / NUM_TWO) * N11;
                deterPrefixData.prefix0.push_back(R0);
            }
        }
    }

    void CalcleTNDBandBns2DeterParam(DeterPrefixData &deterPrefixData)
    {
        if (fBaseParams.splitAxis != SplitAxisEnum::BN2S2) {
            return;
        }

        // 最多允许coreNum列分给不同的核
        if (!SupportTNDBns2(deterPrefixData, deterPrefixData.prefix1.back())) {
            fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
            OP_LOGD("CalcleTNDBandBns2DeterParam", "Not support BNS2, change to BN2GS1S2.");
            return;
        }

        // BNS2分核按顺序分核，存在前后两核收尾分同一列的情况，计算可能分开的列
        std::vector<std::pair<uint64_t, uint64_t>> syncRounds, syncRoundRanges;
        std::fill(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
                  static_cast<int64_t>(-1));
        std::fill(std::begin(fBaseParams.startNeedSyncRound), std::end(fBaseParams.startNeedSyncRound),
                  static_cast<uint64_t>(0));
        std::fill(std::begin(fBaseParams.endNeedSyncRound), std::end(fBaseParams.endNeedSyncRound),
                  static_cast<uint64_t>(0));

        CalcleTNDBandDeterSplitDkOffset(deterPrefixData, syncRounds, syncRoundRanges);
        std::copy(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
                  std::begin(fBaseParams.deterPrefix2));

        CalcleTNDDeterSyncRounds(syncRounds, syncRoundRanges);
    }

    void CalcleTNDDenseDeterSplitDkOffset(DeterPrefixData &deterPrefixData,
                                          std::vector<std::pair<uint64_t, uint64_t>> &syncRounds,
                                          std::vector<std::pair<uint64_t, uint64_t>> &syncRoundRanges)
    {
        int64_t precoreLastBatchStartRound = 0;
        for (uint32_t coreId = 0; coreId < CORE_LIST_NUM; coreId++) {
            if (fBaseParams.deterMaxRound == 0 || coreId > fBaseParams.aicNum - 1) {
                continue;
            }
            auto actualSeqKvlenTensor =
                context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
            const int64_t *kvValue = actualSeqKvlenTensor->GetData<int64_t>();

            TndBandDeterRoundInfo tndBandDeterRoundInfo;
            for (int64_t round = fBaseParams.deterMaxRound; round > 0; round--) {
                auto oriCoordinateInfo = CalTNDDenseIndex<static_cast<uint32_t>(DeterSparseType::DETER_DENSE)>(
                    deterPrefixData, coreId + 1, round, fBaseParams.n1);
                int64_t w, x, y;
                std::tie(w, x, y) = oriCoordinateInfo;

                int64_t batchId = CeilDivideBy(w, fBaseParams.n1);
                if (w == -1 || batchId > fBaseParams.b) {
                    continue;
                }
                int64_t m = deterPrefixData.mNewList[batchId - 1];

                if (fBaseParams.separateDkOffset[coreId] == -1 && x < m && round == fBaseParams.deterMaxRound) {
                    fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
                }
                SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
            }
            if (coreId != 0) {
                uint64_t startSyncRound = precoreLastBatchStartRound;
                uint64_t endSyncRound = tndBandDeterRoundInfo.coreFirstBatchLastRound;
                if (startSyncRound > endSyncRound) {
                    syncRounds.push_back(std::make_pair(startSyncRound, endSyncRound));
                } else {
                    syncRoundRanges.push_back(std::make_pair(startSyncRound, endSyncRound));
                }
            }
            if (coreId == 0) {
                precoreLastBatchStartRound = 1; // 如果在0-1核上涉及切BN轴，那么0核的起点就是第一个roundid = 1
            } else {
                precoreLastBatchStartRound = tndBandDeterRoundInfo.coreLastBatchStartRound;
            }
        }
    }

    bool SupportTNDBns2(DeterPrefixData &deterPrefixData, int64_t round)
    {
        for (int64_t b = 0; b < fBaseParams.b; b++) {
            int64_t m = deterPrefixData.mNewList[b];
            int64_t n = deterPrefixData.nNewList[b];
            if ((round / Gcd(m, round)) >= n) {
                continue;
            }
            return false;
        }
        return true;
    }

    void CalcleTNDBandDeterSplitDkOffset(DeterPrefixData &deterPrefixData,
                                         std::vector<std::pair<uint64_t, uint64_t>> &syncRounds,
                                         std::vector<std::pair<uint64_t, uint64_t>> &syncRoundRanges)
    {
        int64_t precoreLastBatchStartRound = 0;
        int64_t N11 = fBaseParams.n1 / fBaseParams.aicNum;
        int64_t N12 = fBaseParams.n1 % fBaseParams.aicNum;
        for (uint32_t coreId = 0; coreId < CORE_LIST_NUM; coreId++) {
            if (deterPrefixData.prefix1.back() == 0 || coreId >= fBaseParams.aicNum - 1) {
                continue;
            }
            TndBandDeterRoundInfo tndBandDeterRoundInfo;
            for (uint64_t round = deterPrefixData.prefix1.back(); round > 0; round--) {
                auto oriCoordinateInfo = CalTNDDenseIndex<static_cast<uint32_t>(DeterSparseType::DETER_BAND)>(
                    deterPrefixData, coreId + 1, round, fBaseParams.n1 % static_cast<int64_t>(fBaseParams.aicNum));
                int64_t w, x, y;
                std::tie(w, x, y) = oriCoordinateInfo;
                int64_t batchId = CeilDivideBy(w, N12);
                if (w == -1 || batchId > fBaseParams.b) {
                    continue;
                }

                int64_t m = deterPrefixData.mNewList[batchId - 1];
                int64_t n = deterPrefixData.nNewList[batchId - 1];
                int64_t p = deterPrefixData.pNewList[batchId - 1];
                int64_t q = deterPrefixData.qNewList[batchId - 1];

                int64_t wTail = w % N12;
                wTail = wTail != 0 ? wTail : N12;
                w = (CeilDivideBy(w, N12) - 1) * fBaseParams.n1 + N11 * fBaseParams.aicNum + wTail;

                std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
                    coordinateInfo = std::make_tuple(m, n, p, q, x, y, w, coreId, round);
                UpdateSeparateDkOffset(coordinateInfo, tndBandDeterRoundInfo);
            }

            if (coreId != 0) {
                uint64_t startSyncRound = precoreLastBatchStartRound + deterPrefixData.prefix0.back();
                uint64_t endSyncRound = tndBandDeterRoundInfo.coreFirstBatchLastRound + deterPrefixData.prefix0.back();
                if (startSyncRound > endSyncRound) {
                    syncRounds.push_back(std::make_pair(startSyncRound, endSyncRound));
                } else {
                    syncRoundRanges.push_back(std::make_pair(startSyncRound, endSyncRound));
                }
            }
            precoreLastBatchStartRound = tndBandDeterRoundInfo.coreLastBatchStartRound;
        }
    }

    template <const uint32_t deterSparseType>
    std::tuple<int64_t, int64_t, int64_t> CalTNDDenseIndex(DeterPrefixData &deterPrefixData, int64_t coreId,
                                                           int64_t roundId, int64_t N1)
    {
        int64_t unPadRoundMax{deterPrefixData.prefix1[fBaseParams.b + 1]}, ID{(coreId - 1) * unPadRoundMax + roundId},
            w{0};
        while (w < fBaseParams.b && ID > deterPrefixData.prefix1[w + 1] * N1) {
            w += 1;
        }
        int64_t delta = ID - deterPrefixData.prefix1[w] * N1;

        if (w >= fBaseParams.b) {
            return std::make_tuple(-1, -1, -1);
        }

        int64_t m{deterPrefixData.mNewList[w]}, n{deterPrefixData.nNewList[w]}, p, q;
        if constexpr (deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
            p = deterPrefixData.pNewList[w];
            q = deterPrefixData.qNewList[w];
            if (p + q <= m) {
                if (n >= m) {
                    n = p + q - 1;
                } else {
                    m = p + q - 1;
                }
            } else {
                if (p + q <= n) {
                    n = p + q - 1;
                }
            }
        }

        int64_t currentBaseNum = m * n;
        int64_t batchId = w + 1;
        int64_t deltaN = (delta - 1) / currentBaseNum + 1;
        delta = delta % currentBaseNum;
        delta = delta != 0 ? delta : currentBaseNum;

        int64_t g = Gcd(m, unPadRoundMax);
        if (g == 0) {
            return std::make_tuple(-1, -1, -1);
        }
        int64_t t1{unPadRoundMax / g}, t2{m / g}, x{((delta - 1) % m) + 1}, y{(delta - 1) / m + 1};
        if (t1 < n) {
            int64_t n1 = n % t1;
            n1 = n1 == 0 ? t1 : n1;
            if (y <= n - n1) {
                int64_t deltaAdj = CeilDivideBy(y, t1);
                delta += deltaAdj;
                if (delta > deltaAdj * t2 * unPadRoundMax) {
                    delta -= t2 * unPadRoundMax;
                }
                x = ((delta - 1) % m) + 1;
                y = (delta - 1) / m + 1;
            }
        }
        return std::make_tuple((batchId - 1) * N1 + deltaN, x, y);
    }

    void UpdateSeparateDkOffsetLargeM(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> &coordinateInfo,
        TndBandDeterRoundInfo &tndBandDeterRoundInfo)
    {
        auto actualSeqKvlenTensor =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
        const int64_t *kvValue = actualSeqKvlenTensor->GetData<int64_t>();
        int64_t m, n, p, q, x, y, w, coreId, round;
        std::tie(m, n, p, q, x, y, w, coreId, round) = coordinateInfo;
        if (n >= m) {
            if (y - q + 1 <= x && x <= p + y - 1) {
                if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                    fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
                }
                SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
            } else {
                y = AbsCeil((x - (p + y - 1)), (p + q - 1)) * (p + q - 1) + y;
                bool isValid = 1 <= y && y <= n;
                if (fBaseParams.separateDkOffset[coreId] == -1 && isValid && IsSeparateS2(coordinateInfo)) {
                    fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
                }
                if (isValid) {
                    SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
                }
            }
        } else {
            // 优先处理拼出来后有空白格的情况
            if (q + 1 <= y && y <= std::min(n, p + NUM_TWO * q - NUM_TWO)) {
                if (x != p + q - 1 && fBaseParams.separateDkOffset[coreId] == -1) {
                    fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
                }
            } else {
                if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                    fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
                }
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        }
    }

    void UpdateSeparateDkOffsetSmallM(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> &coordinateInfo,
        TndBandDeterRoundInfo &tndBandDeterRoundInfo)
    {
        auto actualSeqKvlenTensor =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
        const int64_t *kvValue = actualSeqKvlenTensor->GetData<int64_t>();
        int64_t m, n, p, q, x, y, w, coreId, round;
        std::tie(m, n, p, q, x, y, w, coreId, round) = coordinateInfo;
        if (p + q <= n) {
            if (x - p + 1 <= y && y <= x + q - 1) {
                if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                    fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
                }
                SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
            } else if (y < x - p + 1 && 1 <= y + p + q - 1 && y + p + q - 1 <= n) {
                y = y + p + q - 1;
                if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                    fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
                }
                SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
            }
        } else {
            if (x - p + 1 <= y && y <= x + q - 1) {
                if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                    fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
                }
                SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
            }
        }
    }

    void UpdateSeparateDkOffset(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> &coordinateInfo,
        TndBandDeterRoundInfo &tndBandDeterRoundInfo)
    {
        int64_t m, p, q;
        std::tie(m, std::ignore, p, q, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
            coordinateInfo;
        if (p + q <= m) {
            UpdateSeparateDkOffsetLargeM(coordinateInfo, tndBandDeterRoundInfo);
        } else {
            UpdateSeparateDkOffsetSmallM(coordinateInfo, tndBandDeterRoundInfo);
        }
    }

    bool IsSeparateS2(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> &coordinateInfo)
    {
        int64_t m, n, p, q, x, y;
        std::tie(m, n, p, q, x, y, std::ignore, std::ignore, std::ignore) = coordinateInfo;
        // 补充的 is_max 运算
        bool isSeparate = true;

        if (p + q <= m && m > n) {
            // 这里要注意None的情况
            if (y <= q) {
                if (x - y >= p - 1) {
                    isSeparate = false;
                }
                // 有空白块的情况交给上面处理了，因为有可能R_max坐标是None
            } else if (y > std::min(n, p + NUM_TWO * q - NUM_TWO)) {
                x = AbsCeil((y - (q + x - 1)), (p + q - 1)) * (p + q - 1) + x;
                if (x == m || x < y - q + 1) {
                    isSeparate = false;
                }
            }
        } else if (p + q <= m && n >= m) {
            if (y <= std::min(n, m + 1 - p)) {
                if (x - y == p - 1) {
                    isSeparate = false;
                }
            } else {
                if (x == m) {
                    isSeparate = false;
                }
            }
        } else {
            if (y <= m - p) {
                if (x - y == p - 1) {
                    isSeparate = false;
                }
            } else {
                if (x == m) {
                    isSeparate = false;
                }
            }
        }
        return isSeparate;
    }

    int64_t GetKeyOffset(const int64_t *kvValue, int64_t w, int64_t y)
    {
        int64_t batchId = CeilDivideBy(w, fBaseParams.n2) - 1;
        int64_t n2Idx = w - batchId * fBaseParams.n2 - 1;
        int64_t s2Idx = y - 1;

        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t s2Offset = 0;

        int64_t seqKvLenPrefix = batchId == 0 ? 0 : kvValue[batchId - 1];
        bOffset = seqKvLenPrefix * fBaseParams.n2 * fBaseParams.d;
        n2Offset = n2Idx * fBaseParams.d;
        s2Offset = s2Idx * fBaseParams.s2Inner * fBaseParams.n2 * fBaseParams.d;
        return bOffset + n2Offset + s2Offset;
    }

    void CalcleTNDDeterSyncRounds(std::vector<std::pair<uint64_t, uint64_t>> &syncRounds,
                                  std::vector<std::pair<uint64_t, uint64_t>> &syncRoundRanges)
    {
        if (syncRounds.size() + syncRoundRanges.size() > CORE_LIST_NUM) {
            fBaseParams.startNeedSyncRound[0] = 1;
            fBaseParams.endNeedSyncRound[0] = std::numeric_limits<uint64_t>::max();
            OP_LOGD("CalcleTNDBandDeterParam", "All rounds need sync!");
            return;
        }
        std::vector<uint64_t> needSyncRoundsTmp = CalculateSyncRound(syncRounds);
        std::vector<uint64_t> needSyncRounds;
        for (uint64_t needSyncRound : needSyncRoundsTmp) {
            bool isValid = true;
            for (auto needSyncRoundRange : syncRoundRanges) {
                if (needSyncRound >= needSyncRoundRange.first && needSyncRound <= needSyncRoundRange.second) {
                    isValid = false;
                    break;
                }
            }
            if (isValid) {
                needSyncRounds.push_back(needSyncRound);
            }
        }

        for (uint32_t i = 0; i < needSyncRounds.size(); i++) {
            fBaseParams.startNeedSyncRound[i] = needSyncRounds[i];
            fBaseParams.endNeedSyncRound[i] = needSyncRounds[i];
        }
        for (uint32_t i = 0; i < syncRoundRanges.size(); i++) {
            auto syncRoundPair = syncRoundRanges[i];
            fBaseParams.startNeedSyncRound[i + needSyncRounds.size()] = syncRoundPair.first;
            fBaseParams.endNeedSyncRound[i + needSyncRounds.size()] = syncRoundPair.second;
        }

        uint64_t allNeedSyncLoopNums = 0;
        for (uint64_t loopIdx = 0; static_cast<int64_t>(loopIdx) < fBaseParams.deterMaxRound; loopIdx++) {
            for (uint32_t i = 0; i < CORE_LIST_NUM; i++) {
                if (fBaseParams.endNeedSyncRound[i] == 0) {
                    break;
                }
                if (loopIdx >= fBaseParams.startNeedSyncRound[i] && loopIdx <= fBaseParams.endNeedSyncRound[i]) {
                    allNeedSyncLoopNums++;
                    break;
                }
            }
        }
        if (fBaseParams.deterMaxRound < static_cast<int64_t>(allNeedSyncLoopNums) * NUM_TWO) {
            fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
        }
    }

    std::vector<uint64_t> CalculateSyncRound(std::vector<std::pair<uint64_t, uint64_t>> syncRounds)
    {
        if (syncRounds.size() == 0) {
            return {};
        }
        if (syncRounds.size() == 1) {
            return {syncRounds[0].first};
        }
        uint64_t minStartSyncRound = syncRounds[0].first;
        for (auto syncRound : syncRounds) {
            minStartSyncRound = syncRound.first < minStartSyncRound ? syncRound.first : minStartSyncRound;
        }

        std::vector<std::pair<uint64_t, uint64_t>> smallSyncRounds;
        for (auto syncRound : syncRounds) {
            if (syncRound.second > minStartSyncRound) {
                smallSyncRounds.push_back(syncRound);
            }
        }
        std::vector<uint64_t> needSyncRounds = CalculateSyncRound(smallSyncRounds);
        needSyncRounds.push_back(minStartSyncRound);
        return needSyncRounds;
    }

    void SetCoreRoundInfo(TndBandDeterRoundInfo &tndBandDeterRoundInfo, uint64_t round, int64_t batchId)
    {
        if (batchId != tndBandDeterRoundInfo.lastBatchId && tndBandDeterRoundInfo.lastBatchId != 0 &&
            tndBandDeterRoundInfo.coreLastBatchStartRound == 0) {
            tndBandDeterRoundInfo.coreLastBatchStartRound = tndBandDeterRoundInfo.lastValidRound;
        }
        if (batchId != tndBandDeterRoundInfo.lastBatchId) {
            tndBandDeterRoundInfo.coreFirstBatchLastRound = round;
        }
        tndBandDeterRoundInfo.lastValidRound = round;
        tndBandDeterRoundInfo.lastBatchId = batchId;
    }

    ge::graphStatus GetBlockInfoOfTNDForBn2() override
    {
        // 二维数组，第一维是batch，第二维的id0存储不乘N的基本块数，id1存每个batch乘N的基本块总数
        std::vector<std::vector<int64_t>> totalBlockInfo(fBaseParams.b, std::vector<int64_t>(TOTAL_BLOCK_DIMENSION));
        // 二维数组，第一维是batch +
        // 2，第一维的倒数两维存下界和总基本块数(包含n2g)，第二维的最后一维存每个batch的基本块数
        std::vector<std::vector<float>> acturalBlockInfo(fBaseParams.b + NUM_THREE,
                                                         std::vector<float>(fBaseParams.s2Outer + 1));
        FillBlockInfoLoadBalanceForBn2(totalBlockInfo, acturalBlockInfo);

        float maxBlockNumPerCore = BinarySearchMaxBlockNumPerCore(acturalBlockInfo);

        int64_t blockStarts[CORE_LIST_NUM];
        int64_t blockEnds[CORE_LIST_NUM];

        if (!CaclePerCoreBlockInfoBn2(totalBlockInfo, acturalBlockInfo, maxBlockNumPerCore, blockStarts, blockEnds)) {
            return ge::GRAPH_FAILED;
        }

        for (int64_t c = 0; c < fBaseParams.blockOuter; c++) {
            OP_LOGD("GetBlockInfoOfTNDForBn2", "blockNum[%ld], blockStarts = %ld , blockEnds = %ld", c, blockStarts[c],
                    blockEnds[c]);
        }

        for (uint32_t c = fBaseParams.blockOuter; c < CORE_LIST_NUM; c++) {
            blockStarts[c] = 0;
            blockEnds[c] = 0;
        }
        std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
        std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

        return ge::GRAPH_SUCCESS;
    }

    void FillBlockInfoLoadBalanceForBn2(std::vector<std::vector<int64_t>> &totalBlockInfo,
                                        std::vector<std::vector<float>> &acturalBlockInfo)
    {
        acturalBlockInfo[fBaseParams.b][0] = 0;     // 存全部累积基本块: bn2g * acutalblocks1s2
        acturalBlockInfo[fBaseParams.b + 1][0] = 0; // 存最大的acutalblocks1s2，用于下界
        OP_LOGD("FillBlockInfoLoadBalanceForBn2", "SparseMode %u, find band index %u", fBaseParams.sparseMode,
                fBaseParams.bandIdx);
        float batchTotalValidBlk;
        std::vector<bool> invalidS1Array;
        for (int64_t i = 0; i < fBaseParams.b; i++) {
            int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
            int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];
            batchTotalValidBlk = 0;

            auto actualS1Outer = (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
            auto actualS2Outer = (actualS2Len + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
            totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;
            invalidS1Array.assign(actualS1Outer, false);
            // 针对S2为0的场景，pre中增加initGm为0的操作
            if ((actualS2Outer == 0) != (actualS1Outer == 0)) {
                fBaseParams.isInvalidCol = (actualS1Outer == 0);
                fBaseParams.isInvalidRow = (actualS2Outer == 0);
            }

            // 对unpad场景的token值做二次校正
            // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
            int64_t actualCalcS1Token, actualCalcS2Token;
            CalcleActualToken(fBaseParams, i, actualCalcS1Token, actualCalcS2Token);

            OP_LOGD(
                "FillBlockInfoLoadBalanceForBn2",
                " b idx = %ld: actualS1Len = %ld, actualS2Len = %ld, actualCalcS1Token = %ld, actualCalcS2Token = %ld",
                i, actualS1Len, actualS2Len, actualCalcS1Token, actualCalcS2Token);

            // unpad 场景下s2Outer是按照最大的s2计算得到的
            for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
                if (fBaseParams.cvS2Inner * j >= actualS2Len) {
                    acturalBlockInfo[i][j] = 0;
                } else {
                    int64_t leftIntersectionPoint =
                        std::max(int64_t(fBaseParams.cvS2Inner * j) - actualCalcS2Token, 0L);
                    int64_t cvBlockTail = fBaseParams.cvS2Inner * (j + 1) > actualS2Len ?
                                              actualS2Len - fBaseParams.cvS2Inner * j :
                                              fBaseParams.cvS2Inner;

                    float acturalS1Begin =
                        leftIntersectionPoint > int64_t(actualS1Len) ? actualS1Len : leftIntersectionPoint;
                    float acturalS1End = static_cast<float>(
                        std::min(int64_t(actualS1Len),
                                 std::max(fBaseParams.cvS2Inner * j + cvBlockTail + actualCalcS1Token, 0L)));
                    float acturalS1Num = acturalS1Begin > acturalS1End ? 0 : acturalS1End - acturalS1Begin;
                    // float acturalS2Num = static_cast<float>(cvBlockTail);
                    acturalBlockInfo[i][j] = acturalS1Num / static_cast<float>(fBaseParams.s1CvInner);
                    batchTotalValidBlk += acturalBlockInfo[i][j];
                    acturalBlockInfo[fBaseParams.b][0] += acturalBlockInfo[i][j] * fBaseParams.n2 * fBaseParams.g;

                    if (acturalS1Begin >= acturalS1End) {
                        fBaseParams.isInvalidCol = true;
                    }
                    // check invalid row or col block for BN2
                    for (size_t k = 0; k < invalidS1Array.size(); k++) {
                        if (k >= acturalS1Begin && k < acturalS1End) {
                            invalidS1Array[k] = true;
                        }
                    }
                }
            }

            // BN2场景下检查是否无效基本块行，用于清零GM
            for (size_t j = 0; j < invalidS1Array.size(); j++) {
                if (!invalidS1Array[j]) {
                    fBaseParams.isInvalidRow = true;
                    break;
                }
            }

            if (i == 0) {
                totalBlockInfo[0][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[0][0];
            } else {
                totalBlockInfo[i][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
            }
            acturalBlockInfo[i][fBaseParams.s2Outer] = batchTotalValidBlk;
            OP_LOGD("FillBlockInfoLoadBalanceForBn2", " batchid = %ld: acturalBlock = %f", i,
                    acturalBlockInfo[i][fBaseParams.s2Outer]);
            acturalBlockInfo[fBaseParams.b + 1][0] =
                acturalBlockInfo[fBaseParams.b + 1][0] < batchTotalValidBlk ?
                    batchTotalValidBlk :
                    acturalBlockInfo[fBaseParams.b + 1][0]; // 逐轮迭代，得到贪心的下界
        }
    }

    bool CaclePerCoreBlockInfoBn2(const std::vector<std::vector<int64_t>> &totalBlockInfo,
                                  const std::vector<std::vector<float>> &acturalBlockInfo,
                                  const float maxBlockNumPerCore, int64_t (&blockStarts)[CORE_LIST_NUM],
                                  int64_t (&blockEnds)[CORE_LIST_NUM])
    {
        float currentSum = 0;
        uint64_t coreIdx = 0;
        int64_t n2g = fBaseParams.n2 * fBaseParams.g;
        int64_t bn2g = fBaseParams.b * n2g;
        for (int64_t i = 0; i < bn2g; i++) {
            int64_t b = i / n2g;
            int64_t n = i % n2g;
            float num = acturalBlockInfo[b][fBaseParams.s2Outer];
            if (coreIdx >= fBaseParams.aicNum) {
                OP_LOGD("CaclePerCoreBlockInfoBn2", " Not support BN2_MULTIBLK.");
                return false;
            } else if (currentSum + num > maxBlockNumPerCore) {
                OP_LOGD("CaclePerCoreBlockInfoBn2", " blockIdx = %ld: acturalBlock = %f", coreIdx, currentSum);
                int64_t preBatchBlockNum = b == 0 ? 0 : totalBlockInfo[b - 1][1];
                int64_t preNGBlockNum = n * totalBlockInfo[b][0];

                blockEnds[coreIdx] = preBatchBlockNum + preNGBlockNum;
                blockStarts[coreIdx + 1] = blockEnds[coreIdx];
                coreIdx += 1;
                currentSum = num;
            } else {
                currentSum += num;
            }
        }
        OP_LOGD("CaclePerCoreBlockInfoBn2", " blockIdx = %ld: acturalBlock = %f", coreIdx, currentSum);

        blockStarts[0] = 0;
        blockEnds[coreIdx] = totalBlockInfo[fBaseParams.b - 1][1];

        fBaseParams.blockOuter = coreIdx + 1;
        return true;
    }

    ge::graphStatus GetSparseUnpadBlockInfo() override
    {
        std::vector<std::vector<std::vector<int64_t>>> calculatedBlockInfo(
            fBaseParams.b,
            std::vector<std::vector<int64_t>>(fBaseParams.s2Outer, std::vector<int64_t>(CALCULATED_BLOCK_DIMENSION)));
        std::vector<std::vector<int64_t>> totalBlockInfo(fBaseParams.b, std::vector<int64_t>(TOTAL_BLOCK_DIMENSION));
        FillBlockInfo(calculatedBlockInfo, totalBlockInfo);

        // block split
        int64_t fusedOuter = calculatedBlockInfo[fBaseParams.b - 1][0][SUM_ALL];
        int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
        int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

        OP_LOGD("GetSparseUnpadBlockInfo", " fusedOuter = %ld: blockFactor = %ld, blockOuter = %ld", fusedOuter,
                blockFactor, blockOuter);
        fBaseParams.blockOuter = blockOuter;
        fBaseParams.blockFactor = blockFactor;
        fBaseParams.maxValidBBLen = blockFactor;
        int64_t bIdx = 0;
        int64_t bTail = 0;
        int64_t n2Idx = 0;
        int64_t n2Tail = 0;
        int64_t gIdx = 0;
        int64_t gTail = 0;
        int64_t s1oIdx = 0;
        int64_t s2oIdx = 0;
        int64_t blockStarts[CORE_LIST_NUM];
        int64_t blockEnds[CORE_LIST_NUM];
        blockStarts[0] = 0;
        blockEnds[blockOuter - 1] = totalBlockInfo[fBaseParams.b - 1][1];
        int64_t s1OuterTmp = 0;
        OP_LOGD("GetSparseUnpadBlockInfo", "Load balancing calculation results in TND scenario:");
        for (int64_t c = 1; c < blockOuter; c++) {
            int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
            uint64_t tndS1S2PrefixSumTmp = 0;
            uint64_t tndS1S2AlignPrefixSumTmp = 0;
            uint64_t tndPrefixSumTmp = 0;
            for (int64_t b = 0; b < fBaseParams.b; b++) {
                if (calculatedBlockInfo[b][0][SUM_ALL] > currentIdx) {
                    bIdx = b;
                    auto s1os2o = calculatedBlockInfo[b][fBaseParams.s2Outer - 1][SUM_S1S2];
                    auto gs1os2o = s1os2o * fBaseParams.g;
                    bTail = (b == 0) ? currentIdx : currentIdx - calculatedBlockInfo[b - 1][0][SUM_ALL];
                    n2Idx = bTail / gs1os2o;
                    n2Tail = bTail % gs1os2o;
                    gIdx = n2Tail / s1os2o;
                    gTail = n2Tail % s1os2o;

                    GetUnpadS1S2OuterIndex(s1oIdx, s2oIdx, gTail, b, calculatedBlockInfo);
                    s1OuterTmp = (fBaseParams.actualSeqQlen[b] + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;

                    tndBaseInfo.tndStartBIdx[c] = b;
                    tndBaseInfo.tndS1S2PrefixSum[c] = tndS1S2PrefixSumTmp;
                    tndBaseInfo.tndS1S2AlignPrefixSum[c] = tndS1S2AlignPrefixSumTmp;
                    tndBaseInfo.tndPrefixSum[c] = tndPrefixSumTmp;
                    break;
                } else {
                    tndS1S2PrefixSumTmp += (fBaseParams.actualSeqQlen[b] * fBaseParams.actualSeqKvlen[b]);
                    tndS1S2AlignPrefixSumTmp +=
                        (fBaseParams.actualSeqQlen[b] *
                         AlignTo(fBaseParams.actualSeqKvlen[b], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
                    int64_t s1Outer = (fBaseParams.actualSeqQlen[b] + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                                      (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
                    int64_t s2Outer = (fBaseParams.actualSeqKvlen[b] + fBaseParams.s2Inner * S2CV_RATIO_DEFAULT - 1) /
                                      (fBaseParams.s2Inner * S2CV_RATIO_DEFAULT);
                    tndPrefixSumTmp += (s1Outer * s2Outer);
                }
            }
            if (bIdx == 0) {
                blockStarts[c] =
                    (n2Idx * fBaseParams.g + gIdx) * totalBlockInfo[bIdx][0] + s2oIdx * s1OuterTmp + s1oIdx;
            } else {
                blockStarts[c] = totalBlockInfo[bIdx - 1][1] +
                                 (n2Idx * fBaseParams.g + gIdx) * totalBlockInfo[bIdx][0] + s2oIdx * s1OuterTmp +
                                 s1oIdx;
            }

            blockEnds[c - 1] = blockStarts[c];
        }

        for (int64_t c = 0; c < blockOuter; c++) {
            OP_LOGD("GetSparseUnpadBlockInfo", "blockNum[%ld], blockStarts = %ld , blockEnds = %ld ", c, blockStarts[c],
                    blockEnds[c]);
        }

        for (uint32_t c = static_cast<uint32_t>(blockOuter); c < CORE_LIST_NUM; c++) {
            blockStarts[c] = 0;
            blockEnds[c] = 0;
        }
        std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
        std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

        return ge::GRAPH_SUCCESS;
    }

    void FillBlockInfo(std::vector<std::vector<std::vector<int64_t>>> &calculatedBlockInfo,
                       std::vector<std::vector<int64_t>> &totalBlockInfo)
    {
        OP_LOGD("FillBlockInfo", " Starting load balancing calculation in TND scenario");
        OP_LOGD("FillBlockInfo", "SparseMode %u, find band index %u", fBaseParams.sparseMode, fBaseParams.bandIdx);
        for (int64_t i = 0; i < fBaseParams.b; i++) {
            int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
            int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];

            auto actualS1Outer = (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
            auto actualS2Outer = (actualS2Len + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
            totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;

            CalValidUnpadBlockInfo(i, calculatedBlockInfo);

            if (i == 0) {
                calculatedBlockInfo[0][0][SUM_ALL] =
                    fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[0][fBaseParams.s2Outer - 1][SUM_S1S2];
                totalBlockInfo[0][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[0][0];
            } else {
                calculatedBlockInfo[i][0][SUM_ALL] =
                    fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[i][fBaseParams.s2Outer - 1][SUM_S1S2] +
                    calculatedBlockInfo[i - 1][0][SUM_ALL];
                totalBlockInfo[i][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
            }
            OP_LOGD("FillBlockInfo", "Up to b idx = %ld , a total of %ld blocks that need to be calculated", i,
                    calculatedBlockInfo[i][0][SUM_ALL]);
        }
    }

    void CalValidUnpadBlockInfo(int64_t batchIdx, std::vector<std::vector<std::vector<int64_t>>> &calculatedBlockInfo)
    {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[batchIdx];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[batchIdx];
        // 对unpad场景的token值做二次校正
        // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
        int64_t actualCalcS1Token, actualCalcS2Token;
        CalcleActualToken(fBaseParams, batchIdx, actualCalcS1Token, actualCalcS2Token);

        // unpad 场景下s2Outer是按照最大的s2计算得到的
        for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
            if (fBaseParams.cvS2Inner * j >= actualS2Len) {
                calculatedBlockInfo[batchIdx][j][BEGIN_IDX] = 0;
                calculatedBlockInfo[batchIdx][j][END_IDX] = 0;
            } else {
                int64_t leftIntersectionPoint = std::max(int64_t(fBaseParams.cvS2Inner * j) - actualCalcS2Token, 0L);
                if (leftIntersectionPoint > int64_t(actualS1Len)) {
                    calculatedBlockInfo[batchIdx][j][BEGIN_IDX] =
                        (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
                } else {
                    calculatedBlockInfo[batchIdx][j][BEGIN_IDX] = leftIntersectionPoint / fBaseParams.s1CvInner;
                }
                int64_t cvBlockTail = fBaseParams.cvS2Inner * (j + 1) > actualS2Len ?
                                          actualS2Len - fBaseParams.cvS2Inner * j :
                                          fBaseParams.cvS2Inner;
                calculatedBlockInfo[batchIdx][j][END_IDX] =
                    int64_t(std::min(int64_t(actualS1Len),
                                     std::max(fBaseParams.cvS2Inner * j + cvBlockTail + actualCalcS1Token, 0L)) +
                            fBaseParams.s1CvInner - 1) /
                    fBaseParams.s1CvInner;
            }

            int64_t tmpLength =
                calculatedBlockInfo[batchIdx][j][END_IDX] > calculatedBlockInfo[batchIdx][j][BEGIN_IDX] ?
                    calculatedBlockInfo[batchIdx][j][END_IDX] - calculatedBlockInfo[batchIdx][j][BEGIN_IDX] :
                    0;
            if (j == 0) {
                calculatedBlockInfo[batchIdx][j][SUM_S1S2] = tmpLength;
            } else {
                calculatedBlockInfo[batchIdx][j][SUM_S1S2] = calculatedBlockInfo[batchIdx][j - 1][SUM_S1S2] + tmpLength;
            }

            calculatedBlockInfo[batchIdx][j][SUM_ALL] = 0; // 初始化清零

            OP_LOGD("FillBlockInfo", " s2Outer idx = %ld: Begin = %ld, End = %ld, Sum_S1S2 = %ld", j,
                    calculatedBlockInfo[batchIdx][j][BEGIN_IDX], calculatedBlockInfo[batchIdx][j][END_IDX],
                    calculatedBlockInfo[batchIdx][j][SUM_S1S2]);
        }
    }

    void GetUnpadS1S2OuterIndex(int64_t &s1oIdx, int64_t &s2oIdx, int64_t gTail, int64_t bIdx,
                                std::vector<std::vector<std::vector<int64_t>>> &calculatedBlockInfo)
    {
        int64_t s1oTail = 0;
        for (int64_t i = 0; i < fBaseParams.s2Outer; i++) {
            if (calculatedBlockInfo[bIdx][i][SUM_S1S2] > gTail) {
                s2oIdx = i;
                s1oTail = (i == 0) ? gTail : gTail - calculatedBlockInfo[bIdx][i - 1][SUM_S1S2];
                s1oIdx = calculatedBlockInfo[bIdx][i][BEGIN_IDX] + s1oTail;
                break;
            }
        }
    }

    bool IsValidUnpad(int64_t blockIdx) override
    {
        int64_t resbaseIdx = blockIdx;
        for (int64_t bIdx = 0; bIdx < fBaseParams.b; bIdx++) {
            int64_t actualS1Len = fBaseParams.actualSeqQlen[bIdx];
            int64_t actualS2Len = fBaseParams.actualSeqKvlen[bIdx];
            int64_t s1OuterTmp = (actualS1Len + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
            int64_t s2OuterTmp = (actualS2Len + fBaseParams.s2Inner * S2CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s2Inner * S2CV_RATIO_DEFAULT);
            int64_t totalBaseIdx = fBaseParams.n2 * fBaseParams.g * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                int64_t gDimTail = resbaseIdx % (s1OuterTmp * s2OuterTmp);
                int64_t s2oDimIdx = gDimTail / s1OuterTmp;
                int64_t s1oDimIdx = gDimTail % s1OuterTmp;
                int64_t s2IdxLeft = s2oDimIdx * fBaseParams.s2Inner * S2CV_RATIO_DEFAULT;
                int64_t s2IdxRight = std::min((s2oDimIdx + 1) * fBaseParams.s2Inner * S2CV_RATIO_DEFAULT, actualS2Len);
                if (fBaseParams.attenMaskOptional != EMPTY_TENSOR) {
                    return CheckSparseLeftAndRight(s1oDimIdx, s2IdxLeft, s2IdxRight, bIdx);
                } else {
                    return true;
                }
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
        return false;
    }

    bool CheckUnpadSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx) override
    {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[bIdx];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[bIdx];
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
            int64_t s2IgnoredEndLen =
                actualS1Len - static_cast<int64_t>(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1));
            int64_t s2EndLen = 0;
            if (actualS2Len > s2IgnoredEndLen) {
                s2EndLen = actualS2Len - s2IgnoredEndLen;
            }
            s2EndLen = std::min(std::max(s2EndLen, static_cast<int64_t>(fBaseParams.prefixN[bIdx])), actualS2Len);
            bool isValid = s2IdxLeft < s2EndLen;
            return isValid;
        }
        int64_t actualCalcS1Token = fBaseParams.s1Token;
        int64_t actualCalcS2Token = fBaseParams.s2Token;
        // sparse_mode == band 或者 RIGHT_DOWN_CASUAL时，token以右下角为基本，需要校正
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) &&
            bIdx != fBaseParams.bandIdx) {
            actualCalcS1Token = static_cast<int64_t>(INT32_MAX) + actualS1Len - actualS2Len;
            actualCalcS2Token = static_cast<int64_t>(0) - actualS1Len + actualS2Len;
        } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL) &&
                   bIdx != fBaseParams.bandIdx) {
            actualCalcS1Token = INT32_MAX;
            actualCalcS2Token = 0;
        } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
                   fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) ||
                   (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) &&
                    bIdx == fBaseParams.bandIdx) ||
                   (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL) &&
                    bIdx == fBaseParams.bandIdx)) {
            actualCalcS1Token = fBaseParams.s1Token + actualS1Len - actualS2Len;
            actualCalcS2Token = fBaseParams.s2Token - actualS1Len + actualS2Len;
        }
        int64_t s2SparseLeft =
            std::max(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * s1oDimIdx - actualCalcS1Token, static_cast<int64_t>(0));
        s2SparseLeft = AlignTo(s2SparseLeft, ALIGN64);
        int64_t s2SparseRight = AlignTo(
            std::min(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1), fBaseParams.s1) + actualCalcS2Token,
            static_cast<int64_t>(64));
        s2SparseRight = std::min(s2SparseRight, actualS2Len);
        bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
        return isValid;
    }

    bool GetBlockInfoOfBNS4TND() override
    {
        std::vector<std::vector<int64_t>> totalBlockInfo(fBaseParams.b, std::vector<int64_t>(TOTAL_BLOCK_DIMENSION));
        std::vector<std::vector<float>> acturalBlockInfo(fBaseParams.b + NUM_TWO,
                                                         std::vector<float>(fBaseParams.s2Outer));
        FillBlockInfoLoadBalance(totalBlockInfo, acturalBlockInfo);

        float maxBlockNumPerCore = BinarySearchMaxBlockNumPerCore(acturalBlockInfo);

        int64_t blockStarts[CORE_LIST_NUM];
        int64_t blockEnds[CORE_LIST_NUM];

        if (!CaclePerCoreBlockInfo(totalBlockInfo, acturalBlockInfo, maxBlockNumPerCore, blockStarts, blockEnds)) {
            return false;
        }

        for (int64_t c = 0; c < fBaseParams.blockOuter; c++) {
            OP_LOGD("GetBlockInfoOfBNS4TND", "blockNum[%ld], blockStarts = %ld , blockEnds = %ld", c, blockStarts[c],
                    blockEnds[c]);
        }

        for (uint32_t c = fBaseParams.blockOuter; c < CORE_LIST_NUM; c++) {
            blockStarts[c] = 0;
            blockEnds[c] = 0;
        }
        std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
        std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));
        fBaseParams.maxValidBBLen = std::ceil(maxBlockNumPerCore);
        return true;
    }

    float BinarySearchMaxBlockNumPerCore(const std::vector<std::vector<float>> &acturalBlockInfo)
    {
        float left = acturalBlockInfo[fBaseParams.b + 1][0];
        float right = acturalBlockInfo[fBaseParams.b][0];
        float mid = 0;
        while (left < right - 1) {
            mid = (left + right) / NUM_TWO;
            if (IsPossible(acturalBlockInfo, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return right;
    }

    bool CaclePerCoreBlockInfo(const std::vector<std::vector<int64_t>> &totalBlockInfo,
                               const std::vector<std::vector<float>> &acturalBlockInfo, const float maxBlockNumPerCore,
                               int64_t (&blockStarts)[CORE_LIST_NUM], int64_t (&blockEnds)[CORE_LIST_NUM])
    {
        float currentSum = 0;
        int64_t coreIdx = 0;
        uint64_t tndS1S2PrefixSumTmp = 0;
        uint64_t tndS1S2AlignPrefixSumTmp = 0;
        uint64_t tndPrefixSumTmp = 0;
        bool isSetSwizzleParam = fBaseParams.b < TND_SWIZZLE_PREFIX_NUM;
        for (int64_t b = 0; b < fBaseParams.b; b++) {
            for (int64_t n = 0; n < fBaseParams.n2 * fBaseParams.g; n++) {
                int64_t actualS1Outer =
                    (fBaseParams.actualSeqQlen[b] + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
                for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
                    float num = acturalBlockInfo[b][j];
                    if (coreIdx >= CORE_LIST_NUM) {
                        OP_LOGD("GetBlockInfoOfBNS4TND", " Not support BN2S2.");
                        return false;
                    } else if (currentSum + num > maxBlockNumPerCore) {
                        OP_LOGD("GetBlockInfoOfBNS4TND", " blockIdx = %ld: acturalBlock = %f", coreIdx, currentSum);
                        int64_t preBatchBlockNum = b == 0 ? 0 : totalBlockInfo[b - 1][1];
                        int64_t preNGBlockNum = n * totalBlockInfo[b][0];
                        int64_t preS2BlockNum = j * actualS1Outer;
                        blockEnds[coreIdx] = preBatchBlockNum + preNGBlockNum + preS2BlockNum;
                        blockStarts[coreIdx + 1] = blockEnds[coreIdx];
                        coreIdx += 1;
                        currentSum = num;
                        tndBaseInfo.tndStartBIdx[coreIdx] = b;
                        tndBaseInfo.tndS1S2PrefixSum[coreIdx] = tndS1S2PrefixSumTmp;
                        tndBaseInfo.tndS1S2AlignPrefixSum[coreIdx] = tndS1S2AlignPrefixSumTmp;
                        tndBaseInfo.tndPrefixSum[coreIdx] = tndPrefixSumTmp;
                    } else {
                        currentSum += num;
                    }
                }
            }
            tndS1S2PrefixSumTmp += (fBaseParams.actualSeqQlen[b] * fBaseParams.actualSeqKvlen[b]);
            tndS1S2AlignPrefixSumTmp +=
                (fBaseParams.actualSeqQlen[b] *
                 AlignTo(fBaseParams.actualSeqKvlen[b], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
            int64_t s1OuterTmp = (fBaseParams.actualSeqQlen[b] + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
            int64_t s2OuterTmp = (fBaseParams.actualSeqKvlen[b] + fBaseParams.s2Inner * S2CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s2Inner * S2CV_RATIO_DEFAULT);
            tndPrefixSumTmp += (s1OuterTmp * s2OuterTmp);
            if (isSetSwizzleParam) {
                SetTndSwizzleParam(b, s1OuterTmp, s2OuterTmp);
            }
        }
        OP_LOGD("GetBlockInfoOfBNS4TND", " blockIdx = %ld: actualBlock = %f", coreIdx, currentSum);
        blockStarts[0] = 0;
        blockEnds[coreIdx] = totalBlockInfo[fBaseParams.b - 1][1];
        fBaseParams.blockOuter = coreIdx + 1;
        return true;
    }

    void SetTndSwizzleParam(int64_t bIdx, int64_t s1OuterTmp, int64_t s2OuterTmp)
    {
        int64_t realS1OuterTmp = s1OuterTmp;
        int64_t realS2OuterTmp = s2OuterTmp;
        // when enable tnd swizzle, fill tnd swizzle info
        if (fBaseParams.sparseType == static_cast<uint8_t>(SparseType::DENSE)) {
            tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
                tndBaseInfo.tndS2BlockPrefixSum[bIdx] +
                CeilDivideBy(s2OuterTmp * fBaseParams.n2 * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)) *
                    s1OuterTmp;
        } else if (fBaseParams.sparseType == static_cast<uint8_t>(SparseType::CASUAL)) {
            // 处理无效列和无效行场景
            if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) &&
                s1OuterTmp < s2OuterTmp) {
                realS2OuterTmp = s1OuterTmp;
            } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) &&
                       s1OuterTmp > s2OuterTmp) {
                realS1OuterTmp = s2OuterTmp;
            }
            int64_t halfN2g = (fBaseParams.n2 * fBaseParams.g) >> 1;
            if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
                int64_t n1 = (realS2OuterTmp << 1) - realS1OuterTmp + 1;
                tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
                    tndBaseInfo.tndS2BlockPrefixSum[bIdx] +
                    CeilDivideBy(n1 * halfN2g, static_cast<int64_t>(fBaseParams.aicNum)) * realS1OuterTmp;
            } else {
                int64_t m1 = (realS1OuterTmp << 1) - realS2OuterTmp + 1;
                tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
                    tndBaseInfo.tndS2BlockPrefixSum[bIdx] +
                    CeilDivideBy(realS2OuterTmp * halfN2g, static_cast<int64_t>(fBaseParams.aicNum)) * m1;
            }
        } else if (fBaseParams.sparseType == static_cast<uint8_t>(SparseType::BAND)) {
            int64_t actualCalcS1Token, actualCalcS2Token;
            CalcleActualToken(fBaseParams, bIdx, actualCalcS1Token, actualCalcS2Token);
            int64_t p = CeilDivideBy(actualCalcS1Token, fBaseParams.s1Inner * fBaseParams.s1CvRatio) + 1;
            int64_t q = CeilDivideBy(actualCalcS2Token, fBaseParams.s2Inner * fBaseParams.s2CvRatio) + 1;
            p = p > s1OuterTmp ? s1OuterTmp : p;
            q = q > s2OuterTmp ? s2OuterTmp : q;
            if (p + q <= s1OuterTmp) {
                tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
                    tndBaseInfo.tndS2BlockPrefixSum[bIdx] + CeilDivideBy(s2OuterTmp * fBaseParams.n2 * fBaseParams.g,
                                                                         static_cast<int64_t>(fBaseParams.aicNum)) *
                                                                (p + q - 1);
            } else {
                tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1] =
                    tndBaseInfo.tndS2BlockPrefixSum[bIdx] + CeilDivideBy(s2OuterTmp * fBaseParams.n2 * fBaseParams.g,
                                                                         static_cast<int64_t>(fBaseParams.aicNum)) *
                                                                s1OuterTmp;
            }
        }
        tndBaseInfo.tndSwizzleS1S2PrefixSum[bIdx + 1] =
            tndBaseInfo.tndSwizzleS1S2PrefixSum[bIdx] +
            (fBaseParams.actualSeqQlen[bIdx] * fBaseParams.actualSeqKvlen[bIdx]);
        tndBaseInfo.tndSwizzleS1S2AlignPrefixSum[bIdx + 1] =
            tndBaseInfo.tndSwizzleS1S2AlignPrefixSum[bIdx] +
            (fBaseParams.actualSeqQlen[bIdx] *
             AlignTo(fBaseParams.actualSeqKvlen[bIdx], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
        OP_LOGD("GetBlockInfoOfBNS4TND", " bIdx = %ld: tndS2BlockPrefixSum = %ld, tndSwizzleS1S2PrefixSum = %d",
                bIdx + 1, tndBaseInfo.tndS2BlockPrefixSum[bIdx + 1], tndBaseInfo.tndSwizzleS1S2PrefixSum[bIdx + 1]);
    }

    bool IsPossible(const std::vector<std::vector<float>> &acturalBlockInfo, const float possibleMax)
    {
        float currentSum = 0;
        uint64_t needCoreNum = 1;
        int64_t n2g = fBaseParams.n2 * fBaseParams.g;
        int64_t bn2g = fBaseParams.b * n2g;
        if (fBaseParams.isBn2MultiBlk) {
            for (int64_t i = 0; i < bn2g; i++) {
                int64_t b = i / n2g;
                float blkNumPerBN = acturalBlockInfo[b][fBaseParams.s2Outer];
                if (currentSum + blkNumPerBN > possibleMax) {
                    needCoreNum += 1;
                    currentSum = blkNumPerBN;
                } else {
                    currentSum += blkNumPerBN;
                }
                if (needCoreNum > fBaseParams.aicNum) {
                    return false;
                }
            }
        } else {
            for (int64_t i = 0; i < bn2g; i++) {
                int64_t b = i / n2g;
                for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
                    float num = acturalBlockInfo[b][j];
                    if (currentSum + num > possibleMax) {
                        needCoreNum += 1;
                        currentSum = num;
                    } else {
                        currentSum += num;
                    }
                    if (needCoreNum > fBaseParams.aicNum) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScoreGrad, FlashAttentionScoreGradTilingVarlenRegbase,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 900);
} // namespace fag
} // namespace optiling
