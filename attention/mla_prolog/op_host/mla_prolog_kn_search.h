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
 * \file mla_prolog_kn_search.h
 * \brief 切KN 等宽分组搜索(独立模块, header-only)。仅由 IsSplitKnSupported() 门控后调用
 */
#ifndef MLA_PROLOG_KN_SEARCH_H
#define MLA_PROLOG_KN_SEARCH_H
#include <algorithm>
#include <cstdint>

namespace optiling {
namespace kn_detail {
constexpr uint32_t MIN_SINGLE_N = 64U;
constexpr uint32_t MAX_N_GROUP = 32U;
constexpr uint32_t MAX_CAND = 256U;
constexpr uint32_t MAX_KG_CACHE = 256U;
constexpr uint32_t L1_A_SIZE = 128U * 1024U;
constexpr uint32_t L1_B_SIZE = 128U * 1024U;
constexpr uint32_t ALIGN_BYTE = 256U;                                   // 随路 ND2NZ 内轴对齐粒度
constexpr uint32_t MAC_PER_CYCLE = 8192U;                               // 每核 MMAD 吞吐(MAC/cycle)
constexpr uint32_t MTE2_BYTE_PER_CYCLE = 64U;                           // 每核 MTE2 管宽(B/cycle)
constexpr uint64_t MAC_TO_EQ_DIV = MAC_PER_CYCLE / MTE2_BYTE_PER_CYCLE; // 128 MAC ≡ 1 字节当量

struct KnCtx {
    uint32_t nSize;
    uint32_t kSize;
    uint32_t mSize;
    uint32_t dtypeSize;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t fractal;
    uint32_t kFractalAlign; // mxfp8: 64; 其它: 32
    uint32_t l1BBudget;     // mxfp8: L1_B_SIZE/2; 其它: 全额
    uint32_t aicNum;
    uint32_t tokPerAiv;
    uint32_t mAligned;
    uint32_t maxStepKA;
    uint32_t maxStepKB;
    uint32_t maxNg;
    uint64_t mnk;
};

struct KnKInfo {
    uint32_t singleK = 0;
    uint32_t kL1Loops = 0;
    bool legal = false;
    bool kAligned = false;
    bool aReuse = false;
};

struct KnCand {
    uint32_t ng = 0;
    uint32_t kg = 0;
    uint32_t coreNum = 0;
    bool kAligned = false;
    bool dominated = false;
    uint64_t loadB = 0;
    uint64_t fixB = 0;
    uint64_t macEq = 0;
    uint64_t aivReadB = 0;
    uint64_t aivVecOps = 0;
    uint64_t l2Obj = 0;
};

inline uint32_t KnCeilDiv(uint32_t a, uint32_t b)
{
    return (b == 0U) ? 0U : (a + b - 1U) / b;
}

inline KnCtx KnMakeCtx(uint32_t nSize, uint32_t kSize, uint32_t mSize, uint32_t dtypeSize, uint32_t baseNKn,
                       bool isMxfp8Scale, uint32_t aicNum, uint32_t aivNum)
{
    KnCtx c;
    c.nSize = nSize;
    c.kSize = kSize;
    c.mSize = mSize;
    c.dtypeSize = dtypeSize;
    c.baseN = (baseNKn != 0U) ? baseNKn : 128U;
    c.baseK = (dtypeSize == 1U) ? 256U : 128U; // 与 kernel MmCq/MmCkvKrParamInit 的 baseK 同源
    c.fractal = (dtypeSize == 1U) ? 32U : 16U; // GetC0Num(dtype)
    c.kFractalAlign = isMxfp8Scale ? 256U : 32U;
    c.l1BBudget = isMxfp8Scale ? (L1_B_SIZE / 2U) : L1_B_SIZE; // mxfp8: scale 绑载常驻 L1B 上半区
    c.aicNum = aicNum;
    const uint32_t aivCores = (aivNum == 0U) ? 0U : std::min(mSize, aivNum);
    c.tokPerAiv = (aivCores == 0U) ? 0U : KnCeilDiv(mSize, aivCores);
    c.mAligned = (mSize + 15U) / 16U * 16U;
    c.maxStepKA = (c.mAligned == 0U || c.baseK == 0U) ? 1U : L1_A_SIZE / (c.mAligned * c.baseK * dtypeSize);
    c.maxStepKB = (c.baseN == 0U || c.baseK == 0U) ? 1U : c.l1BBudget / (c.baseN * c.baseK * dtypeSize);
    c.maxNg = std::min(MAX_N_GROUP, (c.fractal == 0U) ? 0U : nSize / MIN_SINGLE_N);
    c.mnk = static_cast<uint64_t>(mSize) * nSize * kSize;
    return c;
}

inline void KnDeriveK(const KnCtx& c, uint32_t kg, KnKInfo& k)
{
    const uint32_t singleK = KnCeilDiv(c.kSize, kg);
    if (singleK % c.kFractalAlign != 0U) {
        return;
    }
    const int64_t tailK = static_cast<int64_t>(c.kSize) - static_cast<int64_t>(kg - 1U) * singleK;
    if (tailK < static_cast<int64_t>(c.kFractalAlign)) {
        return;
    }
    if (static_cast<uint64_t>(kg) * c.nSize * 4ULL > 64ULL * 1024ULL) {
        return;
    }
    uint32_t stepK = std::min(c.maxStepKA, c.maxStepKB);
    stepK = std::min(stepK, KnCeilDiv(singleK, c.baseK));
    stepK = (stepK < 1U) ? 1U : stepK;
    const uint32_t kL1Step = c.baseK * stepK;
    bool kAligned = true;
    for (uint32_t rem = singleK; rem > 0U;) {
        const uint32_t blk = std::min(rem, kL1Step);
        if (blk % ALIGN_BYTE != 0U) {
            kAligned = false;
            break;
        }
        rem -= blk;
    }
    k.singleK = singleK;
    k.kL1Loops = KnCeilDiv(singleK, kL1Step);
    k.kAligned = kAligned;
    k.aReuse = (k.kL1Loops == 1U) && (static_cast<uint64_t>(c.mSize) * singleK * c.dtypeSize <= L1_A_SIZE);
    k.legal = true;
}

inline void KnGenCandidates(const KnCtx& c, const KnKInfo* cache, KnCand* cands, uint32_t& candNum)
{
    for (uint32_t ng = 1U; ng <= c.maxNg; ng++) {
        if (c.nSize % ng != 0U) { // v2 等宽分割(不均匀分形切分属后续 Step 2)
            continue;
        }
        const uint32_t singleN = c.nSize / ng;
        if (singleN % c.fractal != 0U || singleN < MIN_SINGLE_N) {
            continue;
        }
        const uint32_t maxKg = c.aicNum / ng; // ng*kg <= aicNum(允许预算核)
        const uint32_t nLoops = KnCeilDiv(singleN, c.baseN);
        for (uint32_t kg = 1U; kg <= maxKg; kg++) {
            KnKInfo k;
            if (kg <= MAX_KG_CACHE) {
                k = cache[kg - 1U];
            } else {
                KnDeriveK(c, kg, k);
            }
            if (!k.legal || candNum >= MAX_CAND) {
                continue;
            }
            KnCand& cd = cands[candNum];
            const uint64_t aFactor = ((nLoops > 1U) && !k.aReuse) ? nLoops : 1ULL;
            cd.ng = ng;
            cd.kg = kg;
            cd.coreNum = ng * kg;
            cd.kAligned = k.kAligned;
            cd.loadB = static_cast<uint64_t>(c.mSize) * k.singleK * c.dtypeSize * aFactor +
                       static_cast<uint64_t>(k.singleK) * singleN * c.dtypeSize;
            cd.fixB = static_cast<uint64_t>(c.mSize) * singleN * 4ULL;
            cd.macEq = (cd.coreNum == 0U) ? 0ULL : (c.mnk / cd.coreNum / MAC_TO_EQ_DIV);
            cd.aivReadB = static_cast<uint64_t>(kg) * c.tokPerAiv * c.nSize * 4ULL;
            cd.aivVecOps = static_cast<uint64_t>(kg - 1U) * c.tokPerAiv * c.nSize;
            cd.l2Obj = std::max(cd.fixB, cd.macEq) + cd.aivReadB;
            candNum++;
        }
    }
}

inline void KnMarkDominated(KnCand* cands, uint32_t candNum)
{
    for (uint32_t j = 0U; j < candNum; j++) {
        for (uint32_t i = 0U; i < candNum; i++) {
            if (i == j) {
                continue;
            }
            const KnCand& a = cands[i];
            const KnCand& b = cands[j];
            const bool noWorse = a.loadB <= b.loadB && a.fixB <= b.fixB && a.macEq <= b.macEq &&
                                 a.aivReadB <= b.aivReadB && a.aivVecOps <= b.aivVecOps && (!b.kAligned || a.kAligned);
            const bool strictlyBetter = a.loadB < b.loadB || a.fixB < b.fixB || a.macEq < b.macEq ||
                                        a.aivReadB < b.aivReadB || a.aivVecOps < b.aivVecOps ||
                                        (a.kAligned && !b.kAligned);
            if (noWorse && strictlyBetter) {
                cands[j].dominated = true;
                break;
            }
        }
    }
}

inline bool KnBetter(const KnCand& a, const KnCand& b, bool domKey)
{
    if (a.l2Obj != b.l2Obj) {
        return a.l2Obj < b.l2Obj;
    }
    if (domKey && (a.dominated != b.dominated)) {
        return !a.dominated;
    }
    if (a.kAligned != b.kAligned) {
        return a.kAligned;
    }
    if (a.coreNum != b.coreNum) {
        return a.coreNum > b.coreNum;
    }
    return a.ng < b.ng;
}

inline int32_t KnPickBest(const KnCand* cands, uint32_t candNum)
{
    int32_t best = -1;
    for (uint32_t i = 0U; i < candNum; i++) {
        if (cands[i].dominated) {
            continue;
        }
        if (best < 0 || KnBetter(cands[i], cands[best], false)) {
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}
} // namespace kn_detail

inline bool CalcSplitKnGroups(uint32_t nSize, uint32_t kSize, uint32_t mSize, uint32_t dtypeSize, bool isMxfp8Scale,
                              uint32_t baseNKn, uint32_t aicNum, uint32_t aivNum, uint16_t& nGroup, uint16_t& kGroup)
{
    if (isMxfp8Scale && kSize % 64U != 0U) {
        return false;
    }
    const kn_detail::KnCtx ctx =
        kn_detail::KnMakeCtx(nSize, kSize, mSize, dtypeSize, baseNKn, isMxfp8Scale, aicNum, aivNum);
    kn_detail::KnKInfo cache[kn_detail::MAX_KG_CACHE];
    const uint32_t kgCount = std::min(aicNum, kn_detail::MAX_KG_CACHE);
    for (uint32_t kg = 1U; kg <= kgCount; kg++) {
        kn_detail::KnDeriveK(ctx, kg, cache[kg - 1U]);
    }
    kn_detail::KnCand cands[kn_detail::MAX_CAND];
    uint32_t candNum = 0U;
    kn_detail::KnGenCandidates(ctx, cache, cands, candNum);
    if (candNum == 0U) {
        return false;
    }
    kn_detail::KnMarkDominated(cands, candNum);
    const int32_t chosen = kn_detail::KnPickBest(cands, candNum);
    if (chosen < 0) {
        return false;
    }
    nGroup = static_cast<uint16_t>(cands[chosen].ng);
    kGroup = static_cast<uint16_t>(cands[chosen].kg);
    return true;
}
} // namespace optiling
#endif // MLA_PROLOG_KN_SEARCH_H
