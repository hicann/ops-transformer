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
constexpr uint32_t MIN_SINGLE_N = 64U; // 单核 N 下限：低于 64 时 MM 的 baseN(64/128) 一次都填不满
constexpr uint32_t MAX_N_GROUP = 32U;  // N 方向最多分 32 组
constexpr uint32_t MAX_CAND = 256U; // (ng,kg) 候选数组容量；有效 ng 组合下候选上界约 160，留余量
constexpr uint32_t MAX_KG_CACHE = 256U; // 按 kg 缓存的 K 推导结果容量；kg 只推拿到核数，避免无意义枚举
constexpr uint32_t L1_A_SIZE = 128U * 1024U; // L1A 单ping大小
constexpr uint32_t L1_B_SIZE = 128U * 1024U; // L1B 单ping大小
constexpr uint32_t ALIGN_BYTE = 256U;   // 随路 ND2NZ 的内轴对齐粒度：一次 L1 装载块必须为其整数倍
constexpr uint64_t MAC_TO_EQ_DIV = 32U; // 32 MAC ≡ 1 字节当量

struct KnCtx {
    uint32_t nSize;         // mm 的 N：mm1=Hcq，mm2=Hckv+Dr
    uint32_t kSize;         // mm 的 K：He(权重行数)
    uint32_t mSize;         // 一次处理的 token 数 = stepBatchSize
    uint32_t dtypeSize;     // 权重元素字节数：1(int8/fp8/hif8) / 2(bf16)
    uint32_t baseN;         // N 基本块：mxfp8 为 64，其余 128
    uint32_t baseK;         // K 基本块：1 字节权重 256，2 字节权重 128
    uint32_t fractal;       // NZ 分形的 C0：1 字节 32，2 字节 16
    uint32_t kFractalAlign; // K 方向对齐粒度：mxfp8 需 256(含 scale 块)，其余 32
    uint32_t l1BBudget;     // 可用于 B 的 L1 预算：mxfp8 时 B 的 scale 常驻上半区故减半
    uint32_t aicNum;        // 可用 AIC 数
    uint32_t tokPerAiv;     // 每个 AIV 分到的 token 行数（部分和累加的粒度）
    uint32_t mAligned;      // M 轴按 16 行对齐后的值（MM 的 M fractal 为 16）
    uint32_t maxStepKA;     // 受 L1_A 容量限制的 K 步长上限（单位：baseK 块数）
    uint32_t maxStepKB;     // 受 L1_B 容量限制的 K 步长上限（单位：baseK 块数）
    uint32_t maxNg;         // N 分组上限 = min(MAX_N_GROUP, nSize/MIN_SINGLE_N)
    uint64_t mnk;           // 总 MAC 量 m*n*k
};

struct KnKInfo {
    uint32_t singleK = 0;  // 每个 K 组分到的 K 元素数 = CeilDiv(kSize, kg)
    uint32_t kL1Loops = 0; // 该 K 组被 L1 切成几轮装载
    bool legal = false;    // 该 kg 是否可用（K 对齐、尾组非空、部分和预算均满足）
    bool kAligned = false; // 每轮装载字节数是否都是 ALIGN_BYTE 的整数倍
    bool aReuse = false;   // 整组 K 能否一次常驻 L1_A（跨 N 复用 A，省 A 搬运）
};

struct KnCand {
    uint32_t ng = 0;        // N 分组数（每组 nSize/ng）
    uint32_t kg = 0;        // K 分组数（每组 kSize/kg）
    uint32_t coreNum = 0;   // 占用核数 = ng*kg
    bool kAligned = false;  // 见 KnKInfo::kAligned
    bool dominated = false; // 被其它候选 Pareto 支配，不参与选优
    uint64_t loadB = 0;     // A+B 总装载字节数
    uint64_t fixB = 0;      // FIX 搬出字节数：M×singleN×4B
    uint64_t macEq = 0;     // 计算时间折算的等效字节数
    uint64_t aivReadB = 0;  // AIV 回读部分和的字节数
    uint64_t aivVecOps = 0; // AIV 累加操作数（元素个数）
    uint64_t l2Obj = 0;     // 目标函数 max(fixB, macEq) + aivReadB，越小越好
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
    c.baseN = (baseNKn != 0U) ? baseNKn : 128U; // 128：非 mxfp8 模式的 MM 默认 N 基本块
    c.baseK = (dtypeSize == 1U) ? 256U : 128U; // 1B 权重用 256，2B 权重用 128；与 kernel MmCq/MmCkvKrParamInit 同源
    c.fractal = (dtypeSize == 1U) ? 32U : 16U; // GetC0Num(dtype)：1B 元素 C0=32，2B 元素 C0=16
    c.kFractalAlign = isMxfp8Scale ? 256U : 32U; // mxfp8 的 K 需按 256 对齐(32 元素一块的 scale 组)，其余按 32
    c.l1BBudget = isMxfp8Scale ? (L1_B_SIZE / 2U) : L1_B_SIZE; // mxfp8: scale 绑载常驻 L1B 上半区，可用 B 预算减半
    c.aicNum = aicNum;
    const uint32_t aivCores = (aivNum == 0U) ? 0U : std::min(mSize, aivNum);
    c.tokPerAiv = (aivCores == 0U) ? 0U : KnCeilDiv(mSize, aivCores);
    c.mAligned = (mSize + 15U) / 16U * 16U; // 15/16：M 轴按 16 行(M fractal)向上对齐
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
    if (static_cast<uint64_t>(kg) * c.nSize * 4ULL >
        64ULL * 1024ULL) { // kg 份 N 行（每元素 4B，FP32 部分和）合计不超过 64KB
        return;
    }
    uint32_t stepK = std::min(c.maxStepKA, c.maxStepKB);
    stepK = std::min(stepK, KnCeilDiv(singleK, c.baseK));
    stepK = (stepK < 1U) ? 1U : stepK;
    const uint32_t kL1Step = c.baseK * stepK;
    const bool kAligned = (singleK % ALIGN_BYTE == 0U) &&
                          (singleK < kL1Step || kL1Step % ALIGN_BYTE == 0U); // 满段与尾段都需是 256 元素的整数倍
    k.singleK = singleK;
    k.kL1Loops = KnCeilDiv(singleK, kL1Step);
    k.kAligned = kAligned;
    k.aReuse = (k.kL1Loops == 1U) && (static_cast<uint64_t>(c.mSize) * singleK * c.dtypeSize <= L1_A_SIZE);
    k.legal = true;
}

inline void KnGenCandidates(const KnCtx& c, const KnKInfo* cache, KnCand* cands, uint32_t& candNum)
{
    for (uint32_t ng = 1U; ng <= c.maxNg; ng++) {
        if (c.nSize % ng != 0U) {
            continue;
        }
        const uint32_t singleN = c.nSize / ng;
        if (singleN % c.fractal != 0U || singleN < MIN_SINGLE_N) {
            continue;
        }
        const uint32_t maxKg = c.aicNum / ng;
        const uint32_t nLoops = KnCeilDiv(singleN, c.baseN);
        for (uint32_t kg = 1U; kg <= maxKg; kg++) {
            KnKInfo k;
            if (kg <= MAX_KG_CACHE) {
                k = cache[kg - 1U];
            } else {
                KnDeriveK(c, kg, k);
            }
            if (!k.legal || candNum >= MAX_CAND) { // 容量保护：超出则丢弃后续候选，仅影响最优性
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
            cd.fixB = static_cast<uint64_t>(c.mSize) * singleN * 4ULL; // 4：FP32 部分和，每元素 4B
            cd.macEq = (cd.coreNum == 0U) ? 0ULL : (c.mnk / cd.coreNum / MAC_TO_EQ_DIV);
            cd.aivReadB = static_cast<uint64_t>(kg) * c.tokPerAiv * c.nSize * 4ULL; // 4：FP32 部分和，每元素 4B
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

inline bool KnBetter(const KnCand& a, const KnCand& b)
{
    if (a.l2Obj != b.l2Obj) {
        return a.l2Obj < b.l2Obj;
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
        if (best < 0 || KnBetter(cands[i], cands[best])) {
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}
} // namespace kn_detail

inline bool CalcSplitKnGroups(uint32_t nSize, uint32_t kSize, uint32_t mSize, uint32_t dtypeSize, bool isMxfp8Scale,
                              uint32_t baseNKn, uint32_t aicNum, uint32_t aivNum, uint16_t& nGroup, uint16_t& kGroup)
{
    if (isMxfp8Scale && kSize % 64U != 0U) { // 64：mxfp8 的 K 至少两个 scale 块(32*2)
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
