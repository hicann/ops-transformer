/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit_balance_tiling 求解层: 调度上下文 + 公共逻辑片段
#pragma once

#include "fit_balance_types.h"
#include "comm_model.h"
#include "cut_planner.h"

// 块耗时: tileSize × rowMul × 每行耗时系数(不足 baseM 按 baseM 计)
inline double MatmulTime(const MatmulInputParams& mmInput, const MatmulModelParams& mmModel, uint64_t tileSize,
                         uint64_t rowMul)
{
    tileSize = std::max<uint64_t>(mmInput.baseM, tileSize);
    return static_cast<double>(tileSize) * static_cast<double>(rowMul) * mmModel.matmulGradient;
}

// 反解: 满足目标耗时的块行数(同样受 baseM 下限约束)
inline uint64_t InverseMatmulTime(const MatmulInputParams& mmInput, const MatmulModelParams& mmModel, double targetTime,
                                  uint64_t rowMul)
{
    uint64_t tileSize = static_cast<uint64_t>(targetTime / (static_cast<double>(rowMul) * mmModel.matmulGradient));
    tileSize = std::max<uint64_t>(mmInput.baseM, tileSize);
    return tileSize;
}

// 长短块等长时退化为纯等分
inline void MergeEqualTiles(CutPlanner& tilingM)
{
    if (tilingM.cutRes.shortTileLen == tilingM.cutRes.longTileLen) {
        tilingM.cutRes.shortTileLen = 0U;
        tilingM.cutRes.numShortTile = 0U;
        tilingM.cutRes.numLongTile++;
    }
}

// 线性度好的 shape(k×n 较大, 通信拟合更可信)
inline bool IsGoodLinearityShape(const MatmulInputParams& mmInput)
{
    return mmInput.kValue * mmInput.nValue >= GOOD_LINEARITY_NK_BYTES * BYTES_PER_MB;
}

// 双向配平长块(NONE 形态): 短块与相邻长块构成"通信 ∥ 计算"边界配对,
// 长块按配平等长反解。配对方向由通算顺序决定:
//   计算在前: 短块在后 → comm(长)=mm(短); 短块在前 → mm(长)=comm(短)
//   通信在前: 短块在后 → mm(长)=comm(短); 短块在前 → comm(长)=mm(短)
inline void BalanceLongTileLen(const FitBalanceFlow& flow, const MatmulInputParams& mmInput,
                               const MatmulModelParams& mmModel, const CommModelParams& comm, CutPlanner& tilingM)
{
    const bool commSideCovers = (flow.order == CommComputeOrder::COMPUTE_FIRST) == tilingM.cutRes.shortTileAtBack;
    if (commSideCovers) {
        const double targetTime = MatmulTime(mmInput, mmModel, tilingM.cutRes.shortTileLen, mmModel.mmRankMul);
        tilingM.cutRes.longTileLen = InverseCommTime(comm, targetTime);
    } else {
        const double targetTime = CommTime(comm, tilingM.cutRes.shortTileLen);
        tilingM.cutRes.longTileLen = InverseMatmulTime(mmInput, mmModel, targetTime, mmModel.mmRankMul);
    }
}

// 总量估计: 不切口径(轮数=1)的通信与计算总耗时, 用于短块位置判定与 [5] 的串行基准
struct EstimateTimesResult {
    double totalMatmulTime = 0.0;
    double totalCommTimeNoCut = 0.0;
};
inline EstimateTimesResult EstimateTotalTimes(const MatmulInputParams& mmInput, const MatmulModelParams& mmModel,
                                              const CommModelParams& comm)
{
    EstimateTimesResult estimateResult;
    estimateResult.totalMatmulTime = MatmulTime(mmInput, mmModel, mmInput.mValue, mmModel.mmRankMul);
    estimateResult.totalCommTimeNoCut = TotalCommTime(comm, mmInput.mValue, 1);
    return estimateResult;
}

// 短块位置判定: 本地块形态恒定(FRONT 短块在后收尾 / BACK 短块在前启动通信);
// 无本地块时按 bound 方向选择边界配对方向, kernel 不具备短块前置能力时恒后置。
// 短块在前仅当总长为对齐粒度整数倍时可交付: 长块恒对齐(终态对齐契约)时短块承担
// 总长的非对齐余数, 居首则其边界必被计算块跨界(kernel 一块一槽读取)。
inline bool ChooseShortTileAtBack(const FitBalanceFlow& flow, const EstimateTimesResult& estimate,
                                  const MatmulInputParams& mmInput)
{
    if (flow.localMMPosition == LocalMMPosition::FRONT) {
        return true;
    }
    if (flow.localMMPosition == LocalMMPosition::BACK) {
        return false;
    }
    bool shortAtBack;
    if (flow.order == CommComputeOrder::COMPUTE_FIRST) {
        // 计算 Bound 场景下，短块放在后面，这样最后一块计算（短）和倒数第二块通信（长）可以互相掩盖
        // 通信 Bound 场景下，短块放在前面，这样第二块计算（长）和第一块通信（短）可以互相掩盖
        shortAtBack = estimate.totalMatmulTime >= estimate.totalCommTimeNoCut;
    } else {
        // 通信 Bound 场景下，短块放在后面，这样最后一块通信（短）和倒数第二块计算（长）可以互相掩盖
        // 计算 Bound 场景下，短块放在前面，这样第二块通信（长）和第一块计算（短）可以互相掩盖
        shortAtBack = estimate.totalCommTimeNoCut > estimate.totalMatmulTime;
    }
    // kernel 不支持或输入 shape 不支持短块前置，固定后置(可交付前置要求总长为
    // 切分对齐粒度——即 matmul baseM——的整数倍; 与 MakeCutPlanner 同口径)
    const uint64_t alignLen = mmInput.baseM != 0UL ? mmInput.baseM : static_cast<uint64_t>(M_ALIGN_GRANULARITY);
    if (!flow.allowShortFront || mmInput.mValue % alignLen != 0U) {
        return true;
    }
    return shortAtBack;
}

// 末轮计算按调度块量化计费: mm 按 baseM 块调度, 尾块残行实付 ceil 块数的整块时间。
// 限定 M 轴聚合族: K 轴族 b 高估且切片开销未建模, 量化会错误推向粗切/NoCut。
inline double QuantizedTailTime(const MatmulInputParams& mmInput, const MatmulModelParams& mmModel, uint64_t tileLen,
                                uint64_t rowMul, bool quantize)
{
    if (!quantize) {
        return MatmulTime(mmInput, mmModel, tileLen, rowMul);
    }
    const uint64_t baseM = (mmInput.baseM != 0UL) ? mmInput.baseM : static_cast<uint64_t>(M_ALIGN_GRANULARITY);
    const uint64_t quantLen = ((tileLen + baseM - 1UL) / baseM) * baseM;
    return MatmulTime(mmInput, mmModel, quantLen, rowMul);
}

// 流水线总时估计(切 N 轮的近似): 不可重叠段在流水线两端, 中段取瓶颈侧。
// 单块(含 NoCut 候选)返回 serialTotalTime。
// totalMmTime/serialTotalTime 由调用方一次算好传入(搜索循环内不变)。
inline double EstimateCutTotalTime(const FitBalanceFlow& flow, const MatmulInputParams& mmInput,
                                   const MatmulModelParams& mmModel, const CommModelParams& comm,
                                   const CommCutResult& cut)
{
    // mm 物理惩罚项(标定注入, 默认中性):
    // 候选级 mm 修正, 两项物理惩罚按引擎常量执行(不经标定):
    // (1) 减半惩罚: 候选将触发 host 侧强制减半(头块==baseM ∧ 元素>=1B ∧ 多块)且 N 方向
    //     块数超阈值——baseM 减半使 MMAD 指令数翻倍, 巨 N 时每块变薄、发射开销占比放大
    // (2) A-L2 挤出: B 流量超 L2 且单轮 A 工作集超限——A 行被 B 流持续挤出, DDR 反复读
    const double totalMmTime = MatmulTime(mmInput, mmModel, mmInput.mValue, mmModel.mmRankMul);
    double pipelineCorrectedMm = totalMmTime;
    {
        constexpr double MM_PENALTY_COEF = 1.57;
        constexpr uint64_t HALVE_NTILES_LIMIT = 40;
        constexpr double A_WS_LIMIT_MB = 40.0;
        constexpr double L2_BYTES_MB = 32.0;
        const uint64_t nTiles = (mmInput.baseN > 0U) ? (mmInput.nValue + mmInput.baseN - 1U) / mmInput.baseN : 0U;
        const bool halvingCandidate = cut.totalTileCnt > 1U && cut.longTileLen == mmInput.baseM &&
                                      mmInput.elemAtLeast1Byte && mmInput.baseM % 32UL == 0UL &&
                                      mmInput.baseM < mmInput.mValue * mmModel.mmRankMul;
        const double bBytesMbL2 = static_cast<double>(mmInput.nValue) * static_cast<double>(mmInput.kValue) *
                                  mmInput.mmDtypeBytes / (1024.0 * 1024.0);
        const double aWorkingSetMb = static_cast<double>(cut.longTileLen) * mmModel.mmRankMul *
                                     static_cast<double>(mmInput.kValue) * mmInput.mmDtypeBytes / (1024.0 * 1024.0);
        if ((halvingCandidate && nTiles > HALVE_NTILES_LIMIT) ||
            (bBytesMbL2 > L2_BYTES_MB && aWorkingSetMb > A_WS_LIMIT_MB)) {
            pipelineCorrectedMm = totalMmTime * MM_PENALTY_COEF;
        }
    }
    const double totalCommTimeNoCut = TotalCommTime(comm, mmInput.mValue, 1U);
    const double serialTotalTime = totalCommTimeNoCut + pipelineCorrectedMm;
    // 脆弱带: B 字节量贴近 L2 容量时驻留余量不足, 通信写入挤出 B → 重叠退化。
    // 边界由 L2 容量推导(下界=L2/2, 上界=L2×1.2, 全失效分界=L2×0.84);
    // 仅 M 轴聚合形态生效——K 轴散射形态的写入模式不挤出 B
    constexpr double FRAGILE_BAND_L2_MB = 32.0;
    const double bBytesMb = static_cast<double>(mmInput.nValue) * static_cast<double>(mmInput.kValue) *
                            mmInput.mmDtypeBytes / (1024.0 * 1024.0);
    const bool inFragileBand =
        flow.perRankM && bBytesMb >= FRAGILE_BAND_L2_MB / 2.0 && bBytesMb <= FRAGILE_BAND_L2_MB * 1.2;
    const bool inBrokenBand = inFragileBand && bBytesMb >= FRAGILE_BAND_L2_MB * 0.84;
    if (cut.totalTileCnt <= 1U) {
        if (flow.localMMPosition == LocalMMPosition::FRONT) {
            double noCutCredit = std::min(mmModel.localShareTime, totalCommTimeNoCut);
            if (inFragileBand) {
                noCutCredit = 0.0;
            }
            return serialTotalTime - noCutCredit;
        }
        return serialTotalTime;
    }
    const uint64_t mValue = mmInput.mValue;
    const bool quantizeTail = (comm.commVolumeMul > 1U); // M 轴聚合族限定(与细向下降同域)
    const double totalCommTime = TotalCommTime(comm, mValue, cut.totalTileCnt);
    const uint64_t firstLen = (cut.numShortTile > 0U && !cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
    const uint64_t lastLen = (cut.numShortTile > 0U && cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
    // 轮次碎片化暴露: 单轮数据时间 < 阈值×每轮开销 时轮启动开销无法与 mm 重叠,
    // b×(轮数-1) 串行累加(K 轴族细切场景)
    auto TurnFragExpose = [&flow, &comm, &mmInput, &cut](double firstTurnDataTime) {
        return ((!flow.perRankM && cut.totalTileCnt > 1U && firstTurnDataTime < comm.roundOverheadUs)) ?
                   comm.roundOverheadUs * static_cast<double>(cut.totalTileCnt - 1U) :
                   0.0;
    };
    if (flow.localMMPosition == LocalMMPosition::FRONT) {
        const double commFirst = comm.fixedOverheadUs + CommTime(comm, firstLen);
        // mmLast 只计 remote 份(rankMul-1): 本地份已在 head 的 localMM 中算过, 全量
        // rankMul 口径会重复计尾块的本地行(粗切尾块大 → 重复计多 → est 虚高偏细切)
        const double mmLast = ((mmModel.mmRankMul > 1U) ?
                                   QuantizedTailTime(mmInput, mmModel, lastLen, mmModel.mmRankMul - 1U, quantizeTail) :
                                   QuantizedTailTime(mmInput, mmModel, lastLen, mmModel.mmRankMul, quantizeTail));
        const double localMMScaled = mmModel.localMMTime;
        const double headTime = std::max(commFirst, localMMScaled);
        const double mmRestTime = pipelineCorrectedMm - localMMScaled - mmLast;
        const double fragExpose = TurnFragExpose(CommTime(comm, firstLen) - comm.roundOverheadUs);
        if (inBrokenBand) {
            // 全失效域: 全量 mm + β×comm, β 按首块占比分档(近平局/均衡二块/细切)
            const double firstFrac =
                (mmInput.mValue > 0U) ? static_cast<double>(firstLen) / static_cast<double>(mmInput.mValue) : 1.0;
            const double beta = (firstFrac >= 0.75) ? 1.14 : ((firstFrac >= 0.4) ? 0.83 : 1.21);
            return pipelineCorrectedMm + totalCommTime * beta + fragExpose;
        }
        if (inFragileBand) {
            return headTime + (totalCommTime - commFirst) + mmRestTime + mmLast + fragExpose;
        }
        return headTime + std::max(totalCommTime - headTime, mmRestTime) + mmLast + fragExpose;
    }
    if (flow.localMMPosition == LocalMMPosition::BACK) {
        const double mmFirst = MatmulTime(mmInput, mmModel, firstLen, mmModel.mmRankMul);
        const double commLast = CommTime(comm, lastLen);
        const double mmRest = totalMmTime - mmFirst - mmModel.localShareTime;
        return mmFirst + std::max(mmRest, totalCommTime - commLast) + std::max(commLast, mmModel.localShareTime);
    }
    if (flow.order == CommComputeOrder::COMM_FIRST) {
        const double commFirst = comm.fixedOverheadUs + CommTime(comm, firstLen);
        const double mmLast = QuantizedTailTime(mmInput, mmModel, lastLen, mmModel.mmRankMul, quantizeTail);
        const double fragExpose = TurnFragExpose(CommTime(comm, firstLen) - comm.roundOverheadUs);
        if (inBrokenBand) {
            const double firstFrac =
                (mmInput.mValue > 0U) ? static_cast<double>(firstLen) / static_cast<double>(mmInput.mValue) : 1.0;
            const double beta = (firstFrac >= 0.75) ? 1.14 : ((firstFrac >= 0.4) ? 0.83 : 1.21);
            return pipelineCorrectedMm + totalCommTime * beta + fragExpose;
        }
        if (inFragileBand) {
            return commFirst + (totalCommTime - commFirst) + (pipelineCorrectedMm - mmLast) + mmLast + fragExpose;
        }
        return commFirst + std::max(totalCommTime - commFirst, pipelineCorrectedMm - mmLast) + mmLast + fragExpose;
    }
    const double mmFirst = MatmulTime(mmInput, mmModel, firstLen, mmModel.mmRankMul);
    const double commLast = CommTime(comm, lastLen);
    return mmFirst + std::max(totalMmTime - mmFirst, totalCommTime - commLast) + commLast;
}

// 细向下降: 长块粒度按对齐粒度递减, est 严格变低才继续(平局即停), 变高或触及
// 家族长块下界/轮数上限即停。
constexpr double CUT_DESCENT_EPS_US = 1e-9; // est(µs)严格改善判定容差(代数平坦域fp噪声量级)
inline double DescendToFiner(const FitBalanceFlow& flow, const MatmulInputParams& mmInput,
                             const MatmulModelParams& mmModel, const CommModelParams& comm, CommCutResult& cut)
{
    if (cut.numLongTile == 0U) {
        return 0.0;
    }
    // 下降限定 M 轴聚合族: K 轴族的 est 粒度排序未经验证(实测方向有误), 不下探
    if (comm.commVolumeMul <= 1U) {
        return EstimateCutTotalTime(flow, mmInput, mmModel, comm, cut);
    }
    const uint64_t alignLen = mmInput.baseM != 0UL ? mmInput.baseM : static_cast<uint64_t>(M_ALIGN_GRANULARITY);
    // 家族长块下界, 与 MakeCutPlanner 同口径(M 轴聚合族 1 粒度, K 轴族 2 粒度)
    const uint64_t minLongLen = (comm.commVolumeMul > 1U ? 1U : 2U) * alignLen;
    double bestTime = EstimateCutTotalTime(flow, mmInput, mmModel, comm, cut);
    while (cut.longTileLen >= minLongLen + alignLen) {
        const uint64_t newLen = cut.longTileLen - alignLen;
        CommCutResult cand{};
        cand.shortTileAtBack = cut.shortTileAtBack;
        cand.numLongTile = mmInput.mValue / newLen;
        if (cand.numLongTile == 0U) {
            break;
        }
        cand.longTileLen = newLen;
        cand.shortTileLen = mmInput.mValue - cand.numLongTile * newLen;
        cand.numShortTile = (cand.shortTileLen > 0UL) ? 1UL : 0UL;
        cand.totalTileCnt = cand.numLongTile + cand.numShortTile;
        if (cand.totalTileCnt > MAX_COMM_ROUNDS) {
            break;
        }
        const double candTime = EstimateCutTotalTime(flow, mmInput, mmModel, comm, cand);
        if (candTime < bestTime - CUT_DESCENT_EPS_US) {
            cut = cand;
            bestTime = candTime;
        } else {
            break;
        }
    }
    return bestTime;
}

// Q阶梯全扫: 每个Q生成 (N x Q·baseM + 尾) 均匀族候选, est统一裁判, 平局取粗
inline CommCutResult QLadderSweep(const FitBalanceFlow& flow, const MatmulInputParams& mmInput,
                                  const MatmulModelParams& mmModel, const CommModelParams& comm,
                                  const uint64_t minLongLen)
{
    const uint64_t alignLen = (mmInput.baseM != 0UL) ? mmInput.baseM : static_cast<uint64_t>(M_ALIGN_GRANULARITY);
    const uint64_t mValue = mmInput.mValue;
    CommCutResult bestCut{};
    double bestTime = 0.0;
    bool hasBest = false;
    const uint64_t qMax = mValue / alignLen; // Q上限: 长块不超过m(单块)
    // Q下限: 轮数上限约束 numLong <= Nmax → Q >= m/(Nmax×align), 更小的候选必被轮数检查拒绝
    const uint64_t qMin = std::max((mValue + MAX_COMM_ROUNDS * alignLen - 1UL) / (MAX_COMM_ROUNDS * alignLen), 1UL);
    for (uint64_t q = qMin; q <= qMax; ++q) {
        const uint64_t longLen = q * alignLen;
        const uint64_t numLong = mValue / longLen;
        if (numLong == 0UL) {
            continue;
        }
        if (numLong == 1UL && (mValue - longLen) == 0UL) {
            continue; // NoCut单块形态排除: 单块est分支(serial-credit)与多块不同构, NoCut由串行基准判定
        }
        CommCutResult cand{};
        cand.shortTileAtBack = true;
        cand.numLongTile = numLong;
        cand.longTileLen = longLen;
        cand.shortTileLen = mValue - numLong * longLen;
        cand.numShortTile = (cand.shortTileLen > 0UL) ? 1UL : 0UL;
        cand.totalTileCnt = cand.numLongTile + cand.numShortTile;
        if (cand.totalTileCnt > MAX_COMM_ROUNDS) {
            continue; // 轮数上限跳过
        }
        if (longLen < minLongLen && numLong > 1UL) {
            continue; // 家族长块下界(K轴族2粒度等)
        }
        const double candTime = EstimateCutTotalTime(flow, mmInput, mmModel, comm, cand);
        if (!hasBest || candTime < bestTime) { // 平局保先者(最细Q): 保住减半域的细粒度偏好
            bestCut = cand;
            bestTime = candTime;
            hasBest = true;
        }
    }
    return bestCut;
}

// 离散化 + 轮数搜索 + 切分收益判定: 配平种子先经 [4] 离散化整形出首个候选,
// 再线性枚举轮数预算 [2, N₀) 择优(种子恢复点重切); 总通信时长下界早停;
// 全部候选的预估耗时都比串行高时不切
inline void SelectBestCut(const FitBalanceFlow& flow, const MatmulInputParams& mmInput,
                          const MatmulModelParams& mmModel, const CommModelParams& comm, CutPlanner& tilingM)
{
    const bool localBlockProtected =
        (flow.localMMPosition == LocalMMPosition::FRONT) || (flow.localMMPosition == LocalMMPosition::BACK);
    const bool goodLinearity = localBlockProtected ? true : IsGoodLinearityShape(mmInput);
    if (comm.commVolumeMul >
        1U) { // M轴聚合族: Q阶梯全扫(种子+预算+下降+COF对比四机制归一); K轴族est(b+γ)不可信走原搜索
        // 原型路径: 单条Q阶梯全扫(种子仅用于家族下界口径), NoCut判定沿用串行基准
        const uint64_t alignLen = (mmInput.baseM != 0UL) ? mmInput.baseM : static_cast<uint64_t>(M_ALIGN_GRANULARITY);
        const uint64_t minLongLen = (comm.commVolumeMul > 1U ? 1U : 2U) * alignLen; // 与 MakeCutPlanner 同口径
        const CommCutResult swept = QLadderSweep(flow, mmInput, mmModel, comm, minLongLen);
        const double totalMmTime = MatmulTime(mmInput, mmModel, mmInput.mValue, mmModel.mmRankMul);
        const double serialTotalTime = TotalCommTime(comm, mmInput.mValue, 1U) + totalMmTime;
        if (swept.totalTileCnt == 0U || EstimateCutTotalTime(flow, mmInput, mmModel, comm, swept) >= serialTotalTime) {
            tilingM.NoCutTiling();
        } else {
            tilingM.cutRes = swept;
        }
        return;
    }
    // 配平种子(轮数搜索的恢复点): 长块理想值 + 短块下限 + 位置; 块数尚未离散化
    const CommCutResult balancedSeed = tilingM.cutRes;
    tilingM.FitTileLengthDiscrete(goodLinearity, localBlockProtected);
    MergeEqualTiles(tilingM);

    const uint64_t mValue = mmInput.mValue;
    const double totalMmTime = MatmulTime(mmInput, mmModel, mValue, mmModel.mmRankMul);
    const double serialTotalTime = TotalCommTime(comm, mValue, 1U) + totalMmTime;
    CommCutResult bestCut = tilingM.cutRes;
    double bestTime = EstimateCutTotalTime(flow, mmInput, mmModel, comm, bestCut);
    const uint64_t initialTileCnt = bestCut.totalTileCnt;
    for (uint64_t budget = 2U; budget < initialTileCnt; ++budget) {
        tilingM.cutRes = balancedSeed;
        tilingM.SetMaxTileCnt(budget);
        tilingM.FitTileLengthDiscrete(goodLinearity, localBlockProtected);
        MergeEqualTiles(tilingM);
        const double cutTime = EstimateCutTotalTime(flow, mmInput, mmModel, comm, tilingM.cutRes);
        if (cutTime < bestTime) {
            bestTime = cutTime;
            bestCut = tilingM.cutRes;
        }
        // 早停: 总通信时长(数据总量与每轮启动开销刚性, 重叠不可消除)是同块数方案的耗时下界,
        // 且实际块数随预算单调不降——下界追上当前最优后, 后续候选不可能严格更优
        if (TotalCommTime(comm, mValue, tilingM.cutRes.totalTileCnt) >= bestTime) {
            break;
        }
    }
    tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS);
    if (bestTime >= serialTotalTime) {
        tilingM.NoCutTiling();
    } else {
        tilingM.cutRes = bestCut;
    }
}

constexpr uint64_t MAX_ROUND_COMM_BYTES = 268435456; // HCCL 单次调用通信量上限 (256MB)
constexpr uint64_t CANDIDATE_ALIGN = 256;            // 候选全超限时的兜底对齐粒度

// 单轮受限块长选择: 等长重切候选表(大值优先=少轮次), 轮次预算反选;
// 256MB 上限与轮次预算冲突时返回 0(放弃整形)
inline uint64_t RoundLimitTileLen(const uint64_t mValue, const uint64_t maxTileLen)
{
    constexpr uint64_t TILE_LEN_CANDIDATES[] = {2048, 1024, 512, 256};
    // 优选候选(降序第一个 ≤ maxTileLen); 全部超限则 floor 对齐兜底
    uint64_t tileLen = 0;
    for (const uint64_t cand : TILE_LEN_CANDIDATES) {
        if (cand <= maxTileLen) {
            tileLen = cand;
            break;
        }
    }
    if (tileLen == 0) {
        tileLen = maxTileLen / CANDIDATE_ALIGN * CANDIDATE_ALIGN;
    }
    // 轮次预算: 候选块太小块数超预算时, 反向取满足预算的最小块
    const uint64_t roundsBudget = MAX_COMM_ROUNDS - 1;
    if ((mValue + tileLen - 1) / tileLen > roundsBudget) {
        const uint64_t minLen =
            ((mValue + roundsBudget - 1) / roundsBudget + CANDIDATE_ALIGN - 1) / CANDIDATE_ALIGN * CANDIDATE_ALIGN;
        uint64_t pick = minLen;
        for (const uint64_t cand : TILE_LEN_CANDIDATES) {
            if (cand >= minLen) {
                pick = cand;
                break;
            }
        }
        if (pick > maxTileLen) {
            return 0; // 256MB 上限与轮次预算冲突——放弃整形
        }
        tileLen = pick;
    }
    return tileLen;
}

// 单轮通信量上限整形(CCU 专用): HCCL 单次调用 256MB 上限, 超限块等长重切(尾块吸收余数,
// 短块位置保持); 未超限/单行通信量过大/两约束(256MB 与轮次预算)冲突时放弃整形
inline void ApplySingleRoundCommLimit(CommCutResult& cutRes, const MatmulInputParams& mmInput,
                                      const CommModelParams& comm)
{
    // 单行通信字节数(须含份数 volumeMul, 与 k 的标定口径无关)
    const uint64_t mValue = mmInput.mValue;
    const uint64_t rowBytes =
        comm.commMatrixLen * comm.commDtypeSize * comm.commVolumeMul / comm.commDtypeSizeExpansionFraction;
    if (rowBytes == 0) {
        return;
    }
    if (cutRes.longTileLen * rowBytes <= MAX_ROUND_COMM_BYTES) {
        return; // 未超限: 保持长短配平结构
    }
    const uint64_t maxTileLen = MAX_ROUND_COMM_BYTES / rowBytes;
    if (maxTileLen < CANDIDATE_ALIGN) {
        return; // 单行通信量过大, 任何对齐块都超限——放弃整形
    }
    const uint64_t tileLen = RoundLimitTileLen(mValue, maxTileLen);
    if (tileLen == 0) {
        return;
    }
    // 等长重切: 主块等长 + 尾块余数, 短块位置保持
    cutRes.longTileLen = tileLen;
    cutRes.numLongTile = mValue / tileLen;
    cutRes.shortTileLen = mValue - cutRes.numLongTile * tileLen;
    cutRes.numShortTile = (cutRes.shortTileLen > 0) ? 1 : 0;
    cutRes.totalTileCnt = cutRes.numLongTile + cutRes.numShortTile;
}
