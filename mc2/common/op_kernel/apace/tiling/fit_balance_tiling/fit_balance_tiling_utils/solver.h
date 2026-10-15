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
// 无本地块时按 bound 方向选择边界配对方向(注释见各分支)
inline bool ChooseShortTileAtBack(const FitBalanceFlow& flow, const EstimateTimesResult& estimate)
{
    if (flow.localMMPosition == LocalMMPosition::FRONT) {
        return true;
    }
    if (flow.localMMPosition == LocalMMPosition::BACK) {
        return false;
    }
    if (flow.order == CommComputeOrder::COMPUTE_FIRST) {
        // 计算 Bound 场景下，短块放在后面，这样最后一块计算（短）和倒数第二块通信（长）可以互相掩盖
        // 通信 Bound 场景下，短块放在前面，这样第二块计算（长）和第一块通信（短）可以互相掩盖
        return estimate.totalMatmulTime >= estimate.totalCommTimeNoCut;
    }
    // 通信 Bound 场景下，短块放在后面，这样最后一块通信（短）和倒数第二块计算（长）可以互相掩盖
    // 计算 Bound 场景下，短块放在前面，这样第二块通信（长）和第一块计算（短）可以互相掩盖
    return estimate.totalCommTimeNoCut > estimate.totalMatmulTime;
}

// 流水线总时估计(切 N 轮的近似): 不可重叠段在流水线两端, 中段取瓶颈侧。
// 单块(含 NoCut 候选)返回 serialTotalTime。
// totalMmTime/serialTotalTime 由调用方一次算好传入(搜索循环内不变)。
inline double EstimateCutTotalTime(const FitBalanceFlow& flow, const MatmulInputParams& mmInput,
                                   const MatmulModelParams& mmModel, const CommModelParams& comm,
                                   const CommCutResult& cut)
{
    const double totalMmTime = MatmulTime(mmInput, mmModel, mmInput.mValue, mmModel.mmRankMul);
    const double serialTotalTime = TotalCommTime(comm, mmInput.mValue, 1U) + totalMmTime;
    if (cut.totalTileCnt <= 1U) {
        return serialTotalTime;
    }
    const uint64_t mValue = mmInput.mValue;
    const double totalCommTime = TotalCommTime(comm, mValue, cut.totalTileCnt);
    const uint64_t firstLen = (cut.numShortTile > 0U && !cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
    const uint64_t lastLen = (cut.numShortTile > 0U && cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
    if (flow.localMMPosition == LocalMMPosition::FRONT) {
        // 通信在前 + 本地首块: 首轮通信(含算子级一次性启动开销, 通信链头部无处可藏)不可重叠, 末段计算收尾
        const double commFirst = comm.fixedOverheadUs + CommTime(comm, firstLen);
        const double mmLast = MatmulTime(mmInput, mmModel, lastLen, mmModel.mmRankMul);
        return commFirst + std::max(totalCommTime - commFirst, totalMmTime - mmLast) + mmLast;
    }
    if (flow.localMMPosition == LocalMMPosition::BACK) {
        // 计算在前 + 本地尾块: 首块计算不可重叠, 末段本地窗口与末块通信并行
        const double mmFirst = MatmulTime(mmInput, mmModel, firstLen, mmModel.mmRankMul);
        const double commLast = CommTime(comm, lastLen);
        const double mmRest = totalMmTime - mmFirst - mmModel.localMMTime;
        return mmFirst + std::max(mmRest, totalCommTime - commLast) + std::max(commLast, mmModel.localMMTime);
    }
    if (flow.order == CommComputeOrder::COMM_FIRST) {
        // 无本地块 + 通信在前: 首轮通信(含算子级一次性启动开销, 通信链头部无处可藏)不可重叠, 末块计算收尾
        const double commFirst = comm.fixedOverheadUs + CommTime(comm, firstLen);
        const double mmLast = MatmulTime(mmInput, mmModel, lastLen, mmModel.mmRankMul);
        return commFirst + std::max(totalCommTime - commFirst, totalMmTime - mmLast) + mmLast;
    }
    // 无本地块 + 计算在前: 首块计算不可重叠, 末块通信收尾
    const double mmFirst = MatmulTime(mmInput, mmModel, firstLen, mmModel.mmRankMul);
    const double commLast = CommTime(comm, lastLen);
    return mmFirst + std::max(totalMmTime - mmFirst, totalCommTime - commLast) + commLast;
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
