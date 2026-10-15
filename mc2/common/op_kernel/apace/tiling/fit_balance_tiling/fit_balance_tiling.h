/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file fit_balance_tiling.h
 * \brief fit-balance tiling 公共引擎——对外总接口头文件
 *
 * 通信-计算流水线配平切分引擎
 * 算子侧接入：CALIBRATION 实例 + 枚举入参 + 一行调用 FitBalanceCut。
 *
 * 分文件布局(消费方只 include 本头文件):
 *   fit_balance_tiling_utils/fit_balance_types.h  公共类型/枚举/常量/标定结构
 *   fit_balance_tiling_utils/comm_model.h         通信模型 y = k·x + 轮数·b + c
 *   fit_balance_tiling_utils/matmul_model.h       计算模型(利用率/梯度)
 *   fit_balance_tiling_utils/cut_planner.h        重切整形
 *   fit_balance_tiling_utils/solver.h             长短块求解
 */

#ifndef __APACE_FIT_BALANCE_TILING_H__
#define __APACE_FIT_BALANCE_TILING_H__

#pragma once

#include "fit_balance_tiling_utils/fit_balance_types.h"
#include "fit_balance_tiling_utils/comm_model.h"
#include "fit_balance_tiling_utils/matmul_model.h"
#include "fit_balance_tiling_utils/cut_planner.h"
#include "fit_balance_tiling_utils/solver.h"
#include "../core_occupancy_first_tiling.h"
#include "../tiling_data_struct_define.h"

// 通信标定行查找: 按 (通信引擎, rankDim) 匹配首个命中行, miss 返回 nullptr。
// rankDim=0 为未填行哨兵; 零斜率行同样按未填处理。
inline const FitBalanceCalibration::CommFitRow* FindCommFitRow(const FitBalanceCalibration& calib,
                                                               TilingCommEngine commEngine, uint32_t rankDim)
{
    for (size_t i = 0; i < FitBalanceCalibration::MAX_COMM_FIT_ROWS; ++i) {
        const auto& row = calib.commRows[i];
        if (row.rankDim == 0) {
            break;
        }
        if (row.commEngine == commEngine && row.rankDim == rankDim && row.slopeUsPerMb != 0.0) {
            return &row;
        }
    }
    return nullptr;
}

inline bool IsCommCalibrated(uint32_t rankDim, TilingCommEngine commEngine, const FitBalanceCalibration& calib)
{
    return FindCommFitRow(calib, commEngine, rankDim) != nullptr;
}

// 切分器构造: 对齐粒度/短块下限基准取 baseM; 标定 miss(slope=0)时下限抬到全量 mValue,
// 按不切分处理(兼防 InverseCommTime 除零)。
inline CutPlanner MakeCutPlanner(const FitBalanceTilingArgs& args, const MatmulInputParams& mmInput,
                                 const CommModelParams& comm)
{
    CutPlanner tilingM(args);
    tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS);
    if (comm.slopeUsPerMb == 0.0) {
        tilingM.SetMinLenByMax(mmInput.mValue);
    }
    tilingM.SetMinLenByMax(mmInput.baseM);
    tilingM.SetAlignLength(mmInput.baseM);
    return tilingM;
}

// ============================================================================
// 求解主流程 Schedule: 六段流水
//   [1] Estimate: 总量估计, 决定短块位置
//   [2] SetShort: 设置短块下限
//   [3] SetLong:  长块配平反解
//   [4] Adjust:   离散化整形
//   [5] 轮数搜索: 切分收益判定, 选择最优切分轮次
//   [6] CCU 整形: 单轮 256MB 上限
// ============================================================================

// 核利用率优先候选: 波量纯几何切分(DeriveCommCut), 映射与 CutResultFromCof 一致
// (long = 头块波 x 块数, short = 尾块)。usedCoreNum 取可用 AIC 核数(全量 COF 由
// matmul tiling 决定核数, 此处为引擎侧近似)。推导失败返回零值(单块, 估时退化为
// serial 参与比时)。短块位置同 CutResultFromCof 固定后置。
inline CommCutResult MakeCoreOccupancyCut(const MatmulInputParams& mmInput)
{
    CommCutResult cut{};
    uint64_t headMSize = 0;
    uint64_t headTileCnt = 0;
    uint64_t tailMSize = 0;
    uint64_t tailTileCnt = 0;
    if (!DeriveCommCut(mmInput.mValue, mmInput.nValue, mmInput.baseM, mmInput.baseN, mmInput.coreNum, headMSize,
                       headTileCnt, tailMSize, tailTileCnt)) {
        return cut;
    }
    cut.shortTileAtBack = true;
    cut.longTileLen = headMSize;
    cut.numLongTile = headTileCnt;
    cut.shortTileLen = tailMSize;
    cut.numShortTile = tailTileCnt;
    cut.totalTileCnt = headTileCnt + tailTileCnt;
    return cut;
}

// 键控维度: LocalMMPosition——本地 mm 块在流水线中的位置(不经通信的数据块)。
//   FRONT: 本地块在流水线首端(本卡数据提前计算), 短块退到尾部收尾;
//   BACK:  本地块在流水线尾端(本卡数据滞后计算), 短块前置启动通信;
//   NONE:  无本地块, 短块按 bound 判定放首/尾。
inline CommCutResult Schedule(const FitBalanceFlow& flow, const MatmulInputParams& mmInput,
                              const MatmulModelParams& mmModel, const CommModelParams& comm, CutPlanner& tilingM)
{
    const bool localBlockProtected =
        (flow.localMMPosition == LocalMMPosition::FRONT) || (flow.localMMPosition == LocalMMPosition::BACK);

    // ---- [1] Estimate: 总量估计 + 短块位置判定 ----
    const auto estimate = EstimateTotalTimes(mmInput, mmModel, comm);
    tilingM.cutRes.shortTileAtBack = ChooseShortTileAtBack(flow, estimate);

    // ---- [2] SetShort: 设置短块下限 (由每轮通信启动开销反解——短块独立占一轮通信,
    //      其通信时长须大于等于两倍的每轮启动开销才值得成轮) ----
    tilingM.SetMinLenByMax(InverseCommTime(comm, 2.0 * comm.roundOverheadUs));
    tilingM.cutRes.shortTileLen = tilingM.GetMinLen();
    tilingM.cutRes.numShortTile = 1U;
    // mValue 过小则不切
    if (tilingM.totalLen < tilingM.cutRes.shortTileLen * 2) {
        tilingM.NoCutTiling();
        return tilingM.cutRes;
    }

    // ---- [3] SetLong: 长块配平 ----
    if (localBlockProtected) {
        // 长块的通信恰好掩盖本地块计算(FRONT 首长块掩本地首块 / BACK 末长块掩本地尾块, 同一算式)
        tilingM.cutRes.longTileLen = InverseCommTime(comm, mmModel.localMMTime);
    } else {
        // 长块的数据量与短块的预估耗时配平
        BalanceLongTileLen(flow, mmInput, mmModel, comm, tilingM);
    }
    // ---- [4] Adjust + [5] 轮数搜索: 离散化整形 → 枚举轮数择优 → 切分收益判定 ----
    SelectBestCut(flow, mmInput, mmModel, comm, tilingM);

    // ---- [6] CCU API 上限整形: HCCL 单次调用 256MB 上限 ----
    if (flow.commEngine == TilingCommEngine::CCU) {
        ApplySingleRoundCommLimit(tilingM.cutRes, mmInput, comm);
    }

    return tilingM.cutRes;
}

// 核利用率优先切分结果 → 本引擎切分结果转换。
// 标定 miss 走核利用率优先切分兜底时, 用它统一输出形态。
inline CommCutResult CutResultFromCof(const CommTilingData& data)
{
    CommCutResult result;
    result.shortTileAtBack = true;
    result.longTileLen = data.splitAxisTileSize;
    result.numLongTile = data.splitAxisTileCnt;
    result.shortTileLen = data.splitAxisTailSize;
    result.numShortTile = data.splitAxisTailCnt;
    result.totalTileCnt = data.splitAxisTileCnt + data.splitAxisTailCnt;
    return result;
}

// 旧入参转换(存量算子过渡用): 从旧全局 mc2tiling::TilingArgs 挑字段构造引擎入参;
// commEngine 由调用方解析后补入。新算子直接构造 FitBalanceTilingArgs。
template <typename LegacyArgs>
inline FitBalanceTilingArgs MakeFitBalanceArgs(const LegacyArgs& legacyArgs, TilingCommEngine commEngine)
{
    FitBalanceTilingArgs engineArgs = {};
    engineArgs.mValue = legacyArgs.mValue;
    engineArgs.nValue = legacyArgs.nValue;
    engineArgs.kValue = legacyArgs.kValue;
    engineArgs.rankDim = legacyArgs.rankDim;
    engineArgs.aicCoreNum = legacyArgs.aicCoreNum;
    engineArgs.geAType = legacyArgs.geAType;
    engineArgs.inputADtypeSize = legacyArgs.inputDtypeSize;
    engineArgs.outputDtypeSize = legacyArgs.outputDtypeSize;
    engineArgs.commEngine = commEngine;
    return engineArgs;
}
// 标定组装(多形态/动态标定算子用): dataflow / 通信拟合行 / matmul 拟合三轴分开维护,
// 调用处一行组装。通信行为不随形态变化的算子三份表里共享同一份行表; 未填的通信行
// 保持哨兵(rankDim = 0), 行数超上限在编译期拦截。mmFit 条目不填的项走默认值。
template <size_t COMM_ROW_COUNT>
inline constexpr FitBalanceCalibration MakeFitBalanceCalibration(
    const FitBalanceCalibration::DataflowShape& dataflow,
    const FitBalanceCalibration::CommFitRow (&commRows)[COMM_ROW_COUNT], const FitBalanceCalibration::MatmulFit& mmFit)
{
    static_assert(COMM_ROW_COUNT <= FitBalanceCalibration::MAX_COMM_FIT_ROWS, "commRows exceed row limit");
    FitBalanceCalibration calib{};
    calib.dataflow = dataflow;
    for (size_t i = 0; i < COMM_ROW_COUNT; ++i) {
        calib.commRows[i] = commRows[i];
    }
    calib.mmFit = mmFit;
    return calib;
}

// mm 输入事实: 形状/基本块/字节/核数(由入参与标定基本块构成)
inline MatmulInputParams MakeMatmulInputParams(const FitBalanceTilingArgs& args, const FitBalanceCalibration& calib)
{
    MatmulInputParams mmInput{};
    mmInput.coreNum = args.aicCoreNum;
    mmInput.inMatrixADtypeSize = args.inputADtypeSize;
    mmInput.inMatrixBDtypeSize = args.inputADtypeSize;
    mmInput.outMatrixCDtypeSize = args.outputDtypeSize;
    mmInput.mValue = args.mValue;
    mmInput.nValue = args.nValue;
    mmInput.kValue = args.kValue;
    mmInput.baseM = calib.mmFit.baseBlock.baseM;
    mmInput.baseN = calib.mmFit.baseBlock.baseN;
    mmInput.baseK = calib.mmFit.baseBlock.baseK;
    // Matmul fp4 场景 L2 估算按 1B/元素保守取值(通信侧 fp4 分支自取 1, 不读此字段, 先后无影响)
    if (args.geAType == ge::DataType::DT_FLOAT4_E2M1) {
        mmInput.inMatrixADtypeSize = 1;
        mmInput.inMatrixBDtypeSize = 1;
    }
    return mmInput;
}

// 有效算力: 覆盖值优先(算子侧按自己的运行时属性选定, 依据不限 dtype);
// 缺省走公式——总算力/算术字节(950: fp16 4096 / fp8 8192 / fp4 16384)。
// fp4 打包 2 元素/字节(0.5B/元素), 字节数查询对 fp4 返回 0 需特判
inline double ResolveComputesPerCycle(const FitBalanceTilingArgs& args, const FitBalanceCalibration& calib)
{
    if (calib.mmFit.computesPerCycleOverride > 0.0) {
        return calib.mmFit.computesPerCycleOverride;
    }
    const double arithBytes =
        (args.geAType == ge::DataType::DT_FLOAT4_E2M1) ? 0.5 : static_cast<double>(args.inputADtypeSize);
    // 算术字节缺失(非正常入参)时退化为总算力值, 避免除零
    return (arithBytes > 0.0) ? (calib.mmFit.totalComputesPerCycle / arithBytes) : calib.mmFit.totalComputesPerCycle;
}

// 通信模型参数: 标定行 + 通信形态(矩阵边长取通信承载侧——K 轴=A 输入侧, N 轴=C 输出侧, 字节同源)
// + perRankM 倍数(true: mValue 为每卡份 → ×rankDim; false: 已是全量 → ×1)
inline CommModelParams MakeCommModelParams(const FitBalanceTilingArgs& args, const FitBalanceCalibration& calib,
                                           const CommComputeOrder order, const MatmulInputParams& mmInput,
                                           const uint64_t rankDim)
{
    CommModelParams comm{};
    const auto* commRow = FindCommFitRow(calib, args.commEngine, args.rankDim);
    if (commRow != nullptr) {
        comm.slopeUsPerMb = commRow->slopeUsPerMb;
        comm.roundOverheadUs = commRow->roundOverheadUs;
        comm.fixedOverheadUs = commRow->fixedOverheadUs;
    }
    const bool commOnKAxis = (order == CommComputeOrder::COMM_FIRST);
    comm.commMatrixLen = commOnKAxis ? mmInput.kValue : mmInput.nValue;
    const bool isFp4Input = (args.geAType == ge::DataType::DT_FLOAT4_E2M1);
    if (commOnKAxis) {
        if (isFp4Input) {
            // fp4(0.5B/元素)无法用整数字节表达: 按 1 字节打包 2 元素建模(dtypeSize=1 + fraction=2)
            comm.commDtypeSize = 1;
            comm.commDtypeSizeExpansionFraction = FP4_PACK_RATIO;
        } else {
            comm.commDtypeSize = mmInput.inMatrixADtypeSize;
        }
    } else {
        comm.commDtypeSize = mmInput.outMatrixCDtypeSize;
    }
    comm.commVolumeMul = calib.dataflow.perRankM ? rankDim : 1U;
    return comm;
}

// mm 定参: 梯度与本地块时长一次算清——此后公式对入参纯读
inline void FinalizeMmModelParams(const MatmulInputParams& mmInput, MatmulModelParams& mmModel, const uint64_t shares)
{
    mmModel.matmulGradient = CalcMatmulGradient(mmInput, mmModel, shares);
    // 本地块计算时长。更小块的本地 Matmul 预期利用率会更低
    const double localMMUtil =
        FindCubeUtil(mmInput, mmInput.mValue, 1, MatmulPerfConst::LOCAL_MM_UTIL_EXPONENT_COEFFICIENT);
    // 除零/NaN 防护: 退化 shape(某维为 0)下 util 可为 0 或 NaN, 退回梯度口径估时
    mmModel.localMMTime = (std::abs(localMMUtil) > 1e-10) ?
                              MatmulTime(mmInput, mmModel, mmInput.mValue, 1) / localMMUtil :
                              MatmulTime(mmInput, mmModel, mmInput.mValue, 1);
}

// 对外唯一入口: 四件构造 → 切分器 → 求解, 顺序读完
inline CommCutResult FitBalanceCut(const FitBalanceTilingArgs& args, const FitBalanceCalibration& calib)
{
    const CommComputeOrder order = calib.dataflow.order;
    const LocalMMPosition localMMPosition = calib.dataflow.localMMPosition;
    const uint64_t rankDim = std::max(static_cast<uint64_t>(args.rankDim), MIN_RANK_DIM);

    MatmulInputParams mmInput = MakeMatmulInputParams(args, calib);
    MatmulModelParams mmModel{};
    mmModel.computesPerCycle = ResolveComputesPerCycle(args, calib);
    mmModel.cyclePerMicroSec = calib.mmFit.cyclePerMicroSec;
    const CommModelParams comm = MakeCommModelParams(args, calib, order, mmInput, rankDim);
    mmModel.mmRankMul = comm.commVolumeMul;
    // 利用率按同批计算的数据份数取值(FRONT 有一个 rank 的数据不经通信 → rankDim−1)
    const uint64_t shares = (localMMPosition == LocalMMPosition::FRONT) ? (rankDim - 1U) : comm.commVolumeMul;
    FinalizeMmModelParams(mmInput, mmModel, shares);

    const FitBalanceFlow flow{localMMPosition, order, args.commEngine};
    CutPlanner tilingM = MakeCutPlanner(args, mmInput, comm);
    CommCutResult cut = Schedule(flow, mmInput, mmModel, comm, tilingM);

    // 核利用率优先切分与Fit Balance切分择优返回
    CommCutResult occupancyCut = MakeCoreOccupancyCut(mmInput);
    if (flow.commEngine == TilingCommEngine::CCU) {
        ApplySingleRoundCommLimit(occupancyCut, mmInput, comm);
    }
    if (EstimateCutTotalTime(flow, mmInput, mmModel, comm, occupancyCut) <
        EstimateCutTotalTime(flow, mmInput, mmModel, comm, cut)) {
        cut = occupancyCut;
    }
    return cut;
}
#endif
