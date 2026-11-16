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
 * 算子侧接入：CALIBRATION 实例 + 枚举入参 + 一行调用 GetFitBalanceCut。
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
#include "../matmul_tiling/quant_matmul_tiling.h"
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

// 切分器构造: 短块下限基准取 baseM, 对齐粒度取 256;
// 标定 miss(slope=0)时下限抬到全量 mValue, 按不切分处理(防止 InverseCommTime 除零)。
inline CutPlanner MakeCutPlanner(const FitBalanceTilingArgs& args, const MatmulInputParams& mmInput,
                                 const CommModelParams& comm)
{
    CutPlanner tilingM(args);
    tilingM.SetMaxTileCnt(MAX_COMM_ROUNDS);
    if (comm.slopeUsPerMb == 0.0) {
        tilingM.SetMinLenByMax(mmInput.mValue);
    }
    tilingM.SetMinLenByMax(mmInput.baseM);
    const uint64_t alignLen = mmInput.baseM != 0UL ? mmInput.baseM : static_cast<uint64_t>(M_ALIGN_GRANULARITY);
    // 对齐粒度跟随 matmul baseM(减半同步减半); 0 值防御回退常量
    tilingM.SetAlignLength(alignLen);
    // 长块下限: M 轴聚合族(commVolumeMul>1)取 1 个对齐粒度, K 轴族取 2 个
    tilingM.SetMinLongLen((comm.commVolumeMul > 1U ? 1U : 2U) * alignLen);
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

// 核利用率优先候选: 波量纯几何切分(DeriveCommCut), 短块固定后置
// (long = 头块波 x 块数, short = 尾块)。usedCoreNum 取可用 AIC 核数(全量 COF 由
// matmul tiling 决定核数, 此处为引擎侧近似)。推导失败返回零值(单块, 估时退化为
// serial 参与比时)。短块位置固定后置。
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
    tilingM.cutRes.shortTileAtBack = ChooseShortTileAtBack(flow, estimate, mmInput);

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

inline bool MapCutToHeadTail(const CommCutResult& cut, CommTilingData& data)
{
    const bool shortFront = !cut.shortTileAtBack && cut.numShortTile > 0U;
    if (shortFront) {
        data.splitAxisTileSize = cut.shortTileLen;
        data.splitAxisTileCnt = cut.numShortTile;
        data.splitAxisTailSize = cut.longTileLen;
        data.splitAxisTailCnt = cut.numLongTile;
    } else {
        data.splitAxisTileSize = cut.longTileLen;
        data.splitAxisTileCnt = cut.numLongTile;
        data.splitAxisTailSize = cut.shortTileLen;
        data.splitAxisTailCnt = cut.numShortTile;
    }
    return shortFront;
}

// 旧入参转换(存量算子过渡用): 从旧全局 mc2tiling::TilingArgs 挑字段构造引擎入参;
// commEngine 由调用方解析后补入; 新算子可直接构造 FitBalanceTilingArgs。
// 要注意看算子的 kernel 是否支持把短块放在前面, 定好 allowShortFront 的值, 防止
// 模型产生 kernel 不支持的切分方式(默认 false 恒后置)
template <typename LegacyArgs>
inline FitBalanceTilingArgs MakeFitBalanceArgs(const LegacyArgs& legacyArgs, TilingCommEngine commEngine,
                                               bool allowShortFront = false)
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
    engineArgs.allowShortFront = allowShortFront;
    return engineArgs;
}
// 标定组装(多形态/动态标定算子用): dataflow / 通信拟合行 / matmul 拟合三轴分开维护,
// 调用处一行组装。通信行为不随形态变化的算子三份表里共享同一份行表; 未填的通信行
// 保持哨兵(rankDim = 0), 行数超上限在编译期拦截。mmFit 条目不填的项走默认值。
template <size_t COMM_ROW_COUNT>
inline FitBalanceCalibration MakeFitBalanceCalibration(
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
    // 注入 est 物理项参数(来自标定)
    mmInput.elemAtLeast1Byte = (args.geAType != ge::DataType::DT_FLOAT4_E2M1) && (args.inputADtypeSize >= 1UL);
    mmInput.mmDtypeBytes = (args.geAType == ge::DataType::DT_FLOAT4_E2M1) ?
                               0.5 :
                               (args.inputADtypeSize > 0UL ? static_cast<double>(args.inputADtypeSize) : 1.0);
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
// localDiv: 本地块折算分母(本地块=通信前即可计算的部分)。perRankM=false(K 轴通信)时
// mValue 为本卡全量计算, 本地份额 = 1/rankDim; perRankM=true(M 轴聚合)时本地块即
// mValue 行, 不折算。全量口径会使配平长块反解系统性偏长(种子塌缩)
inline void FinalizeMmModelParams(const MatmulInputParams& mmInput, MatmulModelParams& mmModel, const uint64_t shares,
                                  const uint64_t localDiv)
{
    mmModel.matmulGradient = CalcMatmulGradient(mmInput, mmModel, shares);
    // 流水bound: 计算三条搬运流水耗时, 任一超过 cube bound 则按比例修正行梯度
    mmModel.mmDtypeBytes = mmInput.mmDtypeBytes + 1.0 / 32.0; // 含scale开销
    mmModel.mte1Time = CalcMte1Time(mmInput, mmModel);
    mmModel.mte2Time = CalcMte2Time(mmInput, mmModel);
    mmModel.fixpipeTime = CalcFixpipeTime(mmInput, mmModel);
    // T_cube 为全核聚合量, T_mte1 为单核量需除以核数
    const double tCubeTotal = static_cast<double>(mmInput.mValue) * mmModel.mmRankMul * mmModel.matmulGradient;
    const double tMte1PerCore = mmModel.mte1Time / std::max<uint64_t>(mmInput.coreNum, 1U);
    const double pipelineBound =
        std::max(tCubeTotal, std::max(tMte1PerCore, std::max(mmModel.mte2Time, mmModel.fixpipeTime)));
    if (pipelineBound > tCubeTotal) {
        // 流水bound超过cube bound: 按比例修正行梯度(保持每行梯度形式不变)
        mmModel.matmulGradient *= pipelineBound / std::max(tCubeTotal, 1e-30);
    }
    // 本地块耗时: 行数取 max(baseM, mValue), 每行系数直取目标指数口径(份数恒为 1,
    // 本地块仅本卡 own 行)
    const double pessRowCost = CalcLocalRowCost(mmInput, mmModel, MatmulPerfConst::LOCAL_MM_UTIL_EXPONENT_COEFFICIENT);
    const double physRowCost = CalcLocalRowCost(mmInput, mmModel, MatmulPerfConst::CUBE_UTIL_EXPONENT_COEFFICIENT);
    const double div = static_cast<double>(std::max<uint64_t>(localDiv, 1U));
    const double localRows = static_cast<double>(std::max<uint64_t>(mmInput.baseM, mmInput.mValue));
    // localMMTime 为保守估计(0.6 指数)，在计算长块时使用为了防止最后的长短块整形或者模型
    // 预估的误差导致长块长度不足；FRONT 估时的头部取值也用该保守口径
    mmModel.localMMTime = (std::isfinite(pessRowCost) && pessRowCost > 0.0) ?
                              localRows * pessRowCost / div :
                              MatmulTime(mmInput, mmModel, mmInput.mValue, 1) / div;
    // localShareTime 为物理估计(0.7 指数)，在耗时计算时使用，为了防止高估Local块带来的
    // 掩盖效果，导致模型过于倾向选择不切分
    mmModel.localShareTime = localRows * physRowCost / div;
}

// 对外唯一入口: 四件构造 → 切分器 → 求解, 顺序读完。
// halveBaseM 非空时输出减半决策: 候选满足减半条件(多块 ∧ 头块跨度==baseM ∧
// 元素>=1B)时 est 显式比较 baseM vs baseM/2, 取优者——host 只执行结果
inline CommCutResult GetFitBalanceCut(const FitBalanceTilingArgs& args, const FitBalanceCalibration& calib,
                                      bool* halveBaseM = nullptr)
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
    // 利用率的 M 宽度系数按轴语义: M 轴聚合家族(perRankM=true)FRONT 流的远程块是输出 M
    // 的一部分 → rankDim−1; K 轴通信家族(perRankM=false)远程份额乘的是 K 工作量(已计入
    // kValue), 输出 M 宽度不倍增 → commVolumeMul
    const uint64_t shares =
        (localMMPosition == LocalMMPosition::FRONT && calib.dataflow.perRankM) ? (rankDim - 1U) : comm.commVolumeMul;
    // 本地块折算: perRankM=false(K 轴通信) → 本卡份额 1/rankDim; perRankM=true → 不折算
    const uint64_t localDiv = calib.dataflow.perRankM ? 1U : rankDim;
    FinalizeMmModelParams(mmInput, mmModel, shares, localDiv);

    const FitBalanceFlow flow{localMMPosition, order, args.commEngine, args.allowShortFront, calib.dataflow.perRankM};
    CutPlanner tilingM = MakeCutPlanner(args, mmInput, comm);
    CommCutResult cut = Schedule(flow, mmInput, mmModel, comm, tilingM);

    // 核利用率优先切分与Fit Balance切分择优返回。通信标定 miss 时比较双方一方退化为
    // NoCut、另一方是零通信成本口径下的占位估计, 比较无意义——保持 miss 路径的 NoCut
    // 防御姿态(与配平求解侧的 miss 处理一致), 仅在标定命中时比时
    CommCutResult occupancyCut = MakeCoreOccupancyCut(mmInput);
    // 当核利用率优先只能切出不多于一个长块时，第二轮通信的启动开销会吃掉重叠收益
    // 回退成fit balance 切分（一般为不切分）
    if (occupancyCut.numLongTile <= 1U && occupancyCut.numShortTile > 0U) {
        occupancyCut = cut; // 配平侧(Schedule)结果优先
    } else if (comm.commVolumeMul > 1U) {
        // M 轴族: COF 侧同样细探——终局平局归 COF 的 tie-break 下, 细化候选须经 COF 侧投递
        DescendToFiner(flow, mmInput, mmModel, comm, occupancyCut);
    }
    if (flow.commEngine == TilingCommEngine::CCU) {
        ApplySingleRoundCommLimit(occupancyCut, mmInput, comm);
    }
    if (comm.slopeUsPerMb != 0.0) {
        const double occTime = EstimateCutTotalTime(flow, mmInput, mmModel, comm, occupancyCut);
        const double fitTime = EstimateCutTotalTime(flow, mmInput, mmModel, comm, cut);
        constexpr double OCC_NEAR_TIE_EPS_US = 5.0;
        const bool fitMultiTile = (cut.numLongTile + cut.numShortTile) > 1U;
        if (occTime <= fitTime || (fitMultiTile && occTime - fitTime < OCC_NEAR_TIE_EPS_US)) {
            cut = occupancyCut;
        }
    }
    // 减半决策: 最终切分满足减半门控即减半——头块跨度==baseM 时 baseM 减半使 mm 块数
    // 翻倍填满核波(wave 填充收益, est 梯度模型不可见, 不作 est 比较)
    if (halveBaseM != nullptr) {
        *halveBaseM = false;
        const uint64_t halveHeadSpan =
            (cut.numShortTile > 0U && !cut.shortTileAtBack) ? cut.shortTileLen : cut.longTileLen;
        const bool halveElem1B = (args.geAType != ge::DataType::DT_FLOAT4_E2M1) && (args.inputADtypeSize >= 1UL);
        *halveBaseM = (cut.totalTileCnt > 1U) && (halveHeadSpan == mmInput.baseM) && halveElem1B &&
                      (mmInput.baseM % 32UL == 0UL) && (mmInput.baseM < mmInput.mValue * mmModel.mmRankMul);
    }
    return cut;
}

// 引擎入参组装(编排入口共用): fp4 算术字节传 0(打包半字节由模型内部处理)
inline FitBalanceTilingArgs MakeFitBalanceEngineArgs(uint64_t mValue, uint64_t nValue, uint64_t kValue,
                                                     uint32_t rankDim, uint32_t aicCoreNum, ge::DataType x1Dtype,
                                                     uint64_t inputADtypeSize, uint64_t outputDtypeSize,
                                                     TilingCommEngine commEngine)
{
    FitBalanceTilingArgs args{};
    args.mValue = mValue;
    args.nValue = nValue;
    args.kValue = kValue;
    args.rankDim = rankDim;
    args.aicCoreNum = aicCoreNum;
    args.geAType = x1Dtype;
    args.inputADtypeSize = (x1Dtype == ge::DataType::DT_FLOAT4_E2M1) ? 0UL : inputADtypeSize;
    args.outputDtypeSize = outputDtypeSize;
    args.commEngine = commEngine;
    return args;
}

// 中阶入口: probe 底块已知时的切分+减半决策(不编排 mm tiling 重算; 编排入口内部亦经此)
inline bool GetFitBalanceCutFromMmBase(const FitBalanceTilingArgs& args, const FitBalanceCalibration& calib,
                                       CommCutResult& cut, bool* halveBaseM = nullptr, uint64_t mmBaseM = 0,
                                       uint64_t mmBaseN = 0)
{
    FitBalanceCalibration calibLocal = calib;
    if (mmBaseM > 0UL) {
        calibLocal.mmFit.baseBlock.baseM = mmBaseM;
    }
    if (mmBaseN > 0UL) {
        calibLocal.mmFit.baseBlock.baseN = mmBaseN;
    }
    if (!IsCommCalibrated(args.rankDim, args.commEngine, calibLocal)) {
        return false;
    }
    cut = GetFitBalanceCut(args, calibLocal, halveBaseM);
    return true;
}

// 切分结果防护(幂等): 退化零块→单块; 未对齐单块→"对齐头+未对齐尾"(m>=256)或"纯尾单块"(m<256)
inline void NormalizeDegenerateCut(CommCutResult& cut, uint64_t mValue)
{
    if (cut.numLongTile == 0U && cut.numShortTile == 0U) {
        cut.shortTileAtBack = true;
        cut.totalTileCnt = 1U;
        cut.numLongTile = 1U;
        cut.numShortTile = 0U;
        cut.longTileLen = mValue;
        cut.shortTileLen = 0U;
    }
    if (cut.numLongTile == 1U && cut.numShortTile == 0U && mValue % M_ALIGN_GRANULARITY != 0U) {
        const uint64_t alignedLen = mValue - mValue % M_ALIGN_GRANULARITY;
        if (alignedLen >= M_ALIGN_GRANULARITY) {
            cut.shortTileAtBack = true;
            cut.totalTileCnt = 2U;
            cut.longTileLen = alignedLen;
            cut.numShortTile = 1U;
            cut.shortTileLen = mValue - alignedLen;
        } else {
            cut.shortTileAtBack = true;
            cut.totalTileCnt = 1U;
            cut.numLongTile = 0U;
            cut.longTileLen = 0U;
            cut.numShortTile = 1U;
            cut.shortTileLen = mValue;
        }
    }
}

// 编排结果
struct FitBalanceTilingResult {
    CommCutResult cut;            // 通信切分(fit-balance)
    QuantMatmulTilingData mmTile; // matmul tiling(SWAT 族有效)
    uint32_t usedCoreNum;         // probe 核数(减半重算后保持)
    bool fitBalanceHit;           // 标定是否命中
    bool halveBaseM;              // 是否执行了减半重算
    uint64_t probeBaseM;          // probe 底块(减半前)
    uint64_t probeBaseN;
};

#endif
