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
 * \file reduce_scatter_formulaic_tiling.cpp
 * \brief
 */
#include "mc2_log.h"
#include "reduce_scatter_formulaic_tiling.h"

constexpr static uint64_t AICPU_M_TILE_CAP = 4; // AICPU通路通信切分膨胀较大，限制最大切分轮数
constexpr static uint64_t RANK_SIZE_16P = 16U;  // 16P跨机场景
constexpr static uint64_t EVEN_SPLIT2_M_PER_RANK = 1024U; // 单rank M轴长度上限

constexpr uint64_t APACE_RS_PER_RANK_BASE_M = 256;

void MMPlusReduceScatter::FitApaceTileLengthDiscrete(bool smallDimAlignUp, bool goodLinearityShape)
{
    // MMPlus 的 APACE 分片策略在本类独立维护。
    // APACE 仅调整初始缓存上限与最终余片处理，避免改动其他通路的公共切分规则。
    // 短片已对齐；缓存上限向下对齐，避免公共初分片被非对齐上限截断。
    FormPartition candidate(tilingM_);
    const uint64_t alignLen = tilingM_.GetAlignLength();
    candidate.tileArgs.maxTileLen = std::max(alignLen, candidate.tileArgs.maxTileLen / alignLen * alignLen);
    candidate.GenerateInitialPartition(smallDimAlignUp);
    tilingM_.cutRes = candidate.cutRes;

    auto& cutRes = tilingM_.cutRes;
    auto& tileArgs = tilingM_.tileArgs;
    const uint64_t totalLen = tilingM_.totalLen;
    const uint64_t tiledLen = cutRes.numLongTile * cutRes.longTileLen + cutRes.numShortTile * cutRes.shortTileLen;
    // 初分片向上对齐后可能超出总长，此时没有额外余量参与配平。
    const uint64_t resLen = totalLen > tiledLen ? totalLen - tiledLen : 0U;
    bool largeBackTile =
        (resLen >= cutRes.shortTileLen) && ((SMALL_TILECNT_PAR * cutRes.shortTileLen) >= cutRes.longTileLen);
    goodLinearityShape =
        (goodLinearityShape && (cutRes.shortTileLen >= tileArgs.mAlignLen)) || (!cutRes.shortTileAtBack);
    bool smallCut = cutRes.numLongTile <= SMALL_TILECNT_PAR;
    bool balanceCut = largeBackTile && goodLinearityShape && smallCut;
    if (balanceCut) {
        cutRes.longTileLen = cutRes.shortTileLen;
        cutRes.numLongTile = (totalLen - cutRes.shortTileLen) / cutRes.longTileLen;
        cutRes.numLongTile = std::max(cutRes.numLongTile, static_cast<uint64_t>(1));
        cutRes.totalTileCnt = cutRes.numShortTile + cutRes.numLongTile;
        largeBackTile = false;
    }
    if (cutRes.totalTileCnt == TWO) {
        tilingM_.RedistributeTile();
        largeBackTile = false;
    }
    bool moreFrontTile = largeBackTile && (cutRes.longTileLen <= cutRes.shortTileLen + resLen);
    if (moreFrontTile) {
        cutRes.numLongTile++;
        tilingM_.CalcBackTile();
    }

    tileArgs.maxTileCnt = std::max(tileArgs.maxTileCnt, ONE);
    if (cutRes.totalTileCnt > tileArgs.maxTileCnt) {
        uint64_t tileLen = totalLen / tileArgs.maxTileCnt;
        bool isDivisible = (tileLen * tileArgs.maxTileCnt == totalLen) && (tileLen % tileArgs.mAlignLen == 0U);
        if (isDivisible) {
            cutRes.numLongTile = tileArgs.maxTileCnt;
            cutRes.longTileLen = tileLen;
            cutRes.numShortTile = 0U;
            cutRes.shortTileLen = 0U;
            cutRes.totalTileCnt = tileArgs.maxTileCnt;
        }
    }
    tilingM_.MaxTileCntUniform();
    // 主片保持对齐，剩余行数留给短首片或短尾片，不再均摊。
    tilingM_.CalcShortTile();
    cutRes.totalTileCnt = cutRes.numLongTile + cutRes.numShortTile;
}

void MMPlusReduceScatter::SetComputesPerCycle(double computesPerCycle)
{
    matmulPerf_.mmShapeInfo_.computesPerCycle = computesPerCycle;
    clusterInfo_.computesPerCycle = computesPerCycle;
}

void MMPlusReduceScatter::ConfigureApaceFormulaModel(bool useLegacyFp4Model)
{
    if (useLegacyFp4Model) {
        matmulPerf_.mmShapeInfo_.inMatrixADtypeSize = 1;
        matmulPerf_.mmShapeInfo_.inMatrixBDtypeSize = 1;
        matmulPerf_.calcType_ = MatmulCalcType::QUANT;
        clusterInfo_.inMatrixADtypeSize = matmulPerf_.mmShapeInfo_.inMatrixADtypeSize;
        clusterInfo_.inMatrixBDtypeSize = matmulPerf_.mmShapeInfo_.inMatrixBDtypeSize;
        SetComputesPerCycle(MatmulPerformance::COMPUTES_PER_CYCLE);
    }
    const uint64_t perRankBaseM = APACE_RS_PER_RANK_BASE_M;
    matmulPerf_.mmShapeInfo_.baseM = perRankBaseM;
    clusterInfo_.baseM = perRankBaseM;
    tilingM_.SetAlignLength(perRankBaseM);

    // 基本块更新后，按原顺序重算通信和计算的最小片长。
    rankTileNum_ = commPerf_.GetRankTileNum();
    tilingM_.SetMinLenByMin(perRankBaseM);
    tilingM_.SetMinLenByMax(commPerf_.GetLinearThresholdLen());
    tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankTileNum_));
}

void MMPlusReduceScatter::SetTileAlignment(bool apaceEnabled, bool alignMinTileLen)
{
    keepTileAligned_ = apaceEnabled;
    alignMinTileLen_ = alignMinTileLen;
    if (apaceEnabled) {
        constexpr uint64_t APACE_PER_RANK_M_ALIGN = 256;
        tilingM_.SetAlignLength(APACE_PER_RANK_M_ALIGN);
        tilingM_.SetMinLenByMax(APACE_PER_RANK_M_ALIGN);
        tilingM_.tileArgs.minTileLen = mc2tiling::AlignUp(tilingM_.GetMinLen(), tilingM_.GetAlignLength());
    }
}

void MMPlusReduceScatter::SetCommTimeFactorForA5()
{
    commPerf_.ChangeCommTimeFactorByDivision(REDUCESCATTER_COMMTIME_FACTOR);
    return;
}

void MMPlusReduceScatter::SetCommTimeFactorForOther()
{
    // 通算并行时通信有膨胀，大K场景膨胀明显，做特殊处理
    bool bwGrowthByUtil = matmulPerf_.cubeUtil_ < PART_L2_UTIL;
    bool bwGrowthByShape = clusterInfo_.kValue >= LARGE_K_BOUNDARY && clusterInfo_.nValue > LARGE_N_BOUNDARY;
    bool smallDim = rankDim_ <= MIN_COMM_RANKDIM || rankTileNum_ <= MatmulPerformance::SMALL_RANKTILE;
    bwGrowthByUtil = bwGrowthByUtil && !smallDim;
    bwGrowthByShape = bwGrowthByShape && !smallDim;
    if (bwGrowthByUtil) { // 0.85 is max cube utilizaion
        commPerf_.ChangeCommTimeFactorByDivision(scatterLargerNKCommGrowRatio1);
    } else if (bwGrowthByShape) { // LargeK and LargeN
        commPerf_.ChangeCommTimeFactorByDivision(scatterLargerNKCommGrowRatio2);
    }
    commPerf_.ChangeCommTimeFactorByDivision(commGrowRatio); // 1.15x time of factor
    if (clusterInfo_.socVersion_2201 == SocVersion_2201::SOC910_93) {
        commPerf_.ChangeCommTimeFactorByDivision(0.6); // 0.6x time of factor
    }
    return;
}

void MMPlusReduceScatter::SetCommTimeFactor()
{
    if (npuArch_ == Ops::Base::DAV_3510 || npuArch_ == Ops::Base::DAV_9201) { // A5 & A6
        SetCommTimeFactorForA5();
    } else {
        SetCommTimeFactorForOther();
    }
    return;
}

void MMPlusReduceScatter::EstimateKernelTime()
{
    // 960 AICPU 的通信因子已在构造及 MMRS 模型配置阶段确定。
    if (npuArch_ != Ops::Base::DAV_9201 || !isAicpuComm_) {
        SetCommTimeFactor();
    }
    // 预测耗时
    double totalMatmulTime = EstimateTotalMatmulTime();
    double totalTpTime = EstimateTotalCommTime();
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    bool decreaseMinLen = (totalTpTime > totalMatmulTime * UNBALANCE_RATIO) && (clusterInfo_.nValue >= FOUR_KILO) &&
                          (clusterInfo_.kValue >= ONE_KILO);
    if (decreaseMinLen) { // 中、大shape通过减少初始值的方式鼓励多切
        tilingM_.SetMinLenByMin(tilingM_.GetAlignLength());
    }
    // 确定性通信场景适当少切
    bool increaseMinLen =
        (ratioCalcComm_ < commGrowRatio) ||
        ((totalMatmulTime > totalTpTime * UNBALANCE_RATIO) &&
         (clusterInfo_.mValue >= tilingM_.GetAlignLength() * FOUR)); // 4 = 2 x 2，m太小时增加初始值会导致不切
    increaseMinLen = increaseMinLen && (clusterInfo_.nValue <= FOUR_KILO) && deterministicSoc910B_; // n小时要少切
    if (increaseMinLen) { // 通过增加初始值的方式鼓励少切
        tilingM_.SetMinLenByMax(tilingM_.GetAlignLength() * TWO);
    }

    // 当 M * N < 4 * 1024 * 1024时，matmul性能变差，需要平衡matmul性能与流水掩盖
    // 假如 matmul时间 >= 通讯时间的3.5倍， 尾块配平会保证大主块，不用额外改大matmulMinTileSize参数
    // 假如 matmul时间 < 通讯时间的2倍，计算时间接近通讯时间，此时并行理论收益大：
    // 希望多切分获得并行收益，所以不再另外改大matmulMinTileSize参数
    increaseMinLen =
        (totalMatmulTime >= totalTpTime * TIME_LOWER_RATIO) && (totalMatmulTime < totalTpTime * TIME_UPPER_RATIO);
    if (increaseMinLen) { // 通过增加初始值的方式鼓励少切
        tilingM_.SetMinLenByMax(FOUR_MBYTE / clusterInfo_.nValue / rankTileNum_);
    }
    if (keepTileAligned_) {
        SetTileAlignment(true, alignMinTileLen_);
    }
    OP_LOGD("MatmulReduceScatter",
            "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, "
            "totalMatmulTime %f, totalTP %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f",
            clusterInfo_.mValue, clusterInfo_.nValue, clusterInfo_.kValue, matmulPerf_.cubeUtil_, totalMatmulTime,
            totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_);
}

void MMPlusReduceScatter::SelectTilingMethod()
{
    if (isAicpuComm_ && rankDim_ > 8) { // AICPU大于8P场景通信切分膨胀较大，限制最大切分轮数
        tilingM_.SetMaxTileCnt(AICPU_M_TILE_CAP);
    }
    if (alignMinTileLen_) {
        // 最小片长完成耗时调整后，按 PerBlock 的 M 分组对齐，再生成分片。
        tilingM_.tileArgs.minTileLen = mc2tiling::AlignUp(tilingM_.GetMinLen(), tilingM_.GetAlignLength());
    }
    if (tilingM_.SetShortTileLen()) { // 如果shape太小就不切
        return;
    }
    // 16P AICPU计算bound场景小M场景：通信切分膨胀大，切分轮次越少越好，尽量等分利用aicpu展开缓存
    bool evenSplitTwo = isAicpuComm_ && (rankDim_ == RANK_SIZE_16P) &&
                        (clusterInfo_.mValue <= EVEN_SPLIT2_M_PER_RANK) &&
                        (clusterInfo_.mValue % (TWO * tilingM_.GetAlignLength()) == 0U) &&
                        (clusterInfo_.inMatrixADtypeSize == TWO) && tilingM_.cutRes.shortTileAtBack;
    if (evenSplitTwo) {
        tilingM_.cutRes.longTileLen = tilingM_.totalLen / TWO;
        tilingM_.cutRes.numLongTile = TWO;
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.totalTileCnt = TWO;
        OP_LOGD("MatmulReduceScatter",
                "AICPU 16P even split 2: shortTileAtBack %d, longTileLen %lu, "
                "numLongTile %lu, shortTileLen %lu, numShortTile %lu",
                tilingM_.cutRes.shortTileAtBack, tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
                tilingM_.cutRes.shortTileLen, tilingM_.cutRes.numShortTile);
        return;
    }
    // 流水配平，找到理论切分长度
    if (tilingM_.cutRes.shortTileAtBack) {
        ShortAtEndCalcBoundBalancing();
    } else {
        ShortAtFrontCommBoundBalancing();
    }

    // 生成切分
    bool smallDimAlignUp = (rankDim_ <= MIN_COMM_RANKDIM) && tilingM_.cutRes.shortTileAtBack &&
                           (clusterInfo_.nValue < SMALL_SHAPE_BAR ||
                            clusterInfo_.kValue < SMALL_SHAPE_BAR); // 2p，计算Bound，且N轴或者K轴很小
    bool goodLinearityShape = (clusterInfo_.kValue * clusterInfo_.nValue >= LARGE_NK_BAR_BASE * ONE_MBYTE);
    if (keepTileAligned_) {
        FitApaceTileLengthDiscrete(smallDimAlignUp, goodLinearityShape);
    } else {
        tilingM_.FitTileLengthDiscrete(smallDimAlignUp, goodLinearityShape);
    }
    OP_LOGD("MatmulReduceScatter",
            "Final cut: shortTileAtBack %d, longTileLen %lu, "
            "numLongTile %lu, shortTileLen %lu, numShortTile %lu",
            tilingM_.cutRes.shortTileAtBack, tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
            tilingM_.cutRes.shortTileLen, tilingM_.cutRes.numShortTile);
}
