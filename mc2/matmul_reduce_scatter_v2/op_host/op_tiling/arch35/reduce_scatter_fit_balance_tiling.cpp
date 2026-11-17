/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "reduce_scatter_fit_balance_tiling.h"

constexpr static uint64_t L2_CACHE_SIZE = 128 * ONE_MBYTE;
constexpr static uint64_t FIT_BALANCE_UNBALANCE_RATIO = 2;
constexpr static uint64_t SMALL_K_BOUND = 2048;
constexpr static uint64_t AICPU_M_TILE_CAP = 4; // AICPU通路通信切分膨胀较大，限制最大切分轮数

template <typename MatmulPerf, typename CommPerf>
void MMReduceScatterFitBalanceTiling<MatmulPerf, CommPerf>::SetApaceTileAlignment(bool enabled)
{
    keepTileAligned_ = enabled;
    if (enabled) {
        constexpr uint64_t APACE_PER_RANK_M_ALIGN = 256;
        tilingM_.SetAlignLength(APACE_PER_RANK_M_ALIGN);
        tilingM_.SetMinLenByMax(APACE_PER_RANK_M_ALIGN);
        tilingM_.tileArgs.minTileLen = mc2tiling::AlignUp(tilingM_.GetMinLen(), tilingM_.GetAlignLength());
    }
}

template <typename MatmulPerf, typename CommPerf>
void MMReduceScatterFitBalanceTiling<MatmulPerf, CommPerf>::EstimateMMCommTime()
{
    matmulPerf_.FindCubeUtil(rankTileNum_);
    matmulPerf_.GetMatmulGradient();

    // Find total matmul time and comm time
    double totalMatmulTime = matmulPerf_.MatmulTime(mmInfo_.mValue, rankDim_);
    double totalTpTime = commPerf_.CommTime(mmInfo_.mValue);
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    uint64_t sizeOfComm = mmInfo_.mValue * mmInfo_.nValue * commPerf_.GetCommDTypeSize() * rankDim_ / ONE_MBYTE;
    OP_LOGD("MatmulReduceScatter",
            "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, sizeOfComm %lu, "
            "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
            "rankDim_ %lu, rankTile %lu",
            mmInfo_.mValue, mmInfo_.nValue, mmInfo_.kValue, matmulPerf_.cubeUtil_, sizeOfComm, totalMatmulTime,
            totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_, rankDim_,
            rankTileNum_);
}

template <typename MatmulPerf, typename CommPerf>
void MMReduceScatterFitBalanceTiling<MatmulPerf, CommPerf>::SetShortTileLen()
{
    uint64_t l2UseSize = mmInfo_.mValue * mmInfo_.kValue * mmInfo_.inMatrixADtypeSize +
                         mmInfo_.kValue * mmInfo_.nValue * mmInfo_.inMatrixBDtypeSize;
    isLargerThanL2Cache_ = l2UseSize > L2_CACHE_SIZE;
    if (isLargerThanL2Cache_ && mmInfo_.mValue > tilingM_.GetMinLen() * FOUR) {
        tilingM_.SetMinLenByMax(tilingM_.GetMinLen() * TWO);
    }
    if (mmInfo_.mValue * mmInfo_.nValue <= coreNum_ * matmulPerf_.GetBaseM() * matmulPerf_.GetBaseN() * TWO) {
        tilingM_.SetMinLenByMax(mmInfo_.mValue);
    }

    if (keepTileAligned_) {
        SetApaceTileAlignment(true);
    }
    tilingM_.cutRes.shortTileLen = tilingM_.GetMinLen();
    tilingM_.cutRes.numShortTile = 1U;
}

template <typename MatmulPerf, typename CommPerf>
void MMReduceScatterFitBalanceTiling<MatmulPerf, CommPerf>::SetLongTileLen()
{
    // balancing the pipeline
    if (tilingM_.cutRes.shortTileAtBack) {
        double targetTime = matmulPerf_.MatmulTime(tilingM_.cutRes.shortTileLen, rankTileNum_);
        tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(targetTime);
    } else {
        double targetTime = commPerf_.CommTime(tilingM_.cutRes.shortTileLen);
        tilingM_.cutRes.longTileLen = matmulPerf_.InverseMatmulTime(targetTime, rankTileNum_);
    }
    OP_LOGD("MMReduceScatterFitBalanceTiling", "longTileLen %lu", tilingM_.cutRes.longTileLen);

    // adjust the long tile for better pipeline
    if (isLargerThanL2Cache_ && mmInfo_.mValue % tilingM_.cutRes.shortTileLen == 0 &&
        (mmInfo_.nValue >= TWO * mmInfo_.mValue * rankDim_)) {
        tilingM_.cutRes.longTileLen = tilingM_.cutRes.shortTileLen;
    } else if (isQuantMatmul_ && ratioCalcComm_ > FIT_BALANCE_UNBALANCE_RATIO && mmInfo_.kValue < SMALL_K_BOUND &&
               mmInfo_.mValue % tilingM_.cutRes.shortTileLen == 0) {
        tilingM_.cutRes.longTileLen = tilingM_.cutRes.shortTileLen;
    }
}

template <typename MatmulPerf, typename CommPerf>
void MMReduceScatterFitBalanceTiling<MatmulPerf, CommPerf>::AdjustLongShortTileLen()
{
    if (isAicpuComm_) { // AICPU场景通信切分膨胀较大，限制最大切分轮数
        tilingM_.SetMaxTileCnt(AICPU_M_TILE_CAP);
    }
    bool goodLinearityShape = (mmInfo_.kValue * mmInfo_.nValue >= LARGE_NK_BAR_BASE * ONE_MBYTE);
    if (keepTileAligned_) {
        FitApaceTileLengthDiscrete(false, goodLinearityShape);
    } else {
        tilingM_.FitTileLengthDiscrete(false, goodLinearityShape);
    }
    // When the long and short tiles are equal, the long and short pieces become one.
    if (tilingM_.cutRes.shortTileLen == tilingM_.cutRes.longTileLen) {
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.numLongTile++;
    }
}

template <typename MatmulPerf, typename CommPerf>
void MMReduceScatterFitBalanceTiling<MatmulPerf, CommPerf>::FitApaceTileLengthDiscrete(bool smallDimAlignUp,
                                                                                       bool goodLinearityShape)
{
    // FitBalance 的 APACE 分片策略在本类独立维护。
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

template class MMReduceScatterFitBalanceTiling<MatmulPerformanceArch35, HCCLPerformanceArch35>;
template class MMReduceScatterFitBalanceTiling<MatmulPerformance960dt, HCCLPerformance960dt>;
