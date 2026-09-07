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
 * \file matmul_reduce_scatter_tiling.cpp
 * \brief
 */

#include "matmul_reduce_scatter_tiling_base.h"

using namespace AscendC;
using namespace ge;
using namespace optiling;
using namespace matmul_tiling;
using namespace Ops;
using namespace matmul_reduce_scatter_tiling_key;

namespace {
constexpr char HCCL_DETERMINISTIC[] = "HCCL_DETERMINISTIC";
const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK = {{0, {2, 4, 8}}, {1, {2, 4, 8, 16, 32}}};

constexpr uint32_t BIAS_INDEX = 2;

const std::vector<uint64_t> CALC_ND_BASIC = {6144, 4096, 2048};
constexpr uint64_t NUMBER_TWO = 2;
constexpr uint64_t BLOCK_BYTE_SIZE = 32;
constexpr uint64_t N_ALIGNED = 16;
constexpr uint64_t NCALC_THRES = 16;
constexpr uint64_t MIN_TAIL = 512;
constexpr uint64_t NUMBER_SIXTEEN = 16;

constexpr uint64_t VECTOR_D_BASE = 2048;
constexpr uint64_t UB_SIZE = 196352;

static bool CheckUbOverFlowMC2(uint64_t ubSize, uint64_t nAligned16, uint64_t nValue, const uint64_t &baseN,
                               const uint64_t &baseD, uint64_t dtypeSize)
{
    uint64_t nAlignedLoop = Ops::Transformer::MathUtil::CeilDivision(nAligned16, baseN);
    uint64_t nValueLoop = Ops::Transformer::MathUtil::CeilDivision(nValue, baseN);
    return ((nAlignedLoop != nValueLoop) &&
            ((nAligned16 - ((nValue / baseN) - 1) * baseN) * baseD > (ubSize / NUMBER_TWO / dtypeSize)));
}

static void CalcNd2NzTilingMC2(const mc2tiling::TilingArgs &args, uint64_t ubSize, uint64_t dtypeSize, uint64_t nValue,
                               uint64_t dValue, uint64_t &baseN, uint64_t &baseD)
{
    constexpr uint64_t mataD = 16384;
    constexpr uint64_t minN = 7168;
    // mata 交织场景
    if (dValue % mataD == 0 && nValue >= minN && ubSize == UB_SIZE &&
        (args.aType == matmul_tiling::DataType::DT_FLOAT16 || args.aType == matmul_tiling::DataType::DT_BF16) &&
        args.aicCoreNum >= 24) { // 24 核最小核数
        baseN = 96;              // baseN 经验值 96
        baseD = 512;             // baseD 经验值 512
        return;
    }

    uint64_t vectorCoreNum = NUMBER_TWO * args.aicCoreNum;
    vectorCoreNum = std::max(vectorCoreNum, 1UL);
    uint64_t baseThres = VECTOR_D_BASE / dtypeSize;
    uint64_t c0 = BLOCK_BYTE_SIZE / dtypeSize;
    uint64_t nAligned16 = ops::CeilAlign(nValue, N_ALIGNED);
    uint64_t dAlignedC0 = ops::CeilAlign(dValue, c0);
    if (dValue <= baseThres) {
        baseD = std::max(ops::CeilAlign(dValue, c0), uint64_t(1));
        uint64_t nDim = vectorCoreNum;
        baseN = ubSize / NUMBER_TWO / dtypeSize / baseD;
        uint64_t round = std::max(
            Ops::Transformer::MathUtil::CeilDivision(Ops::Transformer::MathUtil::CeilDivision(nAligned16, nDim), baseN),
            1UL);
        baseN = std::max(
            Ops::Transformer::MathUtil::CeilDivision(Ops::Transformer::MathUtil::CeilDivision(nAligned16, nDim), round),
            NCALC_THRES);
        while (baseN > NCALC_THRES) {
            if (CheckUbOverFlowMC2(ubSize, nAligned16, nValue, baseN, baseD, dtypeSize)) {
                baseN--;
                continue;
            }
            break;
        }
        return;
    }
    uint64_t lastTail = 0;
    // bestBaseD = 4K / dtype, bestBaseN = UB / 2 / 4096
    uint64_t bestBaseN = NCALC_THRES;
    uint64_t bestBaseD = CALC_ND_BASIC.at(1) / dtypeSize;
    for (auto base : CALC_ND_BASIC) {
        baseD = std::max(std::min(dAlignedC0, base / dtypeSize), uint64_t(1));
        uint64_t dLoop = Ops::Transformer::MathUtil::CeilDivision(dAlignedC0, baseD);
        uint64_t dTail = dAlignedC0 % baseD;
        if (dTail > 0 && dTail < (MIN_TAIL / dtypeSize)) {
            if (baseD * dtypeSize ==
                CALC_ND_BASIC.at(0)) { // if dtail < 0.5K && baseD == 3k, give up and return bestbase
                continue;
            }
            dLoop--;
            baseD =
                std::max(ops::CeilAlign(Ops::Transformer::MathUtil::CeilDivision(dAlignedC0, dLoop), c0), uint64_t(1));
            ; // dtail< 0.5K, make baseD larger
        }
        baseN = std::max(ubSize / NUMBER_TWO / dtypeSize / baseD, NUMBER_SIXTEEN);
        if (baseN * baseD * dtypeSize * NUMBER_TWO > ubSize) { // check ub overflow
            continue;
        }
        if (CheckUbOverFlowMC2(ubSize, nAligned16, nValue, baseN, baseD, dtypeSize)) {
            continue;
        }
        uint64_t nLoop = Ops::Transformer::MathUtil::CeilDivision(nAligned16, baseN);
        uint64_t tail = nLoop * dLoop % vectorCoreNum;
        while (baseN > NCALC_THRES) {
            if (CheckUbOverFlowMC2(ubSize, nAligned16, nValue, baseN, baseD, dtypeSize)) { // check ub overflow
                baseN--;
                nLoop = Ops::Transformer::MathUtil::CeilDivision(nAligned16, baseN);
                tail = nLoop * dLoop % vectorCoreNum;
                continue;
            }
            if (tail == 0) {
                return;
            }
            if (tail > lastTail) {
                lastTail = tail;
                bestBaseD = baseD;
                bestBaseN = baseN;
            }
            baseN--;
            nLoop = Ops::Transformer::MathUtil::CeilDivision(nAligned16, baseN);
            tail = nLoop * dLoop % vectorCoreNum;
        }
    }
    baseD = bestBaseD;
    baseN = bestBaseN;
    return;
}

static void PrintTilingData(::TCubeTiling &tiling)
{
    OP_LOGD("MatmulReduceScatter", "tiling.usedCoreNum %d", tiling.usedCoreNum);
    OP_LOGD("MatmulReduceScatter", "tiling.M %d", tiling.M);
    OP_LOGD("MatmulReduceScatter", "tiling.N %d", tiling.N);
    OP_LOGD("MatmulReduceScatter", "tiling.Ka %d", tiling.Ka);
    OP_LOGD("MatmulReduceScatter", "tiling.Kb %d", tiling.Kb);
    OP_LOGD("MatmulReduceScatter", "tiling.singleCoreM %d", tiling.singleCoreM);
    OP_LOGD("MatmulReduceScatter", "tiling.singleCoreN %d", tiling.singleCoreN);
    OP_LOGD("MatmulReduceScatter", "tiling.singleCoreK %d", tiling.singleCoreK);
    OP_LOGD("MatmulReduceScatter", "tiling.baseM %d", tiling.baseM);
    OP_LOGD("MatmulReduceScatter", "tiling.baseN %d", tiling.baseN);
    OP_LOGD("MatmulReduceScatter", "tiling.baseK %d", tiling.baseK);
    OP_LOGD("MatmulReduceScatter", "tiling.depthA1 %d", tiling.depthA1);
    OP_LOGD("MatmulReduceScatter", "tiling.depthB1 %d", tiling.depthB1);
    OP_LOGD("MatmulReduceScatter", "tiling.stepM %d", tiling.stepM);
    OP_LOGD("MatmulReduceScatter", "tiling.stepN %d", tiling.stepN);
    OP_LOGD("MatmulReduceScatter", "tiling.stepka %d", tiling.stepKa);
    OP_LOGD("MatmulReduceScatter", "tiling.stepkb %d", tiling.stepKb);
    OP_LOGD("MatmulReduceScatter", "tiling.isBias %d", tiling.isBias);
    OP_LOGD("MatmulReduceScatter", "tiling.transLength %d", tiling.transLength);
    OP_LOGD("MatmulReduceScatter", "tiling.iterateOrder %s", ((tiling.iterateOrder == 1) ? "orderM" : "orderN"));
    OP_LOGD("MatmulReduceScatter", "tiling.usedL1Size %d", tiling.shareL1Size);
    OP_LOGD("MatmulReduceScatter", "tiling.usedL0CSize %d", tiling.shareL0CSize);
    OP_LOGD("MatmulReduceScatter", "tiling.dbL0C %d", tiling.dbL0C); // set_dbL0C(1)
    OP_LOGD("MatmulReduceScatter", "tiling.usedUBSize %d", tiling.shareUbSize);
    OP_LOGD("MatmulReduceScatter", "tiling.batchM %d", tiling.batchM);
    OP_LOGD("MatmulReduceScatter", "tiling.batchN %d", tiling.batchN);
    OP_LOGD("MatmulReduceScatter", "tiling.singleBatchM %d", tiling.singleBatchM);
    OP_LOGD("MatmulReduceScatter", "tiling.singleBatchN %d", tiling.singleBatchN);
}

static void PrintTilingData(Mc2Tiling::RCSTiling &rcsTiling)
{
    OP_LOGD("MatmulReduceScatter", "rcsTiling.rankDim %d", rcsTiling.rankDim);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.rankID %d", rcsTiling.rankID);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.commtype %d", rcsTiling.commtype);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.subtype %d", rcsTiling.subtype);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.tileCnt %d", rcsTiling.tileCnt);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.tailM %d", rcsTiling.tailM);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.tailCnt %d", rcsTiling.tailCnt);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.biasLen %d", rcsTiling.biasLen);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.isAdd %d", rcsTiling.isAdd);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.rankM %d", rcsTiling.rankM);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.rankN %d", rcsTiling.rankN);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.rankK %d", rcsTiling.rankK);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.gatherIndex %d", rcsTiling.gatherIndex);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.isTransA %d", rcsTiling.isTransposeA);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.isTransB %d", rcsTiling.isTransposeB);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.storageGather %d", rcsTiling.storageGather);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.nd2NzWorkLen %lu", rcsTiling.nd2NzWorkLen);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.cToFloatLen %lu", rcsTiling.cToFloatLen);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.gatherLen %lu", rcsTiling.gatherLen);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.workspaceAddr4 %u", rcsTiling.workspaceAddr4);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.aicCoreNum %u", rcsTiling.aicCoreNum);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.needUbBuffer %u", rcsTiling.needUbBuffer);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.addX3UbCnt %u", rcsTiling.addX3UbCnt);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.commWorkSpaceSize %u", rcsTiling.commWorkSpaceSize);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.isInputCommQuantScale %u", rcsTiling.isInputCommQuantScale);
    OP_LOGD("MatmulReduceScatter", "rcsTiling.dataType %u", rcsTiling.dataType);
}

static void PrintTilingData(Mc2Tiling::TileL2Tiling &tileL2Tiling)
{
    OP_LOGD("MatmulReduceScatter", "tileL2Tiling.mL2TileCnt %d", tileL2Tiling.mL2TileCnt);
    OP_LOGD("MatmulReduceScatter", "tileL2Tiling.nL2TileCnt %d", tileL2Tiling.nL2TileCnt);
    OP_LOGD("MatmulReduceScatter", "tileL2Tiling.mTileBlocks %d", tileL2Tiling.mTileBlocks);
    OP_LOGD("MatmulReduceScatter", "tileL2Tiling.nTileBlocks %d", tileL2Tiling.nTileBlocks);
    OP_LOGD("MatmulReduceScatter", "tileL2Tiling.mTailBlocks %d", tileL2Tiling.mTailBlocks);
    OP_LOGD("MatmulReduceScatter", "tileL2Tiling.nTailBlocks %d", tileL2Tiling.nTailBlocks);
    OP_LOGD("MatmulReduceScatter", "tileL2Tiling.rankTileNum %d", tileL2Tiling.rankTileNum);
    OP_LOGD("MatmulReduceScatter", "tileL2Tiling.calcOrder %d", tileL2Tiling.calcOrder);
    OP_LOGD("MatmulReduceScatter", "tileL2Tiling.enableL2Tile %d", tileL2Tiling.enableL2Tile);
}
} // namespace

namespace optiling {

static ge::graphStatus CalcMatmulTilingReduceScatter(mc2tiling::TilingArgs &args, ::TCubeTiling &cubeTiling,
                                                     Mc2Tiling::TileL2Tiling &l2Tiling);

static uint32_t MC2_SpliteReduceScatter(mc2tiling::TilingArgs &args, uint32_t maxTileCnt = 64)
{
    // 检查允许通信的最大次数
    if (args.commTurn >= maxTileCnt) {
        args.commTurn = maxTileCnt;
    }

    uint64_t tileLen = 1;
    if (args.mValue > args.commTurn) {
        tileLen = args.mValue / args.commTurn;
    }

    if (args.outputDtypeSize == 2) { // 数据长度为2, 则向 2*64 = 128，则向128对齐
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 64);
    } else if (args.outputDtypeSize == 4) {                  // 4 is float32 tpye size
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 32); // 32 is used to align to 128
    }
    if (args.mValue > tileLen) {
        return tileLen;
    }
    return args.mValue;
}

static ge::graphStatus ReduceScatterParamsCheck(const gert::TilingContext *context)
{
    OP_TILING_CHECK(mc2tiling::Mc2TilingUtils::CommonParamCheck(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "common check failed"), return ge::GRAPH_FAILED);

    const gert::StorageShape *aShape = context->GetInputShape(0);
    const gert::StorageShape *bShape = context->GetInputShape(1);
    OP_TILING_CHECK((aShape == nullptr) || (bShape == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "required params"), return ge::GRAPH_FAILED);
    uint64_t valueOne = aShape->GetStorageShape().GetDim(0);
    uint64_t valueTwo = aShape->GetStorageShape().GetDim(1);
    uint64_t valueThree = bShape->GetStorageShape().GetDim(0);
    uint64_t valueFour = bShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(valueOne == 0 || valueTwo == 0 || valueThree == 0 || valueFour == 0,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "shape value"), return ge::GRAPH_FAILED);

    // GetAttrs() 为空时必须提前返回，否则后续会对其结果解空指针
    if (context->GetAttrs() == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    } else {
        auto reduce_op = context->GetAttrs()->GetAttrPointer<char>(1);
        OP_TILING_CHECK(reduce_op == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "reduce_op"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(strcmp(reduce_op, "sum") != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "reduce_op", reduce_op,
                                                              "The value of reduce_op must be sum"),
                        return ge::GRAPH_FAILED);

        auto isTransA = context->GetAttrs()->GetAttrPointer<bool>(2);
        OP_TILING_CHECK(isTransA == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "isTransA"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            *isTransA != false,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "isTransA", *isTransA ? "true" : "false",
                                                  "The value of isTransA must be false"),
            return ge::GRAPH_FAILED);
    }

    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    OP_TILING_CHECK(group == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "group"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetCommAlg(MatmulReduceScatterTilingData &tilingData)
{
    tilingData.socParam.commAlg = COMM_ALG_FULL_MESH;
    return ge::GRAPH_SUCCESS;
}

bool MatmulReduceScatterTilingFuncBase::IsDeterministic()
{
    if (getenv(mc2tiling::HCCL_DETERMINISTIC) == nullptr) {
        return false;
    }
    std::string envStr(getenv(mc2tiling::HCCL_DETERMINISTIC));
    std::transform(envStr.begin(), envStr.end(), envStr.begin(), ::toupper);
    if (envStr == "TRUE") {
        OP_LOGI("MatmulReduceScatter", "HCCL_DETERMINISTIC is set to be true.");
        return true;
    }
    OP_LOGI("MatmulReduceScatter", "HCCL_DETERMINISTIC [%s] is set to be false.", envStr.c_str());
    return false;
}

ge::graphStatus MatmulReduceScatterTilingFuncBase::GetReduceScatterFormulateTileCnt(
    const gert::TilingContext *ctx, MatmulReduceScatterTilingData &tilingData, mc2tiling::TilingArgs &args)
{
    if (ctx->GetAttrs() == nullptr) {
        OP_LOGW(ctx->GetNodeName(), " ctx->GetAttrs is nullptr.");
        return ge::GRAPH_FAILED;
    }

    CutResult mCutScatter = GetCutResult(tilingData, args);
    if (mCutScatter.shortTileAtBack || mCutScatter.numShortTile == 0) {
        tilingData.param.tileCnt = mCutScatter.numLongTile;
        args.mValue = mCutScatter.longTileLen;
        OP_TILING_CHECK(
            CalcMatmulTilingReduceScatter(args, tilingData.tileTiling, tilingData.tileL2Tiling) != ge::GRAPH_SUCCESS,
            OP_LOGE(ctx->GetNodeName(), "CalcMatmulTilingReduceScatter failed"), return ge::GRAPH_FAILED);
        args.baseMLimit = mCutScatter.longTileLen;
        args.mValue = mCutScatter.longTileLen * args.rankTileNum;
        tilingData.param.tailM = mCutScatter.shortTileLen;
        tilingData.param.tailCnt = 0;
        if (mCutScatter.numShortTile > 0) {
            args.mValue = mCutScatter.shortTileLen;
            tilingData.param.tailCnt = mCutScatter.numShortTile;
            auto tailRet = CalcMatmulTilingReduceScatter(args, tilingData.tailTiling, tilingData.tailL2Tiling);
            OP_TILING_CHECK(tailRet != ge::GRAPH_SUCCESS,
                            OP_LOGE(ctx->GetNodeName(), "CalcMatmulTilingReduceScatter failed"),
                            return ge::GRAPH_FAILED);
            args.baseMLimit = mCutScatter.shortTileLen;
            args.mValue = mCutScatter.shortTileLen * args.rankTileNum;
        }
    } else {
        tilingData.param.tileCnt = mCutScatter.numShortTile;
        args.mValue = mCutScatter.shortTileLen;
        OP_TILING_CHECK(
            CalcMatmulTilingReduceScatter(args, tilingData.tileTiling, tilingData.tileL2Tiling) != ge::GRAPH_SUCCESS,
            OP_LOGE(ctx->GetNodeName(), "CalcMatmulTilingReduceScatter failed"), return ge::GRAPH_FAILED);
        args.baseMLimit = mCutScatter.shortTileLen;
        args.mValue = mCutScatter.shortTileLen * args.rankTileNum;
        tilingData.param.tailM = mCutScatter.longTileLen;
        tilingData.param.tailCnt = 0;
        if (mCutScatter.numLongTile > 0) {
            args.mValue = mCutScatter.longTileLen;
            tilingData.param.tailCnt = mCutScatter.numLongTile;
            auto tailRet = CalcMatmulTilingReduceScatter(args, tilingData.tailTiling, tilingData.tailL2Tiling);
            OP_TILING_CHECK(tailRet != ge::GRAPH_SUCCESS,
                            OP_LOGE(ctx->GetNodeName(), "CalcMatmulTilingReduceScatter failed"),
                            return ge::GRAPH_FAILED);
            args.baseMLimit = mCutScatter.longTileLen;
            args.mValue = mCutScatter.longTileLen * args.rankTileNum;
        }
    }
    args.mValue = mCutScatter.longTileLen;
    return ge::GRAPH_SUCCESS;
}

// 第一个参数m
ge::graphStatus MatmulReduceScatterTilingFuncBase::MCSpliteMReduceScatter(gert::TilingContext *ctx,
                                                                          MatmulReduceScatterTilingData &tilingData,
                                                                          mc2tiling::TilingArgs &args)
{
    args.rankTileNum = args.rankDim;

    if (args.enableSplitK) { // 只有1份
        tilingData.param.tileCnt = 1;
        tilingData.param.tailCnt = 0;
        tilingData.param.tailM = 0;
        OP_TILING_CHECK(
            CalcMatmulTilingReduceScatter(args, tilingData.tileTiling, tilingData.tileL2Tiling) != ge::GRAPH_SUCCESS,
            OP_LOGE(ctx->GetNodeName(), "CalcMatmulTilingReduceScatter failed"), return ge::GRAPH_FAILED);
    } else if (args.commTurn != 0) {
        uint64_t splite = MC2_SpliteReduceScatter(args);

        // 现在找到1个合适的切分
        auto tileCnt = args.mValue / splite;  // 切的份数
        auto tileTail = args.mValue % splite; // 尾巴

        tilingData.param.tileCnt = tileCnt;
        args.mValue = splite;
        tilingData.param.tailCnt = 0;
        OP_TILING_CHECK(
            CalcMatmulTilingReduceScatter(args, tilingData.tileTiling, tilingData.tileL2Tiling) != ge::GRAPH_SUCCESS,
            OP_LOGE(ctx->GetNodeName(), "CalcMatmulTilingReduceScatter failed"), return ge::GRAPH_FAILED);
        tilingData.param.tailM = tileTail;
        if (tileTail != 0) {
            args.mValue = tileTail;
            tilingData.param.tailCnt = 1;
            auto tailRet = CalcMatmulTilingReduceScatter(args, tilingData.tailTiling, tilingData.tailL2Tiling);
            OP_TILING_CHECK(tailRet != ge::GRAPH_SUCCESS,
                            OP_LOGE(ctx->GetNodeName(), "CalcMatmulTilingReduceScatter failed"),
                            return ge::GRAPH_FAILED);
        }
        args.mValue = splite;
    } else {
        OP_TILING_CHECK(GetReduceScatterFormulateTileCnt(ctx, tilingData, args) != ge::GRAPH_SUCCESS,
                        OP_LOGE(ctx->GetNodeName(), "GetReduceScatterFormulateTileCnt failed"),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(MC2SetWorkspaceReduceScatter(ctx, tilingData, args) != ge::GRAPH_SUCCESS,
                    OP_LOGE(ctx->GetNodeName(), "MC2SetWorkspaceReduceScatter failed"), return ge::GRAPH_FAILED);

    // mmReduceScatterBiasCast = true时涉及SyncAll，设置batch mode模式，所有核同时启动
    if (tilingData.param.biasLen != 0) {
        uint32_t batch_mode = 1U;
        auto ret = ctx->SetScheduleMode(batch_mode);
        MC2_CHECK_LOG_RET(ctx->GetNodeName(), ret);
    }

    return ge::GRAPH_SUCCESS;
}

static void SetReduceScatterTilingArgs(const gert::TilingContext *context, mc2tiling::TilingArgs &args)
{
    auto coreNum = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetCoreNumAic();
    auto aType = context->GetInputDesc(0)->GetDataType();
    auto bType = context->GetInputDesc(1)->GetDataType();
    auto cType = aType;

    // bias
    const gert::StorageShape *matrixBias = context->GetOptionalInputShape(BIAS_INDEX);
    bool isBias = (matrixBias == nullptr) ? false : true;
    ge::DataType biasType = (matrixBias == nullptr) ? cType : context->GetInputDesc(BIAS_INDEX)->GetDataType();

    const gert::StorageShape *aShape = context->GetInputShape(0);
    const gert::StorageShape *bShape = context->GetInputShape(1);
    uint64_t kValue = aShape->GetStorageShape().GetDim(1);
    uint64_t nValue = bShape->GetStorageShape().GetDim(1);
    uint64_t mValue = aShape->GetStorageShape().GetDim(0);

    if (aShape->GetStorageShape().GetDim(1) != bShape->GetStorageShape().GetDim(0)) {
        OP_LOGD(context->GetNodeName(), "A.shape(1) %lu B.shape(0) %lu, istransB = %d",
                aShape->GetStorageShape().GetDim(1), bShape->GetStorageShape().GetDim(0), args.isBTrans);
        nValue = bShape->GetStorageShape().GetDim(0);
    }

    args.orgMValue = mValue;
    args.orgNValue = nValue;
    args.orgKValue = kValue;
    args.mValue = mValue;

    if (args.commAlg == COMM_ALG_DOUBLE_RING) {
        args.mValue /= DOUBLE_RING_FACTOR;
        OP_LOGD(context->GetNodeName(), "args.mValue is %lu under double ring communication algorithm.", args.mValue);
    }

    args.nValue = nValue;
    args.kValue = kValue;
    args.baseMLimit = -1;
    args.inputDtypeSize = mc2tiling::D_TYPE_SIZE_MAP.at(aType);
    args.outputDtypeSize = mc2tiling::D_TYPE_SIZE_MAP.at(cType);
    args.geAType = aType;
    args.geBType = bType;
    args.geCType = cType;
    args.geBiasType = biasType;
    args.aicCoreNum = coreNum;
    args.enablePad = false;
    args.enableSplitK = false;
    args.isBias = isBias;
    args.aType = mc2tiling::D_TYPE_MAP.at(aType);
    args.bType = mc2tiling::D_TYPE_MAP.at(bType);
    args.cType = mc2tiling::D_TYPE_MAP.at(cType);
    args.biasType = mc2tiling::D_TYPE_MAP.at(biasType); // 因为bias可能不存在，先采用biasType规避
}

struct HcclAicpuOpParam {
    uint8_t res[64];
};

struct KFCNotify {
    // 消息通信
    HcclAicpuOpParam msgSend[16]; // 填充16个
    HcclAicpuOpParam msgCnt[16];
};

struct KFCMsgBody {
    // Rank* aiv * MsgSize * sizeof(消息)
    HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
    HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};

static void GetTilingKey(uint64_t &tilingKey, const MatmulReduceScatterTilingData &tilingData)
{
    bool mmReduceScatterFullMesh = true;
    bool mmReduceScatterNd2nzOpt = false;
    bool mmReduceScatterBiasCast = false;

    if (tilingData.param.biasLen == 0) {
        mmReduceScatterBiasCast = false;
    } else {
        mmReduceScatterBiasCast = true;
    }

    if (tilingData.socParam.isND2NZ == 1) {
        mmReduceScatterNd2nzOpt = true;
    } else {
        mmReduceScatterNd2nzOpt = false;
    }

    if (tilingData.socParam.commAlg == COMM_ALG_FULL_MESH) {
        mmReduceScatterFullMesh = true;
    } else {
        mmReduceScatterFullMesh = false;
    }

    tilingKey = GET_TPL_TILING_KEY(mmReduceScatterFullMesh, mmReduceScatterNd2nzOpt, mmReduceScatterBiasCast);
}

ge::graphStatus MatmulReduceScatterTilingFuncBase::SetMatmulTilingMatmulReduceScatter(
    gert::TilingContext *context, MatmulReduceScatterTilingData &tilingData, mc2tiling::TilingArgs &args)
{
    OP_TILING_CHECK(context->GetInputDesc(0) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input desc x1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context->GetInputDesc(1) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input desc x2"), return ge::GRAPH_FAILED);
    SetReduceScatterTilingArgs(context, args);

    tilingData.param.rankM = args.orgMValue; // 存放用户原始输入的mValue
    tilingData.param.rankN = args.orgNValue; // 存放用户原始输入的nValue
    tilingData.param.rankK = args.orgKValue; // 存放用户原始输入的kValue
    tilingData.param.aicCoreNum = args.aicCoreNum;
    uint64_t baseN = 1;
    uint64_t baseD = 1;
    uint64_t ubSizeValue = 0;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeValue);

    uint64_t nValue = args.isBTrans ? tilingData.param.rankN : tilingData.param.rankK;
    uint64_t dValue = args.isBTrans ? tilingData.param.rankK : tilingData.param.rankN;
    CalcNd2NzTilingMC2(args, ubSizeValue, args.inputDtypeSize, nValue, dValue, baseN, baseD);
    tilingData.socParam.baseBD = baseD;
    tilingData.socParam.baseBN = baseN;
    // 为通信而进行调整搬运
    if (args.cmdType == mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER) {
        if (args.rankDim <= 0 || static_cast<bool>(args.orgMValue % args.rankDim)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "m", std::to_string(args.orgMValue).c_str(),
                ("The value of m must be an integer multiple of rank_size(" + std::to_string(args.rankDim) + ")")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        args.mValue /= args.rankDim; // 必须能够整数切分, 并且不能切K
    } else {
        OP_LOGE(context->GetNodeName(), "args.cmdType error %d", static_cast<int>(args.cmdType));
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(MCSpliteMReduceScatter(context, tilingData, args) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "MCSpliteMReduceScatter failed"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0U;
    GetTilingKey(tilingKey, tilingData);
    OP_LOGD(context->GetNodeName(), "tilingKey is %u, aicCoreNum is %lu.", tilingKey, args.aicCoreNum);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(args.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcMatmulTilingReduceScatter(mc2tiling::TilingArgs &args, ::TCubeTiling &cubeTiling,
                                                     Mc2Tiling::TileL2Tiling &l2Tiling)
{
    uint64_t mValue = args.mValue;
    uint64_t nValue = args.nValue;
    uint64_t kValue = args.kValue;

    matmul_tiling::MultiCoreMatmulTiling mm1;
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.aType, args.isATrans);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.bType, args.isBTrans);
    mm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.cType);
    if (args.isBias) {
        mm1.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.biasType);
        mm1.SetBias(true);
    } else {
        mm1.SetBias(false);
    }
    mm1.SetDim(args.aicCoreNum);

    mm1.SetShape(mValue, nValue, kValue);
    mm1.SetOrgShape(mValue, nValue, kValue);
    mm1.SetBufferSpace(512 * 1024, -1, -1); // 512 * 1024 is buffer size
    mm1.SetSingleShape(-1, -1, -1);
    if (mm1.GetTiling(cubeTiling) == -1) {
        OP_LOGE("MatmulReduceScatter", "mValue %lu, nValue %lu, kValue %lu, aicCoreNum %lu", mValue, nValue, kValue,
                args.aicCoreNum);
        return ge::GRAPH_FAILED;
    }
    mc2tiling::MatmulFormulaicTiling scatterTiling("MatmulReduceScatter");
    scatterTiling.GetCubeTiling(args, cubeTiling, l2Tiling);

    return ge::GRAPH_SUCCESS;
}

static void CalculateNd2nzLen(Mc2Tiling::RCSTiling &config, mc2tiling::TilingArgs &args, uint64_t &nd2nzLen)
{
    constexpr uint64_t alignAddrLen = 512;
    uint32_t gatherIndex = config.gatherIndex;
    if (gatherIndex == 0) { // 转置B
        // 计算ND2NZ需使用空间方法保持与MMV3 tiling计算逻辑一致
        uint64_t alignByte = 256 / args.inputDtypeSize; // 256B 对齐shape
        uint64_t kALign = ops::CeilAlign(static_cast<uint64_t>(config.rankK), alignByte);
        uint64_t nALign = ops::CeilAlign(static_cast<uint64_t>(config.rankN), alignByte);
        nd2nzLen = kALign * nALign * args.inputDtypeSize;
    } else {
        auto alignM = config.rankM + 16;
        auto alignK = config.rankK + 16;
        nd2nzLen = mc2tiling::AlignUp(alignM * alignK * args.inputDtypeSize, alignAddrLen);
    }
}

ge::graphStatus MatmulReduceScatterTilingFuncBase::SetTilingData(const gert::TilingContext *context,
                                                                 MatmulReduceScatterTilingData &tilingData)
{
    const uint32_t opType = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER);
    if (context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char *groupName = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    const std::string rsConfig = GetRsConfig(tilingData);
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, rsConfig, 0);
    mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterTilingFuncBase::MC2SetWorkspaceReduceScatter(
    gert::TilingContext *context, MatmulReduceScatterTilingData &tilingData, mc2tiling::TilingArgs &args)
{
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "get workspace failed"),
                    return ge::GRAPH_FAILED);

    auto &&cfg = tilingData.param;

    // step1: ND2NZ
    uint64_t nd2nzLen = 0;
    CalculateNd2nzLen(cfg, args, nd2nzLen);

    uint64_t storage_a = 0;
    if (args.cmdType == mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER) {
        // A*B 的数据，需要找地方存储，因为 C 的长度 = A*B的长度/ RankDim
        uint64_t gmcFloat = static_cast<uint64_t>(cfg.rankM) * static_cast<uint64_t>(cfg.rankN) *
                            static_cast<uint64_t>(args.outputDtypeSize);
        gmcFloat = mc2tiling::AlignUp<uint64_t>(gmcFloat, 512); // 512 is used to get gm
        OP_LOGD("MatmulReduceScatter", "reduce scatter gmcFloat size %lu.", gmcFloat);

        tilingData.param.nd2NzWorkLen = nd2nzLen;
        tilingData.param.cToFloatLen = gmcFloat;

        storage_a = nd2nzLen + gmcFloat;
        OP_LOGD("MatmulReduceScatter", "reduce scatter storage_a size %lu.", storage_a);
    }

    int biasLen = 0;
    if (args.isBias && args.bType == matmul_tiling::DataType::DT_BFLOAT16) {
        biasLen = mc2tiling::AlignUp(args.orgNValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
    tilingData.param.biasLen = biasLen;
    workspaces[0] = storage_a + 16 * 1024 * 1024 + biasLen; // 16 mb, 1024 * 1024 is one mb
    OP_LOGD("MatmulReduceScatter", "workspaces[0] size %ld, biasLen %d.", workspaces[0], biasLen);
    tilingData.param.dataType = static_cast<uint8_t>(mc2tiling::Mc2TilingUtils::GetDataType(args.geAType));

    if (tilingData.param.rankID == 0) {
        OP_LOGD("MatmulReduceScatter", "workspace size %ld", workspaces[0]);

        PrintTilingData(tilingData.param);
        PrintTilingData(tilingData.tileTiling);
        PrintTilingData(tilingData.tileL2Tiling);
        if (tilingData.param.tailM != 0U) {
            OP_LOGD("MatmulReduceScatter", "have tail");
            PrintTilingData(tilingData.tailTiling);
            PrintTilingData(tilingData.tailL2Tiling);
        }
    }

    OP_TILING_CHECK(SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "SetTilingData failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterTilingFuncBase::MatmulReduceScatterTilingFunc(gert::TilingContext *context)
{
    MatmulReduceScatterTilingData *tilingData = context->GetTilingData<MatmulReduceScatterTilingData>();
    // 对参数进行校验
    OP_TILING_CHECK(ReduceScatterParamsCheck(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "param"), return ge::GRAPH_FAILED);
    int index = 0;
    auto group = context->GetAttrs()->GetAttrPointer<char>(index++);
    auto reduce_op = context->GetAttrs()->GetAttrPointer<char>(index++);
    auto is_trans_a = context->GetAttrs()->GetAttrPointer<bool>(index++);
    auto is_trans_b = context->GetAttrs()->GetAttrPointer<bool>(index++);
    auto comm_turn_ptr = context->GetAttrs()->GetAttrPointer<int64_t>(index++);

    OP_TILING_CHECK(is_trans_b == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "is_trans_b"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(comm_turn_ptr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "comm_turn"),
                    return ge::GRAPH_FAILED);
    auto rankSize = mc2tiling::MatmulFormulaicTiling::GetRankSize(group);
    auto comm_turn = *comm_turn_ptr;
    OP_TILING_CHECK(
        comm_turn != 0,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "comm_turn", std::to_string(comm_turn).c_str(), "0"),
        return ge::GRAPH_FAILED);

    OP_LOGD("MatmulReduceScatter",
            "group is %s, rankSize is %u, reduce_op is %s, is_trans_a is %d, is_trans_b is %d,"
            "comm_turn is %ld.",
            group, rankSize, reduce_op, *is_trans_a, *is_trans_b, comm_turn);

    tilingData->param.rankDim = rankSize;
    tilingData->param.isTransposeA = is_trans_a ? *is_trans_a : 0;
    tilingData->param.isTransposeB = is_trans_b ? *is_trans_b : 0;
    auto commSets = mc2tiling::Mc2TilingUtils::GetCommSets(group);
    tilingData->socParam.isA3 = (commSets == COMM_MESH) ? 0 : 1;
    tilingData->socParam.isStep = 0U;
    tilingData->socParam.isND2NZ = 1U;
    OP_TILING_CHECK(SetCommAlg(*tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), " Set comm algorithm failed."), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), " Communication algorithm is %u.", tilingData->socParam.commAlg);
    OP_LOGI(context->GetNodeName(), " Step comm flag is %u. ND2NZ flag is: %u", tilingData->socParam.isStep,
            tilingData->socParam.isND2NZ);
    if (CheckValidRank(VALID_RANK, tilingData, context, rankSize) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    mc2tiling::TilingArgs args;
    args.isATrans = is_trans_a ? *is_trans_a : 0;
    args.isBTrans = is_trans_b ? *is_trans_b : 0;
    args.cmdType = mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER;
    args.rankDim = rankSize;
    args.commTurn = comm_turn;
    args.commAlg = tilingData->socParam.commAlg;

    tilingData->param.isTransposeA = args.isATrans ? 1 : 0;
    tilingData->param.isTransposeB = args.isBTrans ? 1 : 0;
    tilingData->param.commtype = static_cast<uint32_t>(args.cmdType);

    if (strncmp(reduce_op, "sum", 3) == 0) { // 3 is index
        OP_LOGD("MatmulReduceScatter", "strncmp(reduce_op, sum, 3) == 0");
        tilingData->param.subtype = static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_SUM);
    } else {
        OP_LOGD("MatmulReduceScatter", "strncmp(reduce_op, sum, 3) != 0");
        tilingData->param.subtype = static_cast<uint8_t>(mc2tiling::HcclReduceOp::HCCL_REDUCE_RESERVED);
    }

    OP_TILING_CHECK(SetMatmulTilingMatmulReduceScatter(context, *tilingData, args) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "SetMatmulTilingMatmulReduceScatter failed"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
