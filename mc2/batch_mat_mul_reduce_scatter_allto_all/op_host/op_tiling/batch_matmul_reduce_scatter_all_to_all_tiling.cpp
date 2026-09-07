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
 * \file batch_matmul_reduce_scatter_all_to_all_tiling.cc
 * \brief
 */

#include "batch_matmul_reduce_scatter_all_to_all_tiling.h"
#include "../..//op_kernel/batch_mat_mul_reduce_scatter_allto_all_tiling_key.h"

#include <string>
#include <queue>
#include <vector>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>

#include "op_host/op_tiling/mc2_tiling_common_var.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "batch_matmul_reduce_scatter_all_to_all_formulaic_tiling.h"
#include "common/utils/op_mc2.h"
#include "mc2_moe_utils.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Moe;

namespace {

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t OUTPUT_Y_INDEX = 0;

constexpr uint32_t ATTR_EP_GROUP_INDEX = 0;
constexpr uint32_t ATTR_TP_GROUP_INDEX = 1;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 2;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 3;
constexpr uint32_t ATTR_Y_SHARD_TYPE_INDEX = 4;
constexpr uint32_t ATTR_IS_WEIGHT_TRANS_INDEX = 5;

constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7; // numeric representation of ReduceScatter
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;     // numeric representation of AlltoAll

constexpr int64_t LITE_THRESHOLD = 640;
} // namespace

namespace optiling {

struct BmmTilingConfig {
    gert::TilingContext *context;
    BatchMatMulReduceScatterAlltoAllTilingData &tilingData;
    ReduceScatterAlltoAllBatchInfo &bmmV3BatchInfo;
    ReduceScatterAlltoAllMatmulInfo &mmV3ArgsInfo;
};

static void PrintCommonTilingVariables(BatchMatMulReduceScatterAlltoAllTilingData *tilingData)
{
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - epGroupSize is %u.",
            tilingData->commonTiling.epGroupSize);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - tpGroupSize is %u.",
            tilingData->commonTiling.tpGroupSize);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - expert is %lu.", tilingData->commonTiling.expert);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - EOverEp is %lu.", tilingData->commonTiling.EOverEp);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - C is %lu.", tilingData->commonTiling.C);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - COverTp is %lu.", tilingData->commonTiling.COverTp);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - H is %lu.", tilingData->commonTiling.H);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - HOverTp is %lu.", tilingData->commonTiling.HOverTp);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - MOverTp is %lu.", tilingData->commonTiling.MOverTp);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - aivCoreNum is %u.",
            tilingData->commonTiling.aivCoreNum);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - inputDatasize is %u.",
            tilingData->commonTiling.inputDatasize);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - biasDatasize is %u.",
            tilingData->commonTiling.biasDatasize);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - ubCapacityForAdd is %lu.",
            tilingData->commonTiling.ubCapacityForAdd);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - totalUbSize is %lu.",
            tilingData->commonTiling.totalUbSize);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - isBias is %d.", tilingData->commonTiling.isBias);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - isWeightTrans is %d.",
            tilingData->commonTiling.isWeightTrans);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - yShardFlag is %u.",
            tilingData->commonTiling.yShardFlag);
}

static void PrintSliceTileInfo(BatchMatMulReduceScatterAlltoAllTilingData *tilingData)
{
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local E - Tile length is %lu.",
            tilingData->commonTiling.localTileE.tileLen);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local E - Tile Count is %lu.",
            tilingData->commonTiling.localTileE.tileCnt);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local E - Tail length is %lu.",
            tilingData->commonTiling.localTileE.tailLen);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local E - Tail Count is %lu.",
            tilingData->commonTiling.localTileE.tailCnt);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local C - Tile length is %lu.",
            tilingData->commonTiling.localTileC.tileLen);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local C - Tile Count is %lu.",
            tilingData->commonTiling.localTileC.tileCnt);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local C - Tail length is %lu.",
            tilingData->commonTiling.localTileC.tailLen);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local C - Tail Count is %lu.",
            tilingData->commonTiling.localTileC.tailCnt);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local E - Tile length is %lu.",
            tilingData->commonTiling.domesticTileE.tileLen);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local E - Tile Count is %lu.",
            tilingData->commonTiling.domesticTileE.tileCnt);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local E - Tail length is %lu.",
            tilingData->commonTiling.domesticTileE.tailLen);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local E - Tail Count is %lu.",
            tilingData->commonTiling.domesticTileE.tailCnt);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local C - Tile length is %lu.",
            tilingData->commonTiling.domesticTileC.tileLen);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local C - Tile Count is %lu.",
            tilingData->commonTiling.domesticTileC.tileCnt);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local C - Tail length is %lu.",
            tilingData->commonTiling.domesticTileC.tailLen);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local C - Tail Count is %lu.",
            tilingData->commonTiling.domesticTileC.tailCnt);
}

static matmul_tiling::DataType GetMatMulTilingDataType(ge::DataType geDtype)
{
    if (geDtype == ge::DT_BF16) {
        return matmul_tiling::DataType::DT_BFLOAT16;
    } else if (geDtype == ge::DT_FLOAT16) {
        return matmul_tiling::DataType::DT_FLOAT16;
    } else {
        return matmul_tiling::DataType::DT_FLOAT;
    }
}

static uint32_t GetDataSize(ge::DataType geDtype)
{
    if (geDtype == ge::DT_BF16 || geDtype == ge::DT_FLOAT16) {
        return FP16_DATASIZE;
    } else {
        return FP32_DATASIZE;
    }
}

static void InitTileInfo(BatchMatMulReduceScatterAlltoAllTilingData *tilingData)
{
    if (tilingData->commonTiling.yShardFlag == 1U) {
        tilingData->commonTiling.localTileC.tileLen = tilingData->commonTiling.C / tilingData->commonTiling.tpGroupSize;
    } else {
        tilingData->commonTiling.localTileC.tileLen = tilingData->commonTiling.C;
    }
    tilingData->commonTiling.localTileC.tileCnt = 1U;
    tilingData->commonTiling.localTileC.tailLen = 0U;
    tilingData->commonTiling.localTileC.tailCnt = 0U;

    tilingData->commonTiling.localTileE.tileLen = tilingData->commonTiling.EOverEp;
    tilingData->commonTiling.localTileE.tileCnt = 1U;
    tilingData->commonTiling.localTileE.tailLen = 0U;
    tilingData->commonTiling.localTileE.tailCnt = 0U;

    if (tilingData->commonTiling.yShardFlag == 1U) {
        tilingData->commonTiling.domesticTileC.tileLen =
            tilingData->commonTiling.C / tilingData->commonTiling.tpGroupSize;
    } else {
        tilingData->commonTiling.domesticTileC.tileLen = tilingData->commonTiling.C;
    }
    tilingData->commonTiling.domesticTileC.tileCnt = 1U;
    tilingData->commonTiling.domesticTileC.tailLen = 0U;
    tilingData->commonTiling.domesticTileC.tailCnt = 0U;

    tilingData->commonTiling.domesticTileE.tileLen = tilingData->commonTiling.EOverEp;
    tilingData->commonTiling.domesticTileE.tileCnt = 1U;
    tilingData->commonTiling.domesticTileE.tailLen = 0U;
    tilingData->commonTiling.domesticTileE.tailCnt = 0U;
}

static ge::graphStatus CalculateMaxSplitUB(int64_t ubSize, bool bias_flag, bool xCastFlag, bool biasCastFlag,
                                           BatchMatMulReduceScatterAlltoAllTilingData *tilingData)
{
    uint64_t ubCapacity = 1U;
    uint64_t addLeft = 2U;
    uint64_t addLeftCast = 0U;
    uint64_t addRight = 0U;
    uint64_t addRightCast = 0U;
    uint64_t addOut = 0U;
    uint64_t coeff = 1U;

    if (bias_flag) {
        addRight = FP16_DATASIZE;
        addOut = FP16_DATASIZE;
    }
    if (xCastFlag) {
        addLeftCast = FP32_DATASIZE;
        if (bias_flag) {
            addOut = FP32_DATASIZE;
        }
    }
    if (biasCastFlag) {
        addRightCast = FP32_DATASIZE;
        addOut = FP32_DATASIZE;
    }

    coeff = addLeft + addLeftCast + addRight + addRightCast + addOut;
    ubCapacity = ubSize / coeff;
    ubCapacity = ubCapacity / ALIGN16 * ALIGN16;
    tilingData->commonTiling.ubCapacityForAdd = ubCapacity;
    return ge::GRAPH_SUCCESS;
}

static uint64_t UpdateTilingKey(BatchMatMulReduceScatterAlltoAllTilingData *tilingData, bool isLite)
{
    uint8_t yShardFlag = tilingData->commonTiling.yShardFlag;
    bool isWeightTrans = tilingData->commonTiling.isWeightTrans;
    bool isBias = tilingData->commonTiling.isBias;

    uint64_t tilingKey = GET_TPL_TILING_KEY(yShardFlag, isWeightTrans, isBias, isLite);

    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "TPL Tiling Key Print : Tiling key parameter : xShardFlag is %u.",
            yShardFlag);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "TPL Tiling Key Print : Tiling key parameter : Weight Transpose is %d.",
            isWeightTrans);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "TPL Tiling Key Print : Tiling key parameter : Bias is %d.", isBias);
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "TPL Tiling Key Print : Tiling key parameter : Lite Mode is %d.",
            isLite);

    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "The final tiling Key is: %lu!", tilingKey);
    return tilingKey;
}

static void CompleteBmmStructs(ReduceScatterAlltoAllBatchInfo &BMMV3BatchInfo,
                               ReduceScatterAlltoAllMatmulInfo &MMV3ArgsInfo, uint32_t m, uint32_t n, uint32_t k,
                               uint32_t b)
{
    MMV3ArgsInfo.mValue = m;
    MMV3ArgsInfo.nValue = n;
    MMV3ArgsInfo.kValue = k;
    BMMV3BatchInfo.batchA = b;
    BMMV3BatchInfo.batchB = b;
    BMMV3BatchInfo.batchC = b;
    BMMV3BatchInfo.batchA3 = b;
    BMMV3BatchInfo.batchB3 = b;
    BMMV3BatchInfo.batchC3 = b;
    return;
}

static void GetBatchMatMulReduceScatterAlltoAllFormulateTileCnt(mc2tiling::TilingArgs &formulaicArgs,
                                                                BatchMatMulReduceScatterAlltoAllTilingData *tilingData)
{
    ReduceScatterAll2AllBMMShardH formulaicTiling(formulaicArgs, tilingData->commonTiling.epGroupSize,
                                                  tilingData->commonTiling.tpGroupSize,
                                                  tilingData->commonTiling.EOverEp);
    formulaicTiling.GetTiling();

    if (formulaicTiling.tilingC.cutRes.shortTileAtBack) {
        tilingData->commonTiling.domesticTileC.tileLen = formulaicTiling.tilingC.cutRes.longTileLen;
        tilingData->commonTiling.domesticTileC.tileCnt = formulaicTiling.tilingC.cutRes.numLongTile;
        tilingData->commonTiling.domesticTileC.tailLen = formulaicTiling.tilingC.cutRes.shortTileLen;
        tilingData->commonTiling.domesticTileC.tailCnt = formulaicTiling.tilingC.cutRes.numShortTile;
    } else {
        tilingData->commonTiling.domesticTileC.tileLen = formulaicTiling.tilingC.cutRes.shortTileLen;
        tilingData->commonTiling.domesticTileC.tileCnt = formulaicTiling.tilingC.cutRes.numShortTile;
        tilingData->commonTiling.domesticTileC.tailLen = formulaicTiling.tilingC.cutRes.longTileLen;
        tilingData->commonTiling.domesticTileC.tailCnt = formulaicTiling.tilingC.cutRes.numLongTile;
    }

    if (formulaicTiling.cutE.shortTileAtBack) {
        tilingData->commonTiling.domesticTileE.tileLen = formulaicTiling.cutE.longTileLen;
        tilingData->commonTiling.domesticTileE.tileCnt = formulaicTiling.cutE.numLongTile;
        tilingData->commonTiling.domesticTileE.tailLen = formulaicTiling.cutE.shortTileLen;
        tilingData->commonTiling.domesticTileE.tailCnt = formulaicTiling.cutE.numShortTile;
    } else {
        tilingData->commonTiling.domesticTileE.tileLen = formulaicTiling.cutE.shortTileLen;
        tilingData->commonTiling.domesticTileE.tileCnt = formulaicTiling.cutE.numShortTile;
        tilingData->commonTiling.domesticTileE.tailLen = formulaicTiling.cutE.longTileLen;
        tilingData->commonTiling.domesticTileE.tailCnt = formulaicTiling.cutE.numLongTile;
    }

    if (formulaicTiling.localCutE.shortTileAtBack) {
        tilingData->commonTiling.localTileE.tileLen = formulaicTiling.localCutE.longTileLen;
        tilingData->commonTiling.localTileE.tileCnt = formulaicTiling.localCutE.numLongTile;
        tilingData->commonTiling.localTileE.tailLen = formulaicTiling.localCutE.shortTileLen;
        tilingData->commonTiling.localTileE.tailCnt = formulaicTiling.localCutE.numShortTile;
    } else {
        tilingData->commonTiling.localTileE.tileLen = formulaicTiling.localCutE.shortTileLen;
        tilingData->commonTiling.localTileE.tileCnt = formulaicTiling.localCutE.numShortTile;
        tilingData->commonTiling.localTileE.tailLen = formulaicTiling.localCutE.longTileLen;
        tilingData->commonTiling.localTileE.tailCnt = formulaicTiling.localCutE.numLongTile;
    }
    return;
}

static void GetBatchMatMulReduceScatterAlltoAllFormulateTileCntShard(
    mc2tiling::TilingArgs &formulaicArgs, BatchMatMulReduceScatterAlltoAllTilingData *tilingData)
{
    ReduceScatterAll2AllBMM formulaicTiling(formulaicArgs, tilingData->commonTiling.epGroupSize,
                                            tilingData->commonTiling.tpGroupSize, tilingData->commonTiling.EOverEp);
    formulaicTiling.GetTiling();

    tilingData->commonTiling.domesticTileC.tileLen =
        formulaicTiling.tilingC.cutRes.longTileLen / tilingData->commonTiling.tpGroupSize;
    tilingData->commonTiling.domesticTileC.tileCnt = formulaicTiling.tilingC.cutRes.numLongTile;
    tilingData->commonTiling.domesticTileC.tailLen =
        formulaicTiling.tilingC.cutRes.shortTileLen / tilingData->commonTiling.tpGroupSize;
    tilingData->commonTiling.domesticTileC.tailCnt = formulaicTiling.tilingC.cutRes.numShortTile;

    tilingData->commonTiling.domesticTileE.tileLen = formulaicTiling.cutE.longTileLen;
    tilingData->commonTiling.domesticTileE.tileCnt = formulaicTiling.cutE.numLongTile;
    tilingData->commonTiling.domesticTileE.tailLen = formulaicTiling.cutE.shortTileLen;
    tilingData->commonTiling.domesticTileE.tailCnt = formulaicTiling.cutE.numShortTile;

    tilingData->commonTiling.localTileE.tileLen = formulaicTiling.localCutE.longTileLen;
    tilingData->commonTiling.localTileE.tileCnt = formulaicTiling.localCutE.numLongTile;
    tilingData->commonTiling.localTileE.tailLen = formulaicTiling.localCutE.shortTileLen;
    tilingData->commonTiling.localTileE.tailCnt = formulaicTiling.localCutE.numShortTile;

    return;
}

static ge::graphStatus DoBmmTiling(BmmTilingConfig &config, uint32_t tileLen, uint32_t batch, bool isLocal, bool isTail)
{
    CompleteBmmStructs(config.bmmV3BatchInfo, config.mmV3ArgsInfo, tileLen, config.tilingData.commonTiling.H,
                       config.tilingData.commonTiling.MOverTp, batch);

    auto bmmTilingLambda = [isLocal, isTail, &config]() -> decltype(config.tilingData.domesticTiling.bmmTilingData) & {
        if (isLocal && isTail) {
            return config.tilingData.localTailTiling.bmmTilingData;
        }
        if (isLocal) {
            return config.tilingData.localTiling.bmmTilingData;
        }
        if (isTail) {
            return config.tilingData.domesticTailTiling.bmmTilingData;
        }
        return config.tilingData.domesticTiling.bmmTilingData;
    };
    auto &bmmTilingData = bmmTilingLambda();

    BatchMatMulReduceScatterAlltoAllTiling bmmTiling(config.context, bmmTilingData, config.bmmV3BatchInfo,
                                                     config.mmV3ArgsInfo);
    if (bmmTiling.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(config.context->GetNodeName(), "Do BmmV3Tiling failed under shard-1 %s %s section.",
                isLocal ? "local" : "non-local", isTail ? "tail" : "standard");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetMatmulTilingBatchMatMulReduceScatterAlltoAll(BmmTilingConfig &config,
                                                                       mc2tiling::TilingArgs &formulaicArgs,
                                                                       bool isLite)
{
    // 1. 调用公式化tiling接口，获取local和非local块的切分信息并更新commonTiling
    if (config.tilingData.commonTiling.yShardFlag == 0) {
        GetBatchMatMulReduceScatterAlltoAllFormulateTileCnt(formulaicArgs, &(config.tilingData));
    } else {
        GetBatchMatMulReduceScatterAlltoAllFormulateTileCntShard(formulaicArgs, &(config.tilingData));
    }

    uint32_t factor = (isLite) ? config.tilingData.commonTiling.tpGroupSize : 1U;
    uint32_t localBatch = (isLite) ? config.tilingData.commonTiling.localTileE.tileLen : 1U;
    uint32_t nonLocalBatch = (isLite) ? config.tilingData.commonTiling.domesticTileE.tileLen : 1U;
    // 2. 将切块信息传入BmmV3 tiling，根据后续整改方案更新Bmm所需的相应参数并调用BmmV3
    // Tiling接口，获取local/non-local的BMM tiling信息 Local standard slice BMM tiling
    if (DoBmmTiling(config, config.tilingData.commonTiling.localTileC.tileLen * factor, localBatch, true, false) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Local tail slice BMM tiling
    if (config.tilingData.commonTiling.localTileC.tailLen != 0) {
        if (DoBmmTiling(config, config.tilingData.commonTiling.localTileC.tailLen * factor, localBatch, true, true) !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    factor = (isLite) ?
                 (config.tilingData.commonTiling.tpGroupSize * (config.tilingData.commonTiling.epGroupSize - 1U)) :
                 1U;
    // Non-local standard slice BMM tiling
    if (DoBmmTiling(config, config.tilingData.commonTiling.domesticTileC.tileLen * factor, nonLocalBatch, false,
                    false) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Non-local tail slice BMM tiling
    if (config.tilingData.commonTiling.domesticTileC.tailLen != 0) {
        if (DoBmmTiling(config, config.tilingData.commonTiling.domesticTileC.tailLen * factor, nonLocalBatch, false,
                        true) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static bool CommonCheckTensorShape(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape,
                                   const size_t wDimM)
{
    // 检查 < 0 的范围: x dim C >= 1, dim E H M 会在后面拦截
    if (xShape->GetDim(X_DIM_C) < VALUE_C_MIN) {
        OP_LOGE_FOR_INVALID_SHAPE(nodeName, "x", std::to_string(xShape->GetDim(X_DIM_C)).c_str(),
                                  std::string("not < " + std::to_string(VALUE_C_MIN)).c_str());
        return false;
    }

    // x[2]、w[wDimM] 是 M 轴(reduce 轴)，所以需要相等
    if (xShape->GetDim(X_DIM_M) != weightShape->GetDim(wDimM)) {
        OP_LOGE_FOR_INVALID_SHAPE(
            nodeName, "x or weight dim-M",
            (std::to_string(xShape->GetDim(X_DIM_M)) + " vs " + std::to_string(weightShape->GetDim(wDimM))).c_str(),
            "should be equal");
        return false;
    }

    // x[2]、w[wDimM] 是 M 轴，不支持为 0
    if (xShape->GetDim(X_DIM_M) == 0) {
        OP_LOGE_FOR_INVALID_SHAPE(nodeName, "x or weight dim-M", "0", "> 0");
        return false;
    }

    return true;
}

// GPT 2.2T
static bool YShardCheckTensorShape(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape,
                                   const int64_t yShard, const int64_t epSize, const int64_t tpSize, const size_t wDimH,
                                   const size_t wDimM)
{
    // 检查 shape 维度的范围
    // value E should = [2, 512], x[DIM_E] = E / Ep
    if ((xShape->GetDim(DIM_E) * epSize < VALUE_E_MIN) || (xShape->GetDim(DIM_E) * epSize > VALUE_E_MAX)) {
        OP_LOGE_FOR_INVALID_SHAPE(
            nodeName, "x", std::to_string(xShape->GetDim(DIM_E) * epSize).c_str(),
            (std::string("[") + std::to_string(VALUE_E_MIN) + ", " + std::to_string(VALUE_E_MAX) + "]").c_str());
        return false;
    }
    // w[wDimH] = H, value H should = [1, 65535]
    if ((weightShape->GetDim(wDimH) < VALUE_H_MIN) || (weightShape->GetDim(wDimH) > VALUE_H_MAX)) {
        OP_LOGE_FOR_INVALID_SHAPE(
            nodeName, "weight", std::to_string(weightShape->GetDim(wDimH)).c_str(),
            (std::string("[") + std::to_string(VALUE_H_MIN) + ", " + std::to_string(VALUE_H_MAX) + "]").c_str());
        return false;
    }
    // w[wDimM] = M / Tp, its range should same with H, so it meets M / Tp * H <= 65535 * 65535
    if ((weightShape->GetDim(wDimM) < VALUE_H_MIN) || (weightShape->GetDim(wDimM) > VALUE_H_MAX)) {
        OP_LOGE_FOR_INVALID_SHAPE(
            nodeName, "weight", std::to_string(weightShape->GetDim(wDimM)).c_str(),
            (std::string("[") + std::to_string(VALUE_H_MIN) + ", " + std::to_string(VALUE_H_MAX) + "]").c_str());
        return false;
    }

    // x[0] = E / Ep, w[0] = E / Ep, 所以两者需要相等
    if (xShape->GetDim(DIM_E) != weightShape->GetDim(DIM_E)) {
        OP_LOGE_FOR_INVALID_SHAPE(
            nodeName, "x or weight dim-E",
            (std::to_string(xShape->GetDim(DIM_E)) + " vs " + std::to_string(weightShape->GetDim(DIM_E))).c_str(),
            "should be equal");
        return false;
    }

    if (yShard == 0) {
        // x[1] = Ep * C, 所以 x[1] % Ep 需要等于 0
        if (xShape->GetDim(X_DIM_C) % epSize != 0) {
            OP_LOGE_FOR_INVALID_SHAPE(nodeName, "x", std::to_string(xShape->GetDim(X_DIM_C)).c_str(),
                                      (std::string("multiple of ") + std::to_string(epSize)).c_str());
            return false;
        }
    } else if (yShard == 1) {
        // x[1] = (c / Tp) * Ep * Tp, 所以 x[1] % (Ep * Tp) 需要等于 0
        if (xShape->GetDim(X_DIM_C) % (epSize * tpSize) != 0) {
            OP_LOGE_FOR_INVALID_SHAPE(nodeName, "x", std::to_string(xShape->GetDim(X_DIM_C)).c_str(),
                                      (std::string("multiple of ") + std::to_string(epSize * tpSize)).c_str());
            return false;
        }
    }

    return true;
}

static bool CheckBiasShape(const char *nodeName, const gert::Shape *weightShape, const gert::Shape *biasShape,
                           const int64_t yShard, const int64_t tpSize, const size_t wDimH)
{
    // 检查 dimNum
    if ((biasShape->GetDimNum() != SUPPORT_DIM_NUM) && (biasShape->GetDimNum() != BIAS_SUPPORT_DIM_NUM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "bias", (std::to_string(biasShape->GetDimNum()) + "D").c_str(),
                                     "2D or 3D");
        return false;
    }

    // 检查 shape
    if (biasShape->GetDim(0) != weightShape->GetDim(0)) {
        OP_LOGE_FOR_INVALID_SHAPE(nodeName, "bias dim-0", std::to_string(biasShape->GetDim(0)).c_str(),
                                  std::to_string(weightShape->GetDim(0)).c_str());
        return false;
    }

    size_t biasLastDimValue = 1U;                    // 默认 bias 是二维，所以最后一维的 index 是 1
    if (biasShape->GetDimNum() == SUPPORT_DIM_NUM) { // 三维
        if (biasShape->GetDim(1) != 1) {
            OP_LOGE_FOR_INVALID_SHAPE(nodeName, "bias", std::to_string(biasShape->GetDim(1)).c_str(), "1");
            return false;
        }
        biasLastDimValue = 2; // 三维时候，bias 的最后一维是 2
    }

    if (yShard == 1) {
        if (biasShape->GetDim(biasLastDimValue) != weightShape->GetDim(wDimH)) {
            OP_LOGE_FOR_INVALID_SHAPE(nodeName, "bias", std::to_string(biasShape->GetDim(biasLastDimValue)).c_str(),
                                      std::to_string(weightShape->GetDim(wDimH)).c_str());
            return false;
        }
    } else if (yShard == 0) {
        if (biasShape->GetDim(biasLastDimValue) * tpSize != weightShape->GetDim(wDimH)) {
            OP_LOGE_FOR_INVALID_SHAPE(nodeName, "bias",
                                      std::to_string(biasShape->GetDim(biasLastDimValue) * tpSize).c_str(),
                                      std::to_string(weightShape->GetDim(wDimH)).c_str());
            return false;
        }
    }

    return true;
}

static bool CheckTensorShape(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape,
                             const gert::Shape *biasShape, const int64_t epSize, const int64_t tpSize,
                             const size_t wDimM, const size_t wDimH, const int64_t yShard)
{
    if ((xShape == nullptr) || (weightShape == nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, (xShape == nullptr) ? "xShape" : "weightShape");
        return false;
    }
    if (!DimNumCheck(nodeName, xShape, weightShape)) {
        OP_LOGE(nodeName, "Dim num check failed.");
        return false;
    }

    if (!CommonCheckTensorShape(nodeName, xShape, weightShape, wDimM)) {
        OP_LOGE(nodeName, "common tensor shape check failed.");
        return false;
    }

    if (yShard == 0 || yShard == 1) {
        if (!YShardCheckTensorShape(nodeName, xShape, weightShape, yShard, epSize, tpSize, wDimH, wDimM)) {
            OP_LOGE(nodeName, "yShard = [%ld] tensor shape check failed.", yShard);
            return false;
        }
    } else {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "yShardType", std::to_string(yShard).c_str(), "0 or 1");
        return false;
    }

    if (biasShape != nullptr) {
        if (!(CheckBiasShape(nodeName, weightShape, biasShape, yShard, tpSize, wDimH))) {
            OP_LOGE(nodeName, "bias shape check failed.");
            return false;
        }
    }

    OP_LOGI(nodeName, "tensor shape check success.");
    return true;
}

static bool CheckAttrs(const gert::TilingContext *context, int64_t &epSize, int64_t &tpSize, int64_t &yShard)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return false);

    const char *groupEp = attrs->GetStr(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_GROUP_EP));
    const char *groupTp = attrs->GetStr(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_GROUP_TP));
    const int64_t *epPtr = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_EP_WORLD_SIZE));
    const int64_t *tpPtr = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_TP_WORLD_SIZE));
    const int64_t *yPtr = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_Y_SHARD_TYPE));

    if ((epPtr == nullptr) || (tpPtr == nullptr) || (yPtr == nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "attr pointer");
        return false;
    }

    epSize = *epPtr;
    tpSize = *tpPtr;
    yShard = *yPtr;

    if (!GroupCheck(nodeName, groupEp, groupTp)) {
        OP_LOGE(nodeName, "group size check failed.");
        return false;
    }

    if (!EpTpSizeCheck(epSize, tpSize)) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "tpSize or epSize",
                                  (std::to_string(tpSize) + "/" + std::to_string(epSize)).c_str(), "2/4/8/16/32");
        return false;
    }

    if ((yShard != 1) && (yShard != 0)) { // 当前仅支持 shard = 0 or 1
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "yShardType", std::to_string(yShard).c_str(), "0 or 1");
        return false;
    }

    OP_LOGI(nodeName, "attr info: groupEp %s, groupTp %s, tpSize %ld, epSize %ld, yShard %ld.", groupEp, groupTp,
            tpSize, epSize, yShard);
    return true;
}

static ge::graphStatus TilingCheckBatchMatMulReduceScatterAlltoAll(gert::TilingContext *context)
{
    // 检查 shape 是否为空
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter BmmReduceScatterAlltoAll tiling check impl.");
    const gert::StorageShape *xStorageShape = context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_X));

    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    const gert::StorageShape *weightStorageShape =
        context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_WEIGHT));
    OP_TILING_CHECK(weightStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "weight"),
                    return ge::GRAPH_FAILED);
    const gert::StorageShape *yStorageShape =
        context->GetOutputShape(static_cast<size_t>(ops::BmmReduceScatterAlltoAllOutIdx::K_Y));
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y"), return ge::GRAPH_FAILED);

    // 检查属性
    int64_t epSize = -1;
    int64_t tpSize = -1;
    int64_t yShard = -1;
    if (!CheckAttrs(context, epSize, tpSize, yShard)) {
        OP_LOGE(nodeName, "Attrs check failed.");
        return ge::GRAPH_FAILED;
    }

    // 设 w = [E, M, H], dimH 指 H 轴, dimM 指 M 轴
    size_t wDimM = 1UL;
    size_t wDimH = 2UL; // 1 2 分别代表 weight 没有转置时候的维度值, 2: H 轴, 1: M 轴
    auto attrs = context->GetAttrs();
    bool isWeightTrans = *(attrs->GetAttrPointer<bool>(ATTR_IS_WEIGHT_TRANS_INDEX));
    // w_trans = [E, H, M]
    if (isWeightTrans) {
        size_t wDimMTrans = 2UL;
        size_t wDimHTrans = 1UL; // 2 1 分别代表 weight 转置时候的维度值, 2: M 轴, 1: H 轴
        wDimM = wDimMTrans;
        wDimH = wDimHTrans;
    }

    const gert::StorageShape *biasStorageShape =
        context->GetOptionalInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_BIAS));

    const gert::Shape *xShape = &xStorageShape->GetStorageShape();
    const gert::Shape *weightShape = &weightStorageShape->GetStorageShape();
    const gert::Shape *biasShape = (biasStorageShape == nullptr) ? nullptr : &biasStorageShape->GetStorageShape();

    // tensor shape 检查
    if (!CheckTensorShape(nodeName, xShape, weightShape, biasShape, epSize, tpSize, wDimM, wDimH, yShard)) {
        OP_LOGE(nodeName, "Tensor shape check failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MC2SetWorkspace(gert::TilingContext *context,
                                       BatchMatMulReduceScatterAlltoAllTilingData *tilingData)
{
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "get workspace failed"),
                    return ge::GRAPH_FAILED);

    // 2EC*h + ECH
    uint64_t commOut =
        2UL * tilingData->commonTiling.expert * tilingData->commonTiling.C * tilingData->commonTiling.HOverTp +
        tilingData->commonTiling.expert * tilingData->commonTiling.C * tilingData->commonTiling.H;

    commOut = commOut * static_cast<uint64_t>(tilingData->commonTiling.inputDatasize);

    uint64_t maxLocalELen =
        std::max(tilingData->commonTiling.localTileE.tileLen, tilingData->commonTiling.localTileE.tailLen);
    uint64_t maxLocalCLen =
        std::max(tilingData->commonTiling.localTileC.tileLen, tilingData->commonTiling.localTileC.tailLen);
    uint64_t maxNonLocalELen =
        std::max(tilingData->commonTiling.domesticTileE.tileLen, tilingData->commonTiling.domesticTileE.tailLen);
    uint64_t maxNonLocalCLen =
        std::max(tilingData->commonTiling.domesticTileC.tileLen, tilingData->commonTiling.domesticTileC.tailLen);

    uint64_t transOut = std::max(maxLocalELen * maxLocalCLen * tilingData->commonTiling.H,
                                 static_cast<uint64_t>(tilingData->commonTiling.epGroupSize - 1) * maxNonLocalELen *
                                     maxNonLocalCLen * tilingData->commonTiling.H);

    transOut = transOut * static_cast<uint64_t>(tilingData->commonTiling.inputDatasize);

    workspaces[0] = commOut + transOut + 16 * 1024 * 1024; // 16 mb, 1024 * 1024 is 1 mb
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "workspaces[0] size %ld", workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MC2SetWorkspaceShard(gert::TilingContext *context,
                                            BatchMatMulReduceScatterAlltoAllTilingData *tilingData, bool isLite)
{
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "get workspace failed"),
                    return ge::GRAPH_FAILED);

    // 2EcH + ECH
    uint64_t commOut = 2 * static_cast<uint64_t>(tilingData->commonTiling.expert) *
                           static_cast<uint64_t>(tilingData->commonTiling.COverTp) *
                           static_cast<uint64_t>(tilingData->commonTiling.H) +
                       static_cast<uint64_t>(tilingData->commonTiling.expert) *
                           static_cast<uint64_t>(tilingData->commonTiling.C) *
                           static_cast<uint64_t>(tilingData->commonTiling.H);

    if (isLite) {
        uint64_t localMaxC =
            std::max(tilingData->commonTiling.localTileC.tileLen, tilingData->commonTiling.localTileC.tailLen);
        uint64_t nonLocalMaxC =
            std::max(tilingData->commonTiling.domesticTileC.tileLen, tilingData->commonTiling.domesticTileC.tailLen);
        uint64_t localTransBefore = localMaxC * tilingData->commonTiling.localTileE.tileLen *
                                    tilingData->commonTiling.tpGroupSize * tilingData->commonTiling.MOverTp;
        uint64_t nonLocalTransBefore = nonLocalMaxC * tilingData->commonTiling.domesticTileE.tileLen *
                                       tilingData->commonTiling.tpGroupSize * tilingData->commonTiling.MOverTp *
                                       (tilingData->commonTiling.epGroupSize - 1U);
        uint64_t localTransAfter = localMaxC * tilingData->commonTiling.localTileE.tileLen *
                                   tilingData->commonTiling.tpGroupSize * tilingData->commonTiling.H;
        uint64_t nonLocalTransAfter = nonLocalMaxC * tilingData->commonTiling.domesticTileE.tileLen *
                                      tilingData->commonTiling.tpGroupSize * tilingData->commonTiling.H *
                                      (tilingData->commonTiling.epGroupSize - 1U);
        commOut += std::max(localTransBefore, nonLocalTransBefore) + std::max(localTransAfter, nonLocalTransAfter);
    }

    commOut = commOut * static_cast<uint64_t>(tilingData->commonTiling.inputDatasize);

    workspaces[0] = commOut + 16 * 1024 * 1024; // 16 mb, 1024 * 1024 is 1 mb
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "workspaces[0] size %ld", workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

struct TensorInfo {
    uint64_t inputDatasize = 0;
    uint64_t biasDatasize = 0;
    ge::DataType inputDatatype = ge::DT_FLOAT;
    ge::DataType biasDatatype = ge::DT_FLOAT;
    bool isLite = false;
    const char *epGroup = nullptr;
    const char *tpGroup = nullptr;
    uint64_t aicCoreNum = 0; // AIC 核心数
    uint64_t aivCoreNum = 0; // AIV 核心数
    uint64_t ubSize = 0;     // UB 大小
    uint32_t numBlocks = 1;
    bool isWeightTrans = false;
    // 存放 tiling 公式所需的结构体参数
    ReduceScatterAlltoAllMatmulInfo mmv3ArgsInfo;
    ReduceScatterAlltoAllBatchInfo bmmv3BatchInfo;
};
// 子函数2: 获取平台信息
static void GetPlatformInfo(const gert::TilingContext *context, TensorInfo &tensorInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    tensorInfo.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    tensorInfo.aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tensorInfo.ubSize);
    tensorInfo.numBlocks =
        ascendcPlatform.CalcTschBlockDim(tensorInfo.aivCoreNum, tensorInfo.aicCoreNum, tensorInfo.aivCoreNum);
}

// 子函数3: 获取关键参数并计算，填充tilingdata
static ge::graphStatus CalculateTensorInfo(const gert::TilingContext *context, TensorInfo &tensorInfo,
                                           BatchMatMulReduceScatterAlltoAllTilingData *tilingData,
                                           const gert::StorageShape *biasInputShape)
{
    constexpr int MATMUL_INPUT_M_AXIS = 2;
    const gert::StorageShape *xInputShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *weightInputShape = context->GetInputShape(WEIGHT_INDEX);
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    auto attrs = context->GetAttrs();
    int64_t ep = *(attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX));
    int64_t tp = *(attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX));
    auto yShard = attrs->GetAttrPointer<int64_t>(ATTR_Y_SHARD_TYPE_INDEX);
    tensorInfo.isWeightTrans = *(attrs->GetAttrPointer<bool>(ATTR_IS_WEIGHT_TRANS_INDEX));
    tensorInfo.epGroup = attrs->GetAttrPointer<char>(ATTR_EP_GROUP_INDEX);
    tensorInfo.tpGroup = attrs->GetAttrPointer<char>(ATTR_TP_GROUP_INDEX);
    const char *epGroupStr = tensorInfo.epGroup ? tensorInfo.epGroup : "null";
    const char *tpGroupStr = tensorInfo.tpGroup ? tensorInfo.tpGroup : "null";
    OP_LOGD("BatchMatMulReduceScatterAlltoAll",
            "EP group is %s, ep is %ld, TP group is %s, tp is %ld, weight_trans_flag is %d, y_shard_flag is %ld",
            epGroupStr, ep, tpGroupStr, tp, tensorInfo.isWeightTrans, *yShard);
    int64_t e = xInputShape->GetStorageShape().GetDim(0);
    int64_t E = yShape->GetStorageShape().GetDim(0);
    int64_t C = (*yShard == 1) ? yShape->GetStorageShape().GetDim(1) * tp : yShape->GetStorageShape().GetDim(1);
    int64_t c = (*yShard == 1) ? yShape->GetStorageShape().GetDim(1) : yShape->GetStorageShape().GetDim(1) / tp;
    size_t wDimH = 2U;
    OP_TILING_CHECK(e > MAX_HCCL_HANDLE_LIMIT,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "E/ep", std::to_string(e).c_str(), "<= 32"),
                    return ge::GRAPH_FAILED);
    wDimH = tensorInfo.isWeightTrans ? 1U : 2U;
    int64_t dimH = weightInputShape->GetStorageShape().GetDim(wDimH);
    tensorInfo.inputDatasize = GetDataSize(tensorInfo.inputDatatype);
    tensorInfo.biasDatasize = GetDataSize(tensorInfo.biasDatatype);
    tensorInfo.isLite = (c <= LITE_THRESHOLD) && (*yShard == 1);
    tilingData->commonTiling.COverTp = c;
    tilingData->commonTiling.C = C;
    tilingData->commonTiling.epGroupSize = ep;
    tilingData->commonTiling.tpGroupSize = tp;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.EOverEp = e;
    tilingData->commonTiling.H = dimH;
    tilingData->commonTiling.HOverTp = static_cast<int64_t>(dimH / tp);
    tilingData->commonTiling.MOverTp = xInputShape->GetStorageShape().GetDim(MATMUL_INPUT_M_AXIS); // m
    tilingData->commonTiling.yShardFlag = *yShard;
    tilingData->commonTiling.isBias = biasInputShape == nullptr ? false : true;
    tilingData->commonTiling.inputDatasize = tensorInfo.inputDatasize;
    tilingData->commonTiling.biasDatasize = tensorInfo.biasDatasize;
    tilingData->commonTiling.aivCoreNum = tensorInfo.aivCoreNum;
    tilingData->commonTiling.isWeightTrans = tensorInfo.isWeightTrans;
    tilingData->commonTiling.totalUbSize = tensorInfo.ubSize;
    InitTileInfo(tilingData);
    return ge::GRAPH_SUCCESS;
}

static void SetHcclTiling(const gert::TilingContext *context, BatchMatMulReduceScatterAlltoAllTilingData *tilingData)
{
    std::string reduceScatterConfig = "ReduceScatter=level0:doublering";
    std::string alltoAllConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"), return);
    auto epGroup = attrs->GetAttrPointer<char>(ATTR_EP_GROUP_INDEX);
    auto tpGroup = attrs->GetAttrPointer<char>(ATTR_TP_GROUP_INDEX);

    const uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    const uint32_t opType2 = OP_TYPE_REDUCE_SCATTER;
    const uint32_t reduceType = 0U;
    ge::DataType outputDataType = context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    ge::DataType inputDataType = context->GetInputDesc(X_INDEX)->GetDataType();
    OP_TILING_CHECK(mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
                    OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y", Ops::Base::ToString(outputDataType).c_str(),
                                              "supported HCCL data type"),
                    return);
    OP_TILING_CHECK(mc2tiling::HCCL_DATA_TYPE.find(inputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
                    OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(inputDataType).c_str(),
                                              "supported HCCL data type"),
                    return);

    auto dstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto srcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(inputDataType)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(epGroup, opType1, alltoAllConfig, reduceType, dstDataType, srcDataType);
    hcclCcTilingConfig.GetTiling(tilingData->hcclInitTiling);
    hcclCcTilingConfig.GetTiling(tilingData->alltoAllCcTiling);
    hcclCcTilingConfig.SetGroupName(tpGroup);
    hcclCcTilingConfig.SetOpType(opType2);
    hcclCcTilingConfig.SetAlgConfig(reduceScatterConfig);
    hcclCcTilingConfig.GetTiling(tilingData->reduceScatterCcTiling);
    return;
}

// 子函数5: 设置公式参数
static void SetFormulaicArgs(mc2tiling::TilingArgs &formulaicArgs,
                             BatchMatMulReduceScatterAlltoAllTilingData *tilingData, TensorInfo &tensorInfo)
{
    auto &MMV3ArgsInfo = tensorInfo.mmv3ArgsInfo;
    auto &BMMV3BatchInfo = tensorInfo.bmmv3BatchInfo;
    MMV3ArgsInfo.opName = "BatchMatMulReduceScatterAlltoAll";
    MMV3ArgsInfo.isWeightTrans = tensorInfo.isWeightTrans;
    MMV3ArgsInfo.isBias = false; // 出于性能考虑，当前bias计算在外部进行，算子内不会有bias场景
    MMV3ArgsInfo.aType = tensorInfo.inputDatatype;
    MMV3ArgsInfo.bType = tensorInfo.inputDatatype;
    MMV3ArgsInfo.cType = tensorInfo.inputDatatype;
    MMV3ArgsInfo.biasType = tensorInfo.biasDatatype;
    BMMV3BatchInfo.biasWithBatch = false; // 出于性能考虑，当前bias计算在外部进行，算子内不会有bias场景
    formulaicArgs.mValue = tilingData->commonTiling.C;
    formulaicArgs.nValue = tilingData->commonTiling.H;
    formulaicArgs.kValue = tilingData->commonTiling.MOverTp;
    formulaicArgs.inputDtypeSize = tensorInfo.inputDatasize;
    formulaicArgs.outputDtypeSize = tensorInfo.inputDatasize;
    formulaicArgs.aicCoreNum = tensorInfo.aicCoreNum;
    formulaicArgs.aType = GetMatMulTilingDataType(tensorInfo.inputDatatype);
    formulaicArgs.bType = GetMatMulTilingDataType(tensorInfo.inputDatatype);
    formulaicArgs.cType = GetMatMulTilingDataType(tensorInfo.inputDatatype);
    formulaicArgs.biasType = GetMatMulTilingDataType(tensorInfo.biasDatatype);
}

// 子函数6: 核心tiling逻辑
static ge::graphStatus ComputeCoreTiling(gert::TilingContext *context,
                                         BatchMatMulReduceScatterAlltoAllTilingData *tilingData,
                                         mc2tiling::TilingArgs &formulaicArgs, TensorInfo &tensorInfo)
{
    bool xCastFlag = tensorInfo.inputDatatype == ge::DT_BF16;
    bool biasCastFlag = tensorInfo.biasDatatype == ge::DT_BF16;
    BmmTilingConfig config{context, *tilingData, tensorInfo.bmmv3BatchInfo, tensorInfo.mmv3ArgsInfo};
    // 待修改，等BMM tiling提供接口和修改方案
    OP_TILING_CHECK(
        SetMatmulTilingBatchMatMulReduceScatterAlltoAll(config, formulaicArgs, tensorInfo.isLite) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Set Matmul tiling Failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CalculateMaxSplitUB(tensorInfo.ubSize, tilingData->commonTiling.isBias, xCastFlag, biasCastFlag,
                                        tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Calculate max split UB Failed!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 主函数
static ge::graphStatus BatchMatMulReduceScatterAlltoAllTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(TilingCheckBatchMatMulReduceScatterAlltoAll(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Tiling check shape Failed!"), return ge::GRAPH_FAILED);

    BatchMatMulReduceScatterAlltoAllTilingData *tilingData =
        context->GetTilingData<BatchMatMulReduceScatterAlltoAllTilingData>();
    mc2tiling::TilingArgs formulaicArgs;
    ReduceScatterAlltoAllBatchInfo BMMV3BatchInfo;
    ReduceScatterAlltoAllMatmulInfo MMV3ArgsInfo;
    TensorInfo tensorInfo{};
    // 子函数2 获取aiv,UB
    GetPlatformInfo(context, tensorInfo);
    context->SetBlockDim(tensorInfo.numBlocks);
    // 子函数3 获取参数，计算，填充tilingData
    // 存放inputDatatype，biasDatatype
    auto *biasTensor = context->GetOptionalInputTensor(BIAS_INDEX);
    const gert::StorageShape *biasInputShape = biasTensor ? context->GetOptionalInputShape(BIAS_INDEX) : nullptr;
    tensorInfo.inputDatatype = context->GetInputDesc(X_INDEX)->GetDataType();
    tensorInfo.biasDatatype = (biasInputShape == nullptr) ? tensorInfo.inputDatatype :
                                                            context->GetOptionalInputDesc(BIAS_INDEX)->GetDataType();
    CalculateTensorInfo(context, tensorInfo, tilingData, biasInputShape);
    // 子函数5  设置公式参数
    SetFormulaicArgs(formulaicArgs, tilingData, tensorInfo);
    // 子函数6 tiling分配
    ComputeCoreTiling(context, tilingData, formulaicArgs, tensorInfo);

    if (tilingData->commonTiling.yShardFlag == 0) {
        OP_TILING_CHECK(MC2SetWorkspace(context, tilingData) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "Set workspace Failed!"), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(MC2SetWorkspaceShard(context, tilingData, tensorInfo.isLite) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "Set workspace Failed!"), return ge::GRAPH_FAILED);
    }
    uint64_t tilingKey = UpdateTilingKey(tilingData, tensorInfo.isLite);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(tensorInfo.numBlocks);
    context->GetRawTilingData()->SetDataSize(sizeof(BatchMatMulReduceScatterAlltoAllTilingData));
    SetHcclTiling(context, tilingData);
    PrintCommonTilingVariables(tilingData);
    PrintSliceTileInfo(tilingData);
    return ge::GRAPH_SUCCESS;
}
struct BatchMatMulReduceScatterAlltoAllCompileInfo {};
ge::graphStatus TilingParseForBatchMatMulReduceScatterAlltoAll(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BatchMatMulReduceScatterAlltoAll)
    .Tiling(BatchMatMulReduceScatterAlltoAllTilingFunc)
    .TilingParse<BatchMatMulReduceScatterAlltoAllCompileInfo>(TilingParseForBatchMatMulReduceScatterAlltoAll);
} // namespace optiling
