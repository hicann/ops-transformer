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
 * \file all_to_all_all_gather_batch_matmul_tiling.cpp
 * \brief
 */

#include "allto_all_all_gather_batch_mat_mul_tiling.h"
#include "../../op_kernel/allto_all_all_gather_batch_mat_mul_tiling_key.h"

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "op_host/op_tiling/mc2_tiling_common_var.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "allto_all_all_gather_formulaic_tiling.h"
#include "common/utils/op_mc2.h"
#include "mc2_moe_utils.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Moe;

namespace {

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t OUTPUT_Y1_INDEX = 0;
constexpr uint32_t OUTPUT_Y2_INDEX = 1;
constexpr uint32_t OUTPUT_Y3_INDEX = 2;

constexpr uint32_t ATTR_EP_GROUP_INDEX = 0;
constexpr uint32_t ATTR_TP_GROUP_INDEX = 1;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 2;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 3;
constexpr uint32_t ATTR_X_SHARD_TYPE_INDEX = 4;
constexpr uint32_t ATTR_ACT_TYPE_INDEX = 5;
constexpr uint32_t ATTR_IS_WEIGHT_TRANS_INDEX = 6;
constexpr uint32_t ATTR_OUTPUT_Y2_FLAG_INDEX = 7;
constexpr uint32_t ATTR_OUTPUT_Y3_FLAG_INDEX = 8;

constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8; // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_ALL_GATHER = 6; // numeric representation of AllGather

constexpr uint32_t FASTGELU_MINSIZE = 3 * 256;
constexpr uint32_t GELU = 1;
constexpr uint32_t SILU = 2;
constexpr uint32_t RELU = 3;
constexpr uint32_t FASTGELU = 4;

} // namespace

namespace optiling {

struct TransposeConfig {
    uint64_t ubSize;
    uint64_t tileCWhole;
    uint64_t tileEWhole;
    uint64_t inputDatasize;
    bool isLocal;
    bool isTail;
};

struct ActivationParams {
    uint64_t ubSize;
    bool xCastFlag;
    uint64_t activateType;
};

struct TileShardParams {
    uint64_t tileCWhole;
    uint64_t tileEWhole;
    uint64_t m;
    bool isLocal;
    bool isTail;
};

static void PrintCommonTilingVariables(AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - epGroupSize is %u.",
            tilingData->commonTiling.epGroupSize);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - tpGroupSize is %u.",
            tilingData->commonTiling.tpGroupSize);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - expert is %lu.",
            tilingData->commonTiling.expert);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - EOverEp is %lu.",
            tilingData->commonTiling.EOverEp);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - C is %lu.", tilingData->commonTiling.C);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - COverTp is %lu.",
            tilingData->commonTiling.COverTp);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - H is %lu.", tilingData->commonTiling.H);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - HOverTp is %lu.",
            tilingData->commonTiling.HOverTp);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - MOverTp is %lu.",
            tilingData->commonTiling.MOverTp);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - aivCoreNum is %u.",
            tilingData->commonTiling.aivCoreNum);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - inputDatasize is %u.",
            tilingData->commonTiling.inputDatasize);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - biasDatasize is %u.",
            tilingData->commonTiling.biasDatasize);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - ubCapacityForTrans is %lu.",
            tilingData->commonTiling.ubCapacityForTrans);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - ubCapacityForAddActivate is %lu.",
            tilingData->commonTiling.ubCapacityForAddActivate);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - isBias is %d.",
            tilingData->commonTiling.isBias);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - y2Flag is %d.",
            tilingData->commonTiling.y2Flag);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - y3Flag is %d.",
            tilingData->commonTiling.y3Flag);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - isWeightTrans is %d.",
            tilingData->commonTiling.isWeightTrans);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - activateType is %u.",
            tilingData->commonTiling.activateType);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - xShardFlag is %u.",
            tilingData->commonTiling.xShardFlag);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - fastGeluBuffer is %u.",
            tilingData->commonTiling.fastGeluBuffer);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Common tiling - totalUbSize is %lu.",
            tilingData->commonTiling.totalUbSize);
}

static void PrintSliceTileInfo(AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local E - Tile length is %lu.",
            tilingData->commonTiling.localTileE.tileLen);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local E - Tile Count is %lu.",
            tilingData->commonTiling.localTileE.tileCnt);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local E - Tail length is %lu.",
            tilingData->commonTiling.localTileE.tailLen);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local E - Tail Count is %lu.",
            tilingData->commonTiling.localTileE.tailCnt);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local C - Tile length is %lu.",
            tilingData->commonTiling.localTileC.tileLen);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local C - Tile Count is %lu.",
            tilingData->commonTiling.localTileC.tileCnt);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local C - Tail length is %lu.",
            tilingData->commonTiling.localTileC.tailLen);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Local C - Tail Count is %lu.",
            tilingData->commonTiling.localTileC.tailCnt);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local E - Tile length is %lu.",
            tilingData->commonTiling.domesticTileE.tileLen);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local E - Tile Count is %lu.",
            tilingData->commonTiling.domesticTileE.tileCnt);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local E - Tail length is %lu.",
            tilingData->commonTiling.domesticTileE.tailLen);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local E - Tail Count is %lu.",
            tilingData->commonTiling.domesticTileE.tailCnt);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local C - Tile length is %lu.",
            tilingData->commonTiling.domesticTileC.tileLen);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local C - Tile Count is %lu.",
            tilingData->commonTiling.domesticTileC.tileCnt);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local C - Tail length is %lu.",
            tilingData->commonTiling.domesticTileC.tailLen);
    OP_LOGD("AlltoAllAllGatherBatchMatMul Tiling Check", "Non-Local C - Tail Count is %lu.",
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

static void InitTileInfoInCommonTiling(AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    tilingData->commonTiling.localTileE.tileCnt = 1;
    tilingData->commonTiling.localTileE.tileLen = tilingData->commonTiling.EOverEp;
    tilingData->commonTiling.localTileE.tailCnt = 0;
    tilingData->commonTiling.localTileE.tailLen = 0;

    tilingData->commonTiling.domesticTileE.tileCnt = 1;
    tilingData->commonTiling.domesticTileE.tileLen = tilingData->commonTiling.EOverEp;
    tilingData->commonTiling.domesticTileE.tailCnt = 0;
    tilingData->commonTiling.domesticTileE.tailLen = 0;

    tilingData->commonTiling.localTileC.tileCnt = 1;
    if (tilingData->commonTiling.xShardFlag == 1) {
        tilingData->commonTiling.localTileC.tileLen = tilingData->commonTiling.C / tilingData->commonTiling.tpGroupSize;
    } else {
        tilingData->commonTiling.localTileC.tileLen = tilingData->commonTiling.C;
    }
    tilingData->commonTiling.localTileC.tailCnt = 0;
    tilingData->commonTiling.localTileC.tailLen = 0;

    tilingData->commonTiling.domesticTileC.tileCnt = 1;
    if (tilingData->commonTiling.xShardFlag == 1) {
        tilingData->commonTiling.domesticTileC.tileLen =
            tilingData->commonTiling.C / tilingData->commonTiling.tpGroupSize;
    } else {
        tilingData->commonTiling.domesticTileC.tileLen = tilingData->commonTiling.C;
    }
    tilingData->commonTiling.domesticTileC.tailCnt = 0;
    tilingData->commonTiling.domesticTileC.tailLen = 0;
}

static void UpdateTileInfo(BMM_TileInfo &tileInfo, uint64_t tileCnt, uint64_t tileLen, uint64_t tailCnt,
                           uint64_t tailLen)
{
    tileInfo.tailCnt = tailCnt;
    tileInfo.tailLen = tailLen;
    tileInfo.tileCnt = tileCnt;
    tileInfo.tileLen = tileLen;
}

static void CheckTransposeUBAndUpdateTileShard(TransposeConfig &config,
                                               AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    uint64_t H = tilingData->commonTiling.H;
    uint64_t ep = tilingData->commonTiling.epGroupSize;
    uint64_t tp = tilingData->commonTiling.tpGroupSize;
    uint64_t coreNum = tilingData->commonTiling.aivCoreNum;

    uint64_t ubCapacity = static_cast<uint64_t>(config.ubSize) / config.inputDatasize;
    tilingData->commonTiling.ubCapacityForTrans = ubCapacity;

    uint64_t totalDataSize = (config.isLocal == true) ?
                                 tp * config.tileEWhole * config.tileCWhole * H :
                                 static_cast<uint64_t>(tp) * static_cast<uint64_t>(config.tileEWhole) *
                                     (static_cast<uint64_t>(ep) - 1U) * static_cast<uint64_t>(config.tileCWhole) *
                                     static_cast<uint64_t>(H);
    if (tilingData->commonTiling.xShardFlag == 0) {
        totalDataSize = (config.isLocal == true) ?
                            config.tileEWhole * config.tileCWhole * H :
                            static_cast<uint64_t>(config.tileEWhole) * (static_cast<uint64_t>(ep) - 1U) *
                                static_cast<uint64_t>(config.tileCWhole) * static_cast<uint64_t>(H);
    }

    uint64_t tileLen = (totalDataSize + coreNum - 1) / coreNum;
    uint64_t tileCnt = 1;
    uint64_t tailLen = totalDataSize - tileLen * tileCnt;
    uint64_t tailCnt = (tailLen == 0UL) ? 0UL : 1UL;

    BMM_TileInfo &tileInfo = config.isLocal ? (config.isTail ? tilingData->commonTiling.localTailUbTranspose :
                                                               tilingData->commonTiling.localUbTranspose) :
                                              (config.isTail ? tilingData->commonTiling.domesticTailUbTranspose :
                                                               tilingData->commonTiling.domesticUbTranspose);
    UpdateTileInfo(tileInfo, tileCnt, tileLen, tailCnt, tailLen);
    return;
}

static void CheckAddActivateUB(ActivationParams &actParams, AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    uint64_t ubCapacity = 1U;
    uint64_t actIn = 2U;
    uint64_t actCast = 0U;
    uint64_t addRight = 0U;
    uint64_t fastGelu = 0U;
    uint64_t actOut = 0U;
    uint64_t coeff = 1U;

    if (tilingData->commonTiling.isBias) {
        if (actParams.xCastFlag == true) {
            actCast = FP32_DATASIZE;
            actOut = FP32_DATASIZE;
        } else {
            actOut = FP16_DATASIZE;
        }
        addRight = tilingData->commonTiling.biasDatasize;
    }

    if (actParams.activateType != 0U) {
        if (actParams.xCastFlag == true) {
            actCast = FP32_DATASIZE;
            actOut = FP32_DATASIZE;
        } else {
            actOut = FP16_DATASIZE;
        }
    }

    if ((actParams.activateType == FASTGELU) || (actParams.activateType == GELU)) {
        if (actParams.xCastFlag == true) {
            fastGelu = FP32_DATASIZE;
        } else {
            fastGelu = FP16_DATASIZE;
        }
    }

    coeff = actIn + actCast + actOut + addRight + fastGelu;
    ubCapacity = (actParams.ubSize - FASTGELU_MINSIZE) / static_cast<uint64_t>(coeff);
    ubCapacity = ubCapacity / ALIGN16 * ALIGN16;
    tilingData->commonTiling.ubCapacityForAddActivate = ubCapacity;
    tilingData->commonTiling.fastGeluBuffer = std::max(static_cast<uint64_t>(FASTGELU_MINSIZE), fastGelu * ubCapacity);
}

static void UpdateTileShard(TileShardParams &params, AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    uint64_t ep = tilingData->commonTiling.epGroupSize;
    uint64_t tp = tilingData->commonTiling.tpGroupSize;
    uint64_t coreNum = tilingData->commonTiling.aivCoreNum;

    uint64_t totalDataSize = (params.isLocal == true) ?
                                 static_cast<uint64_t>(tp) * params.tileEWhole * params.tileCWhole * params.m :
                                 static_cast<uint64_t>(tp) * params.tileEWhole * (static_cast<uint64_t>(ep) - 1U) *
                                     static_cast<uint64_t>(params.tileCWhole) * static_cast<uint64_t>(params.m);
    if (tilingData->commonTiling.xShardFlag == 0) {
        totalDataSize = (params.isLocal == true) ?
                            params.tileEWhole * params.tileCWhole * params.m :
                            static_cast<uint64_t>(params.tileEWhole) * (static_cast<uint64_t>(ep) - 1U) *
                                static_cast<uint64_t>(params.tileCWhole) * static_cast<uint64_t>(params.m);
    }

    uint64_t tileLen = (totalDataSize + coreNum - 1U) / static_cast<uint64_t>(coreNum);
    uint64_t tileCnt = 1U;
    uint64_t tailLen = totalDataSize - tileLen * tileCnt;
    uint64_t tailCnt = (tailLen == 0U) ? 0U : 1U;

    BMM_TileInfo &tileInfo =
        params.isLocal ?
            (params.isTail ? tilingData->commonTiling.localTailUbAdd : tilingData->commonTiling.localUbAdd) :
            (params.isTail ? tilingData->commonTiling.domesticTailUbAdd : tilingData->commonTiling.domesticUbAdd);
    UpdateTileInfo(tileInfo, tileCnt, tileLen, tailCnt, tailLen);
    return;
}

static void CheckAddActivateUBAndUpdateTileShard(ActivationParams &actParams, TileShardParams &params,
                                                 AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    CheckAddActivateUB(actParams, tilingData);
    UpdateTileShard(params, tilingData);
}

static uint64_t UpdateTilingKey(AlltoAllAllGatherBatchMatMulTilingData *tilingData, bool y2Flag, bool y3Flag)
{
    uint8_t xShardFlag = tilingData->commonTiling.xShardFlag;
    bool isWeightTrans = tilingData->commonTiling.isWeightTrans;
    bool isBias = tilingData->commonTiling.isBias;

    uint64_t tilingKey = GET_TPL_TILING_KEY(xShardFlag, isWeightTrans, isBias, y2Flag, y3Flag);

    OP_LOGD("AlltoAllAllGatherBatchMatMul", "TPL Tiling Key Print : Tiling key parameter : xShardFlag is %u.",
            xShardFlag);
    OP_LOGD("AlltoAllAllGatherBatchMatMul", "TPL Tiling Key Print : Tiling key parameter : Weight Transpose is %d.",
            isWeightTrans);
    OP_LOGD("AlltoAllAllGatherBatchMatMul", "TPL Tiling Key Print : Tiling key parameter : Bias is %d.", isBias);
    OP_LOGD("AlltoAllAllGatherBatchMatMul", "TPL Tiling Key Print : Tiling key parameter : Y2 output need is %d.",
            y2Flag);
    OP_LOGD("AlltoAllAllGatherBatchMatMul", "TPL Tiling Key Print : Tiling key parameter : Y3 output need is %d.",
            y3Flag);

    OP_LOGD("AlltoAllAllGatherBatchMatMul", "The final tiling key is %lu.", tilingKey);
    return tilingKey;
}

static void CompleteBmmStructs(AlltoAllAllGatherBatchInfo &BMMV3BatchInfo, AlltoAllAllGatherMatmulInfo &MMV3ArgsInfo,
                               uint32_t m, uint32_t n, uint32_t k, uint32_t b)
{
    MMV3ArgsInfo.mValue = m;
    MMV3ArgsInfo.nValue = n;
    MMV3ArgsInfo.kValue = k;
    BMMV3BatchInfo.batchA = b;
    BMMV3BatchInfo.batchA3 = b;
    BMMV3BatchInfo.batchB = b;
    BMMV3BatchInfo.batchB3 = b;
    BMMV3BatchInfo.batchC = b;
    BMMV3BatchInfo.batchC3 = b;

    return;
}

static void GetAlltoAllAllGatherFormulateTileCnt(mc2tiling::TilingArgs &formulaicArgs,
                                                 AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    All2AllAllGatherBMM formulaicTiling(formulaicArgs, tilingData->commonTiling.epGroupSize,
                                        tilingData->commonTiling.tpGroupSize, tilingData->commonTiling.EOverEp);
    formulaicTiling.GetTiling();

    tilingData->commonTiling.localTileE.tileCnt = formulaicTiling.localCutE.numLongTile;
    tilingData->commonTiling.localTileE.tailCnt = formulaicTiling.localCutE.numShortTile;
    tilingData->commonTiling.localTileE.tileLen = formulaicTiling.localCutE.longTileLen;
    tilingData->commonTiling.localTileE.tailLen = formulaicTiling.localCutE.shortTileLen;

    tilingData->commonTiling.domesticTileE.tileCnt = formulaicTiling.cutE.numLongTile;
    tilingData->commonTiling.domesticTileE.tailCnt = formulaicTiling.cutE.numShortTile;
    tilingData->commonTiling.domesticTileE.tileLen = formulaicTiling.cutE.longTileLen;
    tilingData->commonTiling.domesticTileE.tailLen = formulaicTiling.cutE.shortTileLen;

    tilingData->commonTiling.domesticTileC.tileCnt = formulaicTiling.tilingC.cutRes.numLongTile;
    tilingData->commonTiling.domesticTileC.tileLen =
        formulaicTiling.tilingC.cutRes.longTileLen / tilingData->commonTiling.tpGroupSize;
    tilingData->commonTiling.domesticTileC.tailCnt = formulaicTiling.tilingC.cutRes.numShortTile;
    tilingData->commonTiling.domesticTileC.tailLen =
        formulaicTiling.tilingC.cutRes.shortTileLen / tilingData->commonTiling.tpGroupSize;

    return;
}

static void GetAlltoAllAllGatherFormulateTileCntShardH(mc2tiling::TilingArgs &formulaicArgs,
                                                       AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    All2AllAllGatherBMMShardH formulaicTiling(formulaicArgs, tilingData->commonTiling.epGroupSize,
                                              tilingData->commonTiling.tpGroupSize, tilingData->commonTiling.EOverEp);
    formulaicTiling.GetTiling();
    if (formulaicTiling.localCutE.shortTileAtBack) { // local E短块后置
        tilingData->commonTiling.localTileE.tileCnt = formulaicTiling.localCutE.numLongTile;
        tilingData->commonTiling.localTileE.tailCnt = formulaicTiling.localCutE.numShortTile;
        tilingData->commonTiling.localTileE.tileLen = formulaicTiling.localCutE.longTileLen;
        tilingData->commonTiling.localTileE.tailLen = formulaicTiling.localCutE.shortTileLen;
    } else { // local E短块前置
        tilingData->commonTiling.localTileE.tileCnt = formulaicTiling.localCutE.numShortTile;
        tilingData->commonTiling.localTileE.tailCnt = formulaicTiling.localCutE.numLongTile;
        tilingData->commonTiling.localTileE.tileLen = formulaicTiling.localCutE.shortTileLen;
        tilingData->commonTiling.localTileE.tailLen = formulaicTiling.localCutE.longTileLen;
    }

    if (formulaicTiling.cutE.shortTileAtBack) { // non-local E短块后置
        tilingData->commonTiling.domesticTileE.tileCnt = formulaicTiling.cutE.numLongTile;
        tilingData->commonTiling.domesticTileE.tailCnt = formulaicTiling.cutE.numShortTile;
        tilingData->commonTiling.domesticTileE.tileLen = formulaicTiling.cutE.longTileLen;
        tilingData->commonTiling.domesticTileE.tailLen = formulaicTiling.cutE.shortTileLen;
    } else { // non-local E短块前置
        tilingData->commonTiling.domesticTileE.tileCnt = formulaicTiling.cutE.numShortTile;
        tilingData->commonTiling.domesticTileE.tailCnt = formulaicTiling.cutE.numLongTile;
        tilingData->commonTiling.domesticTileE.tileLen = formulaicTiling.cutE.shortTileLen;
        tilingData->commonTiling.domesticTileE.tailLen = formulaicTiling.cutE.longTileLen;
    }

    if (formulaicTiling.tilingC.cutRes.shortTileAtBack) { // non-local C短块后置
        tilingData->commonTiling.domesticTileC.tileCnt = formulaicTiling.tilingC.cutRes.numLongTile;
        tilingData->commonTiling.domesticTileC.tailCnt = formulaicTiling.tilingC.cutRes.numShortTile;
        tilingData->commonTiling.domesticTileC.tileLen = formulaicTiling.tilingC.cutRes.longTileLen;
        tilingData->commonTiling.domesticTileC.tailLen = formulaicTiling.tilingC.cutRes.shortTileLen;
    } else { // non-local C短块前置
        tilingData->commonTiling.domesticTileC.tileCnt = formulaicTiling.tilingC.cutRes.numShortTile;
        tilingData->commonTiling.domesticTileC.tailCnt = formulaicTiling.tilingC.cutRes.numLongTile;
        tilingData->commonTiling.domesticTileC.tileLen = formulaicTiling.tilingC.cutRes.shortTileLen;
        tilingData->commonTiling.domesticTileC.tailLen = formulaicTiling.tilingC.cutRes.longTileLen;
    }

    return;
}

static ge::graphStatus HandleLocalBmmTilingData(gert::TilingContext *context,
                                                AlltoAllAllGatherBatchMatMulTilingData *tilingData,
                                                AlltoAllAllGatherBatchInfo bmmV3BatchInfo,
                                                AlltoAllAllGatherMatmulInfo mmV3ArgsInfo)
{
    // 将切块信息传入BmmV3 tiling，根据后续整改方案更新Bmm所需的相应参数并调用BmmV3
    // Tiling接口，获取local/non-local的BMM tiling信息 Local standard slice BMM tiling
    if (tilingData->commonTiling.xShardFlag == 1U) {
        CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                           tilingData->commonTiling.tpGroupSize * tilingData->commonTiling.localTileC.tileLen,
                           tilingData->commonTiling.MOverTp, tilingData->commonTiling.H,
                           tilingData->commonTiling.localTileE.tileLen);
    } else {
        CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo, tilingData->commonTiling.localTileC.tileLen,
                           tilingData->commonTiling.MOverTp, tilingData->commonTiling.H,
                           tilingData->commonTiling.localTileE.tileLen);
    }
    AlltoAllAllGatherBatchMatMulTiling bmmTilingLocalStd(context, tilingData->localTiling.bmmTilingData, bmmV3BatchInfo,
                                                         mmV3ArgsInfo);
    OP_TILING_CHECK(bmmTilingLocalStd.DoTiling() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Do BmmV3Tiling failed under local standard section."),
                    return ge::GRAPH_FAILED);

    // Local tail slice BMM tiling
    if (tilingData->commonTiling.localTileE.tailLen != 0) {
        if (tilingData->commonTiling.xShardFlag == 1U) {
            CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                               tilingData->commonTiling.tpGroupSize * tilingData->commonTiling.localTileC.tailLen,
                               tilingData->commonTiling.MOverTp, tilingData->commonTiling.H,
                               tilingData->commonTiling.localTileE.tileLen);
        } else {
            CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo, tilingData->commonTiling.localTileC.tileLen,
                               tilingData->commonTiling.MOverTp, tilingData->commonTiling.H,
                               tilingData->commonTiling.localTileE.tailLen);
        }
        AlltoAllAllGatherBatchMatMulTiling bmmTilingLocalTail(context, tilingData->localTailTiling.bmmTilingData,
                                                              bmmV3BatchInfo, mmV3ArgsInfo);
        OP_TILING_CHECK(bmmTilingLocalTail.DoTiling() != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "Do BmmV3Tiling failed under local tail section."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HandleNoneLocalBmmTilingData(gert::TilingContext *context,
                                                    AlltoAllAllGatherBatchMatMulTilingData *tilingData,
                                                    AlltoAllAllGatherBatchInfo bmmV3BatchInfo,
                                                    AlltoAllAllGatherMatmulInfo mmV3ArgsInfo)
{
    // Non-local standard slice BMM tiling
    if (tilingData->commonTiling.xShardFlag == 1U) {
        CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                           (tilingData->commonTiling.epGroupSize - 1) * tilingData->commonTiling.tpGroupSize *
                               tilingData->commonTiling.domesticTileC.tileLen,
                           tilingData->commonTiling.MOverTp, tilingData->commonTiling.H,
                           tilingData->commonTiling.domesticTileE.tileLen);
    } else {
        CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                           (tilingData->commonTiling.epGroupSize - 1) * tilingData->commonTiling.domesticTileC.tileLen,
                           tilingData->commonTiling.MOverTp, tilingData->commonTiling.H,
                           tilingData->commonTiling.domesticTileE.tileLen);
    }
    AlltoAllAllGatherBatchMatMulTiling bmmTilingDomesticStd(context, tilingData->domesticTiling.bmmTilingData,
                                                            bmmV3BatchInfo, mmV3ArgsInfo);
    OP_TILING_CHECK(bmmTilingDomesticStd.DoTiling() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Do BmmV3Tiling failed under domestic standard section."),
                    return ge::GRAPH_FAILED);

    // Non-local tail slice BMM tiling
    if (tilingData->commonTiling.domesticTileC.tailLen != 0) {
        if (tilingData->commonTiling.xShardFlag == 1U) {
            CompleteBmmStructs(bmmV3BatchInfo, mmV3ArgsInfo,
                               (tilingData->commonTiling.epGroupSize - 1) * tilingData->commonTiling.tpGroupSize *
                                   tilingData->commonTiling.domesticTileC.tailLen,
                               tilingData->commonTiling.MOverTp, tilingData->commonTiling.H,
                               tilingData->commonTiling.domesticTileE.tileLen);
        } else {
            CompleteBmmStructs(
                bmmV3BatchInfo, mmV3ArgsInfo,
                (tilingData->commonTiling.epGroupSize - 1) * tilingData->commonTiling.domesticTileC.tailLen,
                tilingData->commonTiling.MOverTp, tilingData->commonTiling.H,
                tilingData->commonTiling.domesticTileE.tileLen);
        }
        AlltoAllAllGatherBatchMatMulTiling bmmTilingDomesticTail(context, tilingData->domesticTailTiling.bmmTilingData,
                                                                 bmmV3BatchInfo, mmV3ArgsInfo);
        OP_TILING_CHECK(bmmTilingDomesticTail.DoTiling() != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "Do BmmV3Tiling failed under domestic tail section."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetMatmulTilingAlltoAllAllGatherBatchMatMul(gert::TilingContext *context,
                                                                   AlltoAllAllGatherBatchMatMulTilingData *tilingData,
                                                                   AlltoAllAllGatherBatchInfo BMMV3BatchInfo,
                                                                   AlltoAllAllGatherMatmulInfo MMV3ArgsInfo,
                                                                   mc2tiling::TilingArgs &formulaicArgs)
{
    // 1. 调用公式化tiling接口，获取local和非local块的切分信息并更新commonTiling
    if (tilingData->commonTiling.xShardFlag == 1U) {
        GetAlltoAllAllGatherFormulateTileCnt(formulaicArgs, tilingData);
    } else {
        GetAlltoAllAllGatherFormulateTileCntShardH(formulaicArgs, tilingData);
    }

    OP_TILING_CHECK(HandleLocalBmmTilingData(context, tilingData, BMMV3BatchInfo, MMV3ArgsInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Handle local bmm tiling data failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        HandleNoneLocalBmmTilingData(context, tilingData, BMMV3BatchInfo, MMV3ArgsInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Handle none local bmm tiling data failed."), return ge::GRAPH_FAILED);

    // shard-0 Non_local tail E slice BMM tiling
    if ((tilingData->commonTiling.xShardFlag == 0U) &&
        (tilingData->commonTiling.domesticTileC.tileLen == tilingData->commonTiling.C) &&
        (tilingData->commonTiling.domesticTileE.tailLen != 0)) {
        CompleteBmmStructs(BMMV3BatchInfo, MMV3ArgsInfo,
                           (tilingData->commonTiling.epGroupSize - 1) * tilingData->commonTiling.domesticTileC.tileLen,
                           tilingData->commonTiling.MOverTp, tilingData->commonTiling.H,
                           tilingData->commonTiling.domesticTileE.tailLen);
        AlltoAllAllGatherBatchMatMulTiling bmmTilingDomesticTailE(
            context, tilingData->domesticTailETiling.bmmTilingData, BMMV3BatchInfo, MMV3ArgsInfo);
        if (bmmTilingDomesticTailE.DoTiling() != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(), "Do BmmV3Tiling failed under shard-0 domestic tail C section.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

const char *K_INNER_DEBUG = "MC2: AlltoAllAllGatherBmm Tiling Shape Debug";

static bool ActTypeCheck(const char *nodeName, const int64_t actType, const bool y3Flag)
{
    if (std::find(ops::ACT_TYPE_SUPPORT_VEC.begin(), ops::ACT_TYPE_SUPPORT_VEC.end(), actType) ==
        ops::ACT_TYPE_SUPPORT_VEC.end()) {
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "actType", std::to_string(actType).c_str(),
            "range [" +
                std::to_string(static_cast<int64_t>(
                    ops::AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE)) +
                ", " +
                std::to_string(static_cast<int64_t>(
                    ops::AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU)) +
                "]");
        return false;
    }

    if (y3Flag &&
        (actType == static_cast<int64_t>(
                        ops::AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "actType", "none", "The value of actType must be non-none when need_activation_feature is true");
        return false;
    }

    return true;
}

static bool CommonCheckTensorShape(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape,
                                   const size_t wDimH)
{
    // 检查每个维度: x dim C >= 1，dim E H M 会在后面限制, 这里不再做校验
    if (xShape->GetDim(X_DIM_C) < VALUE_C_MIN) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "x[1]", std::to_string(xShape->GetDim(X_DIM_C)).c_str(),
                                  "should not < " + std::to_string(VALUE_C_MIN));
        return false;
    }
    // x[2]、w[wDimH] 是 H 轴 (reduce 轴)，不能为 0
    if ((xShape->GetDim(X_DIM_H) == 0) || (weightShape->GetDim(wDimH) == 0)) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "x[H] or weight[H]", "0", "non-zero value is required");
        return false;
    }

    return true;
}

// GPT 2.2T
static bool XShardCheckTensorShape(const char *nodeName, const int64_t xShard, const gert::Shape *xShape,
                                   const gert::Shape *weightShape, const int64_t epSize, const int64_t tpSize,
                                   const size_t wDimH, const size_t wDimM)
{
    // 检查 shape 维度的范围
    // x[DIM_E] = E, value E should = [2, 512]
    if (((xShape->GetDim(DIM_E) < VALUE_E_MIN) || (xShape->GetDim(DIM_E) > VALUE_E_MAX))) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "E", std::to_string(xShape->GetDim(DIM_E)).c_str(),
                                  "[" + std::to_string(VALUE_E_MIN) + ", " + std::to_string(VALUE_E_MAX) + "]");
        return false;
    }
    // w[wDimM] = M / Tp, its range should same with H, so it meets M / Tp * H <= 65535 * 65535
    if ((weightShape->GetDim(wDimM) < VALUE_H_MIN) || (weightShape->GetDim(wDimM) > VALUE_H_MAX)) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "M / Tp", std::to_string(weightShape->GetDim(wDimM)).c_str(),
                                  "[" + std::to_string(VALUE_H_MIN) + ", " + std::to_string(VALUE_H_MAX) + "]");
        return false;
    }
    // x[DIM_E] = E, w[DIM_E] = E / Ep, 所以需要满足 x[DIM_E] = w[DIM_E] * Ep
    if (weightShape->GetDim(DIM_E) * epSize != xShape->GetDim(DIM_E)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "x and weight",
            (std::to_string(xShape->GetDim(DIM_E)) + " and " + std::to_string(weightShape->GetDim(DIM_E))).c_str(),
            "The shape of x[0] must equal weight[0] * epSize");
        return false;
    }

    if (xShard == 0) {
        // x[X_DIM_H] = H / tp, value H should = [1, 65535]
        if ((xShape->GetDim(X_DIM_H) * tpSize < VALUE_H_MIN) || (xShape->GetDim(X_DIM_H) * tpSize > VALUE_H_MAX)) {
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "H", std::to_string(xShape->GetDim(X_DIM_H) * tpSize).c_str(),
                                      "[" + std::to_string(VALUE_H_MIN) + ", " + std::to_string(VALUE_H_MAX) + "]");
            return false;
        }
        // x[X_DIM_H] = H / tp, w[wDimH] = H, 所以 x[X_DIM_H] * tp 需要等于 w[wDimH]
        if (xShape->GetDim(X_DIM_H) * tpSize != weightShape->GetDim(wDimH)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x and weight",
                                                  std::to_string(xShape->GetDim(X_DIM_H) * tpSize).c_str(),
                                                  "The shape of x[H] * tp must equal weight[H]");
            return false;
        }
    } else if (xShard == 1) {
        // x[X_DIM_H] = H, value H should = [1, 65535]
        if ((xShape->GetDim(X_DIM_H) < VALUE_H_MIN) || (xShape->GetDim(X_DIM_H) > VALUE_H_MAX)) {
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "H", std::to_string(xShape->GetDim(X_DIM_H)).c_str(),
                                      "[" + std::to_string(VALUE_H_MIN) + ", " + std::to_string(VALUE_H_MAX) + "]");
            return false;
        }
        // x[X_DIM_H] = H, w[wDimH] = H, 所以 x[X_DIM_H] 需要等于 w[wDimH]
        if (xShape->GetDim(X_DIM_H) != weightShape->GetDim(wDimH)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x and weight",
                                                  std::to_string(xShape->GetDim(X_DIM_H)).c_str(),
                                                  "The shape of x[H] must equal weight[H]");
            return false;
        }
    }

    return true;
}

static bool CheckBiasShape(const char *nodeName, const gert::Shape *weightShape, const gert::Shape *biasShape,
                           const size_t wDimM)
{
    // 检查 dimNum
    if ((biasShape->GetDimNum() != SUPPORT_DIM_NUM) && (biasShape->GetDimNum() != BIAS_SUPPORT_DIM_NUM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "bias", (std::to_string(biasShape->GetDimNum()) + "D").c_str(),
                                     "2D or 3D");
        return false;
    }

    // 检查 shape
    if (biasShape->GetDim(0) != weightShape->GetDim(0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "bias", std::to_string(biasShape->GetDim(0)).c_str(),
                                              "The shape of bias[0] must equal weight[0]");
        return false;
    }

    size_t biasLastDimIdx = 1U; // 默认 bias 是二维，所以最后一维的 index 是 1
    if (biasShape->GetDimNum() == SUPPORT_DIM_NUM) {
        if (biasShape->GetDim(1) != 1) { // 三维时候，中间维度为 1
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "bias", std::to_string(biasShape->GetDim(1)).c_str(),
                                                  "The second dim of bias must be 1 when dim num is 3");
            return false;
        }
        biasLastDimIdx = 2; // 三维时候，bias 的最后一维是 2
    }

    if (biasShape->GetDim(biasLastDimIdx) != weightShape->GetDim(wDimM)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "bias",
                                              std::to_string(biasShape->GetDim(biasLastDimIdx)).c_str(),
                                              "The last dim of bias must equal the corresponding dim of weight");
        return false;
    }

    return true;
}

static bool CheckTensorShape(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape,
                             const gert::Shape *biasShape, const int64_t epSize, const int64_t tpSize,
                             const size_t wDimH, const size_t wDimM, const int64_t xShard)
{
    if ((xShape == nullptr) || (weightShape == nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "xShape or weightShape");
        return false;
    }
    if (!DimNumCheck(nodeName, xShape, weightShape)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "x and weight",
            (std::to_string(xShape->GetDimNum()) + "D and " + std::to_string(weightShape->GetDimNum()) + "D").c_str(),
            "The shape dims of x and weight must be valid");
        return false;
    }

    if (!CommonCheckTensorShape(nodeName, xShape, weightShape, wDimH)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x and weight", "common",
                                              "The shapes of x and weight must be valid");
        return false;
    }

    if (xShard == 0 || xShard == 1) {
        if (!XShardCheckTensorShape(nodeName, xShard, xShape, weightShape, epSize, tpSize, wDimH, wDimM)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x and weight", std::to_string(xShard).c_str(),
                                                  "The shapes of x and weight must be valid for xShard");
            return false;
        }
    }

    if (biasShape != nullptr) {
        if (!(CheckBiasShape(nodeName, weightShape, biasShape, wDimM))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "bias", "bias", "The shape of bias must be valid");
            return false;
        }
    }

    OP_LOGI(nodeName, "Tensor shape check success.");
    return true;
}

static bool CheckIsAttrNull(const char *nodeName, const int64_t *tpWorldSize, const int64_t *epWorldSize,
                            const int64_t *xShardType, const int64_t *actTypePtr, const bool *outputY2Flag,
                            const bool *outputY3Flag)
{
    if ((tpWorldSize == nullptr) || (epWorldSize == nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpWorldSize or epWorldSize");
        return false;
    }

    if ((xShardType == nullptr) || (actTypePtr == nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "xShardType or actTypePtr");
        return false;
    }

    if ((outputY2Flag == nullptr) || (outputY3Flag == nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "outputY2Flag or outputY3Flag");
        return false;
    }
    return true;
}

static bool CheckAttrs(const gert::TilingContext *context, int64_t &epSize, int64_t &tpSize, int64_t &xShard,
                       bool &y2Flag, bool &y3Flag)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return false);
    // get 只有在 index 超出 attr num 的时候才会返回 nullptr
    const char *groupEp = attrs->GetStr(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_GROUP_EP));
    const char *groupTp = attrs->GetStr(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_GROUP_TP));
    const int64_t *tpWorldSize = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_TP_WORLD_SIZE));
    const int64_t *epWorldSize = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_EP_WORLD_SIZE));
    const int64_t *xShardType = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_X_SHARD_TYPE));
    const int64_t *actTypePtr = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_ACT_TYPE));
    const bool *outputY2Flag = attrs->GetBool(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_OUTPUT_Y2_FLAG));
    const bool *outputY3Flag = attrs->GetBool(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_OUTPUT_Y3_FLAG));

    if (!CheckIsAttrNull(nodeName, tpWorldSize, epWorldSize, xShardType, actTypePtr, outputY2Flag, outputY3Flag)) {
        return false;
    }

    tpSize = *tpWorldSize;
    epSize = *epWorldSize;
    xShard = *xShardType;
    y2Flag = *outputY2Flag;
    y3Flag = *outputY3Flag;
    const int64_t actType = *actTypePtr;

    if (!GroupCheck(nodeName, groupEp, groupTp)) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "groupEp or groupTp", "invalid", "valid group");
        return false;
    }
    if (!EpTpSizeCheck(epSize, tpSize)) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "tpSize and epSize",
                                  (std::to_string(tpSize) + " and " + std::to_string(epSize)).c_str(),
                                  "valid tpSize=[2/4/8/16/32], epSize=[2/4/8/16/32]");
        return false;
    }
    if (xShard != 0 && xShard != 1) { // 当前支持 0, 1
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "xShard", std::to_string(xShard).c_str(), "0 or 1");
        return false;
    }
    if (!ActTypeCheck(nodeName, actType, y3Flag)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "actType", std::to_string(actType).c_str(),
                                              "The value of actType must be valid");
        return false;
    }
    OP_LOGI(nodeName, "attrs info: groupEp %s, groupTp %s, tpSize %ld, epSize %ld, xShard %ld, y2Flag %d, y3Flag %d.",
            groupEp, groupTp, tpSize, epSize, xShard, y2Flag, y3Flag);
    return true;
}

// 入参校验
static ge::graphStatus TilingCheckAlltoAllAllGatherBatchMatMul(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "context"), return ge::GRAPH_FAILED);

    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter AlltoAllAllGatherBmm tiling check impl.");

    // 检查 shape 是否为空
    const gert::StorageShape *xStorageShape = context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_X));
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    const gert::StorageShape *weightStorageShape =
        context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_WEIGHT));
    OP_TILING_CHECK(weightStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "weight"),
                    return ge::GRAPH_FAILED);
    const gert::StorageShape *y1StorageShape =
        context->GetOutputShape(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y1));
    OP_TILING_CHECK(y1StorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y1"), return ge::GRAPH_FAILED);

    // 检查属性
    int64_t epSize = -1;
    int64_t tpSize = -1;
    int64_t xShard = -1;
    bool y2Flag = false;
    bool y3Flag = false;
    if (!CheckAttrs(context, epSize, tpSize, xShard, y2Flag, y3Flag)) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "attrs", "invalid", "valid attrs");
        return ge::GRAPH_FAILED;
    }

    // 判断可选参数
    if (y2Flag) {
        const gert::StorageShape *y2StorageShape =
            context->GetOutputShape(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y2));
        OP_TILING_CHECK(y2StorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y2"), return ge::GRAPH_FAILED);
    }
    if (y3Flag) {
        const gert::StorageShape *y3StorageShape =
            context->GetOutputShape(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y3));
        OP_TILING_CHECK(y3StorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y3"), return ge::GRAPH_FAILED);
    }

    //  设 w = [E, H, M], dimH 指 H 轴, dimM 指 M 轴; w_trans = [E, M, H]
    auto attrs = context->GetAttrs();
    bool isWeightTrans = *(attrs->GetAttrPointer<bool>(ATTR_IS_WEIGHT_TRANS_INDEX));
    // weight 转置: 1 代表 M 轴，2 代表 H 轴; weight 没有转置: 1 代表 H 轴，2 代表 M 轴
    size_t wDimH = isWeightTrans ? 2UL : 1UL;
    size_t wDimM = isWeightTrans ? 1UL : 2UL;

    const gert::StorageShape *biasStorageShape =
        context->GetOptionalInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_BIAS));
    const gert::Shape *xShape = &xStorageShape->GetStorageShape();
    const gert::Shape *weightShape = &weightStorageShape->GetStorageShape();
    const gert::Shape *biasShape = (biasStorageShape == nullptr) ? nullptr : &biasStorageShape->GetStorageShape();

    // tensor shape 检查
    if (!CheckTensorShape(nodeName, xShape, weightShape, biasShape, epSize, tpSize, wDimH, wDimM, xShard)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x, weight and bias", "invalid",
                                              "The shapes of x, weight and bias must be valid");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static uint64_t GetCommOutSize(AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    uint64_t commOut = 0UL;
    if (tilingData->commonTiling.xShardFlag == 0) {
        // (E + E/ep) * C * H + E * C * H
        commOut =
            (static_cast<uint64_t>(tilingData->commonTiling.expert) +
             static_cast<uint64_t>(tilingData->commonTiling.EOverEp)) *
                static_cast<uint64_t>(tilingData->commonTiling.C) * static_cast<uint64_t>(tilingData->commonTiling.H) +
            static_cast<uint64_t>(tilingData->commonTiling.expert) * static_cast<uint64_t>(tilingData->commonTiling.C) *
                static_cast<uint64_t>(tilingData->commonTiling.HOverTp);
    } else {
        // (E + E/ep) * C * H + E * C * H/tp
        commOut = (static_cast<uint64_t>(tilingData->commonTiling.expert) +
                   static_cast<uint64_t>(tilingData->commonTiling.EOverEp)) *
                      static_cast<uint64_t>(tilingData->commonTiling.C) *
                      static_cast<uint64_t>(tilingData->commonTiling.H) +
                  static_cast<uint64_t>(tilingData->commonTiling.expert) *
                      static_cast<uint64_t>(tilingData->commonTiling.COverTp) *
                      static_cast<uint64_t>(tilingData->commonTiling.H) +
                  static_cast<uint64_t>(tilingData->commonTiling.EOverEp) *
                      static_cast<uint64_t>(tilingData->commonTiling.epGroupSize) *
                      static_cast<uint64_t>(tilingData->commonTiling.COverTp) *
                      static_cast<uint64_t>(tilingData->commonTiling.H);
    }
    return commOut * static_cast<uint64_t>(tilingData->commonTiling.inputDatasize);
}

static uint64_t GetTransOutSize(AlltoAllAllGatherBatchMatMulTilingData *tilingData, const uint64_t maxLenDomesticTileC,
                                const uint64_t maxLenLocalTileE, const uint64_t maxLenDomesticTileE)
{
    uint64_t transOut = 0UL;
    if (tilingData->commonTiling.xShardFlag == 0) {
        // max of (local_tile_e * local_tile_c * H) and ((ep - 1) * nonLocal_tile_e * nonLocal_tile_c * H)
        transOut =
            std::max(static_cast<uint64_t>(maxLenLocalTileE) *
                         static_cast<uint64_t>(tilingData->commonTiling.localTileC.tileLen) *
                         static_cast<uint64_t>(tilingData->commonTiling.H),
                     static_cast<uint64_t>(tilingData->commonTiling.epGroupSize - 1) *
                         static_cast<uint64_t>(maxLenDomesticTileE) * static_cast<uint64_t>(maxLenDomesticTileC) *
                         static_cast<uint64_t>(tilingData->commonTiling.H));
    } else {
        // max of (local_tile_e * local_tile_c * H * tp) and ((ep - 1) * nonLocal_tile_e * nonLocal_tile_c * H * tp)
        transOut =
            std::max(static_cast<uint64_t>(maxLenLocalTileE) *
                         static_cast<uint64_t>(tilingData->commonTiling.localTileC.tileLen) *
                         static_cast<uint64_t>(tilingData->commonTiling.H) *
                         static_cast<uint64_t>(tilingData->commonTiling.tpGroupSize),
                     static_cast<uint64_t>(tilingData->commonTiling.epGroupSize - 1) *
                         static_cast<uint64_t>(maxLenDomesticTileE) * static_cast<uint64_t>(maxLenDomesticTileC) *
                         static_cast<uint64_t>(tilingData->commonTiling.H) *
                         static_cast<uint64_t>(tilingData->commonTiling.tpGroupSize));
    }
    return transOut * static_cast<uint64_t>(tilingData->commonTiling.inputDatasize);
}

static uint64_t GetBmmOutSize(AlltoAllAllGatherBatchMatMulTilingData *tilingData, const uint64_t maxLenDomesticTileC,
                              const uint64_t maxLenLocalTileE, const uint64_t maxLenDomesticTileE)
{
    uint64_t bmmOut = 0UL;
    // max of (local_tile_e * local_tile_c * M/tp) and ((ep - 1) * nonLocal_tile_e * nonLocal_tile_c * M/tp)
    if (tilingData->commonTiling.xShardFlag == 0) {
        bmmOut = std::max(static_cast<uint64_t>(maxLenLocalTileE) *
                              static_cast<uint64_t>(tilingData->commonTiling.localTileC.tileLen) *
                              static_cast<uint64_t>(tilingData->commonTiling.MOverTp),
                          static_cast<uint64_t>(tilingData->commonTiling.epGroupSize - 1) *
                              static_cast<uint64_t>(maxLenDomesticTileE) * static_cast<uint64_t>(maxLenDomesticTileC) *
                              static_cast<uint64_t>(tilingData->commonTiling.MOverTp));
    } else {
        // max of (local_tile_e * local_tile_c * M/tp * tp) and ((ep - 1) * nonLocal_tile_e * nonLocal_tile_c * M/tp *
        // tp)
        bmmOut = std::max(static_cast<uint64_t>(maxLenLocalTileE) *
                              static_cast<uint64_t>(tilingData->commonTiling.localTileC.tileLen) *
                              static_cast<uint64_t>(tilingData->commonTiling.MOverTp) *
                              static_cast<uint64_t>(tilingData->commonTiling.tpGroupSize),
                          static_cast<uint64_t>(tilingData->commonTiling.epGroupSize - 1) *
                              static_cast<uint64_t>(maxLenDomesticTileE) * static_cast<uint64_t>(maxLenDomesticTileC) *
                              static_cast<uint64_t>(tilingData->commonTiling.MOverTp) *
                              static_cast<uint64_t>(tilingData->commonTiling.tpGroupSize));
    }
    return bmmOut * static_cast<uint64_t>(tilingData->commonTiling.inputDatasize);
}

// calculate workspace for GPT2.2T scenario
static ge::graphStatus MC2SetWorkspaceShard(gert::TilingContext *context,
                                            AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "get workspace failed"),
                    return ge::GRAPH_FAILED);

    const uint64_t commOut = GetCommOutSize(tilingData);

    const uint64_t maxLenDomesticTileC =
        std::max(tilingData->commonTiling.domesticTileC.tileLen, tilingData->commonTiling.domesticTileC.tailLen);
    const uint64_t maxLenLocalTileE =
        std::max(tilingData->commonTiling.localTileE.tileLen, tilingData->commonTiling.localTileE.tailLen);
    const uint64_t maxLenDomesticTileE =
        std::max(tilingData->commonTiling.domesticTileE.tileLen, tilingData->commonTiling.domesticTileE.tailLen);
    const uint64_t transOut = GetTransOutSize(tilingData, maxLenDomesticTileC, maxLenLocalTileE, maxLenDomesticTileE);
    const uint64_t bmmOut = GetBmmOutSize(tilingData, maxLenDomesticTileC, maxLenLocalTileE, maxLenDomesticTileE);

    workspaces[0] = commOut + transOut + bmmOut + 16 * 1024 * 1024; // 16 mb, 1024 * 1024 is 1 mb
    OP_LOGD("AlltoAllAllGatherBatchMatMul",
            "workspaces[0] size is %ld. commOut is %lu, transOut is %lu and"
            " bmmOut is %lu",
            workspaces[0], commOut, transOut, bmmOut);
    return ge::GRAPH_SUCCESS;
}

static void SetDataSliceInfoInCommmonTilingData(const gert::TilingContext *context,
                                                AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    auto attrs = context->GetAttrs();
    const gert::StorageShape *xInputShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *weightInputShape = context->GetInputShape(WEIGHT_INDEX);
    auto ep = *(attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX));
    auto tp = *(attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX));
    auto xShard = attrs->GetAttrPointer<int64_t>(ATTR_X_SHARD_TYPE_INDEX);
    bool isWeightTrans = *(attrs->GetAttrPointer<bool>(ATTR_IS_WEIGHT_TRANS_INDEX));
    int64_t E = xInputShape->GetStorageShape().GetDim(0);
    int64_t e = weightInputShape->GetStorageShape().GetDim(0);
    int64_t C =
        (*xShard == 1) ? xInputShape->GetStorageShape().GetDim(1) * tp : xInputShape->GetStorageShape().GetDim(1);
    int64_t c =
        (*xShard == 1) ? xInputShape->GetStorageShape().GetDim(1) : xInputShape->GetStorageShape().GetDim(1) / tp;
    //  设 w = [E, H, M], dimH 指 H 轴, dimM 指 M 轴
    size_t wDimH = 1UL;
    size_t wDimM = 2UL; // 1 2 分别代表 weight 没有转置时候的维度值, 2: M 轴, 1: H 轴
    // w_trans = [E, M, H]
    if (isWeightTrans) {
        size_t wDimHTrans = 2UL;
        size_t wDimMTrans = 1UL; // 2 1 分别代表 weight 转置时候的维度值, 2: H 轴, 1: M 轴
        wDimH = wDimHTrans;
        wDimM = wDimMTrans;
    }
    int64_t dimH = weightInputShape->GetStorageShape().GetDim(wDimH);
    int64_t h = dimH / tp;
    int64_t m = weightInputShape->GetStorageShape().GetDim(wDimM);

    tilingData->commonTiling.epGroupSize = ep;
    tilingData->commonTiling.tpGroupSize = tp;
    tilingData->commonTiling.expert = E;
    tilingData->commonTiling.EOverEp = e;
    tilingData->commonTiling.C = C;
    tilingData->commonTiling.COverTp = c;
    tilingData->commonTiling.H = dimH;
    tilingData->commonTiling.HOverTp = h;
    tilingData->commonTiling.MOverTp = m;
    tilingData->commonTiling.isWeightTrans = isWeightTrans;
    tilingData->commonTiling.xShardFlag = *xShard;
    return;
}

static void SetDataTypeAndSizeInfoInCommmonTilingData(const gert::TilingContext *context,
                                                      AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    auto attrs = context->GetAttrs();
    const gert::StorageShape *biasInputShape =
        context->GetOptionalInputTensor(BIAS_INDEX) == nullptr ? nullptr : context->GetOptionalInputShape(BIAS_INDEX);
    auto y2Flag = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_Y2_FLAG_INDEX);
    auto y3Flag = attrs->GetAttrPointer<bool>(ATTR_OUTPUT_Y3_FLAG_INDEX);
    auto activate = attrs->GetAttrPointer<int64_t>(ATTR_ACT_TYPE_INDEX);

    ge::DataType inputDatatype = context->GetInputDesc(X_INDEX)->GetDataType();
    ge::DataType biasDatatype =
        (biasInputShape == nullptr) ? inputDatatype : context->GetOptionalInputDesc(BIAS_INDEX)->GetDataType();
    uint32_t inputDatasize = GetDataSize(inputDatatype);
    uint32_t biasDatasize = GetDataSize(biasDatatype);

    tilingData->commonTiling.inputDatasize = inputDatasize;
    tilingData->commonTiling.biasDatasize = biasDatasize;
    tilingData->commonTiling.isBias = biasInputShape == nullptr ? false : true;
    tilingData->commonTiling.y2Flag = *y2Flag;
    tilingData->commonTiling.y3Flag = *y3Flag;
    tilingData->commonTiling.activateType = *activate;
    return;
}

static void SetCommonTilingData(gert::TilingContext *context, uint64_t ubSize, uint64_t aivNum,
                                AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    SetDataSliceInfoInCommmonTilingData(context, tilingData);
    SetDataTypeAndSizeInfoInCommmonTilingData(context, tilingData);
    InitTileInfoInCommonTiling(tilingData);
    tilingData->commonTiling.aivCoreNum = aivNum;
    tilingData->commonTiling.totalUbSize = ubSize;
    return;
}

static void GetChipDataFromPlatform(const gert::TilingContext *context, uint32_t &numBlocks, uint64_t &ubSize,
                                    uint64_t &aicNum, uint64_t &aivNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aicNum = ascendcPlatform.GetCoreNumAic();
    aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    return;
}

static void SetHcclTiling(const gert::TilingContext *context, AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    std::string alltoAllConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string allGatherConfig = "AllGather=level0:doublering";

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"), return);
    auto epGroup = attrs->GetAttrPointer<char>(ATTR_EP_GROUP_INDEX);
    auto tpGroup = attrs->GetAttrPointer<char>(ATTR_TP_GROUP_INDEX);

    const uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    const uint32_t opType2 = OP_TYPE_ALL_GATHER;
    const uint32_t reduceType = 0U;
    ge::DataType outputDataType = context->GetOutputDesc(OUTPUT_Y1_INDEX)->GetDataType();
    ge::DataType inputDataType = context->GetInputDesc(X_INDEX)->GetDataType();
    OP_TILING_CHECK(mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
                    OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y1", Ops::Base::ToString(outputDataType).c_str(),
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
    hcclCcTilingConfig.SetAlgConfig(allGatherConfig);
    hcclCcTilingConfig.GetTiling(tilingData->allGatherCcTiling);
    return;
}

static ge::graphStatus SetBatchMatMulTilingData(gert::TilingContext *context, uint64_t aicNum,
                                                AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    mc2tiling::TilingArgs formulaicArgs;
    AlltoAllAllGatherBatchInfo BMMV3BatchInfo;
    AlltoAllAllGatherMatmulInfo MMV3ArgsInfo;

    const gert::StorageShape *biasInputShape =
        context->GetOptionalInputTensor(BIAS_INDEX) == nullptr ? nullptr : context->GetOptionalInputShape(BIAS_INDEX);
    ge::DataType inputDatatype = context->GetInputDesc(X_INDEX)->GetDataType();
    ge::DataType biasDatatype =
        (biasInputShape == nullptr) ? inputDatatype : context->GetOptionalInputDesc(BIAS_INDEX)->GetDataType();

    MMV3ArgsInfo.opName = "AlltoAllAllGatherBatchMatMul";
    MMV3ArgsInfo.isWeightTrans = tilingData->commonTiling.isWeightTrans;
    MMV3ArgsInfo.isBias = false;
    MMV3ArgsInfo.aType = inputDatatype;
    MMV3ArgsInfo.bType = inputDatatype;
    MMV3ArgsInfo.cType = inputDatatype;
    MMV3ArgsInfo.biasType = biasDatatype;

    BMMV3BatchInfo.biasWithBatch = false;

    formulaicArgs.mValue = tilingData->commonTiling.C;
    formulaicArgs.nValue = tilingData->commonTiling.MOverTp;
    formulaicArgs.kValue = tilingData->commonTiling.H;
    formulaicArgs.inputDtypeSize = GetDataSize(inputDatatype);
    formulaicArgs.outputDtypeSize = GetDataSize(inputDatatype);
    formulaicArgs.aicCoreNum = aicNum;
    formulaicArgs.aType = GetMatMulTilingDataType(inputDatatype);
    formulaicArgs.bType = GetMatMulTilingDataType(inputDatatype);
    formulaicArgs.cType = GetMatMulTilingDataType(inputDatatype);
    formulaicArgs.biasType = GetMatMulTilingDataType(biasDatatype);

    // 待修改，等BMM tiling提供接口和修改方案
    OP_TILING_CHECK(SetMatmulTilingAlltoAllAllGatherBatchMatMul(context, tilingData, BMMV3BatchInfo, MMV3ArgsInfo,
                                                                formulaicArgs) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Set Matmul tiling Failed!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetUbTilingDataInCommonTiling(const gert::TilingContext *context, uint64_t ubSize,
                                          AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    ge::DataType inputDatatype = context->GetInputDesc(X_INDEX)->GetDataType();
    auto inputDatasize = GetDataSize(inputDatatype);
    auto activateType = tilingData->commonTiling.activateType;

    bool xCastFlag = ((inputDatatype == ge::DT_BF16) || (activateType == SILU));
    auto mOverTp = tilingData->commonTiling.MOverTp;
    // GPT2.6场景
    TransposeConfig config = {ubSize,
                              tilingData->commonTiling.localTileC.tileLen,
                              tilingData->commonTiling.localTileE.tileLen,
                              inputDatasize,
                              true,
                              false};
    CheckTransposeUBAndUpdateTileShard(config, tilingData);

    config = {ubSize,
              tilingData->commonTiling.localTileC.tailLen,
              tilingData->commonTiling.localTileE.tailLen,
              inputDatasize,
              true,
              true};
    CheckTransposeUBAndUpdateTileShard(config, tilingData);

    config = {ubSize,
              tilingData->commonTiling.domesticTileC.tileLen,
              tilingData->commonTiling.domesticTileE.tileLen,
              inputDatasize,
              false,
              false};
    CheckTransposeUBAndUpdateTileShard(config, tilingData);

    config = {ubSize,
              tilingData->commonTiling.domesticTileC.tailLen,
              tilingData->commonTiling.domesticTileE.tailLen,
              inputDatasize,
              false,
              true};
    CheckTransposeUBAndUpdateTileShard(config, tilingData);

    ActivationParams actParams = {ubSize, xCastFlag, activateType};
    TileShardParams params = {tilingData->commonTiling.localTileC.tileLen, tilingData->commonTiling.localTileE.tileLen,
                              mOverTp, true, false};
    CheckAddActivateUBAndUpdateTileShard(actParams, params, tilingData);
    params = {tilingData->commonTiling.localTileC.tailLen, tilingData->commonTiling.localTileE.tailLen, mOverTp, true,
              true};
    CheckAddActivateUBAndUpdateTileShard(actParams, params, tilingData);
    params = {tilingData->commonTiling.domesticTileC.tileLen, tilingData->commonTiling.domesticTileC.tileLen, mOverTp,
              false, false};
    CheckAddActivateUBAndUpdateTileShard(actParams, params, tilingData);
    params = {tilingData->commonTiling.domesticTileC.tailLen, tilingData->commonTiling.domesticTileC.tailLen, mOverTp,
              false, true};
    CheckAddActivateUBAndUpdateTileShard(actParams, params, tilingData);

    return;
}

static ge::graphStatus SetContextData(gert::TilingContext *context, uint32_t numBlocks,
                                      AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    uint64_t tilingKey = UpdateTilingKey(tilingData, tilingData->commonTiling.y2Flag, tilingData->commonTiling.y3Flag);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(numBlocks);
    context->GetRawTilingData()->SetDataSize(sizeof(AlltoAllAllGatherBatchMatMulTilingData));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllAllGatherBatchMatMulTilingFunc(gert::TilingContext *context)
{
    // tiling校验shape
    OP_TILING_CHECK(TilingCheckAlltoAllAllGatherBatchMatMul(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Tiling check shape Failed!"), return ge::GRAPH_FAILED);
    AlltoAllAllGatherBatchMatMulTilingData *tilingData =
        context->GetTilingData<AlltoAllAllGatherBatchMatMulTilingData>();
    uint32_t numBlocks = 1U;
    uint64_t ubSize = 0U;
    uint64_t aicNum = 0U;
    uint64_t aivNum = 0U;
    GetChipDataFromPlatform(context, numBlocks, ubSize, aicNum, aivNum);

    SetCommonTilingData(context, ubSize, aivNum, tilingData);
    OP_TILING_CHECK(SetBatchMatMulTilingData(context, aicNum, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Set BatchMatmul tiling failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MC2SetWorkspaceShard(context, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Set workspace Failed!"), return ge::GRAPH_FAILED);
    SetUbTilingDataInCommonTiling(context, ubSize, tilingData);
    OP_TILING_CHECK(SetContextData(context, numBlocks, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Set context data failed!"), return ge::GRAPH_FAILED);

    SetHcclTiling(context, tilingData);
    PrintCommonTilingVariables(tilingData);
    PrintSliceTileInfo(tilingData);
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllAllGatherBatchMatMulCompileInfo {};
ge::graphStatus TilingParseForAlltoAllAllGatherBatchMatMul(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AlltoAllAllGatherBatchMatMul)
    .Tiling(AlltoAllAllGatherBatchMatMulTilingFunc)
    .TilingParse<AlltoAllAllGatherBatchMatMulCompileInfo>(TilingParseForAlltoAllAllGatherBatchMatMul);
} // namespace optiling
