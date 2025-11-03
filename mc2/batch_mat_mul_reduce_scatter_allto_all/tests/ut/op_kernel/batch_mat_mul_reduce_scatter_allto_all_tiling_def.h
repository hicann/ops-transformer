/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_TILING_DEF_H
#define BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "batch_mat_mul_v3_tiling_def.h"

struct Mc2HcommCfgTmp {
    char hcomId[128];
    char tag[128];
    char algName[128];
    uint32_t algType;
    uint8_t opType;
    uint8_t subType;
    uint8_t reserve[10];
    uint8_t hcclReserve[512]; 
};

struct Mc2ServerCfgTmp {
    uint32_t preparePosition;
    uint8_t debugMode;
    uint8_t sendArgIndex;
    uint8_t recvArgIndex;
    uint8_t commOutArgIndex;
    uint32_t reserve;
};

struct TileInfo {
    uint32_t tileCnt;
    uint32_t tileLen;
    uint32_t tailCnt;
    uint32_t tailLen;
};

struct Mc2RSATATiling {
    uint32_t epGroupSize;
    uint32_t tpGroupSize;
    uint32_t expert;
    uint32_t EOverEp;
    uint32_t C;
    uint32_t COverTp;
    uint32_t H;
    uint32_t HOverTp;
    uint32_t MOverTp;
    uint32_t aivCoreNum;
    uint32_t inputDatasize;
    uint32_t biasDatasize;
    uint32_t ubCapacityForTrans;
    uint32_t ubCapacityForAdd;
    uint64_t totalUbSize;
    bool isBias;
    bool isWeightTrans;
    TileInfo localTileE;
    TileInfo domesticTileE;
    TileInfo localTileC;
    TileInfo domesticTileC;
    TileInfo localUbTranspose;
    TileInfo localTailUbTranspose;
    TileInfo domesticUbTranspose;
    TileInfo domesticTailUbTranspose;
    TileInfo localUbAdd;
    TileInfo localTailUbAdd;
    TileInfo domesticUbAdd;
    TileInfo domesticTailUbAdd;
    uint32_t yShardFlag;
};

struct Mc2MatmulTilingData {
    uint32_t rankDim;
    uint32_t rankM;
    uint32_t rankID;
    uint32_t enableL2Tile;
    Mc2BatchMatmulTilingData bmmTilingData;
};

struct BatchMatMulReduceScatterAlltoAllTilingData {
    uint32_t version;
    uint32_t hcommCnt;
    Mc2HcommCfgTmp hcommCfgRS;
    Mc2HcommCfgTmp hcommCfgATA;
    Mc2ServerCfgTmp serverCfg;
    Mc2RSATATiling commonTiling;
    Mc2MatmulTilingData localTiling;
    Mc2MatmulTilingData domesticTiling;
    Mc2MatmulTilingData localTailTiling;
    Mc2MatmulTilingData domesticTailTiling;
};

#endif  // BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_TILING_DEF_H