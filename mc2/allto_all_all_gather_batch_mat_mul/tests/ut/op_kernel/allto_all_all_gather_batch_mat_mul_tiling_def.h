/**	
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.	
 * This file is a part of the CANN Open Software.	
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").	
 * Please refer to the License for details. You may not use this file except in compliance with the License.	
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,	
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.	
 * See LICENSE in the root of the software repository for the full text of the License.	
 */	
#ifndef ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_TILING_DEF_H	
#define ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_TILING_DEF_H	

#include "kernel_tiling/kernel_tiling.h"	
struct L2cacheUseInfo{	
    uint32_t l2CacheFlag;	
};	

struct L2cacheTilePara{	
  uint32_t mTileCntL2;	
  uint32_t nTileCntL2;	
  uint32_t mTileBlock;	
  uint32_t nTileBlock;	
  uint32_t calOrder;	
};	


struct MatMulRunInfo {
    uint32_t transA;	
    uint32_t transB;	
    uint32_t nd2nzA;	
    uint32_t nd2nzB;	
    uint32_t isHf32;	
};	

struct MultiBatchInfo {	
  uint32_t batchUsedCoreNum;	
  uint32_t aBatchDimAll;	
  uint32_t bBatchDimAll;	
  uint32_t cBatchDimAll;	
  uint32_t aBatchDim0;	
  uint32_t bBatchDim0;	
  uint32_t cBatchDim0;	
  uint32_t aBatchDim1;	
  uint32_t bBatchDim1;	
  uint32_t cBatchDim1;	
  uint32_t aBatchDim2;	
  uint32_t bBatchDim2;	
  uint32_t cBatchDim2;	
  uint32_t aBatchDim3;	
  uint32_t bBatchDim3;	
  uint32_t cBatchDim3;	
  uint32_t iterBatch;	
  uint32_t biasWithBatch;	
  uint32_t mOri;	
  uint32_t batchTileBlock;	
  uint32_t aBatch;	
  uint32_t bBatch;	
};	

struct MatmulTilingData{	
  TCubeTiling matmulTiling;	
  L2cacheTilePara tileL2cacheTiling;	
  MatMulRunInfo matmulRunInfo;	
  L2cacheUseInfo l2cacheUseInfo;	
  uint32_t baseAN;	
  uint32_t baseAD;	
  uint32_t baseBN;	
  uint32_t baseBD;	
};	

struct BatchMatmulTilingData {	
    MatmulTilingData matmulTiling;	
    MultiBatchInfo multiBatchInfo;	
};	


struct TileInfo {	
    uint32_t tileCnt;	
    uint32_t tileLen;	
    uint32_t tailCnt;	
    uint32_t tailLen;	
};	

struct Mc2CommonTiling {	
    uint32_t epGroupSize;                  // 每个ep域内的并行运行的专家的个数	
    uint32_t tpGroupSize;	
    uint32_t expert;                        // E	
    uint32_t EOverEp;                       // E/ep	
    uint32_t C;	
    uint32_t COverTp;                       // C/tp	
    uint32_t H;	
    uint32_t HOverTp;                       // H/tp	
    uint32_t MOverTp;                       // M/tp	
    uint32_t aivCoreNum;	
    uint32_t inputDatasize;	
    uint32_t biasDatasize;	
    uint32_t ubCapacityForTrans;	
    uint32_t ubCapacityForAddActivate;	
    bool isBias;	
    bool y2Flag;	
    bool y3Flag;	
    bool isWeightTrans;	
    TileInfo localTileE;             // E 轴本地块切分信息	
    TileInfo domesticTileE;          // E 轴非本地块切分信息	
    TileInfo localTileC;             // C 轴本地块切分信息	
    TileInfo domesticTileC;          // C 轴非本地块切分信息	
    TileInfo localUbTranspose;	
    TileInfo localTailUbTranspose;	
    TileInfo domesticUbTranspose;	
    TileInfo domesticTailUbTranspose;	
    TileInfo localUbAdd;	
    TileInfo localTailUbAdd;	
    TileInfo domesticUbAdd;	
    TileInfo domesticTailUbAdd;	
    uint32_t activateType;                  // 激活措施：0为不激活，1为GELU，3为Relu，4为FastGELU	
    uint32_t xShardFlag;	
    uint32_t fastGeluBuffer;	
    uint32_t totalUbSize;	
};	

struct Mc2MatmulTilingData {	
    uint32_t rankDim;	
    uint32_t rankM;	
    uint32_t rankID;	
    uint32_t enableL2Tile;	
    BatchMatmulTilingData bmmTilingData;	
};	

struct AlltoAllAllGatherBatchMatMulTilingData	
{	
    uint32_t version;	
    uint32_t hcommCnt;	
    Mc2ServerCfg serverCfg;	
    Mc2HcommCfg hcommCfgATA;	
    Mc2HcommCfg hcommCfgAG;	
    Mc2CommonTiling commonTiling;	
    Mc2MatmulTilingData localTiling;	
    Mc2MatmulTilingData domesticTiling;	
    Mc2MatmulTilingData localTailTiling;	
    Mc2MatmulTilingData domesticTailTiling;	
    Mc2MatmulTilingData domesticTailETiling;	
};	

inline void InitAllGatherMatmulTilingData(uint8_t* tiling, AlltoAllAllGatherBatchMatMulTilingData* const_data)	
{	
    memcpy(const_data, tiling, sizeof(AlltoAllAllGatherBatchMatMulTilingData));	
}	

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                        \	
    AlltoAllAllGatherBatchMatMulTilingData tiling_data;                                                 \	
    InitAllGatherMatmulTilingData(tiling_arg, &tiling_data)	

#endif