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
 * \file kda_input_proj_tiling_data.h
 * \brief
 */

#ifndef KDA_INPUT_PROJ_TILING_DATA_H
#define KDA_INPUT_PROJ_TILING_DATA_H

#include <cstdint>

namespace optiling {

#pragma pack(push, 8)

struct alignas(8) KdaInputProjBaseParams {
    uint32_t tSize = 0;      // token 个数 T
    uint32_t hiddenSize = 0; // 隐藏层维度，如 7168
    uint32_t qkvSize = 0;    // qkv 投影输出维度，如 4608
    uint32_t betaSize = 0;   // beta 投影输出维度，如 12
    uint32_t gateSize = 0;   // gate 投影输出维度，如 1536
    uint32_t gSize = 0;      // g 投影输出维度，如 1536
};

// Stage1 AIC：逻辑拼接后 round-robin。填值约定：
//   betaTile == gateTile == gTile == 共用 baseN（nTile）
//   *Dim     = nCnt_* = CeilDiv(N_*, nTile)
//   num*Tile = tDim * nCnt_* ；SetBlockDim = min(aicNum, tileNumAll)
//   hiddenstatesTile / hiddenstatesL0Tile 由 Host 按 Blaze L1 slot + ComputeL0K 写入
struct alignas(8) KdaInputProjMmBggParams {
    uint32_t tTile{0};       // 共用 Cube baseM
    uint32_t betaTile{0};    // = nTile（与 gateTile/gTile 相同）
    uint32_t gateTile{0};    // = nTile
    uint32_t gTile{0};       // = nTile
    uint32_t tDim{0};        // CeilDiv(T, tTile)
    uint32_t betaDim{0};     // nCntBeta
    uint32_t gateDim{0};     // nCntGate
    uint32_t gDim{0};        // nCntG
    uint32_t numBetaTile{0}; // tDim * CeilDiv(N_beta, nTile)
    uint32_t numGateTile{0};
    uint32_t numGTile{0};
    uint32_t hiddenstatesTile{0};   // L1 上 K 方向步长
    uint32_t hiddenstatesL0Tile{0}; // L0 上 K 方向步长
};

// 占位字段保证非空且 8 字节对齐；后续填真实字段时可替换
struct alignas(8) KdaInputProjMxQuantParams {
    uint64_t tilingKey{0};
    int64_t ubSize{0};
    int64_t roundMode{0};
    int64_t blockSize{0};
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t rowTileNum{0};        // row 方向上的切核数
    int64_t colTileNum{0};        // col 方向上的切核数
    int64_t rowNum{0};            // 合轴之后 -2 轴大小
    int64_t colNum{0};            // 合轴之后 -1 轴大小
    int64_t colNormalBlockNum{0}; // 列方向头核处理的块数 (1 x 256)
    int64_t colTailLen{0};        // 列方向尾块长度
    int64_t rowNormalBlockNum{0}; // 行方向头核处理的块数 (1 行)
    int64_t rowTailLen{0};        // 行方向尾块长度
    int64_t maxUbBlockNum{0};     // UB最大能放下的处理块数 (1 x 32) (8 的倍数)
    float dstTypeMax{0.0f};
    float invDstTypeMax{0.0f};
    float maxLowBound{0.0f};
};

struct alignas(8) KdaInputProjQmmQkvParams {
    uint32_t kL1 = 0;
    uint32_t scaleKL1 = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t mTailTile = 1;
    uint32_t nTailTile = 1;
    uint32_t mBaseTailSplitCnt = 1;
    uint32_t nBaseTailSplitCnt = 1;
    uint32_t mTailMain = 0;
    uint32_t nTailMain = 0;
    uint8_t nBufferNum = 0;
    uint8_t dbL0C = 0;
    // 1: B 走 L2 NORMAL（同一核沿 M 复用权重时有收益）；0: streaming，关掉 L2
    uint8_t bMustHitL2 = 1;
};

// Stage2 AIV：1D EleWise inplace Sigmoid(beta)。填值约定：
//   elemNum  = T * N_beta；aivNum = 实际干活 AIV（对齐后可能小于 launched）
//   blockTile / ubTile 均 64 对齐；ubLoop/ubTail 为首核，tailUb* 为尾核
struct alignas(8) KdaInputProjSigmoidParams {
    uint32_t elemNum{0};    // T * N_beta，展平后总元素数
    uint32_t aivNum{0};     // 实际干活的 AIV；其余 GetBlockIdx 早退
    uint32_t blockTile{0};  // 首核元素数，64 对齐
    uint32_t ubTile{0};     // 单次 UB 元素数，64 对齐；InitBuffer 用它
    uint32_t ubLoop{0};     // 首核 UB 循环次数
    uint32_t ubTail{0};     // 首核末次 UB 有效元素数
    uint32_t tailUbLoop{0}; // 尾核 UB 循环次数
    uint32_t tailUbTail{0}; // 尾核末次 UB 有效元素数
};

struct alignas(8) KdaInputProjTilingData {
    KdaInputProjBaseParams baseParams;
    KdaInputProjMmBggParams mmBggParams;
    KdaInputProjMxQuantParams mxQuantParams;
    KdaInputProjQmmQkvParams qmmQkvParams;
    KdaInputProjSigmoidParams sigmoidParams;
};

#pragma pack(pop)

} // namespace optiling

#endif // KDA_INPUT_PROJ_TILING_DATA_H
