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
 * \file fused_causal_conv1d_cut_bh_struct.h
 * \brief FusedCausalConv1dCutBH tiling struct
 */

#ifndef FUSED_CAUSAL_CONV1D_CUT_BH_STRUCT_H
#define FUSED_CAUSAL_CONV1D_CUT_BH_STRUCT_H

struct FusedCausalConv1dCutBHTilingData {
    // Core distribution parameters
    int64_t usedCoreNum;  // Total used core number
    int64_t dimCoreCnt;   // Number of cores for dim direction
    int64_t batchCoreCnt; // Number of cores for batch direction

    // Dim tiling parameters inter-core (non-uniform split)
    int64_t dimMainCoreCnt; // Number of big dim cores (base+1 blocks of 128)
    int64_t dimTailCoreCnt; // Number of small dim cores (base blocks of 128)
    int64_t mainCoredimLen; // Big core dim size: (base+1) * 128
    int64_t tailCoredimLen; // Small core dim size: base * 128

    // Batch tiling parameters inter-core (non-uniform split)
    int64_t batchMainCoreCnt; // Number of big batch cores
    int64_t batchTailCoreCnt; // Number of small batch cores
    int64_t mainCoreBatchNum; // Batch size for big cores
    int64_t tailCoreBatchNum; // Batch size for small cores

    // Intra-core tiling parameters UB loop
    int64_t loopNumBS;                // Loops in BS direction for big cores
    int64_t loopNumDim;               // Loops in Dim direction for big cores
    int64_t ubMainFactorBS;           // UB BS factor for big cores
    int64_t ubTailFactorBS;           // UB BS tail factor for big cores
    int64_t ubMainFactorDim;          // UB Dim factor for big cores
    int64_t ubTailFactorDim;          // UB Dim tail factor for big cores
    int64_t tailBlockloopNumBS;       // Loops in BS direction for tail cores
    int64_t tailBlockloopNumDim;      // Loops in Dim direction for tail cores
    int64_t tailBlockubFactorBS;      // UB BS factor for tail cores
    int64_t tailBlockubTailFactorBS;  // UB BS tail factor for tail cores
    int64_t tailBlockubFactorDim;     // UB Dim factor for tail cores
    int64_t tailBlockubTailFactorDim; // UB Dim tail factor for tail cores

    // Shape information for kernel use
    int64_t batchSize;            // Batch size
    int64_t seqLen;               // Sequence length for 3D input
    int64_t cuSeqLen;             // Cumulative sequence length for 2D input
    int64_t dim;                  // Dimension size
    int64_t kernelSize;           // Kernel size K
    int64_t stateLen;             // State length: second dimension of cacheState (K-1+m)
    int64_t xStride;              // Stride for x tensor - seqLen
    int64_t cacheStride0;         // Stride for cacheState tensor - batch
    int64_t cacheStride1;         // Stride for cacheState tensor - stateLen
    int64_t padSlotId;            // padding batch which will not be calculated
    int64_t xInputMode;           // Input mode: 0 for 3D 1 for 2D
    int64_t hasAcceptTokenNum;    // Whether acceptTokenNum input is provided: 0 for false, 1 for true
    int64_t residualConnection;   // Whether use residual connection: 0 for false, 1 for true
    int64_t apcEnable;            // 是否开启 APC：0=关闭，1=开启
    int64_t blockSize;            // APC block 大小（128 或 256）
    int64_t maxNumBlocks;         // cache_indices 第二维大小（APC 开启时有效）
    int64_t hasCacheIndices;      // cacheIndices 是否有效：0=无（batch_idx 直接作为索引），1=有
    int64_t inplace;              // 0=非原地，1=原地（y 写回 x）
    int64_t convMode;             // 0=Qwen3；1=Pangu v2（前 width-1 个 token 置零）
    int64_t hasNumComputedTokens; // numComputedTokens 是否有效：0=无，1=有
    int64_t maxQueryLen;

    // ---- dimRemainder（last dim core）相关参数 ----
    int64_t dimRemainderElems;  // dim % 128，最后一个 dim 核额外处理的余数元素数（0 表示无余数）
    int64_t lastCoreloopNumDim; // 最后一个 dim 核的 dim 方向循环次数
    int64_t lastCoreubMainFactorDim; // 最后一个 dim 核的 UB 主块 dim 大小
    int64_t lastCoreubTailFactorDim; // 最后一个 dim 核的 UB 尾块 dim 大小
};

#endif
