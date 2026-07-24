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
 * \file fused_causal_conv1d_cut_bsh_struct.h
 * \brief tiling data struct
 */

#ifndef __FUSED_CAUSAL_CONV1D_CUT_BSH_TILING_DATA_H__
#define __FUSED_CAUSAL_CONV1D_CUT_BSH_TILING_DATA_H__

struct FusedCausalConv1dCutBSHTilingData {
    // ===== 核内UB切分参数 =====
    uint64_t loopNumBS;                // 每个核内BS方向的loop循环数
    uint64_t loopNumDim;               // 每个核内Dim方向的loop循环数（2D时固定为1）
    uint64_t ubFactorBS;               // 每个核内BS方向单次循环载入的大小
    uint64_t ubTailFactorBS;           // 每个核内BS方向尾次循环载入的大小
    uint64_t ubFactorDim;              // 每个核内Dim方向单次循环载入的大小
    uint64_t ubTailFactorDim;          // 每个核内Dim方向尾次循环载入的大小
    uint64_t tailBlockloopNumBS;       // 尾核内BS方向的loop循环数
    uint64_t tailBlockloopNumDim;      // 尾核内Dim方向的loop循环数
    uint64_t tailBlockubFactorBS;      // 尾核内BS方向单次循环载入的大小
    uint64_t tailBlockubTailFactorBS;  // 尾核内BS方向尾次循环载入的大小
    uint64_t tailBlockubFactorDim;     // 尾核内Dim方向单次循环载入大小
    uint64_t tailBlockubTailFactorDim; // 尾核内Dim方向尾次循环载入大小

    // ===== dim方向核间切分信息 =====
    uint64_t dimCoreNum;         // dim方向总核数
    uint64_t dimRemainderCores;  // 前多少个核是大核（分配base+1个128-块）
    uint64_t dimBlockFactor;     // 大核的dim大小（(base+1) * 128，前dimRemainderCores个核）
    uint64_t dimBlockTailFactor; // 小核的dim大小（base * 128，后面的核）

    // ===== BS方向核间切分信息 =====
    uint64_t bsCoreNum;         // BS方向总核数
    uint64_t bsRemainderCores;  // BS方向前多少个核是大核（均分策略）
    uint64_t bsBlockFactor;     // BS方向大核处理的长度（含overlap，前bsRemainderCores个核）
    uint64_t bsBlockTailFactor; // BS方向小核处理的长度（后面的核）

    // ===== 核数信息 =====
    uint64_t realCoreNum; // 实际使用核数 = dimCoreNum × bsCoreNum
    uint64_t kernelWidth; // 卷积核宽度 K
    uint64_t cuSeqLen;    // cu_seq_len 大小
    uint64_t dim;         // 特征维度大小
    uint64_t batch;       // batch大小
    int64_t padSlotId;    // padding slot ID，用于 kernel 中跳过 padding batch
    uint64_t stateLen;
    uint64_t xStride;            // x 的 stride（即 dim）
    uint64_t cacheStride0;       // convStates 的 stride[0]
    uint64_t cacheStride1;       // convStates 的 stride[1]
    uint64_t residualConnection; // 是否做残差连接

    uint64_t apcEnabled;           // APC 是否启用（0/1），等价于 cache_indices 是否为 2D
    uint64_t blockSize;            // APC block 大小（0 表示未启用）
    uint64_t maxNumBlocks;         // cache_indices 第二维大小（APC 模式下 > 0）
    uint64_t convMode;             // 0=Qwen3-Next, 1=Pangu v2 输出零填充
    uint64_t inplace;              // 是否原地更新（y 复用 x 的 GM）
    uint64_t hasAcceptTokenNum;    // 是否提供 num_accepted_tokens（MTP）
    uint64_t hasNumComputedTokens; // 是否提供 num_computed_tokens（cache 零初始化判断）
    uint64_t hasCacheIndices;      // 是否提供 cache_indices（为0时用 batch_idx 作为 cache line）

    uint64_t dimRemainderElems;       // dim % 128，最后一个 dim 核额外承担的余数元素数
    uint64_t lastCoreloopNumDim;      // 最后一个 dim 核 Dim 方向循环次数
    uint64_t lastCoreubFactorDim;     // 最后一个 dim 核 UB 主块 dim 大小
    uint64_t lastCoreubTailFactorDim; // 最后一个 dim 核 UB 尾块 dim 大小
};

#endif
