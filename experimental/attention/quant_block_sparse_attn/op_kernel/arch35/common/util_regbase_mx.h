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
 * \file util_regbase_mx.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化运行期参数。
 */
#ifndef UTIL_REGBASE_MX_H
#define UTIL_REGBASE_MX_H

#include <cstdint>
#include "../../quant_block_sparse_attn_common.h"

namespace regbasemx {
constexpr uint32_t QBSA_MX_SCALE_LAST_DIM = 2U;
constexpr uint32_t QBSA_MX_S2_BASE_SIZE = 512U;
constexpr uint32_t QBSA_MX_MIN_KV_BLOCK_SIZE = 64U;
constexpr uint32_t QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK = QBSA_MX_S2_BASE_SIZE / QBSA_MX_MIN_KV_BLOCK_SIZE;

// 单个 (B, N1, S1 block, 512-token logical S2 tile) 的运行期状态。
struct MxRunInfo {
    // 流水状态。
    uint32_t loop = 0U;
    uint32_t mLoop = 0U;
    bool isValid = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;
    // bit0/bit1分别表示256-token subLoop 0/1是否需要真实causal mask。
    uint8_t attenMaskSubLoopBits = 0U;

    // Batch/head/query 坐标。
    uint32_t bIdx = 0U;
    // KV head index。
    uint32_t n2Idx = 0U;
    uint32_t s1Idx = 0U;
    uint32_t realN2Idx = 0U;

    // 实际序列长度及两个 AIV 的 M 轴切分。
    uint64_t actS1Size = 1U;
    uint64_t actS2Size = 1U;
    uint32_t actMSize = 0U;
    uint32_t actVecMSize = 0U;
    uint32_t vecMbaseIdx = 0U;

    // 拼接后 logical S2 tile 的有效长度。
    uint32_t actSingleLoopS2Size = 0U;

    // Sparse block 索引与拼接信息，单 task 最多 8 个 block。
    uint32_t sparseBlockCount = 0U;
    // 有效 block id 升序，负 index 在末尾。
    int64_t sparseBlockIdx[QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK] = {};
    // 原始 KV token 起点，用于 PA/mask 寻址。
    uint64_t sparseBlockTokenOffset[QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK] = {};
    // 拼接后 tile 起点，K/V/scale/mask 共用。
    uint32_t sparseBlockTileOffset[QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK] = {};
    uint32_t sparseBlockRealSize[QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK] = {};
    // true 表示跨 causal 边界或整块不可见，需要搬压缩 mask。
    bool sparseBlockPartialMask[QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK] = {};

    // Q/QScale 的 TND GM offset。
    uint64_t queryTokenBase = 0U;
    uint64_t queryOffset = 0U;
    uint64_t queryScaleOffset = 0U;
};

// 基础 shape 与 TND stride；scale/data offset 默认以元素为单位。
struct MxBaseConstInfo {
    uint32_t dSize = 0U;
    uint32_t dSizeV = 0U;
    uint32_t gSize = 0U;
    uint32_t n2Size = 0U;
    uint32_t realN2Size = 0U;
    uint32_t coreNum = 0U;
    uint32_t aicIdx = 0U;
    uint8_t subBlockIdx = 0U;
    uint32_t n2GD = 0U;
    uint32_t qScaleN1D = 0U;
    // PA BNBD scale 中单个 kv head 的 stride。
    uint32_t kScaleN2D = 0U;
    uint32_t valueScaleN2D = 0U;
    uint32_t attentionOutStride = 0U;
    uint32_t softmaxLseStride = 0U;
    float scaleValue = 0.0F;
};

// PA BNBD 寻址参数。
struct MxPageAttentionConstInfo {
    uint32_t maxBlockNumPerBatch = 0U;
    uint32_t paBlockStride = 0U;
    uint32_t qSparseBlockSize = 0U;
    uint32_t kvSparseBlockSize = 0U;
    uint32_t paBlockSize = 0U;
};

// B_N_Qb_Kb sparse 参数。
struct MxSparseConstInfo {
    uint32_t maxQb = 0U;
    uint32_t maxKb = 0U;
};

// QScale 为 [T,N,D/64,2]；K/V scale 为 PA BNBD 布局。
struct MxScaleConstInfo {
    uint32_t scaleLastDim = QBSA_MX_SCALE_LAST_DIM;
    uint32_t queryScaleDSize = 0U;
    uint32_t keyScaleDSize = 0U;
    uint32_t valueScaleDSize = 0U;
};

struct MxConstInfo : MxBaseConstInfo, MxPageAttentionConstInfo, MxSparseConstInfo, MxScaleConstInfo {};
} // namespace regbasemx

#endif // QUANT_BLOCK_SPARSE_ATTN_COMMON_UTIL_REGBASE_MX_H_
