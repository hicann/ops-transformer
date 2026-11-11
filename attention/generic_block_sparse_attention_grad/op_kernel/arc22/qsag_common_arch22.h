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
 * \file qsag_common_arch22.h
 * \brief Shared constants, task descriptor and global-memory context for arch22 QSAG.
 */
#ifndef ATTENTION_QSAG_COMMON_ARCH22_H
#define ATTENTION_QSAG_COMMON_ARCH22_H

#include "kernel_operator.h"

namespace QSAG {
using namespace AscendC;

inline constexpr uint32_t AIV_PER_CUBE = 2;
inline constexpr uint32_t NZ_WIDTH = 16; // FP16/BF16 NZ 的 C0 维度。
inline constexpr uint32_t BLOCK_BYTES = 32;
inline constexpr uint32_t FLOATS_PER_BLOCK = BLOCK_BYTES / sizeof(float);
inline constexpr uint32_t MAX_HEAD_DIM = 256;
inline constexpr uint32_t TRANSFER_ROWS = 4;
inline constexpr uint32_t MAX_GROUP_ROWS = 256;
inline constexpr uint32_t SLOT_COUNT = 2;
inline constexpr uint32_t PIPELINE_STAGES = 3;
inline constexpr uint32_t LARGE_GROUP_LIMIT = 128;
inline constexpr uint32_t L0_OPERAND_BYTES = 64 * 1024;
inline constexpr uint32_t L0_RESULT_BYTES = 128 * 1024;
inline constexpr uint64_t MAX_ND2NZ_SRC_STRIDE = 65535; // ND2NZ 文档规定的源行步长上限，单位为元素。
inline constexpr uint64_t MAX_FIXPIPE_STRIDE = 0xFFFFFFFFULL; // API 的目的步长为 uint32_t，单位为元素。

// arch22 跨核 flag：ffTS 消息只携带 flagId & 0xf（命名空间 0-15）。
// 0 不使用，11-14 被框架 SyncAll 保留。MIX 1:2 下一条通道只能用一个 flagId：
//   AIV→AIC：同 cube 的两个 AIV 都 Set 同一 flag，AIC 等一次；
//   AIC→AIV：AIC Set 一次即广播给两个 AIV。
inline constexpr uint16_t GATHER_READY = 1;
inline constexpr uint16_t PROBABILITY_READY = 2;
inline constexpr uint16_t RESULT_FREE = 3;
inline constexpr uint16_t SCORE_READY = 4;
inline constexpr uint16_t RESULT_READY = 5;
inline constexpr uint16_t KV_FREE = 6;

// 一个 task 覆盖同一 query/KV head 的一个 token 的 [G, Ktile] 切片。
struct TileTask {
    uint64_t id;
    uint64_t token;
    uint64_t head;
    uint64_t sparseBase;
    uint64_t first;
    uint64_t tailStart;
    uint64_t kbase;
    uint64_t kvHead;
    uint32_t selected;
    bool tail;
    uint32_t querySlot;
};

// Tiling 派生出的、kernel 内只读的尺寸参数。
struct QsagParams {
    uint64_t coreIndex = 0;
    uint64_t aivIndex = 0;
    uint64_t globalAivIndex = 0;
    uint64_t totalQueryTokens = 0;
    uint64_t totalKvTokens = 0;
    uint32_t cubeCoreNum = 0;
    uint32_t headDim = 0;
    uint32_t headDimAligned = 0;
    uint32_t kvTileSize = 0;
    uint32_t groupRows = 0;
    uint32_t rowsPerAiv = 0;
    uint32_t rowCapacityPerAiv = 0;
    uint32_t groupCapacity = 0;
    uint32_t vectorRows = 0;
    uint32_t dTile = 0;
    uint32_t dCache = 0;
    uint32_t queryBufferSlots = 0;
    uint32_t transferElements = 0;
    uint32_t resultSlots = 0;
    uint32_t blockSize = 0;
    uint32_t blockShift = 0;
    bool powerOfTwoBlock = false;
};

// GM 上的全部输入/输出/workspace 视图，cube 与 vector 引擎共享。
template <typename T, typename Tiling>
struct QsagContext {
    const Tiling* tiling = nullptr;
    QsagParams params;

    GlobalTensor<T> query;
    GlobalTensor<T> key;
    GlobalTensor<T> value;
    GlobalTensor<T> dOut;
    GlobalTensor<T> attentionOut;
    GlobalTensor<T> dQuery;
    GlobalTensor<T> dKey;
    GlobalTensor<T> dValue;
    GlobalTensor<T> queryPadded;
    GlobalTensor<T> dOutPadded;

    GlobalTensor<float> lse;
    GlobalTensor<float> di;
    GlobalTensor<float> dQueryWorkspace;
    GlobalTensor<float> dKeyWorkspace;
    GlobalTensor<float> dValueWorkspace;

    GlobalTensor<int32_t> sparseBlockIdx;
    GlobalTensor<int32_t> sparseBlockCount;
    GlobalTensor<int32_t> filteredIndices;
    GlobalTensor<int32_t> filteredCounts;
    GlobalTensor<int32_t> usedQ;
    GlobalTensor<int32_t> usedK;
    GlobalTensor<int64_t> cuSeqQ;
    GlobalTensor<int64_t> cuSeqKv;

    GlobalTensor<T> selectedKey[SLOT_COUNT];
    GlobalTensor<T> selectedValue[SLOT_COUNT];
    GlobalTensor<float> mm4[SLOT_COUNT];
    GlobalTensor<float> mm5[SLOT_COUNT];
    GlobalTensor<float> scoreGm[SLOT_COUNT];
    GlobalTensor<float> gradGm[SLOT_COUNT];
    GlobalTensor<T> probabilityGm[SLOT_COUNT];
    GlobalTensor<T> dsProbabilityGm[SLOT_COUNT];
};

__aicore__ inline uint64_t Min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t AlignBytes(uint32_t bytes)
{
    return (bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

__aicore__ inline uint32_t BlockIndex(const QsagParams& params, uint32_t position)
{
    return params.powerOfTwoBlock ? position >> params.blockShift : position / params.blockSize;
}

__aicore__ inline uint32_t BlockOffset(const QsagParams& params, uint32_t position)
{
    return params.powerOfTwoBlock ? position & (params.blockSize - 1) : position % params.blockSize;
}

// Flattened [token, head] offset in the original Q/KV layout.
template <typename Tiling>
__aicore__ inline uint64_t Offset(const Tiling* tiling, uint64_t token, uint64_t head, bool query, uint64_t dim)
{
    uint64_t heads = query ? tiling->qHeadNum : tiling->kvHeadNum;
    uint64_t seq = query ? tiling->qSeqLen : tiling->kvSeqLen;
    if (tiling->inputLayout == 1) {
        return ((token / seq * heads + head) * seq + token % seq) * dim;
    }
    return (token * heads + head) * dim;
}

// LSE is independent of the Q/Di layout: dense BNS, packed TN.
template <typename Tiling>
__aicore__ inline uint64_t LseOffset(const Tiling* tiling, uint64_t token, uint64_t head)
{
    if (tiling->inputLayout == 0) {
        return token * tiling->qHeadNum + head;
    }
    const uint64_t seq = tiling->qSeqLen;
    return (token / seq * tiling->qHeadNum + head) * seq + token % seq;
}

template <typename Tiling>
__aicore__ inline uint64_t LseHeadStride(const Tiling* tiling)
{
    return tiling->inputLayout == 0 ? 1 : tiling->qSeqLen;
}

} // namespace QSAG
#endif // ATTENTION_QSAG_COMMON_ARCH22_H
