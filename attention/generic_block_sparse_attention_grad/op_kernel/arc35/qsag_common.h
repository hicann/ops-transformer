/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "kernel_operator.h"

namespace QSAG {
using namespace AscendC;

constexpr uint32_t AIV_PER_CUBE = 2;
constexpr uint32_t NZ_WIDTH = 16; // FP16/BF16 NZ 的 C0 维度。
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t FLOATS_PER_BLOCK = BLOCK_BYTES / sizeof(float);
constexpr uint32_t MAX_HEAD_DIM = 256;
constexpr uint32_t MAX_GROUP_ROWS = 256;
constexpr uint32_t SLOT_COUNT = 2;
constexpr uint32_t PIPELINE_STAGES = 3;
constexpr uint32_t SMALL_GROUP_LIMIT = 64;
constexpr uint32_t LARGE_GROUP_LIMIT = 128;
constexpr uint32_t WIDE_D_TILE = 128;
constexpr uint32_t NARROW_D_TILE = 64;
constexpr uint32_t NZ_MIN_ROW_BYTES = 256;
constexpr uint32_t DS_EXTRA_BYTES = 2048;
constexpr uint64_t MAX_ND2NZ_SRC_STRIDE = 65535;       // ND2NZ 文档规定的源行步长上限，单位为元素。
constexpr uint64_t MAX_FIXPIPE_STRIDE = 0xFFFFFFFFULL; // API 的目的步长为 uint32_t，单位为元素。
// mode 4：第二个 AIV 的 flag 编号偏移 16；mode 2：向两个 AIV 广播。
constexpr uint16_t AIV_FLAG_OFFSET = 16;
constexpr uint16_t GATHER_READY = 0;
constexpr uint16_t PROBABILITY_READY = GATHER_READY + SLOT_COUNT;
constexpr uint16_t RESULT_FREE = PROBABILITY_READY + SLOT_COUNT;
constexpr uint16_t KV_FREE = RESULT_FREE + SLOT_COUNT;
constexpr uint16_t SCORE_READY = KV_FREE + SLOT_COUNT;
constexpr uint16_t RESULT_READY = SCORE_READY + SLOT_COUNT;
constexpr uint16_t FILTER_READY = RESULT_READY + SLOT_COUNT;

struct UbShape {
    uint32_t halfCapacity, vectorRows, columns, headDim, blockSize, transferElements;
};

struct UbPlan {
    uint32_t matrixBytes, resultBytes, lseBytes;
    uint32_t pNzBytes, dsNzBytes, topkBytes, totalBytes;
};

struct CubeCapacityPlan {
    uint32_t l1Bytes, l0ABytes, l0BBytes, l0CBytes;
    bool fits;
};

template <typename T>
__aicore__ inline CubeCapacityPlan MakeCubeCapacityPlan(uint32_t groupCapacity, uint32_t columns, uint32_t headDim,
                                                        uint32_t dTile, uint32_t querySlots)
{
    const uint32_t width = headDim < dTile ? headDim : dTile;
    const uint32_t queryElements = groupCapacity * headDim;
    const uint32_t scoreElements = groupCapacity * columns;
    const uint32_t querySlice = groupCapacity * width;
    const uint32_t kvSlice = columns * width;
    CubeCapacityPlan plan;
    plan.l1Bytes =
        (2 * querySlots * queryElements + SLOT_COUNT * (columns * headDim + kvSlice + 2 * scoreElements)) * sizeof(T);
    plan.l0ABytes = (querySlice > scoreElements ? querySlice : scoreElements) * sizeof(T);
    plan.l0BBytes = (querySlice > kvSlice ? querySlice : kvSlice) * sizeof(T);
    const uint32_t queryResult = querySlice > scoreElements ? querySlice : scoreElements;
    plan.l0CBytes = (queryResult > kvSlice ? queryResult : kvSlice) * sizeof(float);
    plan.fits = plan.l1Bytes <= 512 * 1024 && plan.l0ABytes <= 64 * 1024 && plan.l0BBytes <= 64 * 1024 &&
                plan.l0CBytes <= 128 * 1024;
    return plan;
}

__aicore__ inline uint32_t AlignBytes(uint32_t bytes)
{
    return (bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline uint32_t GatherRowCapacity(uint32_t headDim, uint32_t transferElements, uint32_t columns)
{
    constexpr uint32_t registerElements = 256 / sizeof(T);
    const uint32_t roundedDim = (headDim + registerElements - 1) / registerElements * registerElements;
    const uint32_t capacity = (transferElements * sizeof(float) / sizeof(T) - (roundedDim - headDim)) / headDim;
    uint32_t rows = columns > 128 ? 128 : 64;
    while (rows > capacity) {
        rows /= 2;
    }
    return rows;
}

template <typename T>
__aicore__ inline UbPlan MakeUbPlan(const UbShape& shape, bool direct, uint32_t resultSlots)
{
    UbPlan plan;
    plan.matrixBytes = (direct ? shape.halfCapacity : shape.vectorRows) * shape.columns * sizeof(float);
    plan.resultBytes = direct ? shape.columns / AIV_PER_CUBE * shape.headDim * sizeof(float) : 0;
    plan.lseBytes = shape.vectorRows * BLOCK_BYTES;
    const uint32_t nzRowBytes =
        shape.columns * sizeof(T) < NZ_MIN_ROW_BYTES ? NZ_MIN_ROW_BYTES : shape.columns * sizeof(T);
    const uint32_t probabilityBytes = ((direct ? shape.halfCapacity : shape.vectorRows) + 1) * nzRowBytes;
    const uint32_t gatherBytes = direct ?
                                     (GatherRowCapacity<T>(shape.headDim, shape.transferElements, shape.columns) + 1) *
                                         shape.headDim * sizeof(T) :
                                     0;
    plan.pNzBytes = probabilityBytes > gatherBytes ? probabilityBytes : gatherBytes;
    plan.dsNzBytes = plan.pNzBytes + DS_EXTRA_BYTES;
    plan.topkBytes = AlignBytes(shape.columns / shape.blockSize * sizeof(int32_t));
    plan.totalBytes = 2 * SLOT_COUNT * plan.matrixBytes + SLOT_COUNT * plan.lseBytes + plan.pNzBytes + plan.dsNzBytes +
                      (direct ? PIPELINE_STAGES : 1) * plan.topkBytes + 2 * shape.transferElements * sizeof(float) +
                      2 * resultSlots * plan.resultBytes;
    return plan;
}

struct TileTask {
    uint64_t id, token, head, sparseBase, first, tailStart, kbase, kvHead;
    uint32_t selected;
    bool tail;
    uint32_t querySlot, indexSlot, indexCount;
};

__aicore__ inline uint64_t Min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

template <HardEvent E>
__aicore__ inline void Fence()
{
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}

template <typename Tiling>
struct ConstInfo {
    const Tiling* tiling;
    uint64_t core, sub, vcore, tq, tk;
    uint32_t d, dp, n, groupRows, halfRows;
    uint32_t halfCapacity;
    uint32_t groupCapacity, vectorRows, dTile, dCache, querySlots, transferElements;
    uint32_t nzRows, resultSlots, blockSize;
    bool directResult, pairedKV, cacheQueryIndices;

    __aicore__ inline uint64_t Offset(uint64_t token, uint64_t head, bool query, uint64_t dim) const
    {
        if (!query && pairedKV) {
            return (head * tk + token) * dim;
        }
        uint64_t heads = query ? tiling->qHeadNum : tiling->kvHeadNum;
        uint64_t seq = query ? tiling->qSeqLen : tiling->kvSeqLen;
        if (tiling->inputLayout == 1) {
            return ((token / seq * heads + head) * seq + token % seq) * dim;
        }
        return (token * heads + head) * dim;
    }

    // LSE is independent of the Q/Di layout: dense BNS, packed TN.
    __aicore__ inline uint64_t LseOffset(uint64_t token, uint64_t head) const
    {
        if (tiling->inputLayout == 0) {
            return token * tiling->qHeadNum + head;
        }
        const uint64_t seq = tiling->qSeqLen;
        return (token / seq * tiling->qHeadNum + head) * seq + token % seq;
    }

    __aicore__ inline uint64_t LseHeadStride() const
    {
        return tiling->inputLayout == 0 ? 1 : tiling->qSeqLen;
    }

    __aicore__ inline uint32_t ResultSlot(uint32_t slot) const
    {
        return slot & (resultSlots - 1);
    }

    __aicore__ inline uint64_t KvGradRow(uint64_t token, uint64_t head) const
    {
        if (pairedKV) {
            return head * tk + token;
        }
        if (tiling->inputLayout == 1) {
            const uint64_t seq = tiling->kvSeqLen;
            return (token / seq * tiling->kvHeadNum + head) * seq + token % seq;
        }
        return token * tiling->kvHeadNum + head;
    }

    __aicore__ inline uint64_t KvGradRowStride() const
    {
        return (pairedKV || tiling->inputLayout == 1) ? 1 : tiling->kvHeadNum;
    }

    template <typename T>
    __aicore__ inline UbPlan PlanUb(bool direct, uint32_t resultSlots, bool queryIndices = false) const
    {
        auto plan = MakeUbPlan<T>({halfCapacity, vectorRows, n, dp, blockSize, transferElements}, direct, resultSlots);
        if (direct && queryIndices) {
            const uint64_t bytes =
                (static_cast<uint64_t>(tiling->topK) * sizeof(int32_t) + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
            const uint64_t total = plan.totalBytes + PIPELINE_STAGES * bytes - PIPELINE_STAGES * plan.topkBytes;
            if (bytes > plan.topkBytes && total <= tiling->ubSize && total <= 0xFFFFFFFFULL) {
                plan.topkBytes = static_cast<uint32_t>(bytes);
                plan.totalBytes = static_cast<uint32_t>(total);
            }
        }
        return plan;
    }
};

template <typename T>
struct SharedBuffers {
    LocalTensor<float> score[SLOT_COUNT], grad[SLOT_COUNT];
    LocalTensor<float> resultK[SLOT_COUNT], resultV[SLOT_COUNT];
    LocalTensor<T> kl[SLOT_COUNT], vl[SLOT_COUNT], pl[SLOT_COUNT], dsl[SLOT_COUNT];
    GlobalTensor<T> selK[SLOT_COUNT], selV[SLOT_COUNT];
    GlobalTensor<float> mm4[SLOT_COUNT], mm5[SLOT_COUNT];
};
} // namespace QSAG
