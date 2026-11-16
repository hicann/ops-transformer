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
 * \file engram_hash_kernel.h
 * \brief EngramHash kernel 实现 (arch35 / ascend950)
 */

#ifndef ENGRAM_HASH_KERNEL_H
#define ENGRAM_HASH_KERNEL_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "c_api/misc/misc.h"
#include "c_api/sync/sync.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace EngramHashOps {

// 事件 ID: 0 为 ids/mult/参数切片缓冲的 MTE2<->V 往返。
constexpr ::event_t BUF_EVT = static_cast<::event_t>(0);

constexpr uint32_t UB_ALIGN = 32;        // UB 缓冲起始偏移对齐
constexpr int64_t MAX_VF_THREADS = 1024; // 单次 VF 调用的最大线程数 (__launch_bounds__)

__aicore__ inline uint32_t AlignUp32(const uint32_t bytes)
{
    return (bytes + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
}

template <int VF_BOUNDS, int MAXN, typename TIds, typename TMult, typename TScalar>
__simt_vf__ __aicore__ __launch_bounds__(VF_BOUNDS) inline void EngramHashTileVF(
    __gm__ TScalar* outBase, __ubuf__ const TIds* ids, __ubuf__ const TMult* mult, __ubuf__ const TScalar* vocab,
    __ubuf__ const TScalar* offs, int32_t validTokens, int32_t maxNgramSize, int32_t numTables, int32_t numOutCols,
    int32_t chunkStart, int32_t chunkCols, int32_t simtThreads)
{
    const int32_t col = static_cast<int32_t>(threadIdx.x) % chunkCols;
    const int32_t tLoc = static_cast<int32_t>(threadIdx.x) / chunkCols;
    const int32_t tokensPerIter = simtThreads / chunkCols;
    if (tLoc >= tokensPerIter) {
        return; // chunkCols 不整除 simtThreads 时的冗余线程 (避免重复计算)
    }

    const int32_t outCol = chunkStart + col;
    const int32_t lastNgramIdx = outCol / numTables + 1; // n-gram 长度 = last + 1
    const int64_t vocabSize = static_cast<int64_t>(vocab[col]);
    const int64_t offsVal = static_cast<int64_t>(offs[col]);
    const int32_t chainBound = (MAXN > 0) ? MAXN : (lastNgramIdx + 1);
    for (int32_t token = tLoc; token < validTokens; token += tokensPerIter) {
        int64_t hashValue = static_cast<int64_t>(ids[token * maxNgramSize]) * static_cast<int64_t>(mult[0]);
        for (int32_t k = 1; k < chainBound; ++k) {
            if (MAXN <= 0 || k <= lastNgramIdx) {
                hashValue ^= static_cast<int64_t>(ids[token * maxNgramSize + k]) * static_cast<int64_t>(mult[k]);
            }
        }
        int64_t remainder = hashValue % vocabSize;
        if (remainder < 0) {
            remainder += vocabSize;
        }
        outBase[token * numOutCols + col] = static_cast<TScalar>(remainder) + offsVal;
    }
}

#define ENGRAM_HASH_VF_CALL(MAXN, BOUNDS) \
    do { \
        asc_vf_call<EngramHashTileVF<BOUNDS, MAXN, TIds, TMult, TScalar>>( \
            cce::dim3(chunkThreads), outChunkBase, (__ubuf__ TIds*)(ub + idsOff), (__ubuf__ TMult*)(ub + multOff), \
            (__ubuf__ TScalar*)(ub + vocabOff), (__ubuf__ TScalar*)(ub + offsOff), static_cast<int32_t>(validTokens), \
            static_cast<int32_t>(maxNgramSize), static_cast<int32_t>(numTables), static_cast<int32_t>(numOutCols), \
            static_cast<int32_t>(chunkStart), callChunkCols, static_cast<int32_t>(chunkThreads)); \
    } while (0)

#define ENGRAM_HASH_VF_LADDER(MAXN) \
    do { \
        if (chunkThreads <= 64) { \
            ENGRAM_HASH_VF_CALL(MAXN, 64); \
        } else if (chunkThreads <= 256) { \
            ENGRAM_HASH_VF_CALL(MAXN, 256); \
        } else if (chunkThreads <= 512) { \
            ENGRAM_HASH_VF_CALL(MAXN, 512); \
        } else { \
            ENGRAM_HASH_VF_CALL(MAXN, 1024); \
        } \
    } while (0)

template <typename TIds, typename TMult, typename TScalar>
__aicore__ inline void EngramHashProcess(GM_ADDR ngramTokenIds, GM_ADDR multipliers, GM_ADDR vocabSizes,
                                         GM_ADDR offsets, GM_ADDR hashIndices, const EngramHashTilingData* t)
{
    __gm__ TIds* idsGm = (__gm__ TIds*)ngramTokenIds;
    __gm__ TMult* multGm = (__gm__ TMult*)multipliers;
    __gm__ TScalar* vocabGm = (__gm__ TScalar*)vocabSizes;
    __gm__ TScalar* offsGm = (__gm__ TScalar*)offsets;
    __gm__ TScalar* yGm = (__gm__ TScalar*)hashIndices;

    const int64_t numTokens = t->numTokens;
    const int64_t maxNgramSize = t->maxNgramSize;
    const int64_t numTables = t->numTables;
    const int64_t numOutCols = t->numOutCols;
    const int64_t tileTokens = t->tileTokens;
    const int64_t colChunk = t->colChunk;
    const int64_t numTilesPerLayer = t->numTilesPerLayer;
    const int64_t totalBlocks = t->totalBlocks;
    const int64_t usedCoreNums = t->usedCoreNums;
    // 尾列块的列数 (C 不被 colChunk 整除时)。
    const int64_t tailCols = numOutCols - (numOutCols - 1) / colChunk * colChunk;
    const uint32_t maxChunkThreads =
        static_cast<uint32_t>(tileTokens * colChunk < MAX_VF_THREADS ? tileTokens * colChunk : MAX_VF_THREADS);

    const uint32_t idsBytes =
        AlignUp32(static_cast<uint32_t>(tileTokens) * static_cast<uint32_t>(maxNgramSize) * sizeof(TIds));
    const uint32_t multBytes = AlignUp32(static_cast<uint32_t>(maxNgramSize) * sizeof(TMult));
    const uint32_t sliceBytes = AlignUp32(static_cast<uint32_t>(colChunk) * sizeof(TScalar));
    const uint32_t idsOff = 0;
    const uint32_t multOff = idsOff + idsBytes;
    const uint32_t vocabOff = multOff + multBytes;
    const uint32_t offsOff = vocabOff + sliceBytes;

    __ubuf__ uint8_t* ub = (__ubuf__ uint8_t*)0;

    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<::event_t>(BUF_EVT));

    // totalBlocks > 核数时为持久核: 每核按 blockIdx 为起点跨块循环; 否则一核一块。
    for (int64_t block = GetBlockIdx(); block < totalBlocks; block += usedCoreNums) {
        const int64_t layer = block / numTilesPerLayer;
        const int64_t tileIdx = block % numTilesPerLayer;
        const int64_t tokenStart = tileIdx * tileTokens;
        const int64_t validTokens = (tileTokens < numTokens - tokenStart) ? tileTokens : (numTokens - tokenStart);
        const int64_t layerVocabBase = layer * (maxNgramSize - 1) * numTables;
        const int64_t layerOffsBase = layer * numOutCols;
        const int64_t layerMultBase = layer * maxNgramSize;
        const int64_t outTileBase = (layer * numTokens + tokenStart) * numOutCols;

        const uint32_t inBytes =
            static_cast<uint32_t>(validTokens) * static_cast<uint32_t>(maxNgramSize) * sizeof(TIds);
        const uint32_t mb = static_cast<uint32_t>(maxNgramSize) * sizeof(TMult);

        for (int64_t chunkStart = 0; chunkStart < numOutCols; chunkStart += colChunk) {
            const bool isTail = (chunkStart + colChunk > numOutCols);
            const int64_t chunkCols = isTail ? tailCols : colChunk;
            const uint32_t sb = static_cast<uint32_t>(chunkCols) * sizeof(TScalar);
            const uint32_t chunkThreads = static_cast<uint32_t>(
                validTokens * chunkCols < static_cast<int64_t>(maxChunkThreads) ? validTokens * chunkCols :
                                                                                  maxChunkThreads);

            asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<::event_t>(BUF_EVT));
            AscendC::CopyGmToUbufAlignV2<TIds>((__ubuf__ TIds*)(ub + idsOff),
                                               (__gm__ TIds*)(idsGm + tokenStart * maxNgramSize), 1, inBytes, 0, 0, 0,
                                               0, true, 0, false);
            AscendC::CopyGmToUbufAlignV2<uint32_t>((__ubuf__ uint32_t*)(ub + multOff),
                                                   (__gm__ uint32_t*)(multGm + layerMultBase), 1, mb, 0, 0, 0, 0, true,
                                                   0, false);
            AscendC::CopyGmToUbufAlignV2<TScalar>((__ubuf__ TScalar*)(ub + vocabOff),
                                                  (__gm__ TScalar*)(vocabGm + layerVocabBase + chunkStart), 1, sb, 0, 0,
                                                  0, 0, true, 0, false);
            AscendC::CopyGmToUbufAlignV2<TScalar>((__ubuf__ TScalar*)(ub + offsOff),
                                                  (__gm__ TScalar*)(offsGm + layerOffsBase + chunkStart), 1, sb, 0, 0,
                                                  0, 0, true, 0, false);
            asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<::event_t>(BUF_EVT));
            asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<::event_t>(BUF_EVT));

            __gm__ TScalar* outChunkBase = yGm + outTileBase + chunkStart;
            const int32_t callChunkCols = static_cast<int32_t>(isTail ? tailCols : colChunk);
            const int64_t ngramN = maxNgramSize;
            if (ngramN <= 8) {
                ENGRAM_HASH_VF_LADDER(8);
            } else if (ngramN <= 16) {
                ENGRAM_HASH_VF_LADDER(16);
            } else if (ngramN <= 32) {
                ENGRAM_HASH_VF_LADDER(32);
            } else if (ngramN <= 64) {
                ENGRAM_HASH_VF_LADDER(64);
            } else if (ngramN <= 128) {
                ENGRAM_HASH_VF_LADDER(128);
            } else {
                ENGRAM_HASH_VF_LADDER(0);
            }
            asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<::event_t>(BUF_EVT));
        }
    }
    asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<::event_t>(BUF_EVT));
}

} // namespace EngramHashOps

#endif // ENGRAM_HASH_KERNEL_H
