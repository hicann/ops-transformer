/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_INT8_CSA_INPUT_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_INT8_CSA_INPUT_H

#include "mixed_quant_sparse_flash_mla_tq_common.h"
#include "mixed_quant_sparse_flash_mla_tiling_data_arch22.h"

namespace SASKernel {
// Adapts packed INT8 records to the existing arch22 floating-point CSA pipeline.
// Selection is shared by AIC/AIV: only valid tokens occupy the staged Cube tile.
template <typename SAST>
class Int8CsaInput {
public:
    static constexpr uint32_t RECORD_BYTES = 608;
    static constexpr uint32_t HEAD_DIM = 512;
    static constexpr uint32_t NOPE_DIM = 448;
    static constexpr uint32_t GROUP_SIZE = 64;
    optiling::MixedQuantSparseFlashMlaInt8Params params{};
    GlobalTensor<int8_t> oriGm, cmpGm;

    __aicore__ inline void Init(__gm__ uint8_t* ori, __gm__ uint8_t* cmp, __gm__ uint8_t* indices,
                                __gm__ uint8_t* oriTable, __gm__ uint8_t* cmpTable, __gm__ uint8_t* cuQ,
                                __gm__ uint8_t* cuOri, __gm__ uint8_t* cuCmp, __gm__ uint8_t* usedQ,
                                __gm__ uint8_t* usedOri, __gm__ uint8_t* usedCmp, __gm__ uint8_t* residual,
                                __gm__ uint8_t* topkLength, const optiling::MixedQuantSparseFlashMlaInt8Params& p)
    {
        params = p;
        batchCached = false;
        oriPageCached = false;
        cmpPageCached = false;
        oriPageChecked = false;
        cmpPageChecked = false;
        oriPageBounded = false;
        cmpPageBounded = false;
        oriGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(ori));
        cmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(cmp));
        SetOptional(indicesGm, indices);
        SetOptional(oriTableGm, oriTable);
        SetOptional(cmpTableGm, cmpTable);
        SetOptional(cuQGm, cuQ);
        SetOptional(cuOriGm, cuOri);
        SetOptional(cuCmpGm, cuCmp);
        SetOptional(usedQGm, usedQ);
        SetOptional(usedOriGm, usedOri);
        SetOptional(usedCmpGm, usedCmp);
        SetOptional(residualGm, residual);
        SetOptional(topkLengthGm, topkLength);
        hasUsedQ = usedQ != nullptr;
        hasUsedOri = usedOri != nullptr;
        hasUsedCmp = usedCmp != nullptr;
        hasResidual = residual != nullptr;
        hasTopkLength = topkLength != nullptr;
    }

    __aicore__ inline int64_t QPrefix(uint32_t b)
    {
        if constexpr (SAST::layout == SAS_LAYOUT::TND) {
            return cuQGm.GetValue(b);
        }
        return static_cast<int64_t>(b) * params.qSeq;
    }
    __aicore__ inline int64_t QStorage(uint32_t b)
    {
        if constexpr (SAST::layout == SAS_LAYOUT::TND) {
            return cuQGm.GetValue(b + 1) - cuQGm.GetValue(b);
        }
        return params.qSeq;
    }
    __aicore__ inline int64_t QLength(uint32_t b)
    {
        return hasUsedQ ? usedQGm.GetValue(b) : QStorage(b);
    }
    __aicore__ inline int64_t KvLength(uint32_t b, bool cmp)
    {
        if (cmp && hasUsedCmp) {
            return usedCmpGm.GetValue(b);
        }
        if (!cmp && hasUsedOri) {
            return usedOriGm.GetValue(b);
        }
        if constexpr (SAST::kvLayout == SAS_LAYOUT::TND) {
            return cmp ? cuCmpGm.GetValue(b + 1) - cuCmpGm.GetValue(b) : cuOriGm.GetValue(b + 1) - cuOriGm.GetValue(b);
        }
        return cmp ? params.cmpSeq : params.oriSeq;
    }

    // One query's heads share a selected KV sequence. Keep duplicates and stop
    // at the first -1, as in the quant_mode=1 reference contract.
    __aicore__ inline void BeginQuery(uint32_t b, uint32_t s)
    {
        if (!batchCached || batchCacheB != b) {
            batchPrefix = QPrefix(b);
            batchQLength = QLength(b);
            batchOriLength = KvLength(b, false);
            batchCmpLength = KvLength(b, true);
            batchResidual = params.cmpMask == CAUSAL_MASK_MODE && hasResidual ? residualGm.GetValue(b) : 0;
            batchCacheB = b;
            batchCached = true;
        }
        preparedSelection = false;
        query = {};
        query.bIdx = b;
        query.s1Idx = s;
        query.gS1Idx = s * params.heads;
        query.actS1Size = batchQLength;
        query.actOriS2Size = batchOriLength;
        query.actMBaseSize = params.heads;
        query.tensorAOffset = (batchPrefix + s) * params.heads * HEAD_DIM;
        query.attenOutOffset = query.tensorAOffset;
        query.topKBaseOffset = (batchPrefix + s) * params.topk;
        query.isOriOnly = true;
        int64_t oriLength = query.actOriS2Size;
        int64_t center = oriLength - static_cast<int64_t>(query.actS1Size) + s;
        cursor = 0;
        oriEnd = oriLength;
        if (params.oriMask == CAUSAL_MASK_MODE) {
            oriEnd = Max(0, Min(oriLength, center + 1));
        }
        if (params.oriMask == SLIDING_WINDOW_MASK_MODE) {
            cursor = params.winLeft < 0 ? 0 : Min(oriLength, Max(0, center - params.winLeft));
            oriEnd = params.winRight < 0 ? oriLength : Max(0, Min(oriLength, center + params.winRight + 1));
        }
        int64_t cmpLength = batchCmpLength;
        query.cmpS2IdLimit = cmpLength;
        if (params.cmpMask == CAUSAL_MASK_MODE) {
            int64_t residual = batchResidual;
            query.cmpS2IdLimit = Max(0, Min(cmpLength, (cmpLength * params.cmpRatio + residual -
                                                        static_cast<int64_t>(query.actS1Size) + s + 1) /
                                                           params.cmpRatio));
        }
        cmpEnd = Min(params.topk, query.cmpS2IdLimit);
        if (hasTopkLength) {
            cmpEnd = Max(0, Min(cmpEnd, topkLengthGm.GetValue(batchPrefix + s)));
        }
        block = 0;
    }

    // Private ori-cache planner reads this immediately after BeginQuery,
    // before UsePreparedSelection changes cursor/oriEnd to compact row counts.
    __aicore__ inline void GetOriLogicalRange(int64_t& begin, int64_t& end) const
    {
        begin = cursor;
        end = oriEnd;
    }

    __aicore__ inline bool CanPrepareQuery()
    {
        if constexpr (!SAST::pageAttention) {
            return false;
        }
        uint64_t oriCount = static_cast<uint64_t>(Max(0, oriEnd - cursor));
        if (oriCount > MAX_PREPARED_ORI_ROWS || cmpEnd > MAX_PREPARED_CMP_ROWS || oriCount + cmpEnd == 0U) {
            return false;
        }
        uint32_t ob = params.oriBlockSize, cb = params.cmpBlockSize;
        if (ob == 0 || cb == 0 || (ob & (ob - 1U)) != 0 || (cb & (cb - 1U)) != 0 ||
            params.oriStride != static_cast<uint64_t>(ob) * RECORD_BYTES ||
            params.cmpStride != static_cast<uint64_t>(cb) * RECORD_BYTES || params.oriBlocks == 0 ||
            params.cmpBlocks == 0) {
            return false;
        }
        // FP32 is used only for exact integer bounds within its 24-bit range.
        constexpr uint64_t EXACT_LIMIT = 1ULL << 24;
        if (query.actOriS2Size > EXACT_LIMIT || query.cmpS2IdLimit > EXACT_LIMIT ||
            params.oriBlocks > EXACT_LIMIT / ob || params.cmpBlocks > EXACT_LIMIT / cb) {
            return false;
        }
        int64_t oriFirst = cursor / ob;
        int64_t oriLast = (oriEnd + ob - 1U) / ob;
        int64_t cmpPages = (query.cmpS2IdLimit + cb - 1U) / cb;
        return Max(0, oriLast - oriFirst) <= ORI_PAGE_CACHE_CAPACITY &&
               Min(cmpPages, params.cmpTableSize) <= CMP_PAGE_SCRATCH_CAPACITY;
    }

    __aicore__ inline uint32_t ResolveRecords(LocalTensor<int32_t> table, LocalTensor<int32_t> sparse,
                                              LocalTensor<int32_t> work, LocalTensor<uint32_t> addresses,
                                              uint32_t count, uint32_t pages, uint32_t pageFirst, uint32_t blockSize,
                                              uint32_t blocks, uint32_t tokenStart, int64_t limit, bool cmp,
                                              bool cacheTable)
    {
        if (count == 0U || pages == 0U || limit <= 0) {
            return 0U;
        }
        auto tokens = work.template ReinterpretCast<float>();
        auto safe = work[1024];
        auto gatherIndex = work[2048].template ReinterpretCast<uint32_t>();
        auto physical = work[3072];
        auto positions = work[4096].template ReinterpretCast<uint32_t>();
        auto selected = work[5120].template ReinterpretCast<float>();
        auto reduceTmp = work[6144].template ReinterpretCast<float>();
        auto minimum = work[6400].template ReinterpretCast<float>();
        auto maximum = work[6408].template ReinterpretCast<float>();
        if (cmp) {
            Cast(tokens, sparse, RoundMode::CAST_RINT, count);
        } else {
            CreateVecIndex(tokens, static_cast<float>(static_cast<int32_t>(tokenStart)), count);
        }
        PipeBarrier<PIPE_V>();
        ReduceMin(minimum, tokens, reduceTmp, static_cast<int32_t>(count), false);
        PipeBarrier<PIPE_V>();
        ReduceMax(maximum, tokens, reduceTmp, static_cast<int32_t>(count), false);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        float minToken = minimum.GetValue(0), maxToken = maximum.GetValue(0);
        int32_t lower = static_cast<int32_t>(pageFirst * blockSize);
        int32_t upper = static_cast<int32_t>(Min(limit, static_cast<int64_t>(pageFirst + pages) * blockSize));
        if (minToken < static_cast<float>(lower) || maxToken >= static_cast<float>(upper)) {
            return 0xFFFFFFFFU;
        }
        Cast(safe, tokens, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        uint32_t shift = 0;
        for (uint32_t n = blockSize; n > 1U; n >>= 1U) {
            ++shift;
        }
        ShiftRight(gatherIndex, safe.template ReinterpretCast<uint32_t>(), shift, count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(positions, gatherIndex, shift, count);
        PipeBarrier<PIPE_V>();
        Sub(safe, safe, positions.template ReinterpretCast<int32_t>(), count);
        if (pageFirst != 0U) {
            Adds(gatherIndex.template ReinterpretCast<int32_t>(), gatherIndex.template ReinterpretCast<int32_t>(),
                 -static_cast<int32_t>(pageFirst), count);
            PipeBarrier<PIPE_V>();
        }
        ShiftLeft(gatherIndex, gatherIndex, 2U, count);
        PipeBarrier<PIPE_V>();
        Gather(physical, table, gatherIndex, 0U, count);
        PipeBarrier<PIPE_V>();
        const bool tableBounded = cacheTable && (cmp ? cmpPageBounded : oriPageBounded);
        if (!tableBounded) {
            // Prove unchanged cached page IDs once. If an unused page is invalid,
            // this first query takes the existing ordered scalar fallback; later
            // queries still validate their selected IDs exactly as before.
            const bool checkTable = cacheTable && !(cmp ? cmpPageChecked : oriPageChecked);
            const uint32_t checkCount = checkTable ? pages : count;
            Cast(selected, checkTable ? table : physical, RoundMode::CAST_RINT, checkCount);
            PipeBarrier<PIPE_V>();
            ReduceMin(minimum, selected, reduceTmp, static_cast<int32_t>(checkCount), false);
            PipeBarrier<PIPE_V>();
            ReduceMax(maximum, selected, reduceTmp, static_cast<int32_t>(checkCount), false);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            const bool bounded =
                minimum.GetValue(0) >= 0.0f && maximum.GetValue(0) < static_cast<float>(static_cast<int32_t>(blocks));
            if (checkTable) {
                if (cmp) {
                    cmpPageChecked = true;
                    cmpPageBounded = bounded;
                } else {
                    oriPageChecked = true;
                    oriPageBounded = bounded;
                }
            }
            if (!bounded) {
                return 0xFFFFFFFFU;
            }
        }
        Muls(physical, physical, static_cast<int32_t>(blockSize), count);
        PipeBarrier<PIPE_V>();
        Add(physical, physical, safe, count);
        PipeBarrier<PIPE_V>();
        DataCopy(addresses, physical.template ReinterpretCast<uint32_t>(), SASAlign(count, 8U));
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        return count;
    }

    __aicore__ inline void CacheQueryAddresses(LocalTensor<int32_t> scratch, LocalTensor<uint32_t> addresses,
                                               uint32_t& oriRows, uint32_t& cmpRows)
    {
        // A5 resolves page addresses with SIMD. Keep arch22 memory-vector APIs,
        // compact in order, and reserve Vec2's existing temporary range.
        constexpr uint32_t WORK_OFFSET = 1024U + 128U + 8192U;
        static_assert((WORK_OFFSET + 6472U) * sizeof(int32_t) <= 65536U);
        auto sparse = scratch;
        // Current D512 Vec2 temporary range is [1536,2048) bytes. The low
        // 4096B hold prepared row addresses; the final32B hold count publication.
        // Keep page data in disjoint persistent high ranges of the existing8KiB.
        constexpr uint32_t CMP_PAGE_CACHE_OFFSET = 1024U;
        constexpr uint32_t CMP_PAGE_CACHE_CAPACITY = 512U;
        constexpr uint32_t ORI_PAGE_CACHE_OFFSET = 1536U;
        static_assert((ORI_PAGE_CACHE_OFFSET + ORI_PAGE_CACHE_CAPACITY) * sizeof(int32_t) <= 8192U - 32U);
        auto oriTable = addresses[ORI_PAGE_CACHE_OFFSET].template ReinterpretCast<int32_t>();
        auto cmpTable = scratch[1024 + 128];
        auto work = scratch[WORK_OFFSET];
        uint32_t oriFirst = static_cast<uint32_t>(cursor / params.oriBlockSize);
        int64_t oriLast = (oriEnd + params.oriBlockSize - 1) / params.oriBlockSize;
        uint32_t oriPages = static_cast<uint32_t>(Max(0, Min(oriLast, params.oriTableSize) - oriFirst));
        uint32_t cmpPages = static_cast<uint32_t>(
            Min(params.cmpTableSize, (query.cmpS2IdLimit + params.cmpBlockSize - 1) / params.cmpBlockSize));
        CopyIndexSpan(sparse, indicesGm, query.topKBaseOffset, static_cast<uint32_t>(cmpEnd));
        if (!oriPageCached || oriPageBatch != query.bIdx || oriPageFirst != oriFirst || oriPageCount != oriPages) {
            CopyIndexSpan(oriTable, oriTableGm, static_cast<uint64_t>(query.bIdx) * params.oriTableSize + oriFirst,
                          oriPages);
            oriPageBatch = query.bIdx;
            oriPageFirst = oriFirst;
            oriPageCount = oriPages;
            oriPageCached = true;
            oriPageChecked = false;
            oriPageBounded = false;
        }
        const bool cacheCmp = cmpEnd <= MAX_PREPARED_CMP_ROWS && cmpPages <= CMP_PAGE_CACHE_CAPACITY;
        if (cacheCmp) {
            cmpTable = addresses[CMP_PAGE_CACHE_OFFSET].template ReinterpretCast<int32_t>();
            if (!cmpPageCached || cmpPageBatch != query.bIdx || cmpPageCount != cmpPages) {
                CopyIndexSpan(cmpTable, cmpTableGm, static_cast<uint64_t>(query.bIdx) * params.cmpTableSize, cmpPages);
                cmpPageBatch = query.bIdx;
                cmpPageCount = cmpPages;
                cmpPageCached = true;
                cmpPageChecked = false;
                cmpPageBounded = false;
            }
        } else {
            // More than512 prepared cmp rows overlap the cache range. Invalidate
            // before any such address write, and retain the original scratch path.
            cmpPageCached = false;
            cmpPageChecked = false;
            cmpPageBounded = false;
            CopyIndexSpan(cmpTable, cmpTableGm, static_cast<uint64_t>(query.bIdx) * params.cmpTableSize, cmpPages);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
// Keep the same ordered ori/cmp resolution and bounds fallbacks, but
// share the large vector address calculation body across both streams.
#pragma unroll 1
        for (uint32_t stream = 0U; stream < 2U; ++stream) {
            const bool isCmp = stream != 0U;
            uint32_t rows = ResolveRecords(
                isCmp ? cmpTable : oriTable, sparse, work, addresses[isCmp ? 512U : 0U],
                isCmp ? static_cast<uint32_t>(cmpEnd) : static_cast<uint32_t>(oriEnd - cursor),
                isCmp ? cmpPages : oriPages, isCmp ? 0U : oriFirst, isCmp ? params.cmpBlockSize : params.oriBlockSize,
                isCmp ? params.cmpBlocks : params.oriBlocks, isCmp ? 0U : static_cast<uint32_t>(cursor),
                isCmp ? query.cmpS2IdLimit : static_cast<int64_t>(query.actOriS2Size), isCmp, !isCmp || cacheCmp);
            if (isCmp) {
                cmpRows = rows;
            } else {
                oriRows = rows;
            }
        }
        if (oriRows == 0xFFFFFFFFU) {
            oriRows = 0;
            for (int64_t token = cursor; token < oriEnd; ++token) {
                int64_t offset = CachedPageOffset(oriTable, token, oriFirst, oriPages, false);
                if (offset >= 0) {
                    addresses.SetValue(oriRows++, static_cast<uint32_t>(offset / RECORD_BYTES));
                }
            }
        }
        if (cmpRows == 0xFFFFFFFFU) {
            cmpRows = 0;
            for (int64_t i = 0; i < cmpEnd; ++i) {
                int64_t token = sparse.GetValue(static_cast<uint32_t>(i));
                if (token == -1) {
                    break;
                }
                if (token < 0 || token >= query.cmpS2IdLimit) {
                    continue;
                }
                int64_t offset = CachedPageOffset(cmpTable, token, 0U, cmpPages, true);
                if (offset >= 0) {
                    addresses.SetValue(512U + cmpRows++, static_cast<uint32_t>(offset / RECORD_BYTES));
                }
            }
        }
        SetFlag<HardEvent::S_MTE2>(0);
        WaitFlag<HardEvent::S_MTE2>(0);
    }

    __aicore__ inline void UsePreparedSelection(uint32_t oriRows, uint32_t cmpRows)
    {
        preparedSelection = true;
        cursor = 0;
        oriEnd = oriRows;
        cmpEnd = cmpRows;
        query.isOriOnly = true;
    }

    // Pure slicing of actual compacted row counts. This does not inspect or
    // assume index validity; CacheQueryAddresses already produced oriRows/cmpRows.
    // Every untouched RunInfo member is copied from the same BeginQuery snapshot.
    __aicore__ inline void MakePreparedTile(RunInfo& info, uint32_t tileIdx, uint32_t oriTileCount,
                                            uint32_t tileSize) const
    {
        info = query;
        info.isOriOnly = tileIdx < oriTileCount;
        const uint32_t sideTile = info.isOriOnly ? tileIdx : tileIdx - oriTileCount;
        const int64_t start = static_cast<int64_t>(sideTile) * tileSize;
        const int64_t end = info.isOriOnly ? oriEnd : cmpEnd;
        const uint32_t count = static_cast<uint32_t>(end - start < tileSize ? end - start : tileSize);
        info.int8Start = start;
        info.int8End = start + count;
        info.isValid = true;
        info.s2Idx = tileIdx;
        info.isFirstSInnerLoop = tileIdx == 0U;
        info.actualSingleProcessSInnerSize = count;
        info.actualSingleProcessSInnerSizeAlign = SASAlign(count, 32U);
    }

    __aicore__ inline void FinishPreparedTiles(uint32_t tileCount)
    {
        // Match the terminal false-return of NextTile, including both-empty.
        cursor = cmpEnd;
        block = tileCount;
        query.isOriOnly = false;
    }

    __aicore__ inline bool NextTile(RunInfo& info, uint32_t tileSize)
    {
        while (true) {
            info = query;
            info.int8Start = cursor;
            uint32_t count = 0;
            if (preparedSelection) {
                int64_t end = query.isOriOnly ? oriEnd : cmpEnd;
                count = static_cast<uint32_t>(Max(0, Min(end - cursor, tileSize)));
                cursor += count;
            }

            int64_t end = query.isOriOnly ? oriEnd : cmpEnd;
            while (!preparedSelection && cursor < end && count < tileSize) {
                if (!query.isOriOnly && indicesGm.GetValue(query.topKBaseOffset + cursor) == -1) {
                    cmpEnd = cursor;
                    break;
                }
                if (CandidateOffset(info, cursor) >= 0) {
                    ++count;
                }
                ++cursor;
            }
            info.int8End = cursor;
            if (count != 0) {
                info.isValid = true;
                info.s2Idx = block++;
                info.isFirstSInnerLoop = info.s2Idx == 0;
                info.actualSingleProcessSInnerSize = count;
                info.actualSingleProcessSInnerSizeAlign = SASAlign(count, 32U);
                return true;
            }
            if (!query.isOriOnly) {
                return false;
            }
            query.isOriOnly = false;
            cursor = 0;
        }
    }

    __aicore__ inline int64_t CandidateOffset(const RunInfo& info, int64_t candidate)
    {
        bool cmp = !info.isOriOnly;
        int64_t token = cmp ? indicesGm.GetValue(info.topKBaseOffset + candidate) : candidate;
        if (token < 0 || (cmp && token >= info.cmpS2IdLimit)) {
            return -1;
        }
        if constexpr (SAST::pageAttention) {
            uint32_t blockSize = cmp ? params.cmpBlockSize : params.oriBlockSize;
            uint32_t tableSize = cmp ? params.cmpTableSize : params.oriTableSize;
            if (token / blockSize >= tableSize) {
                return -1;
            }
            int64_t tableOffset = static_cast<uint64_t>(info.bIdx) * tableSize + token / blockSize;
            int32_t physicalBlock = cmp ? cmpTableGm.GetValue(tableOffset) : oriTableGm.GetValue(tableOffset);
            if (physicalBlock < 0 ||
                static_cast<uint32_t>(physicalBlock) >= (cmp ? params.cmpBlocks : params.oriBlocks)) {
                return -1;
            }
            return static_cast<int64_t>(physicalBlock) * (cmp ? params.cmpStride : params.oriStride) +
                   token % blockSize * RECORD_BYTES;
        } else if constexpr (SAST::kvLayout == SAS_LAYOUT::TND) {
            return (static_cast<int64_t>(cmp ? cuCmpGm.GetValue(info.bIdx) : cuOriGm.GetValue(info.bIdx)) + token) *
                   RECORD_BYTES;
        }
        return (static_cast<int64_t>(info.bIdx) * (cmp ? params.cmpSeq : params.oriSeq) + token) * RECORD_BYTES;
    }

private:
    static constexpr uint32_t CAUSAL_MASK_MODE = 3;
    static constexpr uint32_t SLIDING_WINDOW_MASK_MODE = 4;
    static constexpr uint32_t MAX_PREPARED_ORI_ROWS = 128U;
    static constexpr uint32_t MAX_PREPARED_CMP_ROWS = 512U;
    static constexpr uint32_t ORI_PAGE_CACHE_CAPACITY = 128U;
    static constexpr uint32_t CMP_PAGE_SCRATCH_CAPACITY = 8192U;

    GlobalTensor<int32_t> indicesGm, oriTableGm, cmpTableGm, cuQGm, cuOriGm, cuCmpGm;
    GlobalTensor<int32_t> usedQGm, usedOriGm, usedCmpGm, residualGm, topkLengthGm;
    bool hasUsedQ = false, hasUsedOri = false, hasUsedCmp = false;
    bool hasResidual = false, hasTopkLength = false;
    RunInfo query{};
    int64_t cursor = 0, oriEnd = 0, cmpEnd = 0;
    uint32_t block = 0;
    bool preparedSelection = false;
    bool batchCached = false;
    uint32_t batchCacheB = 0;
    int64_t batchPrefix = 0, batchQLength = 0, batchOriLength = 0, batchCmpLength = 0, batchResidual = 0;
    bool oriPageCached = false, cmpPageCached = false;
    bool oriPageChecked = false, cmpPageChecked = false;
    bool oriPageBounded = false, cmpPageBounded = false;
    uint32_t oriPageBatch = 0, oriPageFirst = 0, oriPageCount = 0;
    uint32_t cmpPageBatch = 0, cmpPageCount = 0;

    __aicore__ inline void CopyIndexSpan(LocalTensor<int32_t> dst, GlobalTensor<int32_t> src, uint64_t offset,
                                         uint32_t count)
    {
        if (count == 0) {
            return;
        }
        DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(dst, src[offset], copy, pad);
    }

    __aicore__ inline int64_t CachedPageOffset(LocalTensor<int32_t> table, int64_t token, uint32_t first,
                                               uint32_t count, bool cmp)
    {
        uint32_t blockSize = cmp ? params.cmpBlockSize : params.oriBlockSize;
        int64_t logicalBlock = token / blockSize;
        if (logicalBlock < first || logicalBlock - first >= count) {
            return -1;
        }
        int32_t physicalBlock = table.GetValue(static_cast<uint32_t>(logicalBlock - first));
        if (physicalBlock < 0 || static_cast<uint32_t>(physicalBlock) >= (cmp ? params.cmpBlocks : params.oriBlocks)) {
            return -1;
        }
        return static_cast<int64_t>(physicalBlock) * (cmp ? params.cmpStride : params.oriStride) +
               (token - logicalBlock * blockSize) * RECORD_BYTES;
    }

    __aicore__ inline int64_t Min(int64_t a, int64_t b)
    {
        return a < b ? a : b;
    }
    __aicore__ inline int64_t Max(int64_t a, int64_t b)
    {
        return a > b ? a : b;
    }
    __aicore__ inline void SetOptional(GlobalTensor<int32_t>& t, __gm__ uint8_t* address)
    {
        if (address != nullptr) {
            t.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(address));
        }
    }
};
} // namespace SASKernel
#endif
