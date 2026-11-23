/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from the customer msa_indexer_q8k8 PipelineRun, snapshot20261111.
// Retains the original CANN and DeepGEMM notices below. ABI metadata is adapted
// to MsaIndexScore device q-prefix/k-length tensors; output keeps padded stride.
#pragma once

#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include "msa_fp8_pipeline_config.h"

namespace MsaNativeFp8Pipeline {
namespace R = AscendC::MicroAPI;
using namespace AscendC;

constexpr R::CastTrait kHalf{R::RegLayout::ZERO, R::SatMode::NO_SAT, R::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <pipe_t From, pipe_t To>
__aicore__ inline void Fence(int id = 0)
{
    asc_sync_notify(From, To, static_cast<event_t>(id));
    asc_sync_wait(From, To, static_cast<event_t>(id));
}

template <class T>
__aicore__ inline __ubuf__ T* Ub(int bytes)
{
    return reinterpret_cast<__ubuf__ T*>(bytes);
}
__aicore__ inline int Min(int a, int b)
{
    return a < b ? a : b;
}
__aicore__ inline int VisibleRows(int64_t p, int v)
{
    if (p < 0 || v <= 0)
        return 0;
    // Divide before adding; INT64 positions cannot overflow in p+1.
    return static_cast<int>((p / 4 < (v - 1) / 4 ? p / 4 : (v - 1) / 4) + 1);
}

__aicore__ inline int QueryBatch(const MsaFp8PipelineArgs& a, int token)
{
    auto prefix = reinterpret_cast<__gm__ int32_t*>(a.q_prefix);
    int lo = 0, hi = static_cast<int>(a.batches);
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (prefix[mid] <= token)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}
__aicore__ inline void ReduceBatch(__ubuf__ float* score, __ubuf__ half* out, int n, int outputs, bool hasScore,
                                   uint32_t reversed, int queries, int v0, int v1, int v2, int v3);

// Input prefetch and buffer-credit protocol adapted from DeepGEMM-Ascend
// (Copyright 2026 DeepSeek, MIT; see LICENSE.DeepGEMM in this directory), notably
// paged_sparse_mqa_logits.hpp and ascend/sync.hpp. Q is resident, K has six
// L1 stages; L0B/L0C and UB use independent queues and credits.
template <pipe_t From, pipe_t To, int Stages>
__aicore__ inline void NotifyStages()
{
    for (int s = 0; s < Stages; ++s)
        asc_sync_notify(From, To, static_cast<event_t>(s));
}
template <pipe_t From, pipe_t To, int Stages>
__aicore__ inline void WaitStages()
{
    for (int s = 0; s < Stages; ++s)
        asc_sync_wait(From, To, static_cast<event_t>(s));
}

__aicore__ inline int ChunkRows(int rows, int first, int n)
{
    return rows > first ? Min(n, rows - first) : 0;
}

template <class Fp8>
__aicore__ inline void Prefetch(const MsaFp8PipelineArgs& a, int batch, int rows, int firstTile, int tileStep,
                                int chunks, int i, bool bypassK)
{
    if (i >= chunks)
        return;
    const int n = a.summary_tile, stage = i % kKeyStages;
    const int first = (firstTile + i * tileStep) * n;
    const int pages = (ChunkRows(rows, first, n) + kPageRows - 1) / kPageRows;
    auto table = reinterpret_cast<__gm__ int32_t*>(a.table) + batch * a.table_stride;
    auto k = reinterpret_cast<__gm__ Fp8*>(a.k);
    const bool bypass =
        bypassK || kKCachePolicy == 1 || (kKCachePolicy == 2 && a.batches * a.scores * kRowsPerScore * kDim > kL2Bytes);
    const auto hint = bypass ? asc_load_l2_cache_mode::NOTALLOC_KEEP : asc_load_l2_cache_mode::NORMAL_LAST_VICTIM;
    auto dst = reinterpret_cast<__cbuf__ Fp8*>(kKL1 + stage * n * kDim);
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(stage));
    for (int page = 0; page < pages;) {
        const int page0 = table[first / kPageRows + page];
        int batchCount = Min(2, pages - page);
        const int page1 = batchCount == 2 ? table[first / kPageRows + page + 1] : page0;
        const int physical = Min(page0, page1);
        const int64_t gap = page0 > page1 ? static_cast<int64_t>(page0) - page1 : static_cast<int64_t>(page1) - page0;
        if constexpr (kCoalescePages) {
            // Common allocators give contiguous pages or short affine strides
            // (e.g. interleaved requests). Preserve pair boundaries on mixed tables.
            if (page1 > page0 && gap <= 64) {
                while (page + batchCount + 1 < pages &&
                       table[first / kPageRows + page + batchCount] == page0 + batchCount * gap &&
                       table[first / kPageRows + page + batchCount + 1] == page0 + (batchCount + 1) * gap)
                    batchCount += 2;
                if (page + batchCount + 1 == pages &&
                    table[first / kPageRows + page + batchCount] == page0 + batchCount * gap)
                    ++batchCount;
            }
        }
        // DeepGEMM's sorted physical pairs halve ND-to-NZ commands even for
        // reverse page tables. Vector restores logical order before causal masking.
        asc_set_gm2l1_nz_para(batchCount, 1, n, kPageRows);
        asc_copy_gm2l1_nd2nz(dst + page * kPageRows * kDmaBytes, k + physical * a.k_stride, kDim, hint, kPageRows, kDim,
                             gap * a.k_stride, 0);
        page += batchCount;
    }
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(stage));
}

__aicore__ inline __ubuf__ int32_t* CacheTable(const MsaFp8PipelineArgs& a, int batch)
{
    const int64_t bytes = a.table_columns * sizeof(int32_t);
    if (bytes > kTableCacheBytes)
        return nullptr;
    auto src = reinterpret_cast<__gm__ int32_t*>(a.table) + batch * a.table_stride;
    auto dst = Ub<int32_t>(kTableUB);
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(dst), reinterpret_cast<__gm__ uint8_t*>(src), 1, bytes, 0,
                         0, false, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM, bytes, bytes);
    // Scalar UB reads require PIPE_S, not a Vector completion event.
    Fence<PIPE_MTE2, PIPE_S>(kQueryEvent);
    return dst;
}

__simd_vf__ inline void BuildReversals256(__ubuf__ int32_t* table, __ubuf__ uint32_t* masks, uint32_t pages)
{
    auto all = R::CreateMask<uint32_t, R::MaskPattern::ALL>();
    auto sixteen = R::CreateMask<uint32_t, R::MaskPattern::VL16>();
    R::RegTensor<uint32_t> lane, bits, mod, weights, value, zero;
    R::Arange(reinterpret_cast<R::RegTensor<int32_t>&>(lane), 0);
    R::Duplicate(bits, 3u, all);
    R::And(mod, lane, bits, all);
    R::Duplicate(weights, 8u, all);
    R::Duplicate(zero, 0u, all);
#pragma unroll
    for (int bit = 0; bit < 3; ++bit) {
        R::MaskReg selected;
        R::Compares<uint32_t, CMPMODE::EQ>(selected, mod, static_cast<uint32_t>(bit), all);
        R::Duplicate(value, 1u << bit, all);
        R::Select(weights, value, weights, selected);
    }
    R::Duplicate(bits, 7u, all);
    R::And(mod, lane, bits, all);
    R::MaskReg evenMask, oddMask;
    R::Compares<uint32_t, CMPMODE::LT>(evenMask, mod, 4u, all);
    R::Not(oddMask, evenMask, all);
    for (uint32_t page = 0; page < pages; page += 128) {
        R::RegTensor<int32_t> a, b;
        R::LoadAlign<int32_t, R::LoadDist::DIST_DINTLV_B32>(a, b, table + page);
        R::MaskReg reversed, valid;
        R::Compare<int32_t, CMPMODE::GT>(reversed, a, b, all);
        uint32_t pairs = (pages - page) / 2;
        valid = R::UpdateMask<uint32_t>(pairs);
        R::And(reversed, reversed, valid, all);
        R::RegTensor<uint32_t> flags, even, odd, packed, unused;
        R::Select(flags, weights, zero, reversed);
        R::ReduceDataBlock<ReduceType::SUM>(even, flags, evenMask);
        R::ReduceDataBlock<ReduceType::SUM>(odd, flags, oddMask);
        R::Interleave(packed, unused, even, odd);
        R::StoreAlign(masks + page / 8, packed, sixteen);
    }
}

__aicore__ inline __ubuf__ uint32_t* CacheReversals(const MsaFp8PipelineArgs& a, __ubuf__ int32_t* table)
{
    if (!table || a.summary_tile != 256)
        return nullptr;
    auto masks = Ub<uint32_t>(kReversalUB);
    BuildReversals256(table, masks, a.table_columns);
    Fence<PIPE_V, PIPE_S>(kQueryEvent);
    return masks;
}

__aicore__ inline uint32_t ReversedPairs(const MsaFp8PipelineArgs& a, int batch, int first, int rows,
                                         __ubuf__ int32_t* cached)
{
    auto table = reinterpret_cast<__gm__ int32_t*>(a.table) + batch * a.table_stride + first / kPageRows;
    const int pages = (rows + kPageRows - 1) / kPageRows;
    uint32_t bits = 0;
    for (int page = 0; page + 1 < pages; page += 2) {
        const int a0 = cached ? cached[first / kPageRows + page] : table[page];
        const int a1 = cached ? cached[first / kPageRows + page + 1] : table[page + 1];
        bits |= static_cast<uint32_t>(a0 > a1) << (page / 2);
    }
    return bits;
}

template <class Fp8>
__aicore__ inline void PipelineRun(const MsaFp8PipelineArgs& a, int t, int count, int batch, int rows, int firstTile,
                                   int tileStep, int chunks, int owned, bool bypassK)
{
    const int n = a.summary_tile, blocks = n / kRowsPerScore;
    const int m = count <= 2 ? 32 : (count <= 4 ? 64 : 128);
    const int halfQueries = m / (2 * kHeads);
    const int accumulatorBytes = kMaxM * n * sizeof(float);
    const int scoreStride = n + kScorePadding;
    const int scoreBytes = kMaxM / 2 * scoreStride * sizeof(float);
    const int stages = n == 128 ? kSmallScoreStages : kScoreStages;
    const int ubStages = n == 256 ? kLargeUbStages : stages;
    if ASC_IS_AIC {
        NotifyStages<PIPE_MTE1, PIPE_MTE2, kKeyStages>();
        for (int slot = 0; slot < stages; ++slot)
            asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
        auto q = reinterpret_cast<__gm__ Fp8*>(a.q);
        auto ql1 = reinterpret_cast<__cbuf__ Fp8*>(kQL1);
        auto qa = reinterpret_cast<__ca__ Fp8*>(0);
        asc_set_gm2l1_nz_para(1, 1, m, 0);
        if (a.q_stride == kHeads * kDim) {
            asc_copy_gm2l1_nd2nz(ql1, q + t * a.q_stride, kDim, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM,
                                 count * kHeads, kDim, 0, 0);
        } else {
            for (int r = 0; r < count; ++r)
                asc_copy_gm2l1_nd2nz(ql1 + r * kHeads * kDmaBytes, q + (t + r) * a.q_stride, kDim,
                                     asc_load_l2_cache_mode::NORMAL_LAST_VICTIM, kHeads, kDim, 0, 0);
        }
        Fence<PIPE_MTE2, PIPE_MTE1>(kQueryEvent);
        asc_copy_l12l0a(qa, ql1, 0, 0, m / 16, kDim / kDmaBytes, m / 16, m / 16);
        Fence<PIPE_MTE1, PIPE_M>(kQueryEvent);
        for (int i = 0; i < Min(chunks, kKeyStages); ++i)
            Prefetch<Fp8>(a, batch, rows, firstTile, tileStep, chunks, i, bypassK);
        for (int i = 0; i < chunks; ++i) {
            const int slot = i & (stages - 1), keySlot = i % kKeyStages;
            const int ubSlot = n == 256 ? i % kLargeUbStages : slot;
            const int first = (firstTile + i * tileStep) * n;
            const bool active = rows > first;
            auto kb = reinterpret_cast<__cb__ Fp8*>(slot * n * kDim);
            auto kl1 = reinterpret_cast<__cbuf__ Fp8*>(kKL1 + keySlot * n * kDim);
            auto acc = reinterpret_cast<__cc__ float*>(slot * accumulatorBytes);
            asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(keySlot));
            asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
            if (active) {
                asc_copy_l12l0b(kb, kl1, 0, 0, n / 16, kDim / kDmaBytes, n / 16, n / 16);
                asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(slot));
            }
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(keySlot));
            // Refill the released L1 slot BEFORE issuing any FIX wait. MTE2 can
            // run six chunks ahead while Cube/FIX/Vector consume earlier chunks.
            Prefetch<Fp8>(a, batch, rows, firstTile, tileStep, chunks, i + kKeyStages, bypassK);
            if (active) {
                asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(slot));
                // DeepGEMM's unit flags provide the two-way MMA/FIX interlock per
                // L0C tile, without serializing every iteration with explicit fences.
                asc_mmad(acc, qa, kb, m, kDim, n, asc_unit_flag_mode::ENABLE_UPDATE, true, false, true);
            }
            asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
            // Each partner has a local credit. Intra-block flags avoid the FFTS
            // mode-2 round trip on every score tile while retaining both readers.
            asc_sync_intra_wait(PIPE_FIX, ubSlot);
            asc_sync_intra_wait(PIPE_FIX, 16 + ubSlot);
            if (active) {
                asc_set_l0c_copy_nz_para(1, 0, 0);
                asc_copy_l0c2ub(Ub<float>(ubSlot * scoreBytes), acc, n, m, scoreStride, m, 0,
                                static_cast<asc_dual_dst_mode>(1), asc_unit_flag_mode::ENABLE_UPDATE,
                                asc_quant_mode::NoQuant, static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
            }
            asc_sync_intra_arrive(PIPE_FIX, ubSlot);
            asc_sync_intra_arrive(PIPE_FIX, 16 + ubSlot);
        }
        WaitStages<PIPE_MTE1, PIPE_MTE2, kKeyStages>();
        for (int slot = 0; slot < stages; ++slot)
            asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
        for (int slot = 0; slot < ubStages; ++slot) {
            asc_sync_intra_wait(PIPE_FIX, slot);
            asc_sync_intra_wait(PIPE_FIX, 16 + slot);
        }
        Fence<PIPE_FIX, PIPE_MTE2>(kQueryEvent);
    } else {
        const int sub = asc_get_sub_block_id(), begin = sub * halfQueries;
#ifdef MSA_C2UB_CONTROL
        // Diagnostic empty consumer: materialize every full FP32 tile in UB,
        // then sample its first head/64 columns. No max, mask or FP16 conversion.
        const int tiles = (owned * kRowsPerScore + n - 1) / n;
        auto probes = reinterpret_cast<__gm__ float*>(a.out);
        for (int slot = 0; slot < ubStages; ++slot)
            asc_sync_intra_arrive(PIPE_MTE3, sub * 16 + slot);
        for (int i = 0; i < chunks; ++i) {
            const int slot = n == 256 ? i % kLargeUbStages : i & (stages - 1);
            const int tile = firstTile + i * tileStep;
            asc_sync_intra_wait(PIPE_MTE3, slot);
            asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(
                                     probes + ((static_cast<int64_t>(t / kQueries) * tiles + tile) * 2 + sub) * 64),
                                 Ub<uint8_t>(slot * scoreBytes), 1, 64 * sizeof(float),
                                 asc_store_l2_cache_mode::NOTALLOC_CLEAN, 64 * sizeof(float), 64 * sizeof(float));
            asc_sync_intra_arrive(PIPE_MTE3, sub * 16 + slot);
        }
        Fence<PIPE_MTE3, PIPE_V>();
#else
        const int activeQueries = count > begin ? Min(halfQueries, count - begin) : 0;
        auto out = reinterpret_cast<__gm__ half*>(a.out);
        auto cachedTable = CacheTable(a, batch);
        auto cachedReversals = CacheReversals(a, cachedTable);
        int rowBounds[kQueries / 2];
        auto prefix = reinterpret_cast<__gm__ int32_t*>(a.q_prefix);
        auto lengths = reinterpret_cast<__gm__ int32_t*>(a.k_lengths);
        const int qStart = prefix[batch], qLength = prefix[batch + 1] - qStart;
        const int kvLength = lengths[batch];
        const int64_t position = static_cast<int64_t>(kvLength) - qLength + t + begin - qStart;
        for (int r = 0; r < activeQueries; ++r) {
            rowBounds[r] = a.causal ? VisibleRows(position + r, kvLength) : (kvLength / 4 + (kvLength % 4 != 0));
        }
        NotifyStages<PIPE_MTE3, PIPE_V, kOutputStages>();
        for (int slot = 0; slot < ubStages; ++slot)
            asc_sync_intra_arrive(PIPE_V, sub * 16 + slot);
        for (int i = 0; i < chunks; ++i) {
            const int slot = n == 256 ? i % kLargeUbStages : i & (stages - 1);
            const int outputSlot = (i / kFlushTiles) % kOutputStages;
            const int inSlab = i % kFlushTiles;
            const int column = (firstTile + i * tileStep) * blocks;
            const int first = column * kRowsPerScore;
            const int outputs = Min(blocks, owned - column);
            const int tileRows = ChunkRows(rows, first, n);
            const uint32_t reversed =
                cachedReversals ? cachedReversals[firstTile + i * tileStep] & ((1u << ((tileRows + 31) / 64)) - 1u) :
                                  ReversedPairs(a, batch, first, tileRows, cachedTable);
            auto score = Ub<float>(slot * scoreBytes);
            auto encoded = Ub<half>(kOutputUB + outputSlot * kOutputStageBytes) + inSlab * blocks;
            if (!inSlab)
                asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(outputSlot));
            asc_sync_intra_wait(PIPE_V, slot);
            if (activeQueries)
                ReduceBatch(score, encoded, n, outputs, tileRows != 0, reversed, activeQueries,
                            ChunkRows(rowBounds[0], first, tileRows),
                            activeQueries > 1 ? ChunkRows(rowBounds[1], first, tileRows) : 0,
                            activeQueries > 2 ? ChunkRows(rowBounds[2], first, tileRows) : 0,
                            activeQueries > 3 ? ChunkRows(rowBounds[3], first, tileRows) : 0);
            // Score lifetime ends at the last Vector read, independently of MTE3.
            asc_sync_intra_arrive(PIPE_V, sub * 16 + slot);
            if (inSlab + 1 == kFlushTiles || i + 1 == chunks) {
                asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(outputSlot));
                asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(outputSlot));
                if (activeQueries)
                    asc_copy_ub2gm_align(
                        reinterpret_cast<__gm__ uint8_t*>(out + (t + begin) * a.out_stride + column - inSlab * blocks),
                        Ub<uint8_t>(kOutputUB + outputSlot * kOutputStageBytes), activeQueries,
                        (inSlab * blocks + outputs) * sizeof(half), asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                        a.out_stride * sizeof(half), kFlushTiles * blocks * sizeof(half));
                asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(outputSlot));
            }
        }
        WaitStages<PIPE_MTE3, PIPE_V, kOutputStages>();
#endif
    }
}

template <int N, int Queries, bool Full, int OutputTiles = 1>
__simd_vf__ inline void BatchMax(__ubuf__ float* score, __ubuf__ half* out, int outputs, bool hasScore,
                                 uint32_t reversed, int activeQueries, int v0, int v1, int v2, int v3)
{
    constexpr int Parts = N / 64;
    constexpr int ScoreStride = N + kScorePadding;
    static_assert(Queries * Parts <= 16);
    auto all = R::CreateMask<float, R::MaskPattern::ALL>();
    auto sixteen = R::CreateMask<float, R::MaskPattern::VL16>();
    R::RegTensor<float> maximum[Queries][Parts];
    if (Full || hasScore) {
// The first head initializes each chain directly. This removes one max
// and one -inf initialization per vector while preserving NaN behavior.
#pragma unroll
        for (int q = 0; q < Queries; ++q) {
#pragma unroll
            for (int part = 0; part < Parts; ++part)
                R::LoadAlign(maximum[q][part], score + q * kHeads * ScoreStride + part * 64);
        }
#pragma unroll 4
        for (uint16_t head = 1; head < kHeads; ++head) {
#pragma unroll
            for (int q = 0; q < Queries; ++q) {
#pragma unroll
                for (int part = 0; part < Parts; ++part) {
                    R::RegTensor<float> value;
                    R::LoadAlign(value, score + (q * kHeads + head) * ScoreStride + part * 64);
                    R::Max(maximum[q][part], maximum[q][part], value, all);
                }
            }
        }
    } else {
#pragma unroll
        for (int q = 0; q < Queries; ++q) {
#pragma unroll
            for (int part = 0; part < Parts; ++part)
                R::Duplicate(maximum[q][part], kNegative, all);
        }
    }
    R::RegTensor<uint32_t> index, swap, bits;
    R::Arange(reinterpret_cast<R::RegTensor<int32_t>&>(index), 0);
    R::Duplicate(bits, 32u, all);
    R::Xor(swap, index, bits, all);
    // Two masked vcgmax reductions select each half of an 8-float data block.
    // Interleaving their packed results produces the sixteen block4 maxima.
    R::Duplicate(bits, 7u, all);
    R::And(index, index, bits, all);
    R::MaskReg evenBlocks, oddBlocks;
    R::Compares<uint32_t, CMPMODE::LT>(evenBlocks, index, 4u, all);
    R::Not(oddBlocks, evenBlocks, all);
    R::RegTensor<float> negative;
    R::Duplicate(negative, kNegative, all);
#pragma unroll
    for (int q = 0; q < Queries; ++q) {
        if (!Full && q >= activeQueries)
            continue;
        const int valid = q == 0 ? v0 : q == 1 ? v1 : q == 2 ? v2 : v3;
#pragma unroll
        for (int part = 0; part < Parts; ++part) {
            if (!Full && part * 16 >= outputs)
                continue;
            if ((reversed >> part) & 1u)
                R::Gather(maximum[q][part], maximum[q][part], swap);
            if constexpr (!Full) {
                uint32_t count = valid > part * 64 ? valid - part * 64 : 0;
                auto keep = R::UpdateMask<float>(count);
                R::Select(maximum[q][part], maximum[q][part], negative, keep);
            }
            R::RegTensor<float> even, odd, reduced, unused;
            R::ReduceDataBlock<ReduceType::MAX>(even, maximum[q][part], evenBlocks);
            R::ReduceDataBlock<ReduceType::MAX>(odd, maximum[q][part], oddBlocks);
            R::Interleave(reduced, unused, even, odd);
            R::Muls(reduced, reduced, kScale, sixteen);
            R::RegTensor<half> encoded;
            R::Cast<half, float, kHalf>(encoded, reduced, sixteen);
            R::StoreAlign<half, R::StoreDist::DIST_PACK_B32>(out + q * (N / 4) * OutputTiles + part * 16, encoded,
                                                             sixteen);
        }
    }
}

template <int N, int Queries>
__aicore__ inline void ReduceQueries(__ubuf__ float* score, __ubuf__ half* out, int outputs, bool hasScore,
                                     uint32_t reversed, int queries, int v0, int v1, int v2, int v3)
{
    // Interior chunks are almost all of a long-history Q8 call. Derive this
    // predicate from device metadata; tail/partial-query chunks keep the full mask.
    const bool full = queries == Queries && outputs == N / 4 && v0 == N && (Queries < 2 || v1 == N) &&
                      (Queries < 4 || (v2 == N && v3 == N));
    if (full)
        BatchMax<N, Queries, true, N <= kPipelineMaxN ? kFlushTiles : 1>(score, out, outputs, hasScore, reversed,
                                                                         queries, v0, v1, v2, v3);
    else
        BatchMax<N, Queries, false, N <= kPipelineMaxN ? kFlushTiles : 1>(score, out, outputs, hasScore, reversed,
                                                                          queries, v0, v1, v2, v3);
}

template <int N>
__aicore__ inline void ReduceN(__ubuf__ float* score, __ubuf__ half* out, int outputs, bool hasScore, uint32_t reversed,
                               int queries, int v0, int v1, int v2, int v3)
{
    if (queries == 1)
        ReduceQueries<N, 1>(score, out, outputs, hasScore, reversed, queries, v0, v1, v2, v3);
    else if (queries == 2)
        ReduceQueries<N, 2>(score, out, outputs, hasScore, reversed, queries, v0, v1, v2, v3);
    else if constexpr (N <= kPipelineMaxN)
        ReduceQueries<N, 4>(score, out, outputs, hasScore, reversed, queries, v0, v1, v2, v3);
}

__aicore__ inline void ReduceBatch(__ubuf__ float* score, __ubuf__ half* out, int n, int outputs, bool hasScore,
                                   uint32_t reversed, int queries, int v0, int v1, int v2, int v3)
{
    if (n == 64)
        ReduceN<64>(score, out, outputs, hasScore, reversed, queries, v0, v1, v2, v3);
    else if (n == 128)
        ReduceN<128>(score, out, outputs, hasScore, reversed, queries, v0, v1, v2, v3);
    else if (n == 256)
        ReduceN<256>(score, out, outputs, hasScore, reversed, queries, v0, v1, v2, v3);
    else
        ReduceN<512>(score, out, outputs, hasScore, reversed, queries, v0, v1, v2, v3);
}

template <class Fp8, bool Balanced = false, bool Bypass = false>
__aicore__ inline void Pipeline(const MsaFp8PipelineArgs& a)
{
    if (a.tokens <= 0 || a.batches <= 0)
        return;
    asc_init();
    if ASC_IS_AIC {
        asc_set_mmad_direction_n();
    }
    const int groups = (a.tokens + a.query_tile - 1) / a.query_tile;
    const int owned = (a.scores + kScoreAlignment - 1) / kScoreAlignment * kScoreAlignment;
    const int tiles = (owned * kRowsPerScore + a.summary_tile - 1) / a.summary_tile;
    const int cores = get_block_num();
    const int rawBatch = a.bs ? *reinterpret_cast<__gm__ int32_t*>(a.bs) : a.batches;
    const int liveBatch = rawBatch < 0 ? 0 : Min(a.batches, rawBatch);
    // Host-selected literal: no per-tile cache-policy branch in the hot path.
    const bool bypassK = Bypass;
    // Divide the flattened (query group, K tile) domain when whole groups
    // cannot be distributed evenly. A core keeps Q resident within each
    // contiguous segment and reloads Q only at a group boundary.
    constexpr bool balanceTiles = Balanced;
#ifndef MSA_BALANCE_TILE_GRAIN
#define MSA_BALANCE_TILE_GRAIN 1
#endif
    constexpr int grain = MSA_BALANCE_TILE_GRAIN;
    const int units = (tiles + grain - 1) / grain;
    const int64_t work = static_cast<int64_t>(groups) * units;
    const int64_t workBegin = work * block_idx / cores;
    const int64_t workEnd = work * (block_idx + 1) / cores;
    const int firstGroup = balanceTiles ? workBegin / units : block_idx % groups;
    const int lastGroup = balanceTiles ? (workEnd + units - 1) / units : groups;
    const int groupStep = balanceTiles ? 1 : cores;
    for (int group = firstGroup; group < lastGroup; group += groupStep) {
        const int partition = groups < cores ? block_idx / groups : 0;
        const int partitions = groups < cores ? (cores - 1 - group) / groups + 1 : 1;
        const int64_t groupBase = static_cast<int64_t>(group) * units;
        const int segmentBegin = balanceTiles ? (workBegin > groupBase ? workBegin - groupBase : 0) : 0;
        const int segmentEnd = balanceTiles ? Min(units, workEnd - groupBase) : 0;
        const int firstTile = balanceTiles ? Min(tiles, segmentBegin * grain) :
                                             (kContiguousTiles ? tiles * partition / partitions : partition);
        const int tileStep = balanceTiles || kContiguousTiles ? 1 : partitions;
        const int chunks = balanceTiles ? Min(tiles, segmentEnd * grain) - firstTile :
                                          (kContiguousTiles ? tiles * (partition + 1) / partitions - firstTile :
                                                              (tiles - firstTile + tileStep - 1) / tileStep);
        const int end = Min((group + 1) * a.query_tile, a.tokens);
        for (int t = group * a.query_tile; t < end;) {
            const int batch = QueryBatch(a, t);
            auto prefix = reinterpret_cast<__gm__ int32_t*>(a.q_prefix);
            auto lengths = reinterpret_cast<__gm__ int32_t*>(a.k_lengths);
            const int qStart = prefix[batch], qEnd = prefix[batch + 1];
            const int count = Min(end - t, qEnd - t);
            int rows = 0;
            if (batch < 0 || batch >= liveBatch) {
                if ASC_IS_AIV {
                    if (asc_get_sub_block_id() == 0) {
                        auto output = reinterpret_cast<__gm__ uint16_t*>(a.out);
                        for (int i = 0; i < chunks; ++i) {
                            const int column = (firstTile + i * tileStep) * (a.summary_tile / kRowsPerScore);
                            const int length = Min(a.summary_tile / kRowsPerScore, owned - column);
                            for (int r = 0; r < count; ++r)
                                for (int j = 0; j < length; ++j)
                                    output[(t + r) * a.out_stride + column + j] = 0xfc00;
                        }
                    }
                }
                t += count;
                continue;
            }
            const int kvLength = lengths[batch];
            const int64_t position = static_cast<int64_t>(kvLength) - (qEnd - qStart) + t + count - 1 - qStart;
            rows = a.causal ? VisibleRows(position, kvLength) : (kvLength / 4 + (kvLength % 4 != 0));
            rows = Min(rows, Min(a.scores * kRowsPerScore, a.table_columns * kPageRows));
            PipelineRun<Fp8>(a, t, count, batch, rows, firstTile, tileStep, chunks, owned, bypassK);
            t += count;
        }
    }
}
} // namespace MsaNativeFp8Pipeline

/*
DeepGEMM attribution and license:
MIT License

Copyright (c) 2026 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
