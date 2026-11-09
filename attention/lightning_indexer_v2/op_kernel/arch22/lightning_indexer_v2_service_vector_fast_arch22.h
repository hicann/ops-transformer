/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_v2_service_vector_fast_arch22.h
 * \brief vector cal for cube fill vector
 */
#ifndef LIGHTNING_INDEXER_V2_SERVICE_VECTOR_FAST_ARCH22_H
#define LIGHTNING_INDEXER_V2_SERVICE_VECTOR_FAST_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lightning_indexer_v2_common_fast_arch22.h"
#include "lightning_indexer_v2_vector.h"

namespace LIV2FastLongSeqVector {
using namespace AscendC;
constexpr int32_t TOPK = 2048;
constexpr int32_t SCORE_ROWS = 512;
constexpr int32_t SCORE_CHANNEL_COLS = 8;
constexpr int32_t SCORE_RING_DEPTH = 8;
constexpr int32_t SCORE_TILE_ELEMENTS = SCORE_ROWS * 16;
constexpr int32_t Kibibytes(int32_t n)
{
    return n * 1024; // 1 KB = 1024 byte
}
// 一个 AIC 对应两颗 AIV；N=24/32 时每颗 AIV 分别处理 4/3 个 token。
template <int32_t N>
struct KernelConfig {
    static constexpr int32_t SEQUENCES_PER_VECTOR_CORE = 192 / N / 2;
    static constexpr int32_t SCORE_COLUMN_OFFSET = (16 - 192 / N) / 2;
};
// 长序列流水使用固定 UB 分区，编译期校验总量不超过 192 KiB。
template <int32_t Sequences>
struct UbLayout {
    static_assert(Sequences == 3 || Sequences == 4);
    static constexpr int32_t INPUT_BASE = 0;
    static constexpr int32_t INPUT_BYTES = Kibibytes(16);
    static constexpr int32_t CHUNKS_BASE = INPUT_BASE + INPUT_BYTES;
    static constexpr int32_t CHUNKS_BYTES = Sequences * TOPK * sizeof(float);
    static constexpr int32_t INDICES_BASE = CHUNKS_BASE + CHUNKS_BYTES;
    static constexpr int32_t INDICES_BYTES = Kibibytes(8);
    static constexpr int32_t SPILL_BASE = INDICES_BASE + INDICES_BYTES;
    static constexpr int32_t TRANSPOSE_SPILL_BYTES = (SCORE_CHANNEL_COLS - Sequences) * SCORE_ROWS * sizeof(float);
    static constexpr int32_t SPILL_BYTES =
        TRANSPOSE_SPILL_BYTES > INDICES_BYTES ? TRANSPOSE_SPILL_BYTES : INDICES_BYTES;
    static constexpr int32_t SOURCE_HEAD_BASE = SPILL_BASE + SPILL_BYTES;
    static constexpr int32_t SOURCE_HEAD_BYTES = Kibibytes(48);
    static constexpr int32_t SOURCE_TAIL_BASE = SOURCE_HEAD_BASE + SOURCE_HEAD_BYTES;
    static constexpr int32_t SOURCE_TAIL_BYTES = Kibibytes(16);
    static constexpr int32_t DESTINATION_BASE = SOURCE_TAIL_BASE + SOURCE_TAIL_BYTES;
    static constexpr int32_t DESTINATION_BYTES = Kibibytes(16);
    static constexpr int32_t DESTINATION_REST_BASE = DESTINATION_BASE + DESTINATION_BYTES;
    static constexpr int32_t DESTINATION_REST_BYTES = Kibibytes(48);
    static constexpr int32_t BYTES = DESTINATION_REST_BASE + DESTINATION_REST_BYTES;
};

static_assert(UbLayout<4>::BYTES == Kibibytes(192));
static_assert(UbLayout<3>::BYTES <= Kibibytes(192));

constexpr int32_t UB_QUEUE_WORDS = TOPK * 2;
constexpr int32_t UB_QUEUE_FLOATS = UB_QUEUE_WORDS;
constexpr int32_t UB_QUEUE_BLOCKS = UB_QUEUE_FLOATS * sizeof(float) / 32;
constexpr int32_t UB_SCORE_TILE_COUNT = TOPK / SCORE_ROWS;
constexpr int32_t UB_SCORE_CHANNEL_BLOCKS = SCORE_ROWS * SCORE_CHANNEL_COLS * sizeof(float) / 32;
constexpr int32_t UB_OUTPUT_BLOCKS = TOPK * sizeof(uint32_t) / 32;

template <int32_t RepeatTimes, int32_t ElementLength, bool StopWhenExhausted = false>
static __aicore__ inline void MergeSortedFour(AscendC::LocalTensor<float> destination,
                                              AscendC::LocalTensor<float> source0, AscendC::LocalTensor<float> source1,
                                              AscendC::LocalTensor<float> source2, AscendC::LocalTensor<float> source3)
{
    AscendC::MrgSortSrcList<float> sources;
    sources.src1 = source0;
    sources.src2 = source1;
    sources.src3 = source2;
    sources.src4 = source3;

    AscendC::MrgSort4Info params;
    params.elementLengths[0] = ElementLength;
    params.elementLengths[1] = ElementLength;
    params.elementLengths[2] = ElementLength;
    params.elementLengths[3] = ElementLength;
    params.ifExhaustedSuspension = StopWhenExhausted;
    params.validBit = 0x0f;
    params.repeatTimes = RepeatTimes;
    AscendC::MrgSort(destination, sources, params);
}

static __aicore__ inline void InitVectorEvents()
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
}

static __aicore__ inline void InitializeIndices(AscendC::LocalTensor<uint32_t> indices)
{
    AscendC::ArithProgression<int32_t>(indices.template ReinterpretCast<int32_t>(), 0, 1, TOPK);
}

static __aicore__ inline void AdvanceIndices(AscendC::LocalTensor<uint32_t> indices)
{
    AscendC::LocalTensor<int32_t> indicesLocal = indices.template ReinterpretCast<int32_t>();
    AscendC::Adds(indicesLocal, indicesLocal, static_cast<int32_t>(TOPK), TOPK);
    AscendC::PipeBarrier<PIPE_V>();
}

template <int32_t FirstColumn, int32_t SelectedColumns>
static __aicore__ inline void TransposeSelected512X8B32(AscendC::LocalTensor<float> destination,
                                                        AscendC::LocalTensor<float> spill,
                                                        AscendC::LocalTensor<float> source)
{
    static_assert(SelectedColumns == 3 || SelectedColumns == 4);
    static_assert(FirstColumn >= 0 && FirstColumn + SelectedColumns <= SCORE_CHANNEL_COLS);
    uint64_t sourceAddresses[16];
    uint64_t destinationAddresses[16];
    int32_t spillColumn = 0;

    for (int32_t row = 0; row < 16; ++row) {
        sourceAddresses[row] = reinterpret_cast<uint64_t>(source[row * SCORE_CHANNEL_COLS].GetPhyAddr());
    }
    for (int32_t column = 0; column < SCORE_CHANNEL_COLS; ++column) {
        AscendC::LocalTensor<float> columnPointer;
        if (column >= FirstColumn && column < FirstColumn + SelectedColumns) {
            columnPointer = destination[(column - FirstColumn) * TOPK];
        } else {
            columnPointer = spill[spillColumn * SCORE_ROWS];
            ++spillColumn;
        }
        destinationAddresses[2 * column] = reinterpret_cast<uint64_t>(columnPointer.GetPhyAddr());
        destinationAddresses[2 * column + 1] = reinterpret_cast<uint64_t>(columnPointer[8].GetPhyAddr());
    }

    AscendC::TransDataTo5HDParams params(false, false, 32, 2, 16);
    AscendC::TransDataTo5HD<float>(destinationAddresses, sourceAddresses, params);
}

static __aicore__ inline void Sort32Quarter(AscendC::LocalTensor<float> sorted32, AscendC::LocalTensor<float> values,
                                            AscendC::LocalTensor<uint32_t> indices)
{
    AscendC::Sort32(sorted32, values, indices, SCORE_ROWS / 32);
}

static __aicore__ inline void FinishSorted32(AscendC::LocalTensor<float> destination,
                                             AscendC::LocalTensor<float> scratch, AscendC::LocalTensor<float> sorted32)
{
    MergeSortedFour<16, 32>(destination, sorted32, sorted32[32 * 2], sorted32[64 * 2], sorted32[96 * 2]);
    AscendC::PipeBarrier<PIPE_V>();

    MergeSortedFour<4, 128>(scratch, destination, destination[128 * 2], destination[256 * 2], destination[384 * 2]);
    AscendC::PipeBarrier<PIPE_V>();

    MergeSortedFour<1, 512>(destination, scratch, scratch[512 * 2], scratch[1024 * 2], scratch[1536 * 2]);
    AscendC::PipeBarrier<PIPE_V>();
}

static __aicore__ inline void SortChunk(AscendC::LocalTensor<float> destination, AscendC::LocalTensor<float> scratch,
                                        AscendC::LocalTensor<float> values, AscendC::LocalTensor<uint32_t> indices)
{
    AscendC::Sort32(scratch, values, indices, TOPK / 32);
    AscendC::PipeBarrier<PIPE_V>();
    FinishSorted32(destination, scratch, scratch);
}

template <int32_t Sequences>
static __aicore__ inline void FinishPresortedSequences(AscendC::LocalTensor<float> destination,
                                                       AscendC::LocalTensor<float> source)
{
    MergeSortedFour<16 * Sequences, 32>(destination, source, source[32 * 2], source[64 * 2], source[96 * 2]);
    AscendC::PipeBarrier<PIPE_V>();

    MergeSortedFour<4 * Sequences, 128>(source, destination, destination[128 * 2], destination[256 * 2],
                                        destination[384 * 2]);
    AscendC::PipeBarrier<PIPE_V>();

    MergeSortedFour<Sequences, 512>(destination, source, source[512 * 2], source[1024 * 2], source[1536 * 2]);
    AscendC::PipeBarrier<PIPE_V>();
}

static __aicore__ inline void MergeFour(AscendC::LocalTensor<float> destination, AscendC::LocalTensor<float> source)
{
    MergeSortedFour<1, TOPK, true>(destination, source, source[UB_QUEUE_FLOATS], source[2 * UB_QUEUE_FLOATS],
                                   source[3 * UB_QUEUE_FLOATS]);
    AscendC::PipeBarrier<PIPE_V>();
}

static __aicore__ inline void ExtractIndices(AscendC::LocalTensor<uint32_t> destination,
                                             AscendC::LocalTensor<float> source)
{
    uint64_t reservedCount = 0;
    AscendC::GatherMaskParams params(1, TOPK / 32, 8, 0);
    AscendC::GatherMask(destination, source.template ReinterpretCast<uint32_t>(), static_cast<uint8_t>(2), false, 0,
                        params, reservedCount);
}

static __aicore__ inline void FinishEarlyFirstCandidate(AscendC::LocalTensor<float> tail,
                                                        AscendC::LocalTensor<float> destination,
                                                        AscendC::LocalTensor<float> scratch)
{
    AscendC::PipeBarrier<PIPE_V>();
    FinishSorted32(destination, scratch, tail);
    // UB内连续搬运一组TopK候选，每个候选包含value/index两个32位字段
    // blockCount=1；blockLen单位为32字节；源、目的均无额外块间隔
    AscendC::DataCopyParams params(1, UB_QUEUE_BLOCKS, 0, 0);
    AscendC::DataCopy(tail, destination, params);
    AscendC::PipeBarrier<PIPE_V>();
}

template <int32_t EventId, uint32_t BlockCount>
static __aicore__ inline void SubmitVectorStore(AscendC::LocalTensor<float> source, __gm__ float* destination)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EventId);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EventId);
    AscendC::GlobalTensor<float> global;
    global.SetGlobalBuffer(destination);
    AscendC::DataCopyParams params(1, BlockCount, 0, 0);
    AscendC::DataCopy(global, source, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EventId);
}

template <int32_t EventId>
static __aicore__ inline void WaitVectorStore()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EventId);
}

template <int32_t Sequence, int32_t Sequences>
static __aicore__ inline void SubmitAccumulatorHeadLoad(AscendC::LocalTensor<float> sourceHead, __gm__ float* state)
{
    // GM 按 token 交错保存 3 组候选，此处用带 stride 的一次搬运取回同一 token。
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
    AscendC::GlobalTensor<float> global;
    global.SetGlobalBuffer(state + Sequence * UB_QUEUE_FLOATS);
    // GM按[候选组][token][value/index数据]保存
    // 从当前token的起点取回三组候选，连续写入UB
    // blockLen和两侧stride均以32字节为单位
    // 每组搬完后，GM跳过其他Sequences-1个token；UB不留间隔
    AscendC::DataCopyParams params(3, UB_QUEUE_BLOCKS, (Sequences - 1) * UB_QUEUE_BLOCKS, 0);
    AscendC::DataCopy(sourceHead, global, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
}

static __aicore__ inline void WaitAccumulatorHeadLoad()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
}

template <int32_t Sequences, int32_t Sequence>
static __aicore__ inline void PresortQuarter(int32_t tile, AscendC::LocalTensor<float> chunks,
                                             AscendC::LocalTensor<uint32_t> indices,
                                             AscendC::LocalTensor<float> presorted)
{
    Sort32Quarter(presorted[Sequence * UB_QUEUE_FLOATS + tile * SCORE_ROWS * 2],
                  chunks[Sequence * TOPK + tile * SCORE_ROWS], indices[tile * SCORE_ROWS]);
}

template <int32_t N, bool Presort>
static __aicore__ inline void LoadScoreChunk(int32_t chunk, __gm__ float* scoreRing, int32_t* scoreTail,
                                             int32_t realTileCount, int32_t subblock, AscendC::LocalTensor<float> input,
                                             AscendC::LocalTensor<float> chunks, AscendC::LocalTensor<uint32_t> indices,
                                             AscendC::LocalTensor<float> source, AscendC::LocalTensor<float> spill)
{
    constexpr int32_t SEQUENCES_PER_AIV = KernelConfig<N>::SEQUENCES_PER_VECTOR_CORE;
    constexpr int32_t FIRST_COLUMN = KernelConfig<N>::SCORE_COLUMN_OFFSET;

    for (int32_t tile = 0; tile < UB_SCORE_TILE_COUNT; ++tile) {
        const int32_t scoreTile = chunk * UB_SCORE_TILE_COUNT + tile;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);

        if (scoreTile * SCORE_ROWS < realTileCount) {
            AscendC::CrossCoreWaitFlag(*scoreTail % SCORE_RING_DEPTH);
            AscendC::GlobalTensor<float> scoreGlobal;
            scoreGlobal.SetGlobalBuffer(scoreRing + (*scoreTail % SCORE_RING_DEPTH) * SCORE_TILE_ELEMENTS +
                                        subblock * SCORE_ROWS * SCORE_CHANNEL_COLS);
            AscendC::DataCopyParams scoreParams(1, UB_SCORE_CHANNEL_BLOCKS, 0, 0);
            AscendC::DataCopy(input, scoreGlobal, scoreParams);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(*scoreTail % SCORE_RING_DEPTH);
            ++*scoreTail;
        } else {
            AscendC::Duplicate(input, -__builtin_inff(), SCORE_ROWS * SCORE_CHANNEL_COLS);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);

        AscendC::LocalTensor<float> tileValues = chunks[tile * SCORE_ROWS];
        if (subblock == 0) {
            TransposeSelected512X8B32<FIRST_COLUMN, SEQUENCES_PER_AIV>(tileValues, spill, input);
        } else {
            TransposeSelected512X8B32<0, SEQUENCES_PER_AIV>(tileValues, spill, input);
        }
        // realTileCount 是本组最后一个 token 的可见长度；组内 token 的
        // causal 可见长度从 firstVisibleRows 起逐行递增，尾 tile 必须逐 token 屏蔽。
        const int32_t firstVisibleRows = realTileCount - (2 * SEQUENCES_PER_AIV - 1);
        if ((scoreTile + 1) * SCORE_ROWS > firstVisibleRows) {
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t sequence = 0; sequence < SEQUENCES_PER_AIV; ++sequence) {
                int32_t valid = firstVisibleRows + subblock * SEQUENCES_PER_AIV + sequence - scoreTile * SCORE_ROWS;
                valid = (valid < 0) ? 0 : ((valid > SCORE_ROWS) ? SCORE_ROWS : valid);
                if (valid < SCORE_ROWS) {
                    int32_t block = valid / 64;
                    int32_t lane = valid % 64;
                    AscendC::LocalTensor<float> tailStart = tileValues[sequence * TOPK + block * 64];
                    uint64_t tailMask[2] = {lane == 0 ? ~0ULL : (~0ULL << lane), 0};
                    AscendC::Duplicate<float, true>(tailStart, -__builtin_inff(), tailMask, 1, 1, 8);
                    if (block + 1 < SCORE_ROWS / 64) {
                        uint64_t fullMask[2] = {~0ULL, 0};
                        uint8_t repeatTimes = static_cast<uint8_t>(SCORE_ROWS / 64 - block - 1);
                        AscendC::Duplicate<float, true>(tailStart[64], -__builtin_inff(), fullMask, repeatTimes, 1, 8);
                    }
                }
            }
            // Duplicate<..., true> 会修改 vector mask。必须先等待所有尾部
            // Duplicate 完成再恢复；提前恢复会使已下发的尾块屏蔽偶发失效。
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<float>(~0ULL, ~0ULL);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);

        if constexpr (Presort) {
            constexpr int32_t TOKEN_GROUP_0 = 0;
            constexpr int32_t TOKEN_GROUP_1 = 1;
            constexpr int32_t TOKEN_GROUP_2 = 2;

            PresortQuarter<SEQUENCES_PER_AIV, TOKEN_GROUP_0>(tile, chunks, indices, source);
            PresortQuarter<SEQUENCES_PER_AIV, TOKEN_GROUP_1>(tile, chunks, indices, source);
            PresortQuarter<SEQUENCES_PER_AIV, TOKEN_GROUP_2>(tile, chunks, indices, source);
            if constexpr (SEQUENCES_PER_AIV == 4) {
                PresortQuarter<SEQUENCES_PER_AIV, 3>(tile, chunks, indices, source);
            }
        } else {
            PresortQuarter<SEQUENCES_PER_AIV, 0>(tile, chunks, indices, source[3 * UB_QUEUE_FLOATS]);
        }
    }
    if constexpr (Presort) {
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <int32_t N, bool PresortNext>
static __aicore__ inline void StagePresortedCandidates(
    AscendC::LocalTensor<float> source, AscendC::LocalTensor<float> destination, AscendC::LocalTensor<float> chunks,
    AscendC::LocalTensor<uint32_t> indices, AscendC::LocalTensor<float> spill, AscendC::LocalTensor<float> input,
    __gm__ float* gmSet, int32_t nextChunk, __gm__ float* scoreRing, int32_t* scoreTail, int32_t realTileCount,
    int32_t subblock)
{
    // 当前候选组写入 GM 状态的同时预取下一组，供后续四路归并。
    constexpr int32_t SEQUENCES_PER_AIV = KernelConfig<N>::SEQUENCES_PER_VECTOR_CORE;
    FinishPresortedSequences<SEQUENCES_PER_AIV>(destination, source);
    SubmitVectorStore<7, SEQUENCES_PER_AIV * UB_QUEUE_BLOCKS>(destination, gmSet);

    if (nextChunk >= 0) {
        AdvanceIndices(indices);
        LoadScoreChunk<N, PresortNext>(nextChunk, scoreRing, scoreTail, realTileCount, subblock, input, chunks, indices,
                                       source, spill);
    }
    WaitVectorStore<7>();
}

template <int32_t N, int32_t Sequence>
static __aicore__ inline void MergeAccumulatorSequence(AscendC::LocalTensor<float> source,
                                                       AscendC::LocalTensor<float> tail,
                                                       AscendC::LocalTensor<float> destination,
                                                       AscendC::LocalTensor<float> scratch,
                                                       AscendC::LocalTensor<float> chunks,
                                                       AscendC::LocalTensor<uint32_t> indices, __gm__ float* gmState)
{
    constexpr int32_t SEQUENCES_PER_AIV = KernelConfig<N>::SEQUENCES_PER_VECTOR_CORE;
    SubmitAccumulatorHeadLoad<Sequence, SEQUENCES_PER_AIV>(source, gmState);
    if constexpr (Sequence == 0) {
        FinishEarlyFirstCandidate(tail, destination, scratch);
    } else {
        SortChunk(tail, scratch, chunks[Sequence * TOPK], indices);
    }
    WaitAccumulatorHeadLoad();
    if constexpr (Sequence != 0) {
        WaitVectorStore<7>();
    }
    MergeFour(destination, source);
    SubmitVectorStore<7, UB_QUEUE_BLOCKS>(destination, gmState + Sequence * UB_QUEUE_FLOATS);
}

template <int32_t N>
static __aicore__ inline void MergeFourSequences(
    AscendC::LocalTensor<float> source, AscendC::LocalTensor<float> tail, AscendC::LocalTensor<float> destination,
    AscendC::LocalTensor<float> scratch, AscendC::LocalTensor<float> chunks, AscendC::LocalTensor<uint32_t> indices,
    AscendC::LocalTensor<float> spill, AscendC::LocalTensor<float> input, __gm__ float* gmState, int32_t nextChunk,
    __gm__ float* scoreRing, int32_t* scoreTail, int32_t realTileCount, int32_t subblock)
{
    constexpr int32_t SEQUENCES_PER_AIV = KernelConfig<N>::SEQUENCES_PER_VECTOR_CORE;
    MergeAccumulatorSequence<N, 0>(source, tail, destination, scratch, chunks, indices, gmState);
    MergeAccumulatorSequence<N, 1>(source, tail, destination, scratch, chunks, indices, gmState);
    MergeAccumulatorSequence<N, 2>(source, tail, destination, scratch, chunks, indices, gmState);
    if constexpr (SEQUENCES_PER_AIV == 4) {
        MergeAccumulatorSequence<N, 3>(source, tail, destination, scratch, chunks, indices, gmState);
    }

    if (nextChunk >= 0) {
        AdvanceIndices(indices);
        LoadScoreChunk<N, true>(nextChunk, scoreRing, scoreTail, realTileCount, subblock, input, chunks, indices,
                                source, spill);
    }
    WaitVectorStore<7>();
}

template <int32_t N, int32_t Sequence>
static __aicore__ inline void FinishAccumulatorSequence(
    AscendC::LocalTensor<float> source, AscendC::LocalTensor<float> tail, AscendC::LocalTensor<float> destination,
    AscendC::LocalTensor<float> scratch, AscendC::LocalTensor<float> chunks, AscendC::LocalTensor<uint32_t> indices,
    AscendC::LocalTensor<uint32_t> outputBuffer, __gm__ float* gmState, __gm__ uint32_t* gmOutput, int32_t subblock)
{
    constexpr int32_t SEQUENCES_PER_AIV = KernelConfig<N>::SEQUENCES_PER_VECTOR_CORE;
    SubmitAccumulatorHeadLoad<Sequence, SEQUENCES_PER_AIV>(source, gmState);
    if constexpr (Sequence == 0) {
        FinishEarlyFirstCandidate(tail, destination, scratch);
    } else {
        SortChunk(tail, scratch, chunks[Sequence * TOPK], indices);
    }
    WaitAccumulatorHeadLoad();
    MergeFour(destination, source);
    if constexpr (Sequence != 0) {
        WaitVectorStore<5>();
    }
    ExtractIndices(outputBuffer, destination);
    int32_t globalSequence = subblock * SEQUENCES_PER_AIV + Sequence;
    SubmitVectorStore<5, UB_OUTPUT_BLOCKS>(outputBuffer.template ReinterpretCast<float>(),
                                           reinterpret_cast<__gm__ float*>(gmOutput + globalSequence * TOPK));
}

template <int32_t N>
static __aicore__ inline void FinishFourSequences(
    AscendC::LocalTensor<float> source, AscendC::LocalTensor<float> tail, AscendC::LocalTensor<float> destination,
    AscendC::LocalTensor<float> scratch, AscendC::LocalTensor<float> chunks, AscendC::LocalTensor<uint32_t> indices,
    AscendC::LocalTensor<uint32_t> outputBuffer, __gm__ float* gmState, __gm__ uint32_t* gmOutput, int32_t subblock)
{
    constexpr int32_t SEQUENCES_PER_AIV = KernelConfig<N>::SEQUENCES_PER_VECTOR_CORE;
    FinishAccumulatorSequence<N, 0>(source, tail, destination, scratch, chunks, indices, outputBuffer, gmState,
                                    gmOutput, subblock);
    FinishAccumulatorSequence<N, 1>(source, tail, destination, scratch, chunks, indices, outputBuffer, gmState,
                                    gmOutput, subblock);
    FinishAccumulatorSequence<N, 2>(source, tail, destination, scratch, chunks, indices, outputBuffer, gmState,
                                    gmOutput, subblock);
    if constexpr (SEQUENCES_PER_AIV == 4) {
        FinishAccumulatorSequence<N, 3>(source, tail, destination, scratch, chunks, indices, outputBuffer, gmState,
                                        gmOutput, subblock);
    }
    WaitVectorStore<5>();
}

template <int32_t N>
static __aicore__ inline void RunVectorTopk(__gm__ float* scoreRing, __gm__ uint32_t* gmOutput, __gm__ float* gmState,
                                            int32_t paddedS2, int32_t realS2, int32_t subblock, int32_t loopStart,
                                            const AscendC::LocalTensor<float>& workspace)
{
    constexpr int32_t SEQUENCES_PER_AIV = KernelConfig<N>::SEQUENCES_PER_VECTOR_CORE;
    using Ub = UbLayout<SEQUENCES_PER_AIV>;
    // InitSortOutBuf 可能遗留 value/index 交替 mask，进入长序列流水前恢复全量 mask。
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float>(~0ULL, ~0ULL);
    InitVectorEvents();

    AscendC::LocalTensor<float> input = workspace[Ub::INPUT_BASE / sizeof(float)];
    AscendC::LocalTensor<float> chunks = workspace[Ub::CHUNKS_BASE / sizeof(float)];
    AscendC::LocalTensor<uint32_t> indices =
        workspace[Ub::INDICES_BASE / sizeof(float)].template ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> outputBuffer =
        workspace[Ub::SPILL_BASE / sizeof(float)].template ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<float> spill = workspace[Ub::SPILL_BASE / sizeof(float)];
    AscendC::LocalTensor<float> source = workspace[Ub::SOURCE_HEAD_BASE / sizeof(float)];
    AscendC::LocalTensor<float> tail = workspace[Ub::SOURCE_TAIL_BASE / sizeof(float)];
    AscendC::LocalTensor<float> destination = workspace[Ub::DESTINATION_BASE / sizeof(float)];
    AscendC::LocalTensor<float> scratch = workspace[Ub::DESTINATION_REST_BASE / sizeof(float)];

    InitializeIndices(indices);
    int32_t scoreTail = loopStart;
    const int32_t realTileCount = realS2;
    const int32_t chunkCount = paddedS2 / TOPK;

    // 三组候选落 GM，第四组留在 UB；每轮四路归并后复用同一状态区。
    LoadScoreChunk<N, true>(0, scoreRing, &scoreTail, realTileCount, subblock, input, chunks, indices, source, spill);
    StagePresortedCandidates<N, true>(source, destination, chunks, indices, spill, input, gmState, 1, scoreRing,
                                      &scoreTail, realTileCount, subblock);

    for (int32_t chunk = 1; chunk < chunkCount; chunk += 3) {
        StagePresortedCandidates<N, true>(source, destination, chunks, indices, spill, input,
                                          gmState + SEQUENCES_PER_AIV * UB_QUEUE_FLOATS, chunk + 1, scoreRing,
                                          &scoreTail, realTileCount, subblock);
        StagePresortedCandidates<N, false>(source, destination, chunks, indices, spill, input,
                                           gmState + 2 * SEQUENCES_PER_AIV * UB_QUEUE_FLOATS, chunk + 2, scoreRing,
                                           &scoreTail, realTileCount, subblock);

        const int32_t nextChunk = chunk + 3 < chunkCount ? chunk + 3 : -1;
        if (nextChunk >= 0) {
            MergeFourSequences<N>(source, tail, destination, scratch, chunks, indices, spill, input, gmState, nextChunk,
                                  scoreRing, &scoreTail, realTileCount, subblock);
        } else {
            FinishFourSequences<N>(source, tail, destination, scratch, chunks, indices, outputBuffer, gmState, gmOutput,
                                   subblock);
        }
    }

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
    AscendC::PipeBarrier<PIPE_MTE3>();
}

} // namespace LIV2FastLongSeqVector

namespace LIV2Kernel {
using namespace LIV2Common;
using namespace LIServiceVec;
constexpr uint32_t FAST_BASE_TOPK = 2048;
constexpr uint32_t FAST_VECTOR_UB_BYTES = 192 * 1024;
constexpr uint32_t FAST_SORTED_CACHE_OFFSET = 0;
constexpr uint32_t FAST_OUT_WORK_OFFSET = 64 * 1024;
constexpr uint32_t FAST_TMP_OFFSET = 80 * 1024;
constexpr uint32_t FAST_SORT_OUT_OFFSET = 116 * 1024;
constexpr uint32_t FAST_INDEX_OFFSET = 180 * 1024;
constexpr uint32_t FAST_REDUCE_OUT_OFFSET = 182 * 1024;

template <typename LIT>
class LightningIndexerV2ServiceVectorFast {
public:
    using K_T = typename LIT::keyType;
    using MM1_OUT_T = float;

    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitParams(const LIV2FastCommon::ConstInfo& constInfo,
                                      const LIV2TilingData* __restrict tilingData);
    __aicore__ inline void InitDiagVecGlobalTensor(GlobalTensor<float> scoreGm, GlobalTensor<int32_t> indiceOutGm);
    __aicore__ inline void InitCfvTopkState(GlobalTensor<float> state)
    {
        cfvTopkStateGm_ = state;
    }
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1Offset);
    __aicore__ inline void ProcessTopkOnly(const LIV2FastCommon::RunInfo& info);
    __aicore__ inline bool ProcessSequenceTopk(const LIV2FastCommon::RunInfo& info);

private:
    GlobalTensor<float> scoreGm;
    GlobalTensor<float> cfvTopkStateGm_;
    GlobalTensor<int32_t> indiceOutGm;

    TBuf<TPosition::VECCALC> workspaceBuf_;

    LocalTensor<float> workspace_;
    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;

    int32_t blockId_ = -1;
    int32_t groupInner_ = 0;
    int32_t gSize_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t virTopK = 0;
    LIV2FastCommon::ConstInfo constInfo_;

    event_t eventIdVToMTE2Tmp;
    event_t eventIdVToMTE2Ping;
    event_t eventIdVToMTE2Pong;
    event_t eventIdMTE2ToVPing;
    event_t eventIdMTE2ToVPong;
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVectorFast<LIT>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(workspaceBuf_, FAST_VECTOR_UB_BYTES);
    workspace_ = workspaceBuf_.Get<float>();
    // 块级归并始终按 2048 个候选组织 UB。即使输出 TopK 为 1024，
    // MrgBasicBlock/SparseTopK 仍会读写完整的 2048 容量。
    virTopK = FAST_BASE_TOPK;
    tmpUb_ = workspace_[FAST_TMP_OFFSET / sizeof(float)];
    globalTopkIndice_ = workspace_[FAST_INDEX_OFFSET / sizeof(float)].template ReinterpretCast<int32_t>();
    globalTopkUb_ = workspace_[FAST_SORT_OUT_OFFSET / sizeof(float)];
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVectorFast<LIT>::InitParams(const LIV2FastCommon::ConstInfo& constInfo,
                                                                            const LIV2TilingData* __restrict)
{
    constInfo_ = constInfo;
    gSize_ = constInfo.gSize;
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;
    groupInner_ = 16;
    blockId_ = GetBlockIdx();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVectorFast<LIT>::InitDiagVecGlobalTensor(
    GlobalTensor<float> score, GlobalTensor<int32_t> indiceOut)
{
    scoreGm = score;
    indiceOutGm = indiceOut;
}

template <typename LIT>
__aicore__ inline bool LightningIndexerV2ServiceVectorFast<LIT>::ProcessSequenceTopk(
    const LIV2FastCommon::RunInfo& info)
{
    constexpr int32_t SEQUENCE_TOPK_MIN_VISIBLE_ROWS = 32768;
    // 仅完整 S1 tile 且可见区足够长时启用整段流水，其余形状回退逐 512 行路径。
    bool useSequencePipeline =
        (((gSize_ == 24) && (s1BaseSize_ == 8)) || ((gSize_ == 32) && (s1BaseSize_ == 6))) && (s2BaseSize_ == 512) &&
        (constInfo_.attenMaskFlag) && (constInfo_.sparseCount == 2048) && (constInfo_.realSparseCount == 2048) &&
        (info.s2Idx == 0) &&
        ((info.actS2Size - info.actS1Size + (info.gS1Idx + 1) * s1BaseSize_) >= SEQUENCE_TOPK_MIN_VISIBLE_ROWS) &&
        (info.actS1Size - (info.gS1Idx * s1BaseSize_) >= s1BaseSize_);
    if (!useSequencePipeline) {
        return false;
    }
    int32_t visibleRows = info.actS2Size - info.actS1Size + (info.gS1Idx + 1) * s1BaseSize_;
    int32_t chunks = (visibleRows + 2047) / 2048;
    int32_t paddedChunks = 1 + ((chunks - 1 + 2) / 3) * 3;
    AscendC::PipeBarrier<PIPE_V>();
    __gm__ float* score = const_cast<__gm__ float*>(scoreGm.GetPhyAddr());
    __gm__ uint32_t* output =
        reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ int32_t*>(indiceOutGm.GetPhyAddr())) +
        info.indiceOutOffset + info.gS1Idx * s1BaseSize_ * constInfo_.sparseCount;
    __gm__ float* state = const_cast<__gm__ float*>(cfvTopkStateGm_.GetPhyAddr());
    if (gSize_ == 32) {
        LIV2FastLongSeqVector::RunVectorTopk<32>(score, output, state, paddedChunks * 2048, visibleRows, blockId_ & 1,
                                                 info.loop, workspace_);
    } else {
        LIV2FastLongSeqVector::RunVectorTopk<24>(score, output, state, paddedChunks * 2048, visibleRows, blockId_ & 1,
                                                 info.loop, workspace_);
    }
    return true;
}
template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVectorFast<LIT>::ProcessTopkOnly(const LIV2FastCommon::RunInfo& info)
{
    constexpr int32_t SCORE_N_STORE = 16;
    constexpr int32_t SCORE_COLS_PER_AIV = 8;

    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;
    int32_t cuS1ProcNum = cuBaseS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size - cuBaseS1Idx : s1BaseSize_;
    int32_t subBlockIdx = blockId_ & 1;
    int32_t firstHalfCount = CeilDiv(cuS1ProcNum, 2);
    int32_t cuS1ProcNumPerAiv = subBlockIdx == 0 ? firstHalfCount : cuS1ProcNum / 2;
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx + subBlockIdx * firstHalfCount;

    // MM2 把有效 S1 列居中写入 16 列的 Score 行。两颗 AIV 分别读取低/高 8 列；
    // 对于 s1Base=8 及所有尾块，分给单颗 AIV 的有效列在对应半区内均连续。
    int32_t scoreColOffset = (SCORE_N_STORE - cuS1ProcNum) / 2;
    int32_t firstColumnInHalf = subBlockIdx == 0 ? scoreColOffset : 0;
    int64_t scoreGmOffset = (info.loop % LIV2FastCommon::ConstInfo::SCORE_RING_DEPTH) * s2BaseSize_ * SCORE_N_STORE +
                            subBlockIdx * s2BaseSize_ * SCORE_COLS_PER_AIV;

    if (info.s2Idx == 0) {
        ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
    }

    if (cuS1ProcNumPerAiv <= 0) {
        return;
    }

    // channel-split GM [2, S2, 8] -> UB [S2,8]。每颗 AIV 一次连续搬入自己的半区，避免行主序布局下 512 次 32B stride
    // 搬运 搬入物理 tile 的全部有效行；causal mask 的可见长度随 S1 token 变化，在下面逐 token 截断，不能用本 AIV 第一个
    // token 的长度提前截短搬运
    int32_t scoreCopyRows = info.actualSingleProcessSInnerSize;
    LocalTensor<float> scoreHalfUb = tmpUb_;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Tmp);
    AscendC::DataCopyExtParams scoreCopyParams;
    scoreCopyParams.blockCount = 1;
    scoreCopyParams.blockLen = scoreCopyRows * SCORE_COLS_PER_AIV * sizeof(float);
    scoreCopyParams.srcStride = 0;
    scoreCopyParams.dstStride = 0;
    scoreCopyParams.rsv = 0;
    AscendC::DataCopyPadExtParams<float> scorePadParams{false, 0, 0, 0};
    AscendC::DataCopyPad(scoreHalfUb, scoreGm[scoreGmOffset], scoreCopyParams, scorePadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToVPing);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToVPing);
    // 512x8 -> 8x512。AscendC TransDataTo5HD<float> 在 A2/A3 上对应scatter_vnchwconv_b32；仅保留本 AIV 的 3/4
    // 个有效列，其余列写入 spill MergeSort 会在临时区写出 (virTopK + 512) 组 value/index，所需空间为(2048 + 512) * 2 =
    // 5120 个 FP32；转置结果必须从该范围之后开始
    constexpr int32_t MERGE_TMP_ELEMENTS = (FAST_BASE_TOPK + 512) * 2;
    LocalTensor<float> selectedScoreUb = tmpUb_[MERGE_TMP_ELEMENTS];
    LocalTensor<float> spillScoreUb = selectedScoreUb[cuS1ProcNumPerAiv * s2BaseSize_];
    uint64_t srcAddrList[16];
    uint64_t dstAddrList[16];
    for (int32_t row = 0; row < 16; ++row) {
        srcAddrList[row] = reinterpret_cast<uint64_t>(scoreHalfUb[row * SCORE_COLS_PER_AIV].GetPhyAddr());
    }
    int32_t spillColumn = 0;
    for (int32_t column = 0; column < SCORE_COLS_PER_AIV; ++column) {
        LocalTensor<float> columnTensor;
        if (column >= firstColumnInHalf && column < firstColumnInHalf + cuS1ProcNumPerAiv) {
            columnTensor = selectedScoreUb[(column - firstColumnInHalf) * s2BaseSize_];
        } else {
            columnTensor = spillScoreUb[spillColumn * s2BaseSize_];
            ++spillColumn;
        }
        dstAddrList[2 * column] = reinterpret_cast<uint64_t>(columnTensor.GetPhyAddr());
        dstAddrList[2 * column + 1] = reinterpret_cast<uint64_t>(columnTensor[8].GetPhyAddr());
    }
    AscendC::TransDataTo5HDParams transposeParams(false, false, 32, 2, 16);
    AscendC::TransDataTo5HD<float>(dstAddrList, srcAddrList, transposeParams);
    AscendC::PipeBarrier<PIPE_V>();

    constexpr int32_t CACHE_TILE_CAPACITY = 4;
    constexpr int32_t MAX_SEQUENCES_PER_AIV = 4;
    constexpr int32_t SORTED_BLOCK_ELEMENTS = 512 * 2;
    LocalTensor<float> reduceOutBuff = workspace_[FAST_REDUCE_OUT_OFFSET / sizeof(float)];
    int32_t cuS2LenVecAlign = s2BaseSize_; // Score tile 固定 512，尾部以 -inf 补齐。
    LocalTensor<float> sortWorkUb = workspace_[FAST_OUT_WORK_OFFSET / sizeof(float)];
    LocalTensor<float> sortedCacheUb = workspace_[FAST_SORTED_CACHE_OFFSET / sizeof(float)];
    LocalTensor<float> sortScratchUb = sortWorkUb;
    bool activeSequence[MAX_SEQUENCES_PER_AIV] = {false, false, false, false};
    bool endSequence[MAX_SEQUENCES_PER_AIV] = {false, false, false, false};
    int32_t cacheTile = info.s2Idx % CACHE_TILE_CAPACITY;

    for (int32_t innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; ++innerS1Idx) {
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        int32_t cuRealAcSeq = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            // 与原 ProcessVec 保持一致：当前 token 自身可见，因此首 token 也要 +1
            cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1Idx) + 1;
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.realSparseCount);
            continue;
        }
        if (cuS2Len <= 0) {
            continue;
        }
        bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
        activeSequence[innerS1Idx] = true;
        endSequence[innerS1Idx] = isS2End;
        LocalTensor<float> sortScoreUb = reduceOutBuff;
        LocalTensor<int32_t> sortIndiceUb = reduceOutBuff[cuS2LenVecAlign].template ReinterpretCast<int32_t>();
        if (cuS2Len < cuS2LenVecAlign) {
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, cuS2LenVecAlign);
            Duplicate(sortIndiceUb, -1, cuS2LenVecAlign);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Adds(sortScoreUb, selectedScoreUb[innerS1Idx * s2BaseSize_], 0.0f, cuS2Len);
        Adds(sortIndiceUb, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
        AscendC::PipeBarrier<PIPE_V>();
        // 每个 512 点块只做基础排序，先缓存；每 4 个块再更新一次 global TopK
        Sort<float, true>(
            sortedCacheUb[innerS1Idx * CACHE_TILE_CAPACITY * SORTED_BLOCK_ELEMENTS + cacheTile * SORTED_BLOCK_ELEMENTS],
            sortScoreUb, sortIndiceUb.template ReinterpretCast<uint32_t>(), sortScratchUb, cuS2LenVecAlign / 32);
        AscendC::PipeBarrier<PIPE_V>();
    }

    int32_t cacheTileCount = cacheTile + 1;
    int32_t cacheGroupFirst = info.s2Idx - cacheTile;
    for (int32_t innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; ++innerS1Idx) {
        if (!activeSequence[innerS1Idx] || (cacheTile != CACHE_TILE_CAPACITY - 1 && !endSequence[innerS1Idx])) {
            continue;
        }

        LocalTensor<float> cachedBlocks = sortedCacheUb[innerS1Idx * CACHE_TILE_CAPACITY * SORTED_BLOCK_ELEMENTS];
        LocalTensor<float> globalTopk = globalTopkUb_[innerS1Idx * virTopK * 2];
        if (cacheGroupFirst == 0) {
            MrgBasicBlock(globalTopk, cachedBlocks, cacheTileCount, s2BaseSize_);
        } else {
            MrgBasicBlock(sortWorkUb, cachedBlocks, cacheTileCount, s2BaseSize_);
            AscendC::PipeBarrier<PIPE_V>();
            SparseTopK(globalTopk, sortWorkUb, scoreHalfUb, FAST_BASE_TOPK, cacheTileCount * s2BaseSize_);
        }
        AscendC::PipeBarrier<PIPE_V>();

        if (endSequence[innerS1Idx]) {
            int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
            // 复用固定 UB 工作区的前半段，存放 Extract 输出
            LocalTensor<float> outValueUb = sortWorkUb;
            LocalTensor<uint32_t> outIdxUb = outValueUb[virTopK].template ReinterpretCast<uint32_t>();
            Extract(outValueUb, outIdxUb, globalTopk, virTopK / 32);
            LocalTensor<int32_t> idxUb = outValueUb[virTopK].template ReinterpretCast<int32_t>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            LIServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount], idxUb,
                                  constInfo_.sparseCount);
            // 下一 token 的归并会覆盖该区域，复用前须等待 MTE3 读取完成
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Tmp);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVectorFast<LIT>::AllocEventID()
{
    eventIdVToMTE2Tmp = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
    eventIdVToMTE2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
    eventIdVToMTE2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
    eventIdMTE2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
    eventIdMTE2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Tmp);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Ping);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Pong);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVectorFast<LIT>::FreeEventID()
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Tmp);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Ping);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Pong);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Tmp);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Ping);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(eventIdVToMTE2Pong);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToVPing);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToVPong);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVectorFast<LIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    GlobalTensor<int32_t> invalidOut = indiceOutGm[invalidS1Offset];
    AscendC::InitGlobalMemory(invalidOut, constInfo_.realSparseCount, constInfo_.INVALID_IDX);
}

} // namespace LIV2Kernel
#endif
