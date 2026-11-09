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
 * \file quant_lightning_indexer_v2_fast_service_vector_arch22.h
 * \brief Fast template TopK service for AIV on arch22 (int8, TND query).
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_VECTOR_ARCH22_H
#define QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_VECTOR_ARCH22_H

#include "quant_lightning_indexer_v2_fast_service_cube_arch22.h"
#include "quant_lightning_indexer_v2_vector.h"

namespace QLIV2Fast {
static constexpr int QUEUE_WORDS = TOP_K * SCORE_INDEX_PAIR_WORDS;
static constexpr int QUEUE_BYTES = QUEUE_WORDS * sizeof(float);
static constexpr int SCORE_INPUT_BYTES = SCORE_ROWS * SCORE_CHANNEL_COLS * sizeof(float);
static constexpr int VALUE_BYTES = SCORE_INPUT_BYTES;
static constexpr int INDEX_BYTES = SCORE_ROWS * sizeof(uint32_t);
static constexpr int TOP_BYTES = QUEUE_BYTES;
// 排序/归并工作区按 2 倍队列容量预留；kScale 暂存（SCALE_*_SCRATCH_OFFSET）与
// GatherMask 掩码（spill）嵌在本区前段，与排序写入分时复用
static constexpr int WORK_BYTES = 2 * QUEUE_BYTES;
static constexpr int PHASE_BYTES = TOP_BYTES + WORK_BYTES;
// kScale 每行 1 个 half，对应 GetRowGmOffset 的 rowUnitElements 参数位
static constexpr uint32_t SCALE_ROW_ELEMENTS = 1;

struct VectorConstants {
    static constexpr int SORT_UNIT_ELEMENTS = 32;
    static constexpr int MERGE_SOURCE_COUNT = 4;
    static constexpr int SORT_RECORD_WORDS = SCORE_INDEX_PAIR_WORDS;
    static constexpr int VECTOR_REPEAT_ELEMENTS = QLIV2ServiceVec::B32_VEC_ELM_NUM;
    static constexpr int CANDIDATE_ALIGNMENT = 8;
    static constexpr int INITIAL_TOPK_TILES = TOP_K / SCORE_ROWS;
    static constexpr int PREFETCH_GROUP_TILES = 16;
    static constexpr int TRANSPOSE_TENSOR_COUNT = SCORE_STORAGE_COLS;
    static constexpr int TRANSPOSE_SLICES_PER_COLUMN = SCORE_STORAGE_COLS / SCORE_CHANNEL_COLS;
    static constexpr int MASK_WORD_COUNT = 2;
    static constexpr uint8_t ALL_MERGE_SOURCES = (1U << MERGE_SOURCE_COUNT) - 1;
    static constexpr uint8_t TWO_MERGE_SOURCES = 0x03;
    static constexpr uint32_t SCALE_FLOAT_SCRATCH_OFFSET = VALUE_BYTES + INDEX_BYTES;
    static constexpr uint32_t SCALE_HALF_SCRATCH_OFFSET = SCALE_FLOAT_SCRATCH_OFFSET + SCORE_ROWS * sizeof(float);
};

template <int Sequences>
struct UbLayout {
    static_assert(Sequences == 6 || Sequences == 8);
    static constexpr int CANDIDATE_BASE = 0;
    static constexpr int CANDIDATE_BYTES = Sequences * QUEUE_BYTES;
    static constexpr int INPUT_BASE = CANDIDATE_BASE + CANDIDATE_BYTES;
    static constexpr int PHASE_BASE = INPUT_BASE + SCORE_INPUT_BYTES;
    static constexpr int VALUE_BASE = PHASE_BASE;
    static constexpr int INDEX_BASE = VALUE_BASE + VALUE_BYTES;
    static constexpr int SPILL_BASE = INDEX_BASE + INDEX_BYTES;
    static constexpr int TOP_BASE = PHASE_BASE;
    static constexpr int WORK_BASE = TOP_BASE + TOP_BYTES;
    static constexpr int BYTES = PHASE_BASE + PHASE_BYTES;
};

static_assert(UbLayout<8>::BYTES == KiB(192));
static_assert(UbLayout<6>::BYTES == KiB(160));

template <typename T>
static __aicore__ inline void CopyGmToUb(const LocalTensor<T>& destination, const GlobalTensor<T>& source,
                                         uint32_t elementCount)
{
    DataCopy(destination, source, elementCount);
}

template <typename T>
static __aicore__ inline void CopyUbToGm(const GlobalTensor<T>& destination, const LocalTensor<T>& source,
                                         uint32_t elementCount)
{
    DataCopy(destination, source, elementCount);
}

static __aicore__ inline void VectorToScalar()
{
    SetFlag<HardEvent::V_S>(EVENT_ID4);
    WaitFlag<HardEvent::V_S>(EVENT_ID4);
}

static __aicore__ inline void BuildIndices(const LocalTensor<uint32_t>& indices, uint32_t firstIndex)
{
    constexpr uint32_t indexSeedCount = VectorConstants::VECTOR_REPEAT_ELEMENTS;
    // 索引区与 merge 临时区复用同一块 UB，Scalar 写首个完整 repeat 前先结束其上的 Vector 使用
    SetFlag<HardEvent::V_S>(EVENT_ID5);
    WaitFlag<HardEvent::V_S>(EVENT_ID5);
    for (uint32_t index = 0; index < indexSeedCount; ++index) {
        indices.SetValue(index, firstIndex + index);
    }
    SetFlag<HardEvent::S_V>(EVENT_ID6);
    WaitFlag<HardEvent::S_V>(EVENT_ID6);

    // 一次 64 元素 int32 Adds 即可扩展完整种子：各目标块读取不变的 [0, 64) 种子并叠加
    // 本块偏移，得到 firstIndex + [0, 512)，无需掩码逐 lane 生成
    LocalTensor<int32_t> indexBits = indices.template ReinterpretCast<int32_t>();
    for (uint32_t block = 1; block < SCORE_ROWS / indexSeedCount; ++block) {
        Adds(indexBits[block * indexSeedCount], indexBits, static_cast<int32_t>(block * indexSeedCount),
             static_cast<int32_t>(indexSeedCount));
    }
    PipeBarrier<PIPE_V>();
}

static __aicore__ inline void InitIndices(const LocalTensor<uint32_t>& indices)
{
    BuildIndices(indices, 0);
}

static __aicore__ inline void RebuildIndices(const LocalTensor<uint32_t>& indices, uint32_t firstIndex)
{
    BuildIndices(indices, firstIndex);
}

static __aicore__ inline void AdvanceIndices(const LocalTensor<uint32_t>& indices)
{
    LocalTensor<int32_t> indicesTensor = indices.template ReinterpretCast<int32_t>();
    Adds(indicesTensor, indicesTensor, static_cast<int32_t>(SCORE_ROWS), SCORE_ROWS);
    PipeBarrier<PIPE_V>();
}

static __aicore__ inline void TransposeScore512x8(
    const LocalTensor<float> (&destinationList)[VectorConstants::TRANSPOSE_TENSOR_COUNT],
    const LocalTensor<float> (&sourceList)[VectorConstants::TRANSPOSE_TENSOR_COUNT])
{
    // 512×8 score 转 5HD：repeatTimes 为 512/16 个 repeat，每 repeat 处理 16 行；
    // src/dstRepStride 单位为 32B 块（源每行 8 个 float 恰为 1 块，目的每 repeat 16 个元素为 2 块）
    TransDataTo5HDParams transposeParams;
    transposeParams.repeatTimes = SCORE_ROWS / SCORE_STORAGE_COLS;
    transposeParams.dstRepStride = VectorConstants::TRANSPOSE_SLICES_PER_COLUMN;
    transposeParams.srcRepStride = SCORE_STORAGE_COLS;
    TransDataTo5HD<float>(destinationList, sourceList, transposeParams);
}

static __aicore__ inline void FinishSorted32(const LocalTensor<float>& destination, const LocalTensor<float>& scratch,
                                             const LocalTensor<float>& sorted32)
{
    LocalTensor<float> destinationTensor = destination;
    LocalTensor<float> scratchTensor = scratch;
    LocalTensor<float> sorted32Tensor = sorted32;

    constexpr int firstMergeLength = VectorConstants::SORT_UNIT_ELEMENTS;
    constexpr int secondMergeLength = firstMergeLength * VectorConstants::MERGE_SOURCE_COUNT;
    constexpr int finalMergeLength = secondMergeLength * VectorConstants::MERGE_SOURCE_COUNT;

    MrgSort4Info mergeInfo;
    mergeInfo.elementLengths[0] = firstMergeLength;
    mergeInfo.elementLengths[1] = firstMergeLength;
    mergeInfo.elementLengths[2] = firstMergeLength;
    mergeInfo.elementLengths[3] = firstMergeLength;
    mergeInfo.repeatTimes = TOP_K / (firstMergeLength * VectorConstants::MERGE_SOURCE_COUNT);
    mergeInfo.ifExhaustedSuspension = false;
    mergeInfo.validBit = VectorConstants::ALL_MERGE_SOURCES;
    MrgSortSrcList<float> sourceList;
    sourceList.src1 = sorted32Tensor;
    sourceList.src2 = sorted32Tensor[firstMergeLength * VectorConstants::SORT_RECORD_WORDS];
    sourceList.src3 = sorted32Tensor[2 * firstMergeLength * VectorConstants::SORT_RECORD_WORDS];
    sourceList.src4 = sorted32Tensor[3 * firstMergeLength * VectorConstants::SORT_RECORD_WORDS];
    MrgSort<float>(destinationTensor, sourceList, mergeInfo);
    PipeBarrier<PIPE_V>();

    mergeInfo.elementLengths[0] = secondMergeLength;
    mergeInfo.elementLengths[1] = secondMergeLength;
    mergeInfo.elementLengths[2] = secondMergeLength;
    mergeInfo.elementLengths[3] = secondMergeLength;
    mergeInfo.repeatTimes = TOP_K / (secondMergeLength * VectorConstants::MERGE_SOURCE_COUNT);
    sourceList.src1 = destinationTensor;
    sourceList.src2 = destinationTensor[secondMergeLength * VectorConstants::SORT_RECORD_WORDS];
    sourceList.src3 = destinationTensor[2 * secondMergeLength * VectorConstants::SORT_RECORD_WORDS];
    sourceList.src4 = destinationTensor[3 * secondMergeLength * VectorConstants::SORT_RECORD_WORDS];
    MrgSort<float>(scratchTensor, sourceList, mergeInfo);
    PipeBarrier<PIPE_V>();

    mergeInfo.elementLengths[0] = finalMergeLength;
    mergeInfo.elementLengths[1] = finalMergeLength;
    mergeInfo.elementLengths[2] = finalMergeLength;
    mergeInfo.elementLengths[3] = finalMergeLength;
    mergeInfo.repeatTimes = TOP_K / (finalMergeLength * VectorConstants::MERGE_SOURCE_COUNT);
    sourceList.src1 = scratchTensor;
    sourceList.src2 = scratchTensor[finalMergeLength * VectorConstants::SORT_RECORD_WORDS];
    sourceList.src3 = scratchTensor[2 * finalMergeLength * VectorConstants::SORT_RECORD_WORDS];
    sourceList.src4 = scratchTensor[3 * finalMergeLength * VectorConstants::SORT_RECORD_WORDS];
    MrgSort<float>(destinationTensor, sourceList, mergeInfo);
    PipeBarrier<PIPE_V>();
}

static __aicore__ inline void SortChunk(const LocalTensor<float>& destination, const LocalTensor<float>& scratch,
                                        const LocalTensor<float>& values, const LocalTensor<uint32_t>& indices)
{
    LocalTensor<float> scratchTensor = scratch;
    LocalTensor<float> valueTensor = values;
    LocalTensor<uint32_t> indexTensor = indices;
    Sort32(scratchTensor, valueTensor, indexTensor, TOP_K / VectorConstants::SORT_UNIT_ELEMENTS);
    PipeBarrier<PIPE_V>();
    FinishSorted32(destination, scratch, scratch);
}

static __aicore__ inline void MergeTop(const LocalTensor<float>& destination, const LocalTensor<float>& top,
                                       const LocalTensor<float>& candidate)
{
    LocalTensor<float> destinationTensor = destination;
    LocalTensor<float> topTensor = top;
    LocalTensor<float> candidateTensor = candidate;
    MrgSort4Info mergeInfo;
    // 仅 src1/src2 两个源参与归并：elementLengths 后两项与 src3/src4 为 API 要求的
    // 占位，validBit 只使能前两源
    mergeInfo.elementLengths[0] = TOP_K;
    mergeInfo.elementLengths[1] = TOP_K;
    mergeInfo.elementLengths[2] = 0;
    mergeInfo.elementLengths[3] = 0;
    mergeInfo.repeatTimes = 1;
    mergeInfo.ifExhaustedSuspension = true;
    mergeInfo.validBit = VectorConstants::TWO_MERGE_SOURCES;
    MrgSortSrcList<float> sourceList;
    sourceList.src1 = topTensor;
    sourceList.src2 = candidateTensor;
    sourceList.src3 = candidateTensor;
    sourceList.src4 = candidateTensor;
    MrgSort<float>(destinationTensor, sourceList, mergeInfo);
    PipeBarrier<PIPE_V>();
}

static __aicore__ inline LocalTensor<float> GetCandidateQueue(const LocalTensor<float>& candidates, int sequence)
{
    LocalTensor<float> queue = candidates[sequence * QUEUE_WORDS];
    queue.SetSize(QUEUE_WORDS);
    return queue;
}

static __aicore__ inline LocalTensor<float> GetCandidateValues(const LocalTensor<float>& candidates, int sequence)
{
    LocalTensor<float> values = GetCandidateQueue(candidates, sequence);
    values.SetSize(TOP_K);
    return values;
}

static __aicore__ inline LocalTensor<uint32_t> GetCandidateIndices(const LocalTensor<float>& candidates, int sequence)
{
    LocalTensor<uint32_t> indices = GetCandidateQueue(candidates, sequence)[TOP_K].template ReinterpretCast<uint32_t>();
    indices.SetSize(TOP_K);
    return indices;
}

static __aicore__ inline void SubmitTopLoad(const LocalTensor<float>& destination, const GlobalTensor<float>& source)
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    CopyGmToUb(destination, source, QUEUE_WORDS);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
}

static __aicore__ inline void WaitTopLoad()
{
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
}

static __aicore__ inline void StoreQueue(const GlobalTensor<float>& destination, const LocalTensor<float>& source)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    CopyUbToGm(destination, source, QUEUE_WORDS);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

static __aicore__ inline float ReadThreshold(const LocalTensor<float>& queue, int topK)
{
    PipeBarrier<PIPE_V>();
    VectorToScalar();
    return queue.GetValue((topK - 1) * VectorConstants::SORT_RECORD_WORDS);
}

static __aicore__ inline void ClearCandidate(const LocalTensor<float>& values)
{
    LocalTensor<float> valueTensor = values;
    Duplicate(valueTensor, -__builtin_inff(), TOP_K);
}

template <int Sequences>
static __aicore__ inline void ClearAllCandidates(const LocalTensor<float>& candidates)
{
    for (int sequence = 0; sequence < Sequences; ++sequence) {
        ClearCandidate(GetCandidateValues(candidates, sequence));
    }
    PipeBarrier<PIPE_V>();
}

static __aicore__ inline int CompactCandidates(const LocalTensor<float>& destinationValues,
                                               const LocalTensor<uint32_t>& destinationIndices,
                                               const LocalTensor<float>& values, const LocalTensor<uint32_t>& indices,
                                               const LocalTensor<float>& maskStorage, float threshold)
{
    LocalTensor<uint8_t> maskTensor = maskStorage.template ReinterpretCast<uint8_t>();
    LocalTensor<float> valueTensor = values;
    LocalTensor<uint32_t> indexTensor = indices;
    LocalTensor<float> destinationValueTensor = destinationValues;
    LocalTensor<uint32_t> destinationIndexTensor = destinationIndices;

    CompareScalar(maskTensor, valueTensor, threshold, CMPMODE::GE, SCORE_ROWS);
    PipeBarrier<PIPE_V>();

    // 单次 repeat 完成整块筛选：源与掩码按 32B 块连续读取
    GatherMaskParams gatherParams;
    gatherParams.repeatTimes = 1;
    gatherParams.src0BlockStride = 1;
    gatherParams.src0RepeatStride = HardwareConstants::VECTOR_BLOCKS_PER_REPEAT;
    gatherParams.src1RepeatStride = 1;
    uint64_t selectedCount = 0;
    GatherMask(destinationValueTensor, valueTensor, maskTensor.template ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(SCORE_ROWS), gatherParams, selectedCount);
    PipeBarrier<PIPE_V>();
    GatherMask(destinationIndexTensor, indexTensor, maskTensor.template ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(SCORE_ROWS), gatherParams, selectedCount);
    PipeBarrier<PIPE_V>();
    return static_cast<int>(selectedCount);
}

static __aicore__ inline void SubmitScoreTile(const GlobalTensor<float>& scoreRing, int& scoreTail, int subblock,
                                              const LocalTensor<float>& scoreInput)
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if (scoreTail % SCORE_SYNC_PERIOD == 0) {
        CrossCoreWaitFlag<CrossCoreMessageId::MODE, PIPE_S>(CrossCoreMessageId::SCORE_READY);
    }
    CopyGmToUb(
        scoreInput,
        scoreRing[(scoreTail % SCORE_RING_DEPTH) * SCORE_TILE_ELEMENTS + subblock * SCORE_ROWS * SCORE_CHANNEL_COLS],
        SCORE_INPUT_BYTES / sizeof(float));
    ++scoreTail;
    if (scoreTail % SCORE_SYNC_PERIOD == 0) {
        NotifyScoreFree();
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

static __aicore__ inline void FinishScoreTile(
    const LocalTensor<float> (&destinationList)[VectorConstants::TRANSPOSE_TENSOR_COUNT],
    const LocalTensor<float> (&sourceList)[VectorConstants::TRANSPOSE_TENSOR_COUNT])
{
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    TransposeScore512x8(destinationList, sourceList);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

// 用掩码把 [validCount, totalCount) 填充为 value，保留 [0, validCount) 前缀
// PostprocessScoreTile 的 -inf 补齐与 WriteOutputs 的无效索引填充共用
template <typename T>
static __aicore__ inline void FillInvalidTail(const LocalTensor<T>& destination, int validCount, int totalCount,
                                              T value)
{
    constexpr int repeatElements = VectorConstants::VECTOR_REPEAT_ELEMENTS;
    const int alignedStart = validCount & ~(repeatElements - 1);
    const int firstLane = validCount - alignedStart;
    const int remaining = totalCount - alignedStart;
    const int firstCount = remaining < repeatElements ? remaining : repeatElements;
    // 标量 uint64_t 重载是 count-mask 语义；这里必须用数组重载传入位掩码，
    // 才能只替换首个 repeat 的尾部而不改动有效前缀。repeatTimes=1、目的 repeat
    // 步长 1，即只写目的端首个 64 元素 repeat 的被掩码部分
    uint64_t firstMask[VectorConstants::MASK_WORD_COUNT] = {
        (static_cast<uint64_t>(-1) << firstLane) & (static_cast<uint64_t>(-1) >> (repeatElements - firstCount)),
        0,
    };
    Duplicate(destination[alignedStart], value, firstMask, 1, 1, HardwareConstants::VECTOR_BLOCKS_PER_REPEAT);
    const int remainingStart = alignedStart + repeatElements;
    if (remaining > repeatElements) {
        Duplicate(destination[remainingStart], value, remaining - repeatElements);
    }
}

template <int N>
static __aicore__ inline void PostprocessScoreTile(const LocalTensor<float>& values, int tile,
                                                   const VectorPostContext& context)
{
    constexpr int sequences = KernelConfig<N>::SEQUENCES_PER_AIV;
    using Layout = UbLayout<sequences>;
    if (static_cast<int>(context.subblock) * sequences >= static_cast<int>(context.validSequences)) {
        for (int sequence = 0; sequence < sequences; ++sequence) {
            LocalTensor<float> row(TPosition::VECCALC, Layout::VALUE_BASE + sequence * SCORE_ROWS * sizeof(float),
                                   SCORE_ROWS);
            Duplicate(row.template ReinterpretCast<int32_t>(), QLIV2ServiceVec::NEG_INF, SCORE_ROWS);
        }
        PipeBarrier<PIPE_V>();
        return;
    }
    constexpr uint32_t scaleFloatAddr = Layout::PHASE_BASE + VectorConstants::SCALE_FLOAT_SCRATCH_OFFSET;
    constexpr uint32_t scaleHalfAddr = Layout::PHASE_BASE + VectorConstants::SCALE_HALF_SCRATCH_OFFSET;
    LocalTensor<float> scaleFloat(TPosition::VECCALC, scaleFloatAddr, SCORE_ROWS);
    LocalTensor<half> scaleHalf(TPosition::VECCALC, scaleHalfAddr, SCORE_ROWS);
    const GlobalTensor<half>& scaleGm = context.keyScale;
    const GlobalTensor<int32_t>& blockTableGm = context.blockTable;

    const int tileStart = tile * SCORE_ROWS;
    int scaleLen = QLIV2Common::Max(0, QLIV2Common::Min(static_cast<int>(context.actualS2) - tileStart, SCORE_ROWS));
    int copied = 0;
    // 不启用补齐：左/右补零数量与填充值参数在未启用时无效，恒取 0
    DataCopyPadExtParams<half> pad{false, 0, 0, 0};
    while (copied < scaleLen) {
        const int logicalRow = tileStart + copied;
        int rows = scaleLen - copied;
        const uint64_t sourceOffset =
            GetRowGmOffset(blockTableGm, context.batch, context.maxBlocksPerBatch, context.blockSize, context.kHeadNum,
                           context.n2Index, context.scalePrefix, context.kScaleStride, SCALE_ROW_ELEMENTS,
                           context.pageAttention, logicalRow, rows);
        // 逐行搬 kScale：blockCount 为行数，blockLen 为每行 2 字节（1 个 half），
        // srcStride 单位为字节，是行间跳过其余 kHeadNum-1 个 head 的距离；kHeadNum==1 时整段连续
        DataCopyExtParams copy;
        copy.blockCount = static_cast<uint16_t>(rows);
        copy.blockLen = sizeof(half);
        copy.srcStride = static_cast<uint32_t>((context.kHeadNum - 1) * sizeof(half));
        copy.dstStride = 0;
        copy.rsv = 0;
        if (context.kHeadNum == 1) {
            copy.blockCount = 1;
            copy.blockLen = static_cast<uint32_t>(rows * sizeof(half));
            copy.srcStride = 0;
        }
        const int copyBytes = static_cast<int>(rows * sizeof(half));
        const uint64_t sourceBytes = sourceOffset * sizeof(half);
        const bool sourceAligned = (sourceBytes & (HardwareConstants::VECTOR_BLOCK_BYTES - 1)) == 0;
        if ((context.kHeadNum == 1) && sourceAligned &&
            ((copyBytes & (HardwareConstants::VECTOR_BLOCK_BYTES - 1)) == 0)) {
            CopyGmToUb(scaleHalf[copied], context.keyScale[sourceOffset], static_cast<uint32_t>(rows));
        } else {
            DataCopyPad(scaleHalf[copied], scaleGm[sourceOffset], copy, pad);
        }
        copied += rows;
    }
    if (scaleLen > 0) {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        Cast(scaleFloat, scaleHalf, RoundMode::CAST_NONE, scaleLen);
        PipeBarrier<PIPE_V>();
    }

    for (int sequence = 0; sequence < sequences; ++sequence) {
        int valid = scaleLen;
        const int localSequence = static_cast<int>(context.subblock) * sequences + sequence;
        if (localSequence >= static_cast<int>(context.validSequences)) {
            valid = 0;
        } else if (context.causal) {
            const int s1Index = static_cast<int>(context.s1TileStart) + localSequence;
            int visible = static_cast<int>(context.actualS2) - static_cast<int>(context.actualS1) + s1Index + 1;
            int causalValid = visible - tileStart;
            valid = QLIV2Common::Max(0, QLIV2Common::Min(valid, causalValid));
        }
        LocalTensor<float> row(TPosition::VECCALC, Layout::VALUE_BASE + sequence * SCORE_ROWS * sizeof(float),
                               SCORE_ROWS);
        if (valid > 0) {
            Mul(row, row, scaleFloat, valid);
            PipeBarrier<PIPE_V>();
        }
        if (valid < SCORE_ROWS) {
            LocalTensor<int32_t> rowBits = row.template ReinterpretCast<int32_t>();
            FillInvalidTail(rowBits, valid, SCORE_ROWS, QLIV2ServiceVec::NEG_INF);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <int Sequences>
static __aicore__ inline void AppendInitialTile(const LocalTensor<float>& candidates, const LocalTensor<float>& values,
                                                const LocalTensor<uint32_t>& indices, int tile)
{
    for (int sequence = 0; sequence < Sequences; ++sequence) {
        LocalTensor<float> destinationValues = GetCandidateValues(candidates, sequence)[tile * SCORE_ROWS];
        LocalTensor<float> sourceValues = values[sequence * SCORE_ROWS];
        LocalTensor<float> destinationValueTensor = destinationValues;
        LocalTensor<float> sourceValueTensor = sourceValues;
        Adds(destinationValueTensor, sourceValueTensor, 0.0f, SCORE_ROWS);

        LocalTensor<int32_t> destinationIndices =
            GetCandidateIndices(candidates, sequence)[tile * SCORE_ROWS].template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> sourceIndices = indices.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> destinationIndexTensor = destinationIndices;
        LocalTensor<int32_t> sourceIndexTensor = sourceIndices;
        Adds(destinationIndexTensor, sourceIndexTensor, static_cast<int32_t>(0), SCORE_ROWS);
    }
    PipeBarrier<PIPE_V>();
}

template <int Sequences>
static __aicore__ inline void CompactLoadedTile(int tile, int tileCount, const LocalTensor<float>& candidates,
                                                const LocalTensor<float>& values, const LocalTensor<uint32_t>& indices,
                                                const LocalTensor<float>& spill, float (&thresholds)[Sequences],
                                                int (&storedElements)[Sequences])
{
    constexpr int maskWords = (SCORE_ROWS / HardwareConstants::BITS_PER_BYTE) / sizeof(float);
    for (int sequence = 0; sequence < Sequences; ++sequence) {
        const int selected = CompactCandidates(GetCandidateValues(candidates, sequence)[storedElements[sequence]],
                                               GetCandidateIndices(candidates, sequence)[storedElements[sequence]],
                                               values[sequence * SCORE_ROWS], indices, spill[sequence * maskWords],
                                               thresholds[sequence]);
        storedElements[sequence] += QLIV2Common::Align(selected, VectorConstants::CANDIDATE_ALIGNMENT);
    }
    if (tile + 1 < tileCount) {
        AdvanceIndices(indices);
    }
}

template <int Sequences>
static __aicore__ inline void FlushCandidate(int sequence, const LocalTensor<float>& candidates,
                                             const LocalTensor<float>& top, const LocalTensor<float>& work,
                                             const GlobalTensor<float>& topkWorkspace, float (&thresholds)[Sequences],
                                             int topK)
{
    GlobalTensor<float> state = topkWorkspace[sequence * QUEUE_WORDS];
    SubmitTopLoad(top, state);
    SortChunk(GetCandidateQueue(candidates, sequence), work, GetCandidateValues(candidates, sequence),
              GetCandidateIndices(candidates, sequence));
    WaitTopLoad();
    MergeTop(work, top, GetCandidateQueue(candidates, sequence));
    thresholds[sequence] = ReadThreshold(work, topK);
    StoreQueue(state, work);
    ClearCandidate(GetCandidateValues(candidates, sequence));
    PipeBarrier<PIPE_V>();
}

template <int Sequences>
static __aicore__ inline bool PrepareCandidateCapacity(const LocalTensor<float>& candidates,
                                                       const LocalTensor<float>& top, const LocalTensor<float>& work,
                                                       const GlobalTensor<float>& topkWorkspace,
                                                       float (&thresholds)[Sequences], int (&storedElements)[Sequences],
                                                       int topK)
{
    bool changedPhase = false;
    for (int sequence = 0; sequence < Sequences; ++sequence) {
        if (storedElements[sequence] > TOP_K - SCORE_ROWS) {
            FlushCandidate<Sequences>(sequence, candidates, top, work, topkWorkspace, thresholds, topK);
            storedElements[sequence] = 0;
            changedPhase = true;
        }
    }
    return changedPhase;
}

// 一个 Score tile 的前半段处理：容量不足时先换出 candidate 并重建索引，
// 再完成转置与 kScale/因果掩码后处理。压缩写回 candidate 由 CompactLoadedTile 负责，
// 调用方可在两者之间插入下一次 Score tile 的预取
template <int N>
static __aicore__ inline void ProcessScoreTile(
    const LocalTensor<float>& values, const LocalTensor<float>& candidates, const LocalTensor<float>& top,
    const LocalTensor<float>& work, const LocalTensor<uint32_t>& indices, const GlobalTensor<float>& topkWorkspace,
    const LocalTensor<float> (&destinationList)[VectorConstants::TRANSPOSE_TENSOR_COUNT],
    const LocalTensor<float> (&sourceList)[VectorConstants::TRANSPOSE_TENSOR_COUNT], int tile, int topK,
    const VectorPostContext& postContext, float (&thresholds)[KernelConfig<N>::SEQUENCES_PER_AIV],
    int (&storedElements)[KernelConfig<N>::SEQUENCES_PER_AIV])
{
    constexpr int sequences = KernelConfig<N>::SEQUENCES_PER_AIV;
    if (PrepareCandidateCapacity<sequences>(candidates, top, work, topkWorkspace, thresholds, storedElements, topK)) {
        RebuildIndices(indices, tile * SCORE_ROWS);
    }
    FinishScoreTile(destinationList, sourceList);
    PostprocessScoreTile<N>(values, tile, postContext);
}

template <int N>
static __aicore__ inline void ProcessPrefetchedGroup(
    const GlobalTensor<float>& scoreRing, int& scoreTail, int subblock, int firstTile, int tileCount,
    const LocalTensor<float>& scoreInput, const LocalTensor<float>& values,
    const LocalTensor<float> (&destinationList)[VectorConstants::TRANSPOSE_TENSOR_COUNT],
    const LocalTensor<float> (&sourceList)[VectorConstants::TRANSPOSE_TENSOR_COUNT],
    const LocalTensor<float>& candidates, const LocalTensor<float>& top, const LocalTensor<float>& work,
    const LocalTensor<uint32_t>& indices, const LocalTensor<float>& spill, const GlobalTensor<float>& topkWorkspace,
    const VectorPostContext& postContext, int topK, float (&thresholds)[KernelConfig<N>::SEQUENCES_PER_AIV],
    int (&storedElements)[KernelConfig<N>::SEQUENCES_PER_AIV])
{
    constexpr int sequences = KernelConfig<N>::SEQUENCES_PER_AIV;
    constexpr int groupTiles = VectorConstants::PREFETCH_GROUP_TILES;
    for (int offset = 0; offset < groupTiles; ++offset) {
        const int tile = firstTile + offset;
        ProcessScoreTile<N>(values, candidates, top, work, indices, topkWorkspace, destinationList, sourceList, tile,
                            topK, postContext, thresholds, storedElements);
        if (offset + 1 < groupTiles) {
            SubmitScoreTile(scoreRing, scoreTail, subblock, scoreInput);
        }
        CompactLoadedTile<sequences>(tile, tileCount, candidates, values, indices, spill, thresholds, storedElements);
    }
}

template <int Sequences>
static __aicore__ inline void WriteOutputs(const GlobalTensor<uint32_t>& output, int validSequences,
                                           const GlobalTensor<float>& topkWorkspace,
                                           const LocalTensor<float>& candidates, const LocalTensor<float>& top,
                                           int topK, const VectorPostContext& postContext)
{
    using Layout = UbLayout<Sequences>;
    // 本 AIV 首行在 batch 内的右对齐因果可见长度。用有符号计算：靠前的 query 行可能没有可见 key
    int64_t firstVisible = static_cast<int64_t>(postContext.actualS2);
    if (postContext.causal) {
        firstVisible -= static_cast<int64_t>(postContext.actualS1);
        firstVisible +=
            static_cast<int64_t>(postContext.s1TileStart) + static_cast<int64_t>(postContext.subblock) * Sequences + 1;
    }
    const GlobalTensor<uint32_t>& outputGm = output;
    for (int sequence = 0; sequence < validSequences; ++sequence) {
        SubmitTopLoad(top, topkWorkspace[sequence * QUEUE_WORDS]);
        WaitTopLoad();
        LocalTensor<uint32_t> indices = GetCandidateValues(candidates, sequence).template ReinterpretCast<uint32_t>();
        // 与慢模板共用同一实现：GatherMask 提取 value/index 交错队列中的奇数索引
        QLIV2ServiceVec::ExtractIndex(indices, top.template ReinterpretCast<uint32_t>(), TOP_K);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> indicesLocal(TPosition::VECCALC, Layout::CANDIDATE_BASE + sequence * QUEUE_BYTES, TOP_K);
        const int64_t visible = firstVisible + (postContext.causal ? sequence : 0);
        if (visible < topK) {
            const int validCount = visible > 0 ? static_cast<int>(visible) : 0;
            // 仅靠分数掩码无法给出要求的 -1 索引；也不能用 score == -inf 判定，
            // 合法输入同样可能算出 -inf
            FillInvalidTail(indicesLocal, validCount, topK, static_cast<uint32_t>(QLIV2ServiceVec::INVALID_INDEX));
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // 与慢模板共用同一搬出实现（DataCopyPad，单 block 连续写）
        QLIV2ServiceVec::CopyOut(outputGm[sequence * topK], indicesLocal, topK);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

template <int N>
static __aicore__ inline void RunVectorTopk(const GlobalTensor<float>& scoreRing, const GlobalTensor<uint32_t>& output,
                                            const GlobalTensor<float>& topkWorkspace, int s2, int topK,
                                            int validSequences, int subblock, int& scoreTail,
                                            const VectorPostContext& postContext)
{
    constexpr int sequences = KernelConfig<N>::SEQUENCES_PER_AIV;
    using Layout = UbLayout<sequences>;

    LocalTensor<float> candidates(TPosition::VECCALC, Layout::CANDIDATE_BASE, Layout::CANDIDATE_BYTES / sizeof(float));
    LocalTensor<float> scoreInput(TPosition::VECCALC, Layout::INPUT_BASE, SCORE_INPUT_BYTES / sizeof(float));
    LocalTensor<float> values(TPosition::VECCALC, Layout::VALUE_BASE, VALUE_BYTES / sizeof(float));
    LocalTensor<uint32_t> indices(TPosition::VECCALC, Layout::INDEX_BASE, SCORE_ROWS);
    LocalTensor<float> spill(TPosition::VECCALC, Layout::SPILL_BASE,
                             (Layout::BYTES - Layout::SPILL_BASE) / sizeof(float));
    LocalTensor<float> top(TPosition::VECCALC, Layout::TOP_BASE, TOP_BYTES / sizeof(float));
    LocalTensor<float> work(TPosition::VECCALC, Layout::WORK_BASE, WORK_BYTES / sizeof(float));

    // 每次 S2 扫描只构建一次固定转置视图：operator[] 保持 LocalTensor 描述符语义，
    // 同时避免在逐 Score tile 的热点循环中重复构造描述符
    LocalTensor<float> sourceList[VectorConstants::TRANSPOSE_TENSOR_COUNT];
    LocalTensor<float> destinationList[VectorConstants::TRANSPOSE_TENSOR_COUNT];
    for (int row = 0; row < VectorConstants::TRANSPOSE_TENSOR_COUNT; ++row) {
        sourceList[row] = scoreInput[row * SCORE_CHANNEL_COLS];
    }
    for (int column = 0; column < SCORE_CHANNEL_COLS; ++column) {
        destinationList[VectorConstants::TRANSPOSE_SLICES_PER_COLUMN * column] = values[column * SCORE_ROWS];
        destinationList[VectorConstants::TRANSPOSE_SLICES_PER_COLUMN * column + 1] =
            values[column * SCORE_ROWS + SCORE_CHANNEL_COLS];
    }

    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    InitIndices(indices);

    const int tileCount = s2 / SCORE_ROWS;
    ClearAllCandidates<sequences>(candidates);
    for (int tile = 0; tile < VectorConstants::INITIAL_TOPK_TILES; ++tile) {
        SubmitScoreTile(scoreRing, scoreTail, subblock, scoreInput);
        FinishScoreTile(destinationList, sourceList);
        PostprocessScoreTile<N>(values, tile, postContext);
        AppendInitialTile<sequences>(candidates, values, indices, tile);
        if (tile + 1 < tileCount) {
            AdvanceIndices(indices);
        }
    }

    float thresholds[sequences];
    for (int sequence = 0; sequence < sequences; ++sequence) {
        SortChunk(top, work, GetCandidateValues(candidates, sequence), GetCandidateIndices(candidates, sequence));
        thresholds[sequence] = ReadThreshold(top, topK);
        StoreQueue(topkWorkspace[sequence * QUEUE_WORDS], top);
    }
    ClearAllCandidates<sequences>(candidates);

    if (tileCount > VectorConstants::INITIAL_TOPK_TILES) {
        RebuildIndices(indices, VectorConstants::INITIAL_TOPK_TILES * SCORE_ROWS);
    }

    // 每 sequence 已压缩写入 candidate 队列的元素个数（按 CANDIDATE_ALIGNMENT 对齐累计）
    int storedElements[sequences] = {};
    constexpr int prefetchGroup = VectorConstants::PREFETCH_GROUP_TILES;
    int tile = VectorConstants::INITIAL_TOPK_TILES;
    for (; tile + prefetchGroup <= tileCount; tile += prefetchGroup) {
        SubmitScoreTile(scoreRing, scoreTail, subblock, scoreInput);
        ProcessPrefetchedGroup<N>(scoreRing, scoreTail, subblock, tile, tileCount, scoreInput, values, destinationList,
                                  sourceList, candidates, top, work, indices, spill, topkWorkspace, postContext, topK,
                                  thresholds, storedElements);
    }
    for (; tile < tileCount; ++tile) {
        SubmitScoreTile(scoreRing, scoreTail, subblock, scoreInput);
        ProcessScoreTile<N>(values, candidates, top, work, indices, topkWorkspace, destinationList, sourceList, tile,
                            topK, postContext, thresholds, storedElements);
        CompactLoadedTile<sequences>(tile, tileCount, candidates, values, indices, spill, thresholds, storedElements);
    }

    for (int sequence = 0; sequence < sequences; ++sequence) {
        if (storedElements[sequence] != 0) {
            FlushCandidate<sequences>(sequence, candidates, top, work, topkWorkspace, thresholds, topK);
        }
    }
    WriteOutputs<sequences>(output, validSequences, topkWorkspace, candidates, top, topK, postContext);

    SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

} // namespace QLIV2Fast

#endif // QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_VECTOR_ARCH22_H
