/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SFA_ARCH22_SPLIT_KV_MERGE_H
#define SFA_ARCH22_SPLIT_KV_MERGE_H

#include "kernel_operator.h"

namespace AttentionCommon {
using namespace AscendC;

constexpr uint32_t SPLIT_KV_HEAD_DIM = 512;       // 每个 head 的 D 维
constexpr uint32_t SPLIT_KV_HEADS_PER_CHUNK = 16; // 32 KiB / (512 个 float/head) = 16

// 合并一条 query 行的 splitCount 份 partial softmax 结果。
template <bool SUPPORT_PUBLIC_LSE, bool HANDLE_EMPTY_PARTITIONS, typename OUT_T>
__aicore__ inline void MergeSparseSplitKV(uint32_t queryCount, uint32_t queryIdx, uint32_t splitCount, uint32_t heads,
                                          GlobalTensor<float>& partialOut, GlobalTensor<float>& partialMax,
                                          GlobalTensor<float>& partialSum, GlobalTensor<OUT_T>& finalOut,
                                          GlobalTensor<float>& finalMax, GlobalTensor<float>& finalSum,
                                          bool returnSoftmaxLse, LocalTensor<float> partialOutUb,
                                          LocalTensor<float> accum, LocalTensor<float> stats, LocalTensor<float> work,
                                          LocalTensor<OUT_T> result)
{
    constexpr uint32_t DIM = SPLIT_KV_HEAD_DIM;
    constexpr uint32_t HEADS_PER_CHUNK = SPLIT_KV_HEADS_PER_CHUNK;
    constexpr uint32_t STATS_ARRAY_COUNT = 3U; // Partition maxima, sums, and merge weights.
    constexpr uint32_t FP32_PER_BLOCK = ONE_BLK_SIZE / sizeof(float);
    constexpr uint32_t FP32_PER_REPEAT = DEFAULT_REPEAT_STRIDE * FP32_PER_BLOCK;
    const uint32_t sumRowOffset = splitCount;
    const uint32_t weightRowOffset = 2U * splitCount; // Weights follow the max and sum arrays.
    // Reserve the tail of the existing stats buffer for a second partial-output tile.
    const uint32_t statsElements = stats.GetSize() - HEADS_PER_CHUNK * DIM;
    LocalTensor<float> partialOutPong = stats[statsElements];
    const uint32_t statsHeadsPerChunk =
        (statsElements / (STATS_ARRAY_COUNT * splitCount) / FP32_PER_BLOCK) * FP32_PER_BLOCK;
    // 两个 AIV 子核各处理一半 head。
    const uint32_t headsPerSubBlock = (heads + 1) / 2;
    const uint32_t subBlockStart = GetSubBlockIdx() * headsPerSubBlock;
    const uint32_t subBlockEnd = Min(heads, subBlockStart + headsPerSubBlock);
    for (uint32_t headBase = subBlockStart; headBase < subBlockEnd; headBase += statsHeadsPerChunk) {
        const uint32_t statsHeads = Min(statsHeadsPerChunk, subBlockEnd - headBase);
        const uint32_t statsStride = (statsHeads + FP32_PER_BLOCK - 1) / FP32_PER_BLOCK * FP32_PER_BLOCK;
        const DataCopyExtParams statsCopy{1, static_cast<uint32_t>(statsHeads * sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> statsPad{false, 0, 0, 0};
        // 阶段 1：搬入本 chunk 各 partition 的 max/sum。
        LocalTensor<float> max = work;
        LocalTensor<float> sum = work[statsStride];
        for (uint32_t split = 0; split < splitCount; ++split) {
            const uint64_t offset = (split * queryCount + queryIdx) * heads + headBase;
            DataCopyPad(stats[split * statsStride], partialMax[offset], statsCopy, statsPad);
            DataCopyPad(stats[(sumRowOffset + split) * statsStride], partialSum[offset], statsCopy, statsPad);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        // 阶段 2：归约全局 max，算权重并累加 sum。
        DataCopy(max, stats, statsStride);
        Duplicate(sum, 0.0f, statsHeads);
        PipeBarrier<PIPE_V>();
        for (uint32_t split = 1; split < splitCount; ++split) {
            Max(max, max, stats[split * statsStride], statsHeads);
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t split = 0; split < splitCount; ++split) {
            LocalTensor<float> weight = stats[(weightRowOffset + split) * statsStride];
            Sub(weight, stats[split * statsStride], max, statsHeads);
            PipeBarrier<PIPE_V>();
            Exp(weight, weight, statsHeads);
            PipeBarrier<PIPE_V>();
            Mul(weight, weight, stats[(sumRowOffset + split) * statsStride], statsHeads);
            PipeBarrier<PIPE_V>();
            Add(sum, sum, weight, statsHeads);
            PipeBarrier<PIPE_V>();
        }
        bool allEmpty = false;
        if constexpr (HANDLE_EMPTY_PARTITIONS) {
            // sum[head_id]是一个该分片的head的q_head x (分片 keys)的exp(score-max_score))之和,
            // 所以若分片不是一个0长度的key分片，则第0个head的sum不可能等于0。 因此此处用 sum.GetValue(0) 为 0
            // 判断该分片是否为一个空分片。
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            allEmpty = sum.GetValue(0) == 0.0f;
            if (allEmpty) {
                Duplicate(sum, 1.0f, statsHeads);
                PipeBarrier<PIPE_V>();
            }
        }
        for (uint32_t split = 0; split < splitCount; ++split) {
            LocalTensor<float> weight = stats[(weightRowOffset + split) * statsStride];
            Div(weight, weight, sum, statsHeads);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (HANDLE_EMPTY_PARTITIONS) {
            if (allEmpty) {
                Duplicate(max, 0.0f, statsHeads);
                Duplicate(sum, 0.0f, statsHeads);
                PipeBarrier<PIPE_V>();
            }
        }
        if constexpr (SUPPORT_PUBLIC_LSE) {
            if (returnSoftmaxLse) {
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(finalMax[queryIdx * heads + headBase], max, statsCopy);
                DataCopyPad(finalSum[queryIdx * heads + headBase], sum, statsCopy);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
        }
        // headBase/statsHeads 已在本子核的 head 半区内。
        const uint32_t firstHead = headBase;
        const uint32_t endHead = Min(headBase + statsHeads, subBlockEnd);
        // 阶段 3：按权重累加 partial 输出并写回。
        for (uint32_t head = firstHead; head < endHead; head += HEADS_PER_CHUNK) {
            const uint32_t chunkHeads = Min(HEADS_PER_CHUNK, endHead - head);
            const uint32_t elements = chunkHeads * DIM;
            const uint64_t firstOffset = (queryIdx * heads + head) * DIM;
            DataCopy(partialOutUb, partialOut[firstOffset], elements);
            SetFlag<HardEvent::MTE2_V>(0);
            Duplicate(accum, 0.0f, elements);
            PipeBarrier<PIPE_V>();
            for (uint32_t split = 0; split < splitCount; ++split) {
                const uint32_t bufferIdx = split % 2;
                LocalTensor<float> currentOut = bufferIdx == 0 ? partialOutUb : partialOutPong;
                WaitFlag<HardEvent::MTE2_V>(bufferIdx);
                // Prefetch the next partition's O while merging the current O.
                if (split + 1 < splitCount) {
                    LocalTensor<float> nextOut = bufferIdx == 0 ? partialOutPong : partialOutUb;
                    const uint64_t offset = (((split + 1) * queryCount + queryIdx) * heads + head) * DIM;
                    DataCopy(nextOut, partialOut[offset], elements);
                    SetFlag<HardEvent::MTE2_V>(bufferIdx ^ 1);
                }
                // Max/sum are no longer needed; reuse work for one weight block per head.
                Brcb(work, stats[(weightRowOffset + split) * statsStride + head - headBase],
                     (chunkHeads + FP32_PER_BLOCK - 1) / FP32_PER_BLOCK, {1, FP32_PER_BLOCK});
                PipeBarrier<PIPE_V>();
                // Broadcast one weight block per head across its output row.
                BinaryRepeatParams mulParams{1, 1, 0, DIM / FP32_PER_BLOCK, DIM / FP32_PER_BLOCK, 1};
                for (uint32_t column = 0; column < DIM; column += FP32_PER_REPEAT) {
                    Mul(currentOut[column], currentOut[column], work, FP32_PER_REPEAT, chunkHeads, mulParams);
                }
                PipeBarrier<PIPE_V>();
                Add(accum, accum, currentOut, elements);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
            }
            if constexpr (IsSameType<OUT_T, bfloat16_t>::value) {
                Cast(result, accum, RoundMode::CAST_RINT, elements);
            } else {
                Cast(result, accum, RoundMode::CAST_ROUND, elements);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(finalOut[(queryIdx * heads + head) * DIM], result, elements);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }
}
} // namespace AttentionCommon
#endif
