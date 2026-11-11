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
 * \file qsag_vector_engine.h
 * \brief AIV side of arch22 QSAG: Pre/Post, Gather/Scatter, row statistics and P/dS.
 */
#ifndef ATTENTION_QSAG_VECTOR_ENGINE_ARCH22_H
#define ATTENTION_QSAG_VECTOR_ENGINE_ARCH22_H

#include "qsag_common_arch22.h"
#include "qsag_probability.h"

namespace QSAG {

template <typename T, typename Tiling>
class VectorEngine {
public:
    __aicore__ inline void Init(TPipe* pipe, const QsagParams& params)
    {
        // UB：score/grad 单缓冲（每 AIV 只持有自己的半行）、LSE/Di 双槽、
        // P/dS 连续 ND 输出（arch22 无 A5 的 NZ 队列间隔）和索引缓存。
        const uint32_t matrixBytes = params.vectorRows * params.kvTileSize * sizeof(float);
        const uint32_t rowStatsBytes = params.vectorRows * BLOCK_BYTES;
        const uint32_t probabilityBytes = params.vectorRows * params.kvTileSize * sizeof(T);
        const uint32_t indexBytes = AlignBytes(params.kvTileSize / params.blockSize * sizeof(int32_t));
        const uint32_t inputRowBytes = TRANSFER_ROWS * MAX_HEAD_DIM * sizeof(T);
        const uint32_t floatRowBytes = TRANSFER_ROWS * MAX_HEAD_DIM * sizeof(float);
        const uint32_t rowScratchBytes = 2 * inputRowBytes + 2 * floatRowBytes + TRANSFER_ROWS * BLOCK_BYTES;
        const uint32_t stagingBytes = params.transferElements * sizeof(float);
        const uint32_t broadcastBytes = VEC_ROW_SEGMENT * sizeof(float);
        // 两档容量核算（910B UB 192KiB）：G<=128 约 145KiB，G>128（Ktile=64）约 133KiB。
        const uint32_t ubBytes = 2 * matrixBytes + SLOT_COUNT * rowStatsBytes + 2 * probabilityBytes + indexBytes +
                                 rowScratchBytes + 2 * stagingBytes + 2 * broadcastBytes;
        pipe->InitBuffer(ubBuffer_, ubBytes);
        uint32_t offset = 0;
        score_ = TakeUb<float>(matrixBytes, offset);
        grad_ = TakeUb<float>(matrixBytes, offset);
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            rowStatsLocal_[slot] = TakeUb<float>(rowStatsBytes, offset);
        }
        probabilityNd_ = TakeUb<T>(probabilityBytes, offset);
        dsProbabilityNd_ = TakeUb<T>(probabilityBytes, offset);
        indicesLocal_ = TakeUb<int32_t>(indexBytes, offset);
        inputRowLocal_ = TakeUb<T>(inputRowBytes, offset);
        dOutRowLocal_ = TakeUb<T>(inputRowBytes, offset);
        floatRowLocal0_ = TakeUb<float>(floatRowBytes, offset);
        floatRowLocal1_ = TakeUb<float>(floatRowBytes, offset);
        diPerRowLocal_ = TakeUb<float>(TRANSFER_ROWS * BLOCK_BYTES, offset);
        gatherStaging0_ = TakeUb<float>(2 * stagingBytes, offset);
        gatherStaging1_ = gatherStaging0_[params.transferElements];
        lseBroadcast_ = TakeUb<float>(broadcastBytes, offset);
        diBroadcast_ = TakeUb<float>(broadcastBytes, offset);
    }

    __aicore__ inline void Pre(QsagContext<T, Tiling>& ctx)
    {
        const QsagParams& params = ctx.params;
        ZeroWorkspace(ctx.dQueryWorkspace, params.totalQueryTokens * ctx.tiling->qHeadNum * params.headDimAligned,
                      params);
        ZeroWorkspace(ctx.dKeyWorkspace, params.totalKvTokens * ctx.tiling->kvHeadNum * params.headDimAligned, params);
        ZeroWorkspace(ctx.dValueWorkspace, params.totalKvTokens * ctx.tiling->kvHeadNum * params.headDimAligned,
                      params);
        const uint64_t totalRows = params.totalQueryTokens * ctx.tiling->qHeadNum;
        for (uint64_t row = params.globalAivIndex * TRANSFER_ROWS; row < totalRows;
             row += ctx.tiling->cubeCoreNum * AIV_PER_CUBE * TRANSFER_ROWS) {
            const uint32_t rows = Min(TRANSFER_ROWS, totalRows - row);
            // The flattened row and Di follow the actual Q layout; LSE uses its own layout.
            ReadRow(inputRowLocal_, ctx.attentionOut, row * params.headDim, rows, params);
            ReadRow(dOutRowLocal_, ctx.dOut, row * params.headDim, rows, params);
            Cast(floatRowLocal0_, inputRowLocal_, RoundMode::CAST_NONE, rows * params.headDimAligned);
            Cast(floatRowLocal1_, dOutRowLocal_, RoundMode::CAST_NONE, rows * params.headDimAligned);
            PipeBarrier<PIPE_V>();
            Mul(floatRowLocal0_, floatRowLocal0_, floatRowLocal1_, rows * params.headDimAligned);
            PipeBarrier<PIPE_V>();
            ReduceDiRows(rows, params);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(ctx.di[row * FLOATS_PER_BLOCK], diPerRowLocal_, rows * FLOATS_PER_BLOCK);
            if (params.headDim != params.headDimAligned) {
                DataCopy(ctx.dOutPadded[row * params.headDimAligned], dOutRowLocal_, rows * params.headDimAligned);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                ReadRow(inputRowLocal_, ctx.query, row * params.headDim, rows, params);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(ctx.queryPadded[row * params.headDimAligned], inputRowLocal_, rows * params.headDimAligned);
            }
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        ZeroGmPadding(ctx);
        PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<false>();
    }

    __aicore__ inline void Post(QsagContext<T, Tiling>& ctx)
    {
        const QsagParams& params = ctx.params;
        PostTensor(ctx, ctx.dQuery, ctx.dQueryWorkspace, params.totalQueryTokens * ctx.tiling->qHeadNum,
                   ctx.tiling->softmaxScale);
        PostTensor(ctx, ctx.dKey, ctx.dKeyWorkspace, params.totalKvTokens * ctx.tiling->kvHeadNum,
                   ctx.tiling->softmaxScale);
        PostTensor(ctx, ctx.dValue, ctx.dValueWorkspace, params.totalKvTokens * ctx.tiling->kvHeadNum, 1.0f);
    }

    __aicore__ inline void Gather(QsagContext<T, Tiling>& ctx, const TileTask& task)
    {
        const QsagParams& params = ctx.params;
        const uint32_t slot = task.id % SLOT_COUNT;
        // 上一轮同槽的 K/V 已由 Cube 消费完，才允许重新搬入。
        AscendC::WaitEvent(KV_FREE);
        GatherRows(ctx, params, task.sparseBase, task.first, task.selected, task.tail, task.tailStart, task.kbase,
                   task.kvHead, slot);
        // 1:2 模式下同 cube 双 AIV 必须发同一 flag（AIC 端该 flag 需两侧都到达才置位）。
        CrossCoreSetFlag<2, PIPE_MTE3>(GATHER_READY);
    }

    // LSE/Di 行统计按 (token, head) owner 复用，覆盖该 owner 下的全部 tile。
    __aicore__ inline void EnsureRowStats(QsagContext<T, Tiling>& ctx, uint64_t token, uint64_t head)
    {
        if (rowStatsValid_ && rowStatsToken_ == token && rowStatsHead_ == head) {
            return;
        }
        CacheRowStats(ctx, token, head, ctx.params.groupRows);
        rowStatsToken_ = token;
        rowStatsHead_ = head;
        rowStatsValid_ = true;
    }

    __aicore__ inline void ProcessProbability(QsagContext<T, Tiling>& ctx, const TileTask& task)
    {
        const QsagParams& params = ctx.params;
        const uint32_t slot = task.id % SLOT_COUNT;
        EnsureRowStats(ctx, task.token, task.head);
        AscendC::WaitEvent(SCORE_READY);
        // score_/grad_ are single-buffered: finish the previous tile's V reads before MTE2 overwrites them.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        // 每个 AIV 读回自己那半行（GM ND 行主序，rowsPerAiv*kvTileSize 连续且 32B 对齐）。
        DataCopy(score_, ctx.scoreGm[slot][params.aivIndex * params.rowsPerAiv * params.kvTileSize],
                 params.rowsPerAiv * params.kvTileSize);
        DataCopy(grad_, ctx.gradGm[slot][params.aivIndex * params.rowsPerAiv * params.kvTileSize],
                 params.rowsPerAiv * params.kvTileSize);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        const uint32_t firstRow = params.aivIndex * params.rowsPerAiv;
        const uint32_t validRows =
            firstRow < params.groupRows ? Min(params.rowsPerAiv, params.groupRows - firstRow) : 0;
        // Only actual head rows take part in the KV reduction. Rows in
        // [validRows, rowCapacityPerAiv) may hold Mmad output of padded Q/dO rows;
        // exp(0 - lse) would turn them into non-zero P and pollute MM4/MM5.
        if (validRows < params.rowCapacityPerAiv) {
            Duplicate(score_[validRows * params.kvTileSize], 0.0f,
                      (params.rowCapacityPerAiv - validRows) * params.kvTileSize);
            Duplicate(grad_[validRows * params.kvTileSize], 0.0f,
                      (params.rowCapacityPerAiv - validRows) * params.kvTileSize);
            PipeBarrier<PIPE_V>();
        }
        ProbabilityGradRows(score_, grad_, rowStatsLocal_[0], rowStatsLocal_[1], lseBroadcast_, diBroadcast_,
                            ctx.tiling->softmaxScale, params.kvTileSize, params.rowsPerAiv, validRows, task.selected);
        PipeBarrier<PIPE_V>();
        // probabilityNd_/dsProbabilityNd_ also source the previous task's UB->GM copy
        // (MTE3). In the drain path two ProcessProbability calls are back-to-back, so
        // wait for that copy before overwriting the buffers.
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        Cast(probabilityNd_, score_, RoundMode::CAST_ROUND, params.rowsPerAiv * params.kvTileSize);
        Cast(dsProbabilityNd_, grad_, RoundMode::CAST_ROUND, params.rowsPerAiv * params.kvTileSize);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(ctx.probabilityGm[slot][params.aivIndex * params.rowsPerAiv * params.kvTileSize], probabilityNd_,
                 params.rowsPerAiv * params.kvTileSize);
        DataCopy(ctx.dsProbabilityGm[slot][params.aivIndex * params.rowsPerAiv * params.kvTileSize], dsProbabilityNd_,
                 params.rowsPerAiv * params.kvTileSize);
        CrossCoreSetFlag<2, PIPE_MTE3>(PROBABILITY_READY);
    }

    __aicore__ inline void Scatter(QsagContext<T, Tiling>& ctx, const TileTask& task)
    {
        const QsagParams& params = ctx.params;
        const uint32_t slot = task.id % SLOT_COUNT;
        AscendC::WaitEvent(RESULT_READY);
        if (!task.tail) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            DataCopyExtParams indexParams{1, static_cast<uint32_t>(task.selected / params.blockSize * sizeof(int32_t)),
                                          0, 0, 0};
            DataCopyPad(indicesLocal_, ctx.filteredIndices[task.sparseBase + task.first], indexParams,
                        DataCopyPadExtParams<int32_t>{});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        const uint32_t begin = params.aivIndex * (params.kvTileSize / AIV_PER_CUBE);
        const uint32_t end = Min(task.selected, begin + params.kvTileSize / AIV_PER_CUBE);
        for (uint32_t row = begin; row < end;) {
            const uint32_t rows = Min(params.transferElements / params.headDimAligned, end - row);
            DataCopy(gatherStaging0_, ctx.mm4[slot][row * params.headDimAligned], rows * params.headDimAligned);
            DataCopy(gatherStaging1_, ctx.mm5[slot][row * params.headDimAligned], rows * params.headDimAligned);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            SetAtomicAdd<float>();
            for (uint32_t copied = 0; copied < rows;) {
                const uint32_t position = row + copied;
                uint32_t run =
                    task.tail ? rows - copied : Min(rows - copied, params.blockSize - BlockOffset(params, position));
                const uint64_t token =
                    task.tail ?
                        task.tailStart + position :
                        static_cast<uint64_t>(indicesLocal_.GetValue(BlockIndex(params, position))) * params.blockSize +
                            BlockOffset(params, position);
                const uint64_t offset =
                    Offset(ctx.tiling, task.kbase + token, task.kvHead, false, params.headDimAligned);
                uint64_t stride = ctx.tiling->inputLayout == 1 ?
                                      0 :
                                      (ctx.tiling->kvHeadNum - 1) * params.headDimAligned * sizeof(float);
                if (stride > MAX_FIXPIPE_STRIDE) {
                    run = 1;
                    stride = 0;
                }
                DataCopyExtParams copyParams{static_cast<uint16_t>(run),
                                             static_cast<uint32_t>(params.headDimAligned * sizeof(float)), 0,
                                             static_cast<uint32_t>(stride), 0};
                DataCopyPad(ctx.dKeyWorkspace[offset], gatherStaging0_[copied * params.headDimAligned], copyParams);
                DataCopyPad(ctx.dValueWorkspace[offset], gatherStaging1_[copied * params.headDimAligned], copyParams);
                copied += run;
            }
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            row += rows;
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(RESULT_FREE);
    }

public:
    // 按 causal 可见长度过滤一行稀疏索引：保留 [0, visible/Bs) 内的块号，
    // 压缩到 filteredIndices 的前缀并写回有效块数，供 AIC/AIV 主循环读取。
    __aicore__ inline void FilterQuery(QsagContext<T, Tiling>& ctx, uint64_t row, uint64_t visible, int32_t rawCount)
    {
        const QsagParams& params = ctx.params;
        const uint64_t complete = visible / params.blockSize;
        const uint32_t count =
            rawCount <= 0 ? 0 : static_cast<uint32_t>(Min(static_cast<uint64_t>(rawCount), ctx.tiling->topK));
        auto input = gatherStaging0_.template ReinterpretCast<int32_t>();
        auto output = input[params.transferElements];
        uint32_t kept = 0;
        const uint64_t base = row * ctx.tiling->topK;
        for (uint32_t first = 0; first < count; first += params.transferElements) {
            const uint32_t size = static_cast<uint32_t>(
                Min(static_cast<uint64_t>(params.transferElements), static_cast<uint64_t>(count - first)));
            DataCopyExtParams load{1, static_cast<uint32_t>(size * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(input, ctx.sparseBlockIdx[base + first], load, DataCopyPadExtParams<int32_t>{});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            uint32_t valid = 0;
            for (uint32_t i = 0; i < size; ++i) {
                const int32_t block = input.GetValue(i);
                if (block >= 0 && static_cast<uint64_t>(block) < complete) {
                    output.SetValue(valid++, block);
                }
            }
            if (valid != 0) {
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopyExtParams store{1, static_cast<uint32_t>(valid * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(ctx.filteredIndices[base + kept], output, store);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            kept += valid;
        }
        output.SetValue(0, static_cast<int32_t>(kept));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams storeCount{1, sizeof(int32_t), 0, 0, 0};
        DataCopyPad(ctx.filteredCounts[row], output, storeCount);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

private:
    // 每行 LSE/Di 放进独立的 32B 块（首元素有效，其余为 0）。
    __aicore__ inline void CacheRowStats(QsagContext<T, Tiling>& ctx, uint64_t token, uint64_t head, uint32_t rows)
    {
        const QsagParams& params = ctx.params;
        const uint32_t firstRow = params.aivIndex * params.rowsPerAiv;
        const uint32_t valid = firstRow < rows ? Min(params.rowsPerAiv, rows - firstRow) : 0;
        const uint64_t lseStride = LseHeadStride(ctx.tiling);
        const uint64_t diStride = ctx.tiling->inputLayout == 1 ? ctx.tiling->qSeqLen : 1;
        const uint64_t maxStrideElements = MAX_FIXPIPE_STRIDE / sizeof(float) + 1;
        if (lseStride > maxStrideElements || diStride > maxStrideElements / FLOATS_PER_BLOCK) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for (uint32_t row = 0; row < params.rowsPerAiv; ++row) {
                float lse = 0.0f;
                float di = 0.0f;
                if (row < valid) {
                    const uint64_t offset = Offset(ctx.tiling, token, head + firstRow + row, true, 1);
                    lse = ctx.lse.GetValue(LseOffset(ctx.tiling, token, head + firstRow + row));
                    di = ctx.di.GetValue(offset * FLOATS_PER_BLOCK);
                }
                rowStatsLocal_[0].SetValue(row * FLOATS_PER_BLOCK, lse);
                rowStatsLocal_[1].SetValue(row * FLOATS_PER_BLOCK, di);
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            return;
        }
        if (valid < params.rowsPerAiv) {
            Duplicate(rowStatsLocal_[0][valid * FLOATS_PER_BLOCK], 0.0f,
                      (params.rowsPerAiv - valid) * FLOATS_PER_BLOCK);
            Duplicate(rowStatsLocal_[1][valid * FLOATS_PER_BLOCK], 0.0f,
                      (params.rowsPerAiv - valid) * FLOATS_PER_BLOCK);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if (valid != 0) {
            const uint64_t offset = Offset(ctx.tiling, token, head + firstRow, true, 1);
            const uint64_t lseOffset = LseOffset(ctx.tiling, token, head + firstRow);
            DataCopyExtParams copyParams{static_cast<uint16_t>(valid), sizeof(float),
                                         static_cast<uint32_t>((lseStride - 1) * sizeof(float)), 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, FLOATS_PER_BLOCK - 1, 0.0f};
            DataCopyPad(rowStatsLocal_[0], ctx.lse[lseOffset], copyParams, pad);
            copyParams.srcStride = static_cast<uint32_t>((diStride * FLOATS_PER_BLOCK - 1) * sizeof(float));
            DataCopyPad(rowStatsLocal_[1], ctx.di[offset * FLOATS_PER_BLOCK], copyParams, pad);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void GatherRows(QsagContext<T, Tiling>& ctx, const QsagParams& params, uint64_t sparseBase,
                                      uint64_t first, uint32_t count, bool tail, uint64_t tailStart, uint64_t kbase,
                                      uint64_t head, uint32_t slot)
    {
        if (!tail) {
            const uint32_t entries = count / params.blockSize;
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            DataCopyExtParams indexParams{1, static_cast<uint32_t>(entries * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(indicesLocal_, ctx.filteredIndices[sparseBase + first], indexParams,
                        DataCopyPadExtParams<int32_t>{});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        // Each AIV owns a contiguous half tile. Coalesce rows inside each
        // sparse block; no copy crosses an index/block or ownership boundary.
        const uint32_t begin = params.aivIndex * (params.kvTileSize / AIV_PER_CUBE);
        const uint32_t end = begin + params.kvTileSize / AIV_PER_CUBE;
        auto gatherKey = gatherStaging0_.template ReinterpretCast<T>();
        auto gatherValue = gatherStaging1_.template ReinterpretCast<T>();
        for (uint32_t row = begin; row < end;) {
            uint32_t rows = Min(params.transferElements / params.headDimAligned, end - row);
            if (row < count) {
                rows = Min(rows, count - row);
                for (uint32_t copied = 0; copied < rows;) {
                    const uint32_t position = row + copied;
                    uint32_t run =
                        tail ? rows - copied : Min(rows - copied, params.blockSize - BlockOffset(params, position));
                    const uint64_t token =
                        tail ? tailStart + position :
                               static_cast<uint64_t>(indicesLocal_.GetValue(BlockIndex(params, position))) *
                                       params.blockSize +
                                   BlockOffset(params, position);
                    const uint64_t offset = Offset(ctx.tiling, kbase + token, head, false, params.headDim);
                    uint64_t stride =
                        ctx.tiling->inputLayout == 1 ? 0 : (ctx.tiling->kvHeadNum - 1) * params.headDim * sizeof(T);
                    if (stride > MAX_FIXPIPE_STRIDE) {
                        run = 1;
                        stride = 0;
                    }
                    DataCopyExtParams copyParams{static_cast<uint16_t>(run),
                                                 static_cast<uint32_t>(params.headDim * sizeof(T)),
                                                 static_cast<uint32_t>(stride), 0, 0};
                    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(params.headDimAligned - params.headDim),
                                                static_cast<T>(0)};
                    DataCopyPad(gatherKey[copied * params.headDimAligned], ctx.key[offset], copyParams, pad);
                    DataCopyPad(gatherValue[copied * params.headDimAligned], ctx.value[offset], copyParams, pad);
                    copied += run;
                }
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            } else {
                Duplicate(gatherKey, static_cast<T>(0), rows * params.headDimAligned);
                Duplicate(gatherValue, static_cast<T>(0), rows * params.headDimAligned);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            }
            DataCopy(ctx.selectedKey[slot][row * params.headDimAligned], gatherKey, rows * params.headDimAligned);
            DataCopy(ctx.selectedValue[slot][row * params.headDimAligned], gatherValue, rows * params.headDimAligned);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            row += rows;
        }
    }

    __aicore__ inline void ReadRow(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset, uint32_t rows,
                                   const QsagParams& params)
    {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyExtParams copyParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(params.headDim * sizeof(T)), 0,
                                     0, 0};
        DataCopyPad(dst, src[offset], copyParams,
                    DataCopyPadExtParams<T>{true, 0, static_cast<uint8_t>(params.headDimAligned - params.headDim),
                                            static_cast<T>(0)});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void ZeroWorkspace(GlobalTensor<float> gm, uint64_t count, const QsagParams& params)
    {
        constexpr uint32_t zeroElements = TRANSFER_ROWS * MAX_HEAD_DIM;
        Duplicate(floatRowLocal0_, 0.0f, zeroElements);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        for (uint64_t start = params.globalAivIndex * zeroElements; start < count;
             start += params.cubeCoreNum * AIV_PER_CUBE * zeroElements) {
            const uint32_t length = Min(zeroElements, count - start);
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(float)), 0, 0, 0};
            DataCopyPad(gm[start], floatRowLocal0_, copyParams);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    // di = sum(out .* dout)：PFA MsdRowSum 树归约范式（910B 2D repeat 的 repStride 为 32B block）。
    __aicore__ inline void ReduceDiRows(uint32_t rows, const QsagParams& params)
    {
        uint32_t blockCount = params.headDimAligned / FLOATS_PER_BLOCK;
        const uint32_t remain = params.headDimAligned % FLOATS_PER_BLOCK;
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = params.headDimAligned / FLOATS_PER_BLOCK;
        repeatParams.src1RepStride = params.headDimAligned / FLOATS_PER_BLOCK;
        repeatParams.dstRepStride = params.headDimAligned / FLOATS_PER_BLOCK;
        // 尾块折入：非 8 的整数倍时把最后一段加到第 0 块。
        if (blockCount > 0 && remain > 0) {
            Add(floatRowLocal0_, floatRowLocal0_, floatRowLocal0_[blockCount * FLOATS_PER_BLOCK], remain, rows,
                repeatParams);
            PipeBarrier<PIPE_V>();
        }
        // 块对树归约，每轮两两相加，最终每行只剩首块。
        for (uint32_t loopCount = blockCount / 2; loopCount > 0; loopCount = blockCount / 2) {
            blockCount = (blockCount + 1) / 2;
            for (uint32_t block = 0; block < loopCount; ++block) {
                Add(floatRowLocal0_[block * FLOATS_PER_BLOCK], floatRowLocal0_[block * FLOATS_PER_BLOCK],
                    floatRowLocal0_[(block + blockCount) * FLOATS_PER_BLOCK], FLOATS_PER_BLOCK, rows, repeatParams);
            }
            PipeBarrier<PIPE_V>();
        }
        // dav_c220 的 dstRepStride 单位为元素、srcRepStride 单位为 32B 块：
        // 每行结果落到 diPerRowLocal_ 的第 r 个 32B 块。
        WholeReduceSum(diPerRowLocal_, floatRowLocal0_,
                       (params.headDimAligned < FLOATS_PER_BLOCK) ? params.headDimAligned : FLOATS_PER_BLOCK, rows,
                       FLOATS_PER_BLOCK, 1, params.headDimAligned / FLOATS_PER_BLOCK);
    }

    // AIC Fixpipe 只写 [0, 2*rowsPerAiv)；MM4/MM5 的 k 维覆盖 [2*rowsPerAiv, gcap)。
    // 若该区间含 L0C 垃圾（NaN/Inf），MM4/MM5 中 NaN x 0 = NaN 会毒化所有结果行，
    // 两个 AIV 在 Pre() 对四区 x 双槽的 padding 行幂等清零（disjoint 分片）。
    __aicore__ inline void ZeroGmPadding(QsagContext<T, Tiling>& ctx)
    {
        const QsagParams& params = ctx.params;
        const uint32_t firstPaddingRow = AIV_PER_CUBE * params.rowsPerAiv;
        const uint32_t paddingRowsPerAiv = (params.groupCapacity - firstPaddingRow) / AIV_PER_CUBE;
        if (paddingRowsPerAiv == 0) {
            return;
        }
        const uint32_t rowOffset = firstPaddingRow + params.aivIndex * paddingRowsPerAiv;
        const uint32_t paddingLength = paddingRowsPerAiv * params.kvTileSize;
        Duplicate(gatherStaging0_, 0.0f, paddingLength);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            DataCopy(ctx.scoreGm[slot][rowOffset * params.kvTileSize], gatherStaging0_, paddingLength);
            DataCopy(ctx.gradGm[slot][rowOffset * params.kvTileSize], gatherStaging0_, paddingLength);
        }
        auto zeroT = gatherStaging0_.template ReinterpretCast<T>();
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            DataCopy(ctx.probabilityGm[slot][rowOffset * params.kvTileSize], zeroT, paddingLength);
            DataCopy(ctx.dsProbabilityGm[slot][rowOffset * params.kvTileSize], zeroT, paddingLength);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void PostTensor(QsagContext<T, Tiling>& ctx, GlobalTensor<T> output, GlobalTensor<float> source,
                                      uint64_t rows, float scale)
    {
        const QsagParams& params = ctx.params;
        for (uint64_t row = params.globalAivIndex * TRANSFER_ROWS; row < rows;
             row += ctx.tiling->cubeCoreNum * AIV_PER_CUBE * TRANSFER_ROWS) {
            const uint32_t batchRows = Min(TRANSFER_ROWS, rows - row);
            DataCopy(floatRowLocal0_, source[row * params.headDimAligned], batchRows * params.headDimAligned);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Muls(floatRowLocal0_, floatRowLocal0_, scale, batchRows * params.headDimAligned);
            PipeBarrier<PIPE_V>();
            Cast(inputRowLocal_, floatRowLocal0_, RoundMode::CAST_ROUND, batchRows * params.headDimAligned);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams copyParams{static_cast<uint16_t>(batchRows),
                                         static_cast<uint32_t>(params.headDim * sizeof(T)), 0, 0, 0};
            DataCopyPad(output[row * params.headDim], inputRowLocal_, copyParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    template <typename Element>
    __aicore__ inline LocalTensor<Element> TakeUb(uint32_t bytes, uint32_t& offset)
    {
        auto tensor = ubBuffer_.GetWithOffset<Element>(bytes / sizeof(Element), offset);
        offset += AlignBytes(bytes);
        return tensor;
    }

    TBuf<TPosition::VECCALC> ubBuffer_;
    LocalTensor<float> score_;
    LocalTensor<float> grad_;
    LocalTensor<float> rowStatsLocal_[SLOT_COUNT];
    LocalTensor<T> probabilityNd_;
    LocalTensor<T> dsProbabilityNd_;
    LocalTensor<int32_t> indicesLocal_;
    LocalTensor<T> inputRowLocal_;
    LocalTensor<T> dOutRowLocal_;
    LocalTensor<float> floatRowLocal0_;
    LocalTensor<float> floatRowLocal1_;
    LocalTensor<float> diPerRowLocal_;
    LocalTensor<float> gatherStaging0_;
    LocalTensor<float> gatherStaging1_;
    LocalTensor<float> lseBroadcast_;
    LocalTensor<float> diBroadcast_;
    bool rowStatsValid_ = false;
    uint64_t rowStatsToken_ = 0;
    uint64_t rowStatsHead_ = 0;
};

} // namespace QSAG
#endif // ATTENTION_QSAG_VECTOR_ENGINE_ARCH22_H
