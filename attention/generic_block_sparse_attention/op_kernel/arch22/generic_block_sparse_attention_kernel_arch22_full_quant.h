/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_FULL_QUANT_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_FULL_QUANT_H

#include "../attn_infra/gemm/block/block_mmad_qk_full_quant.hpp"
#include "../attn_infra/gemm/block/block_mmad_pv_full_quant.hpp"
#include "../attn_infra/epilogue/block/block_epilogue_online_softmax_full_quant.hpp"
#include "../attn_infra/epilogue/block/block_epilogue_rescale_o_full_quant.hpp"
#include "../generic_block_sparse_attention_fd_utils.h"
#include "generic_block_sparse_attention_fd_combine_arch22.h"

namespace GbsaKernelArch22::Int8 {

struct FullQuantScales {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR probability;
};

// Adapt fixed-capacity batches to the same task cursor used by FP16/TND.
struct FixedSequenceOffsets {
    uint32_t capacity;
    __aicore__ inline uint64_t GetValue(uint32_t index) const
    {
        return static_cast<uint64_t>(index) * capacity;
    }
};

// A private view of the flat metadata. Only full-quant kernels consume this plan.
struct FullQuantFdSchedule {
    int32_t combineTaskNum = 0;
    uint32_t activeCoreNum = 0;
    GbsaMetadata::CombineSchedule combineSchedules[GbsaMetadata::MAX_COMBINE_TASK_NUM];
    GbsaMetadata::DecodeSchedule decode;
    uint32_t partialTask = 0;
    uint32_t partialCount = 0;

    __aicore__ inline bool Build(__gm__ const GbsaMetadata::Metadata* meta, uint32_t core, uint32_t maxBlocks,
                                 uint32_t availableCores)
    {
        const int32_t tasks = meta->saTotalTaskNum;
        const int32_t cores = meta->fdActiveCoreNum;
        activeCoreNum = cores;
        const int32_t limit = meta->decodePerCoreTaskNum;
        if (tasks < 2 || tasks > cores || limit <= 0 || meta->combineTaskNum == 0) {
            return false;
        }
        // Each unsplit task contributes one task-core pair, each split task contributes its partial count.
        if (meta->fdPartialTaskNum + tasks - meta->combineTaskNum == cores) {
            return false;
        }
        const int32_t total = meta->fdTotalFlatTaskNum;
        const bool uniform = static_cast<uint64_t>(tasks) * maxBlocks == static_cast<uint32_t>(total);
        if (uniform && maxBlocks % limit == 0) {
            return false;
        }
        if (uniform && maxBlocks > static_cast<uint32_t>(limit)) {
            const uint32_t parts = 1U + (maxBlocks - 1U) / limit;
            const uint64_t required = static_cast<uint64_t>(tasks) * parts;
            // Reuse idle launched cores only when the existing partial workspace can hold the plan.
            if (required > availableCores || required > static_cast<uint32_t>(meta->fdPartialTaskNum)) {
                return false;
            }
            activeCoreNum = static_cast<uint32_t>(required);
            const uint32_t task = core / parts, part = core % parts;
            decode = {static_cast<int32_t>(task), static_cast<int32_t>(task + 1U),
                      static_cast<int32_t>(static_cast<uint64_t>(part) * maxBlocks / parts),
                      static_cast<int32_t>(static_cast<uint64_t>(part + 1U) * maxBlocks / parts)};
            partialTask = core;
            partialCount = parts;
            for (int32_t taskId = 0; taskId < tasks; ++taskId) {
                const int32_t start = taskId * parts;
                combineSchedules[combineTaskNum++] = {taskId, start, start, static_cast<int32_t>(parts)};
            }
            return true;
        }
        int32_t firstCore = 0, partialStart = 0, flatStart = 0, sourceCore = 0;
        bool crossesBoundary = false, sourceCrossesBoundary = false;
        for (int32_t task = 0; task < tasks; ++task) {
            int32_t flatEnd = total;
            if (uniform) {
                flatEnd = flatStart + maxBlocks;
            } else if (task + 1 < tasks) {
                // The first core reaching the next task encodes its flat starting offset.
                while (sourceCore < cores && meta->decodeSchedules[sourceCore].baseTaskEnd <= task + 1) {
                    ++sourceCore;
                }
                if (sourceCore >= cores) {
                    return false;
                }
                const auto& range = meta->decodeSchedules[sourceCore];
                sourceCrossesBoundary |= range.baseTaskEnd - range.baseTaskStart > 1;
                if (range.baseTaskStart == task + 1) {
                    flatEnd = sourceCore * limit - range.firstBlockStart;
                } else if (range.baseTaskEnd == task + 2) {
                    const int64_t coreEnd = static_cast<int64_t>(sourceCore + 1) * limit;
                    flatEnd = static_cast<int32_t>(coreEnd < total ? coreEnd : total) - range.lastBlockEnd;
                } else {
                    return false;
                }
            }
            const int32_t blocks = flatEnd - flatStart;
            if (blocks <= 0) {
                return false;
            }
            const int32_t parts = 1 + (blocks - 1) / limit;
            if (parts > cores - firstCore) {
                return false;
            }
            if (parts > 1) {
                combineSchedules[combineTaskNum++] = {task, firstCore, partialStart, parts};
            }
            const int32_t part = static_cast<int32_t>(core) - firstCore;
            if (part >= 0 && part < parts) {
                decode = {task, task + 1, static_cast<int32_t>(static_cast<int64_t>(part) * blocks / parts),
                          static_cast<int32_t>(static_cast<int64_t>(part + 1) * blocks / parts)};
                partialTask = partialStart + part;
                partialCount = parts > 1 ? parts : 0;
            }
            crossesBoundary |= flatStart % limit != 0;
            firstCore += parts;
            partialStart += parts > 1 ? parts : 0;
            flatStart = flatEnd;
        }
        return crossesBoundary && (uniform || sourceCrossesBoundary) && firstCore == cores &&
               partialStart <= meta->fdPartialTaskNum;
    }
};

template <class ElementO, bool outputLse = false>
class GbsaFullQuantKernelArch22 {
    using Tiling = GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData;
    static constexpr uint16_t QK_READY = 1;
    static constexpr uint16_t P_READY = 2;
    static constexpr uint16_t PV_READY = 3;
    static constexpr uint16_t TASK_DONE = 4;

public:
    __aicore__ inline void operator()(const GbsaKernelParamsArch22& params, const FullQuantScales& scales,
                                      const Tiling& tiling)
    {
        const bool hasResidual = tiling.residualBlockMode == 1U;
        if (params.blockTable != nullptr) {
            if (hasResidual) {
                DispatchRows<true, true>(params, scales, tiling);
            } else {
                DispatchRows<true, false>(params, scales, tiling);
            }
        } else if (hasResidual) {
            DispatchRows<false, true>(params, scales, tiling);
        } else {
            DispatchRows<false, false>(params, scales, tiling);
        }
    }

private:
    template <bool isPaged, bool hasResidualBlock>
    __aicore__ inline void DispatchRows(const GbsaKernelParamsArch22& params, const FullQuantScales& scales,
                                        const Tiling& tiling)
    {
        // A single Cube row-fractal has constant strides; larger groups share
        // the same implementation with capacity-driven row/column tiling.
        if (tiling.groupSize <= 16U) {
            if (PackedKv::Enabled(tiling)) {
                if (tiling.blockShapeY >= 4U) {
                    Run<isPaged, hasResidualBlock, true, true, true>(params, scales, tiling);
                } else {
                    Run<isPaged, hasResidualBlock, true, false, true>(params, scales, tiling);
                }
            } else if (tiling.blockShapeY >= 4U) {
                Run<isPaged, hasResidualBlock, true, true, false>(params, scales, tiling);
            } else {
                Run<isPaged, hasResidualBlock, true, false, false>(params, scales, tiling);
            }
        } else {
            Run<isPaged, hasResidualBlock, false>(params, scales, tiling);
        }
    }

    template <bool isPaged, bool hasResidualBlock, bool singleRowTile, bool compactUb = false, bool packedMode = false>
    __aicore__ inline void Run(const GbsaKernelParamsArch22& params, const FullQuantScales& scales,
                               const Tiling& tiling)
    {
        const auto* meta = reinterpret_cast<__gm__ const GbsaMetadata::Metadata*>(params.metaData);
        if (!GsaFd::ValidateMetadata(meta, &tiling, AscendC::GetBlockNum())) {
            return;
        }
        Bind(params, scales, tiling);
        constexpr bool packedKv = packedMode;
        if (packedKv) {
#ifdef __DAV_C220_VEC__
            PackedKv::PackAll<isPaged>(resource_, params, tiling);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(5);
#else
            AscendC::CrossCoreWaitFlag(5);
#endif
            // Every Cube first waits its two local Vector packers; the all-core barrier then covers all pages.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
            key_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.workSpace + PackedKv::Offset(tiling)));
            value_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.workSpace + PackedKv::Offset(tiling) +
                                                                    PackedKv::Bytes(tiling)));
        }
        // Above 127, saturation depends on the partial max; retain the
        // whole-KV quantization order rather than merging saturated partials.
        const bool fdEnabled =
            tiling.fdStaticEnabled != 0U &&
            (static_cast<uint32_t>(meta->fdScheduleFlags) & GbsaMetadata::FD_SCHEDULE_ENABLED) != 0U &&
            pScale_.GetValue(0) <= 127.0f;
        // Match FP16: keep all GQA rows in one task and reuse its K/V loads.
        constexpr uint32_t rowPartitions = 1U;
        const uint32_t baseTasks = static_cast<uint32_t>(meta->saTotalTaskNum);
        const uint32_t rowBlocks = (tiling.groupSize + 15U) / 16U;
        AscendC::GlobalTensor<float> partialO, partialLse, lse;
        partialO.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workSpace + tiling.fdPartialOOffset));
        partialLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workSpace + tiling.fdPartialLseOffset));
        lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.softmaxLse));
        const uint32_t rows = singleRowTile ? 16U : (tiling.groupSize + 15U) / 16U * 16U;
        const uint32_t sBytes = rows * FullQuantLayout::KV_TILE * sizeof(int32_t);
        const uint32_t pBytes = rows * FullQuantLayout::KV_TILE;
        const uint32_t oBytes = rows * FullQuantLayout::HEAD_DIM * sizeof(int32_t);
        const bool batchedQ = tiling.qLayout != 0U;
        const bool headMajorQ = tiling.qLayout == 2U;
        const bool headMajorKv = packedKv || tiling.kvLayout == 2U || tiling.kvLayout == 3U;
        const uint64_t qHeadStride =
            headMajorQ ? static_cast<uint64_t>(tiling.qSeqCapacity) * tiling.embeddingSize : tiling.embeddingSize;
        const uint32_t scaleHeadStride = headMajorQ ? tiling.qSeqCapacity : 1U;
        const uint64_t kvTokenStride =
            headMajorKv ? tiling.embeddingSize : static_cast<uint64_t>(tiling.kvHeads) * tiling.embeddingSize;
        const uint64_t kvHeadStride =
            headMajorKv ?
                static_cast<uint64_t>(isPaged ? tiling.blockSize : tiling.kvSeqCapacity) * tiling.embeddingSize :
                tiling.embeddingSize;
        FixedSequenceOffsets fixedQ{tiling.qSeqCapacity};
        uint32_t core = AscendC::GetBlockIdx();
#ifdef __DAV_C220_VEC__
        core /= AscendC::GetSubBlockNum();
        GbsaFullQuantVectorArch22<ElementO, singleRowTile, compactUb> vector(resource_);
        typename decltype(vector)::EpilogueOnlineSoftmax softmax{vector};
        typename decltype(vector)::EpilogueRescaleO rescaleO{vector};
#else
        GbsaFullQuantCubeArch22<singleRowTile> cube(resource_, tiling.embeddingSize, tiling.kvHeads, headMajorKv,
                                                    packedKv, tiling.blockShapeY);
        typename decltype(cube)::BlockMmadQK qk{cube};
        typename decltype(cube)::BlockMmadPV pv{cube};
        cube.Init();
#endif
        GsaFd::TaskStorageCursor cursor;
        uint32_t taskStart = core;
        uint32_t taskEnd = baseTasks * rowPartitions;
        uint32_t taskStep = AscendC::GetBlockNum();
        uint32_t firstBlock = 0U, lastBlock = 0U;
        FullQuantFdSchedule aligned;
        const bool useAligned =
            fdEnabled && aligned.Build(meta, core, tiling.topK + (hasResidualBlock ? 1U : 0U), AscendC::GetBlockNum());
        if (fdEnabled) {
            taskStart = taskEnd;
            taskStep = 1U;
            if (useAligned && core < aligned.activeCoreNum) {
                taskStart = aligned.decode.baseTaskStart;
                taskEnd = aligned.decode.baseTaskEnd;
                firstBlock = aligned.decode.firstBlockStart;
                lastBlock = aligned.decode.lastBlockEnd;
            } else if (!useAligned && core < static_cast<uint32_t>(meta->fdActiveCoreNum)) {
                const auto& schedule = meta->decodeSchedules[core];
                taskStart = schedule.baseTaskStart;
                taskEnd = schedule.baseTaskEnd;
                firstBlock = schedule.firstBlockStart;
                lastBlock = schedule.lastBlockEnd;
            }
        }
#ifdef __DAV_C220_CUBE__
        bool taskDonePending = false;
#endif
        for (uint32_t work = taskStart; work < taskEnd; work += taskStep) {
            const uint32_t task = work / rowPartitions;
            const uint32_t part = work % rowPartitions;
            const uint32_t groupRowStart = part * rowBlocks / rowPartitions * 16U;
            const uint32_t groupRowEndRound = (part + 1U) * rowBlocks / rowPartitions * 16U;
            const uint32_t localRows =
                singleRowTile ?
                    tiling.groupSize :
                    (groupRowEndRound < tiling.groupSize ? groupRowEndRound : tiling.groupSize) - groupRowStart;
            uint32_t partialTask = aligned.partialTask, partialCount = aligned.partialCount;
            const bool isPartial = useAligned ?
                                       partialCount > 0U :
                                       fdEnabled && GsaFd::FindPartialTask(meta, task, core, partialTask, partialCount);
            uint32_t token = 0;
            uint32_t tokenInBatch = 0;
            uint32_t batch = 0;
            uint32_t head = 0;
            const bool decoded = batchedQ ? cursor.Decode(task, tiling.kvHeads, tiling.batch, fixedQ, sequsedQ_, true,
                                                          token, tokenInBatch, batch, head) :
                                            cursor.Decode(task, tiling.kvHeads, tiling.batch, cuSeqLengthsQ_, sequsedQ_,
                                                          params.sequsedQ != nullptr, token, tokenInBatch, batch, head);
            if (!decoded) {
                continue;
            }
            const uint64_t queryOffset =
                headMajorQ ?
                    (static_cast<uint64_t>(batch) * tiling.numHeads * tiling.qSeqCapacity + tokenInBatch) *
                            tiling.embeddingSize +
                        (head * tiling.groupSize + groupRowStart) * qHeadStride :
                    (static_cast<uint64_t>(token) * tiling.numHeads + head * tiling.groupSize + groupRowStart) *
                        tiling.embeddingSize;
            const uint64_t scaleOffset = queryOffset / tiling.embeddingSize;
            const uint32_t queryLength =
                !batchedQ && params.sequsedQ == nullptr ?
                    static_cast<uint32_t>(cuSeqLengthsQ_.GetValue(batch + 1) - cuSeqLengthsQ_.GetValue(batch)) :
                    sequsedQ_.GetValue(batch);
            uint64_t kvStart = 0;
            uint32_t kvLength = 0;
            if constexpr (isPaged) {
                kvLength = sequsedKv_.GetValue(batch);
            } else {
                kvStart =
                    batchedQ ? static_cast<uint64_t>(batch) * tiling.kvSeqCapacity : cuSeqLengthsKv_.GetValue(batch);
                const uint32_t storageLength =
                    batchedQ ? tiling.kvSeqCapacity : cuSeqLengthsKv_.GetValue(batch + 1) - kvStart;
                kvLength = params.sequsedKv == nullptr ? storageLength : sequsedKv_.GetValue(batch);
            }
            const int64_t causalEnd = static_cast<int64_t>(kvLength) - queryLength + tokenInBatch + 1;
            const uint32_t visibleEnd =
                tiling.maskType == 0U ? kvLength : (causalEnd > 0 ? static_cast<uint32_t>(causalEnd) : 0U);
            const uint32_t queryBlock =
                batchedQ ? token : GsaFd::StorageQBlock(cuSeqLengthsQ_, batch, tokenInBatch, token, 1);
            const uint64_t sparseRow = static_cast<uint64_t>(head) * tiling.qBlockNum + queryBlock;
            int32_t count = counts_.GetValue(sparseRow);
            const uint32_t selected =
                count <= 0 ? 0 :
                             (static_cast<uint32_t>(count) < tiling.topK ? static_cast<uint32_t>(count) : tiling.topK);
            const uint32_t virtualCount =
                selected + static_cast<uint32_t>(hasResidualBlock && visibleEnd % tiling.blockShapeY != 0U);
            const uint32_t rawBegin =
                fdEnabled && task == taskStart ? (firstBlock < virtualCount ? firstBlock : virtualCount) : 0U;
            const uint32_t rawEnd = fdEnabled && task + 1U == taskEnd ?
                                        (lastBlock < virtualCount ? lastBlock : virtualCount) :
                                        virtualCount;
            const uint32_t explicitBegin = rawBegin < selected ? rawBegin : selected;
            const uint32_t explicitEnd = rawEnd < selected ? rawEnd : selected;
            const uint32_t localSelected = explicitEnd - explicitBegin;
            auto indices = indices_[sparseRow * tiling.topK + explicitBegin];
            const uint64_t sourceKvHeadStride = packedKv ?
                                                    (isPaged ? static_cast<uint64_t>(tiling.maxBlocksPerBatch) * 128U :
                                                               static_cast<uint64_t>((kvLength + 127U) / 128U * 128U)) *
                                                        tiling.embeddingSize :
                                                    kvHeadStride;
            const uint64_t dataKvStart = packedKv ?
                                             (isPaged ? static_cast<uint64_t>(batch) * tiling.maxBlocksPerBatch * 128U :
                                                        (kvStart / 128U + batch) * 128U) :
                                             kvStart;
            const uint64_t dataKPageStride = packedKv && isPaged ?
                                                 static_cast<uint64_t>(128U) * tiling.kvHeads * tiling.embeddingSize :
                                                 tiling.kStride0;
            const uint64_t dataVPageStride = packedKv && isPaged ? dataKPageStride : tiling.vStride0;
#ifdef __DAV_C220_CUBE__
            // PA is resolved once by the packer. Cube now reads logical blocks without a page-table lookup.
            constexpr bool sourcePaged = isPaged && !packedMode;
#else
            // Kscale retains its original physical-page input format.
            constexpr bool sourcePaged = isPaged;
#endif
            SparseSource<sourcePaged, hasResidualBlock, AscendC::GlobalTensor<int32_t>> source(
                indices, table_[static_cast<uint64_t>(batch) * tiling.maxBlocksPerBatch], tiling.blockShapeY,
                visibleEnd, tiling.maxBlocksPerBatch, localSelected, tiling.embeddingSize, kvTokenStride,
                sourceKvHeadStride, rawBegin <= selected && rawEnd > selected);
            const uint32_t validBlocks = source.CountValid(localSelected);
            const uint32_t blocksPerTile = FullQuantLayout::KV_TILE / tiling.blockShapeY;
            const uint32_t rounds = (validBlocks + blocksPerTile - 1) / blocksPerTile;
            auto gS = scores_[static_cast<uint64_t>(core) * FullQuantLayout::STAGES * sBytes / sizeof(int32_t)];
            auto gP = probability_[static_cast<uint64_t>(core) * FullQuantLayout::STAGES * pBytes];
            auto gOTmp = temporary_[static_cast<uint64_t>(core) * FullQuantLayout::STAGES * oBytes / sizeof(int32_t)];
#ifdef __DAV_C220_VEC__
            vector.Init(localRows, qScale_[scaleOffset], vScale_[head * tiling.embeddingSize], pScale_,
                        tiling.embeddingSize, scaleHeadStride);
            auto taskKScale = kScale_;
            uint64_t scaleKvStart = kvStart;
            uint64_t scaleKvTokens = tiling.totalKvTokens;
            if (batchedQ && !isPaged) {
                taskKScale = kScale_[static_cast<uint64_t>(batch) * tiling.kvHeads * tiling.kvSeqCapacity];
                scaleKvStart = 0U;
                scaleKvTokens = tiling.kvSeqCapacity;
            }
            const bool useKScaleCache = softmax.template PrepareHeadKScaleCache<isPaged>(
                taskKScale, source, batch, tiling.kvHeads, head, kvLength, scaleKvStart, scaleKvTokens,
                tiling.blockShapeY, !fdEnabled && queryLength > 1U && rounds > 0U && taskStep % tiling.kvHeads == 0U);
#else
            qk.LoadQuery(query_[queryOffset], localRows, head, qHeadStride);
#endif
            Tile tiles[FullQuantLayout::STAGES];
#ifdef __DAV_C220_VEC__
            CopyPlan spanPlan;
#else
            // compactUb dispatch guarantees Y>=4, hence at most32 sparse records per128-token chunk.
            using CubePlan = std::conditional_t<compactUb, CompactVCopyPlan, VCopyPlan>;
            CubePlan kvPlans[FullQuantLayout::STAGES];
#endif
            // Lifetime proof for the three GM slots:
            // QK(t+3) follows PV(t), which requires P(t), hence Softmax consumed S(t).
            // Softmax(t+3) waits for QK(t+3), by which point PrefetchP(t) consumed P(t).
            // PV(t+3) requires later P-ready events, issued after Vector Update(t).
            // Alpha(t) is consumed by Update(t) before Softmax(t+3) on each AIV.
            uint32_t selectionCursor = 0;
            // Use FP16'gS two-round lookahead uniformly for all layouts and shapes.
            const uint32_t lead = rounds < FullQuantLayout::PRE_LAUNCH ? rounds : FullQuantLayout::PRE_LAUNCH;
            for (uint32_t step = 0; step < rounds + lead; ++step) {
                if (step < rounds) {
                    const uint32_t stage = step % FullQuantLayout::STAGES;
#ifdef __DAV_C220_VEC__
                    if (vector.HasRows()) {
                        if (useKScaleCache) {
                            tiles[stage] = softmax.PrefetchCachedKScale(source, selectionCursor, validBlocks, step,
                                                                        tiling.blockShapeY);
                        } else {
                            // Sub-32-byte scale blocks need many padded/gather records.
                            // Keep those records in dead FP16 scratch; larger blocks keep the stack plan.
                            if (tiling.blockShapeY * sizeof(float) < 32U) {
                                auto ubPlan = vector.GetCopyPlan();
                                vector.BeginCopyPlan();
                                tiles[stage] = source.NextTile(selectionCursor, validBlocks, &ubPlan);
                                softmax.template PrefetchKScale<isPaged>(taskKScale, tiles[stage], tiling.kvHeads, head,
                                                                         step, scaleKvStart, scaleKvTokens,
                                                                         tiling.blockShapeY, ubPlan);
                                vector.EndCopyPlan();
                            } else {
                                tiles[stage] = source.NextTile(selectionCursor, validBlocks, &spanPlan);
                                softmax.template PrefetchKScale<isPaged>(taskKScale, tiles[stage], tiling.kvHeads, head,
                                                                         step, scaleKvStart, scaleKvTokens,
                                                                         tiling.blockShapeY, spanPlan);
                            }
                        }
                    }
                    softmax.PrepareQScale(tiling.scaleValue);
                    AscendC::CrossCoreWaitFlag(QK_READY);
                    softmax.Softmax(gS[stage * sBytes / sizeof(int32_t)], gP[stage * pBytes], tiles[stage], step,
                                    tiling.scaleValue);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(P_READY);
#else
                    tiles[stage] = source.DescribeTile(selectionCursor, validBlocks);
                    qk.template Qk<sourcePaged>(key_, gS[stage * sBytes / sizeof(int32_t)], tiles[stage],
                                                dataKPageStride, dataVPageStride, tiling.kvHeads, head, dataKvStart,
                                                source, kvPlans[stage]);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(QK_READY);
#endif
                }
                if (step >= lead) {
                    const uint32_t round = step - lead;
                    const uint32_t stage = round % FullQuantLayout::STAGES;
#ifdef __DAV_C220_VEC__
                    AscendC::CrossCoreWaitFlag(PV_READY);
                    rescaleO.Update(gOTmp[stage * oBytes / sizeof(int32_t)], round);
#else
                    // Match the regular FP16 pipeline: PV(round) consumes only P(round).
                    // Waiting for a future P here would stall an already runnable PV.
                    pv.PrefetchV(value_, tiles[stage], tiling.kvHeads, kvPlans[stage], 0);
                    if (taskDonePending) {
                        // Next-query Q/K/QK may overlap the old Vector tail; old O must be consumed before new PV.
                        AscendC::CrossCoreWaitFlag(TASK_DONE);
                        taskDonePending = false;
                    }
                    AscendC::CrossCoreWaitFlag(P_READY);
                    pv.PrefetchP(gP[stage * pBytes], stage, tiles[stage]);
                    pv.Pv(value_, gOTmp[stage * oBytes / sizeof(int32_t)], tiles[stage], tiling.kvHeads, stage,
                          kvPlans[stage]);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(PV_READY);
#endif
                }
            }
#ifdef __DAV_C220_VEC__
            if (isPartial) {
                rescaleO.StorePartial(partialO, partialLse, partialTask, tiling.groupSize, tiling.fdLseSubStride);
            } else {
                rescaleO.Store(output_[queryOffset], qHeadStride);
                if constexpr (outputLse) {
                    // FP16 contract: batched LSE is [B, Hq, Sq, 1] for both Q layouts.
                    const uint64_t lseOffset =
                        batchedQ ?
                            (static_cast<uint64_t>(batch) * tiling.numHeads + head * tiling.groupSize + groupRowStart) *
                                    tiling.qSeqCapacity +
                                tokenInBatch :
                            scaleOffset;
                    rescaleO.StoreLse(lse[lseOffset], batchedQ ? tiling.qSeqCapacity : 1U);
                }
            }
            vector.Finish();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(TASK_DONE);
#else
            // An empty query has no PV at which to consume the preceding done event.
            if (taskDonePending) {
                AscendC::CrossCoreWaitFlag(TASK_DONE);
            }
            qk.ReleaseQuery();
            taskDonePending = true;
#endif
        }
#ifdef __DAV_C220_CUBE__
        if (taskDonePending) {
            AscendC::CrossCoreWaitFlag(TASK_DONE);
        }
        cube.Finish();
#endif
        if (fdEnabled) {
#ifdef __DAV_C220_VEC__
            if (!useAligned) {
                WriteEmptyUnscheduled(meta, params, tiling, vector, core);
            }
#endif
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
#ifdef __DAV_C220_VEC__
            GenericBlockSparseAttentionFdCombineArch22<ElementO, NpuArch::Arch::Int8StaticResource, outputLse> combine(
                resource_);
            if (useAligned && batchedQ) {
                combine(&aligned, &tiling, partialLse, partialO, output_, lse, fixedQ, sequsedQ_, true);
            } else if (useAligned) {
                combine(&aligned, &tiling, partialLse, partialO, output_, lse, cuSeqLengthsQ_, sequsedQ_,
                        params.sequsedQ != nullptr);
            } else if (batchedQ) {
                combine(meta, &tiling, partialLse, partialO, output_, lse, fixedQ, sequsedQ_, true);
            } else {
                combine(meta, &tiling, partialLse, partialO, output_, lse, cuSeqLengthsQ_, sequsedQ_,
                        params.sequsedQ != nullptr);
            }
#endif
        }
    }

    template <bool singleRowTile, bool compactUb>
    __aicore__ inline void WriteEmptyUnscheduled(__gm__ const GbsaMetadata::Metadata* meta,
                                                 const GbsaKernelParamsArch22& params, const Tiling& tiling,
                                                 GbsaFullQuantVectorArch22<ElementO, singleRowTile, compactUb>& vector,
                                                 uint32_t core)
    {
        typename GbsaFullQuantVectorArch22<ElementO, singleRowTile, compactUb>::EpilogueRescaleO rescaleO{vector};
        const bool batched = tiling.qLayout != 0U;
        const uint64_t headStride = tiling.qLayout == 2U ?
                                        static_cast<uint64_t>(tiling.qSeqCapacity) * tiling.embeddingSize :
                                        tiling.embeddingSize;
        FixedSequenceOffsets fixed{tiling.qSeqCapacity};
        GsaFd::TaskStorageCursor cursor;
        for (uint32_t task = core; task < static_cast<uint32_t>(meta->saTotalTaskNum); task += AscendC::GetBlockNum()) {
            bool scheduled = false;
            for (uint32_t i = 0U; i < static_cast<uint32_t>(meta->fdActiveCoreNum); ++i) {
                const auto& schedule = meta->decodeSchedules[i];
                scheduled |= task >= static_cast<uint32_t>(schedule.baseTaskStart) &&
                             task < static_cast<uint32_t>(schedule.baseTaskEnd);
            }
            if (scheduled) {
                continue;
            }
            uint32_t token = 0U, inBatch = 0U, batch = 0U, head = 0U;
            const bool decoded = batched ? cursor.Decode(task, tiling.kvHeads, tiling.batch, fixed, sequsedQ_, true,
                                                         token, inBatch, batch, head) :
                                           cursor.Decode(task, tiling.kvHeads, tiling.batch, cuSeqLengthsQ_, sequsedQ_,
                                                         params.sequsedQ != nullptr, token, inBatch, batch, head);
            if (!decoded) {
                continue;
            }
            const uint64_t offset =
                tiling.qLayout == 2U ?
                    (static_cast<uint64_t>(batch) * tiling.numHeads * tiling.qSeqCapacity + inBatch) *
                            tiling.embeddingSize +
                        head * tiling.groupSize * headStride :
                    (static_cast<uint64_t>(token) * tiling.numHeads + head * tiling.groupSize) * tiling.embeddingSize;
            vector.Init(tiling.groupSize, qScale_[offset / tiling.embeddingSize], vScale_[head * tiling.embeddingSize],
                        pScale_, tiling.embeddingSize, tiling.qLayout == 2U ? tiling.qSeqCapacity : 1U);
            rescaleO.Store(output_[offset], headStride);
            if constexpr (outputLse) {
                AscendC::GlobalTensor<float> lse;
                lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.softmaxLse));
                const uint64_t lseOffset =
                    batched ? (static_cast<uint64_t>(batch) * tiling.numHeads + head * tiling.groupSize) *
                                      tiling.qSeqCapacity +
                                  inBatch :
                              offset / tiling.embeddingSize;
                rescaleO.StoreLse(lse[lseOffset], batched ? tiling.qSeqCapacity : 1U);
            }
            vector.Finish();
        }
    }

    __aicore__ inline void Bind(const GbsaKernelParamsArch22& params, const FullQuantScales& scales,
                                const Tiling& tiling)
    {
        query_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.q));
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.k));
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.v));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementO*>(params.o));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sparseBlockIdx));
        counts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sparseBlockCount));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.blockTable));
        cuSeqLengthsQ_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengths));
        cuSeqLengthsKv_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengthsKv));
        sequsedQ_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedQ));
        sequsedKv_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedKv));
        qScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales.query));
        kScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales.key));
        vScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales.value));
        pScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales.probability));
        scores_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workSpace));
        probability_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.workSpace + tiling.mm1OutSize));
        temporary_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(params.workSpace + tiling.mm1OutSize + tiling.smOnlineOutSize));
    }

    NpuArch::Arch::Int8StaticResource resource_;
    AscendC::GlobalTensor<int8_t> query_, key_, value_, probability_;
    AscendC::GlobalTensor<int32_t> indices_, counts_, table_, sequsedQ_, sequsedKv_, scores_, temporary_;
    AscendC::GlobalTensor<int64_t> cuSeqLengthsQ_, cuSeqLengthsKv_;
    AscendC::GlobalTensor<float> qScale_, kScale_, vScale_, pScale_;
    AscendC::GlobalTensor<ElementO> output_;
};

} // namespace GbsaKernelArch22::Int8
#endif
