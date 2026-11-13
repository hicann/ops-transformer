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
#include "qsag_common.h"
#include "qsag_cube_op.h"
#include "qsag_vec_op.h"

namespace QSAG {

template <typename T, typename Tiling>
class Kernel {
    struct QueryCache {
        uint64_t batch = 0;
        uint64_t qbase = 0;
        uint64_t qend = 0;
        uint64_t kbase = 0;
        uint64_t qlen = 0;
        uint64_t klen = 0;
        bool valid = false;
    } queryCache_;
    ConstInfo<Tiling> info_;
    SharedBuffers<T> shared_;
    VectorOp<T, Tiling> vector_;
    CubeOp<T, Tiling> cube_;
    GlobalTensor<int32_t> counts_, filteredCounts_, usedQ_, usedK_;
    GlobalTensor<int64_t> cuQ_, cuK_;
    uint32_t querySlot_ = 0;
    uint32_t indexOwnerSlot_ = 0;

    __aicore__ inline bool QueryInfo(uint64_t token, uint64_t& batch, uint64_t& qlocal, uint64_t& kbase, uint64_t& qlen,
                                     uint64_t& klen)
    {
        if (info_.tiling->inputLayout != 0) {
            batch = token / info_.tiling->qSeqLen;
            qlocal = token % info_.tiling->qSeqLen;
            kbase = batch * info_.tiling->kvSeqLen;
            qlen = info_.tiling->qSeqLen;
            klen = info_.tiling->kvSeqLen;
        } else {
            // 轮转分核保证本核 token 单调递增；只在跨 batch 时读取序列信息。
            auto& cache = queryCache_;
            if (token >= cache.qend) {
                cache.valid = false;
                while (cache.batch < info_.tiling->batchNum) {
                    cache.qend = static_cast<uint64_t>(cuQ_.GetValue(cache.batch + 1));
                    if (token < cache.qend) {
                        break;
                    }
                    ++cache.batch;
                }
                if (cache.batch == info_.tiling->batchNum) {
                    return false;
                }
                cache.qbase = cuQ_.GetValue(cache.batch);
                cache.kbase = cuK_.GetValue(cache.batch);
                cache.qlen = cache.qend - cache.qbase;
                cache.klen = cuK_.GetValue(cache.batch + 1) - cache.kbase;
                if (info_.tiling->hasUsedQ) {
                    int32_t used = usedQ_.GetValue(cache.batch);
                    if (used < 0 || static_cast<uint64_t>(used) > cache.qlen) {
                        return false;
                    }
                    cache.qlen = used;
                }
                if (info_.tiling->hasUsedKv) {
                    int32_t used = usedK_.GetValue(cache.batch);
                    if (used < 0 || static_cast<uint64_t>(used) > cache.klen) {
                        return false;
                    }
                    cache.klen = used;
                }
                cache.valid = true;
            }
            if (!cache.valid) {
                return false;
            }
            batch = cache.batch;
            qlocal = token - cache.qbase;
            kbase = cache.kbase;
            qlen = cache.qlen;
            klen = cache.klen;
        }
        return qlocal < qlen && qlocal < info_.tiling->qCapacity;
    }

    __aicore__ inline uint64_t VisibleLength(uint64_t qlocal, uint64_t qlen, uint64_t klen) const
    {
        if (info_.tiling->maskType == 0) {
            return klen;
        }
        int64_t causal = static_cast<int64_t>(klen) - static_cast<int64_t>(qlen) + static_cast<int64_t>(qlocal) + 1;
        return causal <= 0 ? 0 : Min(klen, static_cast<uint64_t>(causal));
    }

    __aicore__ inline void FilterPatterns()
    {
        queryCache_ = {};
        const uint64_t rows = info_.tq * info_.tiling->kvHeadNum;
        for (uint64_t work = info_.vcore; work < rows; work += info_.tiling->cubeCoreNum * AIV_PER_CUBE) {
            const uint64_t token = work / info_.tiling->kvHeadNum;
            const uint64_t head = work % info_.tiling->kvHeadNum;
            uint64_t batch, qlocal, kbase, qlen, klen;
            if (!QueryInfo(token, batch, qlocal, kbase, qlen, klen)) {
                continue;
            }
            uint64_t row = (batch * info_.tiling->kvHeadNum + head) * info_.tiling->qCapacity + qlocal;
            vector_.FilterQuery(row, VisibleLength(qlocal, qlen, klen), counts_.GetValue(row));
        }
        queryCache_ = {};
    }

    __aicore__ inline void ProcessGather(const TileTask& task)
    {
        if ASCEND_IS_AIV {
            uint32_t slot = task.id % SLOT_COUNT;
            CrossCoreWaitFlag<2, PIPE_MTE3>(KV_FREE + slot);
            vector_.Gather(task);
            CrossCoreSetFlag<4, PIPE_MTE3>(GATHER_READY + slot + info_.sub * AIV_FLAG_OFFSET);
        }
    }

    __aicore__ inline void ProcessMm12(const TileTask& task, bool firstTile)
    {
        if ASCEND_IS_AIC {
            uint32_t slot = task.id % SLOT_COUNT;
            CrossCoreWaitFlag<4, PIPE_MTE2>(GATHER_READY + slot);
            CrossCoreWaitFlag<4, PIPE_MTE2>(GATHER_READY + slot + AIV_FLAG_OFFSET);
            cube_.Mm12(task, firstTile);
            CrossCoreSetFlag<2, PIPE_FIX>(SCORE_READY + slot);
        }
    }

    __aicore__ inline void ProcessSoftmax(const TileTask& task)
    {
        if ASCEND_IS_AIV {
            uint32_t slot = task.id % SLOT_COUNT;
            vector_.PrepareSoftmax(task);
            CrossCoreWaitFlag<2, PIPE_V>(SCORE_READY + slot);
            vector_.Softmax(task.selected, slot);
            CrossCoreSetFlag<4, PIPE_MTE3>(PROBABILITY_READY + slot + info_.sub * AIV_FLAG_OFFSET);
        }
    }

    __aicore__ inline void ProcessMm345(const TileTask& task)
    {
        if ASCEND_IS_AIC {
            uint32_t slot = task.id % SLOT_COUNT;
            CrossCoreWaitFlag<4, PIPE_MTE1>(PROBABILITY_READY + slot);
            CrossCoreWaitFlag<4, PIPE_MTE1>(PROBABILITY_READY + slot + AIV_FLAG_OFFSET);
            cube_.Mm345(task);
            CrossCoreSetFlag<2, PIPE_FIX>(RESULT_READY + slot);
        }
    }

    __aicore__ inline void ProcessScatter(const TileTask& task)
    {
        if ASCEND_IS_AIV {
            uint32_t slot = task.id % SLOT_COUNT;
            if (info_.directResult) {
                CrossCoreWaitFlag<2, PIPE_V>(RESULT_READY + slot);
            } else {
                CrossCoreWaitFlag<2, PIPE_MTE2>(RESULT_READY + slot);
                Fence<HardEvent::MTE2_MTE3>();
            }
            vector_.Scatter(task);
            CrossCoreSetFlag<4, PIPE_MTE3>(RESULT_FREE + info_.ResultSlot(slot) + info_.sub * AIV_FLAG_OFFSET);
        }
    }

    __aicore__ inline void DrainPipeline(TileTask tasks[PIPELINE_STAGES], uint64_t nextTask, uint64_t firstTask)
    {
        if (nextTask == firstTask) {
            return;
        }
        const auto& last = tasks[(nextTask - 1) % PIPELINE_STAGES];
        ProcessSoftmax(last);
        ProcessMm345(last);
        if (nextTask - firstTask >= 2) {
            ProcessScatter(tasks[(nextTask - 2) % PIPELINE_STAGES]);
        }
        ProcessScatter(last);
    }

public:
    __aicore__ inline void Run(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dout, GM_ADDR out, GM_ADDR lse,
                               GM_ADDR indices, GM_ADDR counts, GM_ADDR cuQ, GM_ADDR cuK, GM_ADDR usedQ, GM_ADDR usedK,
                               GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const Tiling* tiling, TPipe* pipe)
    {
        info_.tiling = tiling;
        info_.blockSize = static_cast<uint32_t>(info_.tiling->blockSize);
        queryCache_ = {};
        info_.d = info_.tiling->headDim;
        info_.dp = info_.tiling->headDimAlign;
        // 保留 host 预留的 workspace；小稀疏块仍使用原来的较宽 tile。
        info_.n = info_.tiling->kTile == 256 && info_.blockSize >= 8 && info_.dp > 64 ? 128 : info_.tiling->kTile;
        info_.groupRows = info_.tiling->qGroup;
        info_.halfRows = (info_.groupRows + AIV_PER_CUBE - 1) / AIV_PER_CUBE;
        info_.groupCapacity = (AIV_PER_CUBE * info_.halfRows + NZ_WIDTH - 1) / NZ_WIDTH * NZ_WIDTH;
        info_.halfCapacity = info_.groupCapacity / AIV_PER_CUBE;
        if (info_.groupRows <= LARGE_GROUP_LIMIT) {
            info_.vectorRows = LARGE_GROUP_LIMIT / AIV_PER_CUBE; // 64
            info_.dTile = WIDE_D_TILE;
        } else {
            info_.vectorRows = MAX_GROUP_ROWS / AIV_PER_CUBE; // 128
            info_.dTile = NARROW_D_TILE;
        }
        info_.dCache = Min(info_.dTile, info_.dp);
        info_.querySlots = info_.groupRows <= SMALL_GROUP_LIMIT ? SLOT_COUNT : 1;
        info_.nzRows = info_.vectorRows + 1;
        // Bound extra scratch to 32 KiB / 16 KiB for the two UB capacity classes.
        info_.transferElements = info_.groupRows <= LARGE_GROUP_LIMIT ? 4096 : 2048;

        info_.core = GetBlockIdx();
        info_.sub = 0;
        if ASCEND_IS_AIV {
            info_.core = GetBlockIdx() / AIV_PER_CUBE;
            info_.sub = GetSubBlockIdx();
        }
        info_.vcore = info_.core * AIV_PER_CUBE + info_.sub;

        if (info_.tiling->inputLayout == 0) {
            // TND
            info_.tq = info_.tiling->qSeqLen;
            info_.tk = info_.tiling->kvSeqLen;
        } else {
            info_.tq = info_.tiling->batchNum * info_.tiling->qSeqLen;
            info_.tk = info_.tiling->batchNum * info_.tiling->kvSeqLen;
        }

        counts_.SetGlobalBuffer((__gm__ int32_t*)counts);
        filteredCounts_.SetGlobalBuffer((__gm__ int32_t*)(workspace + info_.tiling->filteredCountsOffset));
        cuQ_.SetGlobalBuffer((__gm__ int64_t*)cuQ);
        cuK_.SetGlobalBuffer((__gm__ int64_t*)cuK);
        usedQ_.SetGlobalBuffer((__gm__ int32_t*)usedQ);
        usedK_.SetGlobalBuffer((__gm__ int32_t*)usedK);

        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            uint64_t offset = (info_.core * SLOT_COUNT + slot) * info_.n * info_.dp;
            shared_.selK[slot].SetGlobalBuffer((__gm__ T*)(workspace + info_.tiling->selectedKOffset) + offset);
            shared_.selV[slot].SetGlobalBuffer((__gm__ T*)(workspace + info_.tiling->selectedVOffset) + offset);
            shared_.mm4[slot].SetGlobalBuffer((__gm__ float*)(workspace + info_.tiling->mm4Offset) + offset);
            shared_.mm5[slot].SetGlobalBuffer((__gm__ float*)(workspace + info_.tiling->mm5Offset) + offset);
        }
        info_.resultSlots = SLOT_COUNT;
        info_.directResult = info_.dp <= 128 && info_.dp <= info_.dTile;
        if (info_.directResult && info_.template PlanUb<T>(true, info_.resultSlots).totalBytes > info_.tiling->ubSize) {
            // 尝试走直接路径
            // 如果双槽对应的UB需求超出容量，计算单槽的UB需求是否满足
            info_.resultSlots = 1;
            info_.directResult = info_.template PlanUb<T>(true, info_.resultSlots).totalBytes <= info_.tiling->ubSize;
        }
        if (!info_.directResult) {
            // 单槽也不满足，退回普通路径
            info_.resultSlots = info_.n > 128 ? 1 : SLOT_COUNT;
        }
        if (!info_.directResult && info_.groupRows <= SMALL_GROUP_LIMIT && info_.dp > WIDE_D_TILE) {
            const auto cubePlan =
                MakeCubeCapacityPlan<T>(info_.groupCapacity, info_.n, info_.dp, MAX_HEAD_DIM, info_.querySlots);
            if (cubePlan.fits) {
                const uint32_t originalTransfer = info_.transferElements;
                const uint32_t originalSlots = info_.resultSlots;
                // Prefer the larger transfer batch; shrink it only when the
                // full result cannot fit. Gather derives its rows from this.
                for (uint32_t attempt = 0; attempt < 2; ++attempt) {
                    info_.transferElements = attempt == 0 ? originalTransfer : originalTransfer / 2;
                    info_.resultSlots = info_.n > 128 ? 1 : SLOT_COUNT;
                    if (info_.template PlanUb<T>(true, info_.resultSlots).totalBytes > info_.tiling->ubSize) {
                        info_.resultSlots = 1;
                    }
                    if (info_.template PlanUb<T>(true, info_.resultSlots).totalBytes <= info_.tiling->ubSize) {
                        info_.directResult = true;
                        info_.dTile = MAX_HEAD_DIM;
                        info_.dCache = info_.dp;
                        break;
                    }
                }
                if (!info_.directResult) {
                    info_.transferElements = originalTransfer;
                    info_.resultSlots = originalSlots;
                }
            }
        }
        info_.nzRows = (info_.directResult ? info_.halfCapacity : info_.vectorRows) + 1;
        info_.pairedKV = info_.directResult && info_.d == info_.dp;
        info_.cacheQueryIndices =
            info_.directResult && info_.template PlanUb<T>(true, info_.resultSlots, true).topkBytes >
                                      AlignBytes(info_.n / info_.blockSize * sizeof(int32_t));
        vector_.Init(key, value, dout, out, lse, indices, dq, dk, dv, workspace, info_, shared_, pipe);
        cube_.Init(query, dout, workspace, info_, shared_, pipe);
        if ASCEND_IS_AIV {
            FilterPatterns();
            vector_.Begin(pipe);
            CrossCoreSetFlag<4, PIPE_MTE3>(FILTER_READY + info_.sub * AIV_FLAG_OFFSET);
        }
        if ASCEND_IS_AIC {
            // Counts are read on Scalar, after every AIV has finished the preprocessing barrier.
            CrossCoreWaitFlag<4, PIPE_S>(FILTER_READY);
            CrossCoreWaitFlag<4, PIPE_S>(FILTER_READY + AIV_FLAG_OFFSET);
        }

        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<2, PIPE_FIX>(KV_FREE + slot);
            }
            if ASCEND_IS_AIV {
                if (slot < info_.resultSlots) {
                    CrossCoreSetFlag<4, PIPE_MTE3>(RESULT_FREE + info_.ResultSlot(slot) + info_.sub * AIV_FLAG_OFFSET);
                }
            }
        }
        TileTask tasks[PIPELINE_STAGES];
        uint64_t nextTask = 0;
        const uint64_t queryRows = info_.tq * info_.tiling->kvHeadNum;
        const uint32_t queryParts = info_.tiling->queryParts;
        const uint32_t queryGroups = info_.tiling->queryGroups;
        const uint64_t firstQuery = info_.core < queryGroups * queryParts ? info_.core / queryParts : queryRows;
        const uint32_t queryPart = info_.core % queryParts;
        queryCache_ = {};
        for (uint64_t work = firstQuery; work < queryRows; work += queryGroups) {
            const uint64_t token = work / info_.tiling->kvHeadNum;
            const uint64_t h = work % info_.tiling->kvHeadNum;
            uint64_t batch, qlocal, kbase, qlen, klen;
            if (!QueryInfo(token, batch, qlocal, kbase, qlen, klen)) {
                continue;
            }
            uint64_t limit = VisibleLength(qlocal, qlen, klen);
            uint64_t complete = limit / info_.blockSize, tailLen = limit % info_.blockSize;
            uint64_t perTile = info_.n / info_.blockSize;
            uint64_t patternRow = (batch * info_.tiling->kvHeadNum + h) * info_.tiling->qCapacity + qlocal;
            uint64_t quota = static_cast<uint32_t>(filteredCounts_.GetValue(patternRow));
            uint64_t sparseBase = patternRow * info_.tiling->topK;
            uint64_t fullTiles = (quota + perTile - 1) / perTile;
            uint32_t rows = info_.groupRows;
            uint64_t head = h * info_.tiling->qGroup;
            if ASCEND_IS_AIV {
                // AIV缓存LSE和Di
                if (info_.querySlots != SLOT_COUNT) {
                    vector_.CacheQueryStatsVector(token, head, rows);
                }
            }
            uint64_t firstTask = nextTask;
            uint64_t pipelineStart = info_.querySlots == SLOT_COUNT ? 0 : firstTask;
            const uint64_t tileCount = fullTiles + (tailLen != 0);
            const uint64_t tileBegin = tileCount * queryPart / queryParts;
            const uint64_t tileEnd = tileCount * (queryPart + 1) / queryParts;
            for (uint64_t tile = tileBegin; tile < tileEnd; ++tile) {
                bool tail = tile == fullTiles;
                uint64_t first = tile * perTile;
                uint32_t entries = tail ? 0 : Min(perTile, quota - first);
                uint32_t selected = tail ? tailLen : entries * info_.blockSize;
                auto& current = tasks[nextTask % PIPELINE_STAGES];
                current = {nextTask,
                           token,
                           head,
                           sparseBase,
                           first,
                           complete * info_.blockSize,
                           kbase,
                           h,
                           selected,
                           tail,
                           querySlot_,
                           indexOwnerSlot_,
                           static_cast<uint32_t>(quota),
                           nextTask == firstTask};
                ProcessGather(current);
                ProcessMm12(current, nextTask == firstTask);
                if (nextTask > pipelineStart) {
                    const auto& previous = tasks[(nextTask - 1) % PIPELINE_STAGES];
                    ProcessSoftmax(previous);
                    ProcessMm345(previous);
                }
                if (nextTask - pipelineStart >= 2) {
                    ProcessScatter(tasks[(nextTask - 2) % PIPELINE_STAGES]);
                }
                ++nextTask;
            }
            if (info_.querySlots != SLOT_COUNT) {
                DrainPipeline(tasks, nextTask, firstTask);
            }
            if (nextTask != firstTask) {
                querySlot_ = (querySlot_ + 1) % info_.querySlots;
                indexOwnerSlot_ = (indexOwnerSlot_ + 1) % PIPELINE_STAGES;
            }
        }
        if (info_.querySlots == SLOT_COUNT) {
            DrainPipeline(tasks, nextTask, 0);
        }
        // 空输入也走同一收尾：消费初始或最后一次释放的空闲令牌。
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            if ASCEND_IS_AIC {
                if (slot < info_.resultSlots) {
                    CrossCoreWaitFlag<4, PIPE_FIX>(RESULT_FREE + info_.ResultSlot(slot));
                    CrossCoreWaitFlag<4, PIPE_FIX>(RESULT_FREE + info_.ResultSlot(slot) + AIV_FLAG_OFFSET);
                }
            }
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag<2, PIPE_MTE3>(KV_FREE + slot);
            }
        }
        if ASCEND_IS_AIV {
            vector_.Finish(pipe);
        }
        if ASCEND_IS_AIC {
            cube_.Destroy(pipe);
        }
    }
};
} // namespace QSAG
