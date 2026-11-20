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
 * \file qsag_orchestrator.h
 * \brief arch22 QSAG orchestrator: task pipeline, cross-core flags and AIC/AIV dispatch.
 *
 * The cube work lives in qsag_cube_engine.h and the vector work in
 * qsag_vector_engine.h; this file only owns the shared task loop and
 * the AIC<->AIV handshake.
 */
#ifndef ATTENTION_QSAG_KERNEL_ARCH22_H
#define ATTENTION_QSAG_KERNEL_ARCH22_H

#include "qsag_common_arch22.h"
#ifdef __DAV_C220_CUBE__
#include "qsag_cube_engine.h"
#else
#include "qsag_vector_engine.h"
#endif

namespace QSAG {

template <typename T, typename Tiling>
class Kernel {
public:
    __aicore__ inline void Run(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dout, GM_ADDR out, GM_ADDR lse,
                               GM_ADDR indices, GM_ADDR counts, GM_ADDR cuQ, GM_ADDR cuK, GM_ADDR usedQ, GM_ADDR usedK,
                               GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const Tiling* tiling, TPipe* pipe)
    {
        InitParams(tiling);
        InitGlobalTensors(query, key, value, dout, out, lse, indices, counts, cuQ, cuK, usedQ, usedK, dq, dk, dv,
                          workspace);

#ifdef __DAV_C220_CUBE__
        cube_.Init(pipe, context_.params);
        // Pre() 末尾的 SyncAll<false> 需 AIC 侧在此 rendezvous（对齐生产 k_out/GBSAG 的
        // 双侧调用模式）；AIC 侧自带 PipeBarrier<PIPE_ALL>，两侧同点汇聚。
        AscendC::SyncAll<false>();
#else
        vector_.Init(pipe, context_.params);
        // 过滤必须在主循环读取 filteredCounts 之前完成；Pre() 末尾的 SyncAll
        // 与 AIC 侧汇聚，天然充当过滤结果的全局屏障。
        // NONE 无因果裁剪：接口要求块索引本身有效，直接消费原始前缀即可，免拷贝。
        if (context_.tiling->maskType != 0) {
            FilterPatterns();
        }
        vector_.Pre(context_);
#endif

        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
#ifdef __DAV_C220_CUBE__
            CrossCoreSetFlag<2, PIPE_FIX>(KV_FREE);
#else
            if (slot < context_.params.resultSlots) {
                CrossCoreSetFlag<2, PIPE_MTE3>(RESULT_FREE);
            }
#endif
        }

        TileTask tasks[PIPELINE_STAGES];
        uint64_t nextTask = 0;
        for (uint64_t kvHead = 0; kvHead < tiling->kvHeadNum; ++kvHead) {
            // Keep one KV head's K/V and gradient working set hot across queries.
            queryCache_ = {};
            for (uint64_t token = context_.params.coreIndex; token < context_.params.totalQueryTokens;
                 token += tiling->cubeCoreNum) {
                uint64_t batch = 0;
                uint64_t qlocal = 0;
                uint64_t kbase = 0;
                uint64_t qlen = 0;
                uint64_t klen = 0;
                if (!QueryInfo(token, batch, qlocal, kbase, qlen, klen)) {
                    continue;
                }
                const uint64_t limit = CausalLimit(qlen, klen, qlocal);
                const uint64_t complete = limit / context_.params.blockSize;
                const uint64_t tailLength = limit % context_.params.blockSize;
                const uint64_t blocksPerTile = context_.params.kvTileSize / context_.params.blockSize;
                const uint64_t patternRow = (batch * tiling->kvHeadNum + kvHead) * tiling->qCapacity + qlocal;
                // FilterPatterns 已按 causal 可见范围把索引压缩到前缀，count 即有效块数。
                const int32_t count = context_.filteredCounts.GetValue(patternRow);
                if (count < 0) {
                    continue;
                }
                const uint64_t quota = static_cast<uint64_t>(count);
                const uint64_t sparseBase = patternRow * tiling->topK;
                const uint64_t tileCount = (quota + blocksPerTile - 1) / blocksPerTile;
                const uint64_t head = kvHead * tiling->qGroup;
#ifndef __DAV_C220_CUBE__
                if (context_.params.queryBufferSlots != SLOT_COUNT) {
                    vector_.EnsureRowStats(context_, token, head);
                }
#endif
                const uint64_t firstTask = nextTask;
                const uint64_t pipelineStart = context_.params.queryBufferSlots == SLOT_COUNT ? 0 : firstTask;
                for (uint64_t tile = 0; tile < tileCount + (tailLength != 0); ++tile) {
                    const bool tail = tile == tileCount;
                    const uint64_t first = tile * blocksPerTile;
                    const uint32_t entries = tail ? 0 : Min(blocksPerTile, quota - first);
                    const uint32_t selected =
                        tail ? static_cast<uint32_t>(tailLength) : entries * context_.params.blockSize;
                    // Gather/Scatter load the tile indices once into UB; avoid
                    // serial GM scalar scans on the AIC and both AIVs here.
                    auto& current = tasks[nextTask % PIPELINE_STAGES];
                    current = {nextTask, token,  head,     sparseBase, first,     complete * context_.params.blockSize,
                               kbase,    kvHead, selected, tail,       querySlot_};
#ifdef __DAV_C220_CUBE__
                    cube_.ProcessMm12(context_, current, nextTask == firstTask);
#else
                    vector_.Gather(context_, current);
#endif
                    if (nextTask > pipelineStart) {
                        const auto& previous = tasks[(nextTask - 1) % PIPELINE_STAGES];
#ifdef __DAV_C220_CUBE__
                        cube_.ProcessMm345(context_, previous);
#else
                        vector_.ProcessProbability(context_, previous);
                        if (nextTask - pipelineStart >= 2) {
                            vector_.Scatter(context_, tasks[(nextTask - 2) % PIPELINE_STAGES]);
                        }
#endif
                    }
                    ++nextTask;
                }
                if (context_.params.queryBufferSlots != SLOT_COUNT) {
                    DrainPipeline(tasks, nextTask, firstTask);
                }
                // Empty owners do not consume a Q/dO buffer slot.
                if (nextTask != firstTask) {
                    querySlot_ = (querySlot_ + 1) % context_.params.queryBufferSlots;
                }
            }
        }
        if (context_.params.queryBufferSlots == SLOT_COUNT) {
            DrainPipeline(tasks, nextTask, 0);
        }
        // 空输入也走同一收尾：消费初始或最后一次释放的空闲令牌。
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
#ifdef __DAV_C220_CUBE__
            if (slot < context_.params.resultSlots) {
                AscendC::WaitEvent(RESULT_FREE);
            }
#else
            AscendC::WaitEvent(KV_FREE);
#endif
        }
#ifndef __DAV_C220_CUBE__
        PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<true>();
        vector_.Post(context_);
#endif
    }

private:
    struct QueryCache {
        uint64_t batch = 0;
        uint64_t qbase = 0;
        uint64_t qend = 0; // 存储边界；usedQ 截短不改变 batch 边界。
        uint64_t kbase = 0;
        uint64_t qlen = 0;
        uint64_t klen = 0;
        bool valid = false;
    };

    __aicore__ inline void InitParams(const Tiling* tiling)
    {
        context_.tiling = tiling;
        QsagParams& params = context_.params;
        params.cubeCoreNum = static_cast<uint32_t>(tiling->cubeCoreNum);
        params.blockSize = static_cast<uint32_t>(tiling->blockSize);
        params.blockShift = 0;
        while ((1U << params.blockShift) < params.blockSize) {
            ++params.blockShift;
        }
        params.powerOfTwoBlock = (1U << params.blockShift) == params.blockSize;
        params.headDim = static_cast<uint32_t>(tiling->headDim);
        params.headDimAligned = static_cast<uint32_t>(tiling->headDimAlign);
        params.kvTileSize = static_cast<uint32_t>(tiling->kTile);
        params.groupRows = static_cast<uint32_t>(tiling->qGroup);
        params.rowsPerAiv = (params.groupRows + AIV_PER_CUBE - 1) / AIV_PER_CUBE;
        params.groupCapacity = static_cast<uint32_t>(tiling->gCapacity);
        params.rowCapacityPerAiv = params.groupCapacity / AIV_PER_CUBE;
        params.dTile = static_cast<uint32_t>(tiling->dTile);
        // 只保留本 AIV 实际持有的 padded Q 行；释放的 UB 在 G<=64 时容纳
        // 半 tile 的 FP32 dK/dV staging。上限与原有分档保持一致兜底。
        params.vectorRows = Min(params.rowCapacityPerAiv, MAX_GROUP_ROWS / AIV_PER_CUBE);
        params.dCache = static_cast<uint32_t>(Min(params.dTile, params.headDimAligned));
        params.queryBufferSlots = static_cast<uint32_t>(tiling->qDoSlots);
        params.transferElements = params.groupRows <= 64 ? 16384 : params.groupRows <= LARGE_GROUP_LIMIT ? 8192 : 2048;
        // Two result slots overlap MM4/MM5 with Scatter for short D or larger sparse blocks.
        // Long D with small blocks keeps one in-flight result to avoid Scatter pressure.
        params.resultSlots =
            params.kvTileSize <= 128 || params.headDimAligned <= 128 || params.blockSize >= 32 ? SLOT_COUNT : 1;

#ifdef __DAV_C220_CUBE__
        params.coreIndex = GetBlockIdx();
        params.aivIndex = 0;
#else
        params.coreIndex = GetBlockIdx() / AIV_PER_CUBE;
        params.aivIndex = GetSubBlockIdx();
#endif
        params.globalAivIndex = params.coreIndex * AIV_PER_CUBE + params.aivIndex;
        if (tiling->inputLayout == 0) {
            // TND 的序列维已经是所有 batch 的 token 总数。
            params.totalQueryTokens = tiling->qSeqLen;
            params.totalKvTokens = tiling->kvSeqLen;
        } else {
            params.totalQueryTokens = tiling->batchNum * tiling->qSeqLen;
            params.totalKvTokens = tiling->batchNum * tiling->kvSeqLen;
        }
    }

    __aicore__ inline void InitGlobalTensors(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dout, GM_ADDR out,
                                             GM_ADDR lse, GM_ADDR indices, GM_ADDR counts, GM_ADDR cuQ, GM_ADDR cuK,
                                             GM_ADDR usedQ, GM_ADDR usedK, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                             GM_ADDR workspace)
    {
        const QsagParams& params = context_.params;
        const Tiling* tiling = context_.tiling;
        context_.query.SetGlobalBuffer((__gm__ T*)query);
        context_.key.SetGlobalBuffer((__gm__ T*)key);
        context_.value.SetGlobalBuffer((__gm__ T*)value);
        context_.dOut.SetGlobalBuffer((__gm__ T*)dout);
        context_.attentionOut.SetGlobalBuffer((__gm__ T*)out);
        context_.lse.SetGlobalBuffer((__gm__ float*)lse);
        context_.dQuery.SetGlobalBuffer((__gm__ T*)dq);
        context_.dKey.SetGlobalBuffer((__gm__ T*)dk);
        context_.dValue.SetGlobalBuffer((__gm__ T*)dv);
        context_.sparseBlockIdx.SetGlobalBuffer((__gm__ int32_t*)indices);
        context_.sparseBlockCount.SetGlobalBuffer((__gm__ int32_t*)counts);
        if (tiling->maskType == 0) {
            context_.filteredIndices.SetGlobalBuffer((__gm__ int32_t*)indices);
            context_.filteredCounts.SetGlobalBuffer((__gm__ int32_t*)counts);
        } else {
            context_.filteredIndices.SetGlobalBuffer((__gm__ int32_t*)(workspace + tiling->filteredIndicesOffset));
            context_.filteredCounts.SetGlobalBuffer((__gm__ int32_t*)(workspace + tiling->filteredCountsOffset));
        }
        context_.cuSeqQ.SetGlobalBuffer((__gm__ int64_t*)cuQ);
        context_.cuSeqKv.SetGlobalBuffer((__gm__ int64_t*)cuK);
        context_.usedQ.SetGlobalBuffer((__gm__ int32_t*)usedQ);
        context_.usedK.SetGlobalBuffer((__gm__ int32_t*)usedK);

        context_.di.SetGlobalBuffer((__gm__ float*)(workspace + tiling->diOffset));
        context_.dQueryWorkspace.SetGlobalBuffer((__gm__ float*)(workspace + tiling->dqOffset));
        context_.dKeyWorkspace.SetGlobalBuffer((__gm__ float*)(workspace + tiling->dkOffset));
        context_.dValueWorkspace.SetGlobalBuffer((__gm__ float*)(workspace + tiling->dvOffset));
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            const uint64_t offset = (params.coreIndex * SLOT_COUNT + slot) * params.kvTileSize * params.headDimAligned;
            context_.selectedKey[slot].SetGlobalBuffer((__gm__ T*)(workspace + tiling->selectedKOffset) + offset);
            context_.selectedValue[slot].SetGlobalBuffer((__gm__ T*)(workspace + tiling->selectedVOffset) + offset);
            context_.mm4[slot].SetGlobalBuffer((__gm__ float*)(workspace + tiling->mm4Offset) + offset);
            context_.mm5[slot].SetGlobalBuffer((__gm__ float*)(workspace + tiling->mm5Offset) + offset);
        }
        context_.queryPadded.SetGlobalBuffer((__gm__ T*)(workspace + tiling->qPadOffset));
        context_.dOutPadded.SetGlobalBuffer((__gm__ T*)(workspace + tiling->dOPadOffset));
        // GM 中转四区双槽：[cores][2][gcap][ktile]。
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            const uint64_t offset = (params.coreIndex * SLOT_COUNT + slot) * params.groupCapacity * params.kvTileSize;
            context_.scoreGm[slot].SetGlobalBuffer((__gm__ float*)(workspace + tiling->scoreGmOffset) + offset);
            context_.gradGm[slot].SetGlobalBuffer((__gm__ float*)(workspace + tiling->gradGmOffset) + offset);
            context_.probabilityGm[slot].SetGlobalBuffer((__gm__ T*)(workspace + tiling->probabilityGmOffset) + offset);
            context_.dsProbabilityGm[slot].SetGlobalBuffer((__gm__ T*)(workspace + tiling->dsProbabilityGmOffset) +
                                                           offset);
        }
    }

    __aicore__ inline bool QueryInfo(uint64_t token, uint64_t& batch, uint64_t& qlocal, uint64_t& kbase, uint64_t& qlen,
                                     uint64_t& klen)
    {
        const Tiling* tiling = context_.tiling;
        if (tiling->inputLayout != 0) {
            batch = token / tiling->qSeqLen;
            qlocal = token % tiling->qSeqLen;
            kbase = batch * tiling->kvSeqLen;
            qlen = tiling->qSeqLen;
            klen = tiling->kvSeqLen;
        } else {
            // 轮转分核保证本核 token 单调递增；只在跨 batch 时读取序列信息。
            auto& cache = queryCache_;
            if (token >= cache.qend) {
                cache.valid = false;
                while (cache.batch < tiling->batchNum) {
                    cache.qend = static_cast<uint64_t>(context_.cuSeqQ.GetValue(cache.batch + 1));
                    if (token < cache.qend) {
                        break;
                    }
                    ++cache.batch;
                }
                if (cache.batch == tiling->batchNum) {
                    return false;
                }
                cache.qbase = context_.cuSeqQ.GetValue(cache.batch);
                cache.kbase = context_.cuSeqKv.GetValue(cache.batch);
                cache.qlen = cache.qend - cache.qbase;
                cache.klen = context_.cuSeqKv.GetValue(cache.batch + 1) - cache.kbase;
                if (tiling->hasUsedQ) {
                    const int32_t used = context_.usedQ.GetValue(cache.batch);
                    if (used < 0 || static_cast<uint64_t>(used) > cache.qlen) {
                        return false;
                    }
                    cache.qlen = used;
                }
                if (tiling->hasUsedKv) {
                    const int32_t used = context_.usedK.GetValue(cache.batch);
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
        return qlocal < qlen && qlocal < tiling->qCapacity;
    }

    __aicore__ inline uint64_t CausalLimit(uint64_t qlen, uint64_t klen, uint64_t qlocal) const
    {
        if (context_.tiling->maskType != 1) {
            return klen;
        }
        const int64_t causal =
            static_cast<int64_t>(klen) - static_cast<int64_t>(qlen) + static_cast<int64_t>(qlocal) + 1;
        return causal <= 0 ? 0 : Min(klen, static_cast<uint64_t>(causal));
    }

    // AIV 预扫描：每行保留 [0, visible/Bs) 内合法的索引，压缩成前缀并写回有效块数。
    // 两个 AIV 按 globalAivIndex 分片，结果由 Pre() 末尾的 SyncAll 对所有核可见。
    __aicore__ inline void FilterPatterns()
    {
#ifndef __DAV_C220_CUBE__
        for (uint64_t kvHead = 0; kvHead < context_.tiling->kvHeadNum; ++kvHead) {
            queryCache_ = {};
            for (uint64_t token = context_.params.globalAivIndex; token < context_.params.totalQueryTokens;
                 token += context_.params.cubeCoreNum * AIV_PER_CUBE) {
                uint64_t batch = 0, qlocal = 0, kbase = 0, qlen = 0, klen = 0;
                if (!QueryInfo(token, batch, qlocal, kbase, qlen, klen)) {
                    continue;
                }
                const uint64_t row =
                    (batch * context_.tiling->kvHeadNum + kvHead) * context_.tiling->qCapacity + qlocal;
                vector_.FilterQuery(context_, row, CausalLimit(qlen, klen, qlocal),
                                    context_.sparseBlockCount.GetValue(row));
            }
        }
        queryCache_ = {};
#endif
    }

    __aicore__ inline void DrainPipeline(TileTask tasks[PIPELINE_STAGES], uint64_t nextTask, uint64_t firstTask)
    {
        if (nextTask == firstTask) {
            return;
        }
        const auto& last = tasks[(nextTask - 1) % PIPELINE_STAGES];
#ifdef __DAV_C220_CUBE__
        cube_.ProcessMm345(context_, last);
#else
        vector_.ProcessProbability(context_, last);
        if (nextTask - firstTask >= 2) {
            vector_.Scatter(context_, tasks[(nextTask - 2) % PIPELINE_STAGES]);
        }
        vector_.Scatter(context_, last);
#endif
    }

    QsagContext<T, Tiling> context_;
#ifdef __DAV_C220_CUBE__
    CubeEngine<T, Tiling> cube_;
#else
    VectorEngine<T, Tiling> vector_;
#endif
    QueryCache queryCache_;
    uint32_t querySlot_ = 0;
};

} // namespace QSAG
#endif // ATTENTION_QSAG_KERNEL_ARCH22_H
