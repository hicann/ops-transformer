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
 * \file msa_index_score_task.h
 * \brief varlen + sparse_mode 因果裁剪的任务枚举器。
 *
 * atten_mask（sparse_mode=3）采用与 LightningIndexer 相同的 rightDownCausal 解析：
 *   visible_key_end(q) = kv_len - q_len + t_off + 1
 * 不物化 / 不加载 [2048,2048] 模板。
 *
 * start_loc[b] 为当前 query 所在逻辑 block 索引，仅用于 local_mask（强制高分），
 * 不再作为 token 级 prefix 偏移。
 */

#ifndef MSA_INDEX_SCORE_TASK_H
#define MSA_INDEX_SCORE_TASK_H

#include "kernel_operator.h"
#include "msa_index_score_common.h"

// Keep A8C8 request planning and packet accounting out of A2/A3 floating-point kernels.
#if (__CCE_AICORE__ != 310) && defined(ORIG_DTYPE_QUERY) && \
    (ORIG_DTYPE_QUERY == DT_FLOAT16 || ORIG_DTYPE_QUERY == DT_BF16)
#define MSA_TASK_USE_LEGACY_FLOAT_SCHEDULER 1
#else
#define MSA_TASK_USE_LEGACY_FLOAT_SCHEDULER 0
#endif

namespace MsaIndexScoreNs {

__aicore__ inline uint32_t MsaCeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

// C2UB 非量化 F322F16 单目标：两次 Fixpipe 的 mSize 切分（每段偶数），AIV 必须相同。
__aicore__ inline void MsaC2UbHalfSplit(uint32_t mActual, uint32_t& m0, uint32_t& m1)
{
    const uint32_t mEven = (mActual + 1U) & ~1U;
    m0 = mEven / 2U;
    if ((m0 & 1U) != 0U) {
        m0 += 1U;
    }
    m1 = mEven - m0;
}

__aicore__ inline uint32_t MsaMinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t MsaMaxU32(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

__aicore__ inline int32_t MsaClampI32(int32_t v, int32_t lo, int32_t hi)
{
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

/// C2UB 按行把 M-tile 分给两个 AIV。GROUP 为行分组粒度：
/// 128 路径取 1（沿用 MsaC2UbHalfSplit 的偶数对半），BSND 取 numQHeads，
/// 保证同一 token 的全部 head 落在同一个 AIV 上，head 间 maxpool 不跨核。
/// Fixpipe 与 epilogue 必须用同一个切分，否则两侧对不上。
__aicore__ inline void MsaC2UbGroupSplit(uint32_t mActual, uint32_t group, uint32_t& m0, uint32_t& m1)
{
    if (group <= 1U) {
        MsaC2UbHalfSplit(mActual, m0, m1);
        return;
    }
    const uint32_t groups = mActual / group;
    m0 = ((groups + 1U) / 2U) * group;
    if (m0 > mActual) {
        m0 = mActual;
    }
    m1 = mActual - m0;
}

/// 一个任务 = 一个请求内的一个 M-tile，可选再沿 KV stile 切开（950 短 decode）。
/// M 维 = 请求内 (token, head) 扁平行索引，head 为低位。
struct MsaTask {
    uint32_t batchIdx;
    uint32_t cuQStart;
    uint32_t mStart;
    uint32_t mActual;
    uint32_t globalRowBase;
    int32_t startLoc;   // 当前 query 所在逻辑 block 索引（local_mask）
    int32_t kvLen;      // 压缩后的 key 行数。compress_factor=1 时等于原始 token 数
    int32_t kvTokenLen; // actual_seq_klen 的原始 token 数。只在 compress_factor>1 的因果位置里使用
    int32_t qLen;
    uint32_t cuKStart; // TND：本请求在 packed key 中的压缩行起点；PA / BSND / BNSD：0
    uint32_t sparseMode;
    uint32_t initBlocks;
    uint32_t localBlocks;
    uint32_t fullEndBlk;
    uint32_t visibleEndBlk;
    uint32_t sBlkBegin; // 本 chunk 计算的 page 半开区间
    uint32_t sBlkEnd;
    uint32_t sStileBegin; // GM 路径 stile 循环
    uint32_t sStileEnd;
    uint32_t numSTiles;        // 含不可见尾，必须写 -inf 的 S-tile 数
    uint32_t numComputeSTiles; // AIC 真正做 QKᵀ 的 S-tile 数（<= numSTiles）
    uint32_t sChunkIdx;        // 本 M-tile 的 S 切分序号
    uint32_t sChunks;          // 本 M-tile 的 S 切分数
    bool writeTailFill;        // 本 M-tile 的最后 chunk 负责 [visible, stride) 填 -inf
};

/// BLOCK_SIZE 为逻辑 block 的 token 数，编译期常量。
/// 128 路径（TND / BBND / BNBD）实例化 <MSA_BLOCK_SIZE>，除法仍折叠成移位，
/// 与模板化之前的代码等价；BSND 实例化 <MSA_BLOCK16_SIZE>。
template <uint32_t BLOCK_SIZE = MSA_BLOCK_SIZE>
class MsaTaskScheduler {
public:
    __aicore__ inline MsaTaskScheduler() {}

    // rotateChunk：A2/A3 把每个请求的 chunk 0 错开到不同核。950 保持 false。
    // nStiles 不能整除 chunks 时，多出来的 stile 否则会每个请求都落在核 0..extra-1 上。
    template <bool REQUEST_PLAN = false>
    __aicore__ inline void Init(
        uint32_t batch, uint32_t numQHeads, uint32_t maxBlocksPerBatch, uint32_t scoreBlockStride, uint32_t sparseMode,
        uint32_t initBlocks, uint32_t localBlocks, uint32_t keyLayout, uint32_t kvChunks,
        const AscendC::GlobalTensor<int32_t>& actualSeqQlen, const AscendC::GlobalTensor<int32_t>& actualSeqKlen,
        const AscendC::GlobalTensor<int32_t>& startLoc, bool rotateChunk = false, uint32_t qMaxSeq = 0U,
        uint32_t kMaxSeq = 0U, uint32_t tableWidth = 0U, uint32_t chunkBudget = 0U, uint32_t pageSize = MSA_BLOCK_SIZE,
        uint32_t totalQ = ~0U, uint32_t compressFactor = 1U, uint32_t hostTotalTasks = 0U,
        uint32_t requestPlanCores = 0U, uint32_t requestPlanTaskIdx = 0U, bool checkBatchPacketRows = false)
    {
        // actual_seq_klen 是压缩前的 token 数。compress_factor>1 时 key 行数 = ceil(token / cf)，
        // 一个逻辑 block 仍是 BLOCK_SIZE 个 token，落到 key 上是 rowsPerBlock_ 行。
        if constexpr (BLOCK_SIZE == MSA_BLOCK16_SIZE) {
            compressFactor_ = (compressFactor < 1U) ? 1U : compressFactor;
        }
        rowsPerBlock_ = BLOCK_SIZE / compressFactor_;
        if (rowsPerBlock_ == 0U) {
            rowsPerBlock_ = 1U;
        }
        // 一个 S-tile 的宽度由 Cube 的 N 决定：128 路径是 MSA_BLOCKS_PER_STILE 个块，
        // 块长 16 的一窗是 128 个 key 行，压缩后装得下更多块。
        blocksPerStile_ = (BLOCK_SIZE == MSA_BLOCK16_SIZE) ? (MSA_BLOCK_SIZE / rowsPerBlock_) : MSA_BLOCKS_PER_STILE;
        batch_ = batch;
        numQHeads_ = numQHeads;
        maxBlocksPerBatch_ = maxBlocksPerBatch;
        scoreBlockStride_ = scoreBlockStride;
        sparseMode_ = sparseMode;
        initBlocks_ = initBlocks;
        localBlocks_ = localBlocks;
        keyLayout_ = keyLayout;
        kvChunks_ = (kvChunks < 1U) ? 1U : kvChunks;
        rotateChunk_ = rotateChunk;
        qMaxSeq_ = qMaxSeq;
        kMaxSeq_ = kMaxSeq;
        tableWidth_ = tableWidth;
        pageSize_ = pageSize;
        totalQ_ = totalQ;
        gActualSeqQlen_ = actualSeqQlen;
        gActualSeqKlen_ = actualSeqKlen;
        gStartLoc_ = startLoc;

        // host 已算好 totalTasks 时只拉 batch0 元数据，避免扫遍 batch 的 GetValue。
        if (BLOCK_SIZE == MSA_BLOCK16_SIZE && hostTotalTasks > 0U) {
            totalTasks_ = hostTotalTasks;
            curBatch_ = 0;
            taskBase_ = 0;
            LoadBatchMeta(0);
            curTasks_ = (batch_ == 1U) ? hostTotalTasks : CountBatchTasks();
            return;
        }

#if !MSA_TASK_USE_LEGACY_FLOAT_SCHEDULER
        tailWaveFactor_ = 1U;
        tailWaveStart_ = 0U;
        useRequestPlan_ = false;
        batchPacketRows_ = checkBatchPacketRows;
        if constexpr (REQUEST_PLAN) {
            if (batch_ > 1U && batch_ <= requestPlanCores && requestPlanCores <= 24U && chunkBudget == 0U &&
                keyLayout_ == MSA_KEY_LAYOUT_BBND && kvChunks_ > 1U && sparseMode_ == MSA_SPARSE_MODE_RIGHT_DOWN &&
                initBlocks_ == 0U && localBlocks_ == 0U &&
                (totalQ_ == ~0U || static_cast<uint64_t>(totalQ_) * numQHeads_ <= batch_ * 64U)) {
                useRequestPlan_ = TryPlanRequestChunks(requestPlanCores, requestPlanTaskIdx);
                if (useRequestPlan_) {
                    if (requestPlanTaskIdx < totalTasks_) {
                        LoadBatchMetadata(curBatch_, true);
                    }
                    return;
                }
            }
        }
        // A rejected planner may have left its sweep before EOF. Reuse the full
        // legacy task-count loop in that case, including empty requests.
        batchPacketRows_ = checkBatchPacketRows;
#endif
        totalTasks_ = 0;
        for (uint32_t b = 0; b < batch_; ++b) {
            LoadBatch(b);
            totalTasks_ += curTasks_;
#if !MSA_TASK_USE_LEGACY_FLOAT_SCHEDULER
            AccumulateBatchPacketRows();
#endif
        }
        Reset();
    }

    // Split only the final partial M wave when actual task startup and S-tile
    // counts predict a smaller busiest-core workload.
    __aicore__ inline void ConfigureTailWaveSplit(uint32_t cores)
    {
        constexpr uint32_t coreCount = 24U;
        const uint32_t tiles = MsaCeilDiv(curRows_, MSA_ROW_TILE_M);
        if (cores != coreCount || batch_ != 1U || kvChunks_ != 1U || useRequestPlan_ ||
            keyLayout_ != MSA_KEY_LAYOUT_BBND || pageSize_ != MSA_BLOCK_SIZE ||
            sparseMode_ != MSA_SPARSE_MODE_RIGHT_DOWN || initBlocks_ != 0U || localBlocks_ != 0U ||
            tiles < 2U * coreCount || tiles > 4U * coreCount || qLen_ <= 0 || kvLen_ <= 0) {
            return;
        }
        const uint32_t remainder = tiles % coreCount;
        if (remainder == 0U) {
            return;
        }
        uint32_t divisor = coreCount;
        uint32_t next = remainder;
        while (next != 0U) {
            const uint32_t value = divisor % next;
            divisor = next;
            next = value;
        }
        const uint32_t factor = coreCount / divisor;
        if (factor < 2U || factor > 3U) {
            return;
        }
        const uint32_t prefix = tiles - remainder;
        uint32_t originalWork[coreCount] = {};
        uint32_t candidateWork[coreCount] = {};
        for (uint32_t tile = 0U; tile < tiles; ++tile) {
            uint32_t start = 0U;
            uint32_t rows = 0U;
            GetMTile(tile, start, rows);
            const uint32_t stiles = MsaCeilDiv(VisibleEndBlkOf(start, rows), MSA_BLOCKS_PER_STILE);
            originalWork[tile % coreCount] += stiles + 1U;
            if (tile < prefix) {
                candidateWork[tile % coreCount] += stiles + 1U;
                continue;
            }
            if (stiles < factor) {
                return;
            }
            const uint32_t first = prefix + (tile - prefix) * factor;
            for (uint32_t part = 0U; part < factor; ++part) {
                const uint32_t count = stiles / factor + (part < stiles % factor ? 1U : 0U);
                candidateWork[(first + part) % coreCount] += count + 1U;
            }
        }
        uint32_t originalPeak = 0U;
        uint32_t candidatePeak = 0U;
        for (uint32_t core = 0U; core < coreCount; ++core) {
            originalPeak = MsaMaxU32(originalPeak, originalWork[core]);
            candidatePeak = MsaMaxU32(candidatePeak, candidateWork[core]);
        }
        if (candidatePeak + 1U >= originalPeak) {
            return;
        }
        tailWaveStart_ = prefix;
        tailWaveFactor_ = factor;
        totalTasks_ = prefix + remainder * factor;
        curTasks_ = totalTasks_;
    }

    __aicore__ inline bool BatchPacketRows() const
    {
        return batchPacketRows_ && totalTasks_ != 0U;
    }

    __aicore__ inline uint32_t TotalTasks() const
    {
        return totalTasks_;
    }

    __aicore__ inline bool IsSingleFullCausalPrefill(uint32_t launchCores, int32_t maxQTokens) const
    {
        return batch_ == 1U && kvChunks_ == 1U && sparseMode_ == MSA_SPARSE_MODE_RIGHT_DOWN && qLen_ > 0 &&
               qLen_ <= maxQTokens && qLen_ == kvLen_ && curRows_ % MSA_ROW_TILE_M == 0U && totalTasks_ > launchCores;
    }

    __aicore__ inline void Reset()
    {
        curBatch_ = 0;
        taskBase_ = 0;
        LoadBatch(0);
    }

    /// 按 sparse_mode 计算某 query token 的可见 key 行上界（半开区间右端，压缩行粒度）。
    /// 压缩行 c 覆盖 token [c*cf, (c+1)*cf)，可见当且仅当 c*cf <= position。
    /// position 用 actual_seq_klen 的原始 token 数，行数上界是 ceil(token/cf)。
    /// cf=1 时原始长度就是行数，公式退化为 kvLen - qLen + tOff + 1。
    __aicore__ inline int32_t VisibleKeyEndOf(int32_t tOff) const
    {
        if (sparseMode_ == MSA_SPARSE_MODE_RIGHT_DOWN) {
            const int32_t cap = kvLen_ < 0 ? 0 : kvLen_;
            if (compressFactor_ == 1U) {
                // rightDownCausal：j <= kvLen - qLen + tOff  <=>  end = kvLen - qLen + tOff + 1
                return MsaClampI32(kvLen_ - qLen_ + tOff + 1, 0, cap);
            }
            const int32_t cf = static_cast<int32_t>(compressFactor_);
            const int32_t position = kvTokenLen_ - qLen_ + tOff;
            return position < 0 ? 0 : MsaClampI32(position / cf + 1, 0, cap);
        }
        // defaultMask：仅受压缩行数截断
        return kvLen_ < 0 ? 0 : kvLen_;
    }

    __aicore__ inline void Decode(uint32_t taskIdx, MsaTask& task)
    {
        if (BLOCK_SIZE == MSA_BLOCK16_SIZE && batch_ == 1U && kvChunks_ <= 1U && totalTasks_ == 1U) {
            uint32_t mStart = 0;
            uint32_t mActual = 0;
            GetMTile(0, mStart, mActual);
            FillTask(task, mStart, mActual, 0U, 1U);
            return;
        }
        while (taskIdx >= taskBase_ + curTasks_) {
            taskBase_ += curTasks_;
            ++curBatch_;
            LoadBatch(curBatch_);
        }

        const uint32_t local = taskIdx - taskBase_;
        uint32_t mStart = 0;
        uint32_t mActual = 0;
        uint32_t sChunkIdx = 0;
        uint32_t sChunks = 1;
#if !MSA_TASK_USE_LEGACY_FLOAT_SCHEDULER
        if (useRequestPlan_) {
            mActual = curRows_;
            sChunkIdx = local;
            sChunks = curTasks_;
        } else if (tailWaveFactor_ > 1U && local >= tailWaveStart_) {
            const uint32_t tail = local - tailWaveStart_;
            GetMTile(tailWaveStart_ + tail / tailWaveFactor_, mStart, mActual);
            sChunkIdx = tail % tailWaveFactor_;
            sChunks = tailWaveFactor_;
        } else
#endif
            if (kvChunks_ <= 1U) {
            GetMTile(local, mStart, mActual);
        } else {
            uint32_t acc = 0;
            const uint32_t mTiles = MsaCeilDiv(curRows_, MSA_ROW_TILE_M);
            for (uint32_t t = 0; t < mTiles; ++t) {
                GetMTile(t, mStart, mActual);
                const uint32_t vis = VisibleEndBlkOf(mStart, mActual);
                sChunks = KvChunksOf(vis);
                if (local < acc + sChunks) {
                    sChunkIdx = local - acc;
                    break;
                }
                acc += sChunks;
            }
        }
        if (rotateChunk_ && sChunks > 1U) {
            sChunkIdx = (sChunkIdx + curBatch_) % sChunks;
        }
        FillTask(task, mStart, mActual, sChunkIdx, sChunks);
    }

private:
    __aicore__ inline void AccumulateBatchPacketRows()
    {
        if (batchPacketRows_ && qLen_ != 0) {
            const uint64_t rows = static_cast<uint64_t>(qLen_) * numQHeads_;
            batchPacketRows_ = rows >= MSA_PACKET_MIN_M && rows <= MSA_PACKET_MAX_M;
        }
    }

    // Each accepted request has one M tile and at most one task per core.
    __aicore__ inline bool TryPlanRequestChunks(uint32_t cores, uint32_t taskIdx)
    {
        uint32_t stiles[24];
        uint32_t work = 0U;
        uint32_t legacyTasks = 0U;
        uint32_t legacySinglePeak = 0U;
        const uint32_t blockCap = BlockCap();
        for (uint32_t b = 0; b < batch_; ++b) {
            LoadBatchMetadata(b, false);
            AccumulateBatchPacketRows();
            if (curRows_ == 0U || curRows_ > 64U || kvLen_ <= 0) {
                return false;
            }
            // The last query of the single M tile sees all valid KV pages for
            // both accepted sparse modes. Preserve the scheduler's page cap.
            const uint32_t pages = MsaMinU32(MsaCeilDiv(static_cast<uint32_t>(kvLen_), BLOCK_SIZE), blockCap);
            const uint32_t count = MsaCeilDiv(pages, MSA_BLOCKS_PER_STILE);
            if (count == 0U) {
                return false;
            }
            stiles[b] = count;
            work += count;
            const uint32_t legacyChunks = MsaMinU32(count, kvChunks_);
            legacyTasks += legacyChunks;
            legacySinglePeak = MsaMaxU32(legacySinglePeak, MsaCeilDiv(count, legacyChunks));
        }

        const uint32_t cap = MsaCeilDiv(2U * work, 2U * cores - batch_);
        uint32_t finalCap = cap;
        uint32_t tasks = BuildRequestChunks(stiles, cap, taskIdx);
        if (tasks > cores) {
            // sum ceil(stiles/cap) <= floor((work-B)/cap) + B.
            const uint32_t safeCap = MsaCeilDiv(work - batch_ + 1U, cores - batch_ + 1U);
            finalCap = safeCap;
            tasks = BuildRequestChunks(stiles, safeCap, taskIdx);
        }
        if (tasks > cores) {
            return false;
        }

        {
            // Measure in units of one S-tile per task startup. The old peak
            // is at least either of the following two lower bounds.
            const uint32_t legacyLower = MsaMaxU32(legacySinglePeak + 1U, MsaCeilDiv(work + legacyTasks, cores));
            uint32_t plannedPeak = 0U;
            for (uint32_t b = 0U; b < batch_; ++b) {
                plannedPeak = MsaMaxU32(plannedPeak, MsaCeilDiv(stiles[b], MsaCeilDiv(stiles[b], finalCap)));
            }
            if (legacyLower <= plannedPeak + 2U) {
                return false;
            }
        }
        totalTasks_ = tasks;
        return true;
    }

    __aicore__ inline uint32_t BuildRequestChunks(const uint32_t* stiles, uint32_t cap, uint32_t taskIdx)
    {
        uint32_t tasks = 0U;
        for (uint32_t b = 0; b < batch_; ++b) {
            const uint32_t chunks = MsaCeilDiv(stiles[b], cap);
            if (chunks > 24U || tasks + chunks > 24U) {
                return 25U;
            }
            {
                if (tasks <= taskIdx && taskIdx - tasks < chunks) {
                    curBatch_ = b;
                    taskBase_ = tasks;
                    curTasks_ = chunks;
                }
            }
            tasks += chunks;
        }
        return tasks;
    }

    __aicore__ inline void GetMTile(uint32_t tileIdx, uint32_t& mStart, uint32_t& mActual) const
    {
        mStart = tileIdx * MSA_ROW_TILE_M;
        mActual = MsaMinU32(MSA_ROW_TILE_M, (curRows_ > mStart) ? (curRows_ - mStart) : 0U);
        if (mActual > 0U && mActual < MSA_M_ALIGN && mStart >= MSA_M_ALIGN) {
            mStart -= (MSA_M_ALIGN - mActual);
            mActual = MSA_M_ALIGN;
        }
    }

    __aicore__ inline uint32_t VisibleEndBlkOf(uint32_t mStart, uint32_t mActual) const
    {
        if (mActual == 0U) {
            return 0U;
        }
        const uint32_t tHi = (mStart + mActual - 1U) / numQHeads_;
        const int32_t visibleKeyEndHi = VisibleKeyEndOf(static_cast<int32_t>(tHi));
        uint32_t visibleEndBlk = BlocksOfRows(static_cast<uint32_t>(visibleKeyEndHi));
        return MsaMinU32(visibleEndBlk, BlockCap());
    }

    /// 可见 key 行数 → 逻辑块数。cf=1 时除数是编译期常量，仍折叠成移位。
    __aicore__ inline uint32_t BlocksOfRows(uint32_t rows) const
    {
        return (compressFactor_ == 1U) ? MsaCeilDiv(rows, BLOCK_SIZE) : MsaCeilDiv(rows, rowsPerBlock_);
    }

    /// 分页页表宽度是物理页数，逻辑块上界还要乘 pageSize/rowsPerBlock。
    /// score 末维是 16 对齐后的输出宽度，分页时只作写入上界，避免写出分配的 score。
    __aicore__ inline uint32_t BlockCap() const
    {
        if (MsaIsPaKeyLayout(keyLayout_) && tableWidth_ > 0U) {
            const uint32_t perPage = (compressFactor_ == 1U) ? (pageSize_ / BLOCK_SIZE) : (pageSize_ / rowsPerBlock_);
            const uint64_t tableBlocks = static_cast<uint64_t>(tableWidth_) * perPage;
            return tableBlocks < maxBlocksPerBatch_ ? static_cast<uint32_t>(tableBlocks) : maxBlocksPerBatch_;
        }
        return maxBlocksPerBatch_;
    }

    // 块长 16 的 score 是 fp16。MTE 一次写出要 32B 对齐，列起点和列数都得是 16 的倍数。
    // cf=1 时一个 stile 只有 8 个块，按 8 切开会让 fp16 地址落在 16B 上。块长 128 仍按 stile。
    __aicore__ inline uint32_t ScoreStoreUnit() const
    {
        if constexpr (BLOCK_SIZE == MSA_BLOCK16_SIZE) {
            return blocksPerStile_ < 16U ? 16U : blocksPerStile_;
        }
        return blocksPerStile_;
    }

    __aicore__ inline uint32_t KvChunksOf(uint32_t visibleEndBlk) const
    {
        if (kvChunks_ <= 1U) {
            return 1U;
        }
        const uint32_t nStiles = MsaCeilDiv(visibleEndBlk, ScoreStoreUnit());
        if (nStiles <= 1U) {
            return 1U;
        }
        return MsaMinU32(kvChunks_, nStiles);
    }

    __aicore__ inline void AssignSRange(uint32_t visibleEndBlk, uint32_t sChunkIdx, uint32_t sChunks,
                                        MsaTask& task) const
    {
        const uint32_t unit = ScoreStoreUnit();
        const uint32_t nStiles = MsaCeilDiv(visibleEndBlk, unit);
        if (sChunks <= 1U || nStiles == 0U) {
            task.sBlkBegin = 0;
            task.sBlkEnd = visibleEndBlk;
            task.writeTailFill = true;
        } else {
            uint32_t sc = sChunks;
            if (sc > nStiles) {
                sc = nStiles;
            }
            const uint32_t base = nStiles / sc;
            const uint32_t extra = nStiles % sc;
            uint32_t st0 = 0;
            uint32_t nst = 0;
            if (sChunkIdx < extra) {
                st0 = sChunkIdx * (base + 1U);
                nst = base + 1U;
            } else {
                st0 = extra * (base + 1U) + (sChunkIdx - extra) * base;
                nst = base;
            }
            task.sBlkBegin = st0 * unit;
            task.sBlkEnd = MsaMinU32((st0 + nst) * unit, visibleEndBlk);
            task.writeTailFill = (sChunkIdx + 1U == sc);
        }
        task.sStileBegin = task.sBlkBegin / blocksPerStile_;
        task.sStileEnd = MsaCeilDiv(task.sBlkEnd, blocksPerStile_);
    }

    __aicore__ inline void FillTask(MsaTask& task, uint32_t mStart, uint32_t mActual, uint32_t sChunkIdx,
                                    uint32_t sChunks) const
    {
        task.batchIdx = curBatch_;
        task.cuQStart = cuQStart_;
        task.startLoc = startLoc_;
        task.kvLen = kvLen_;
        task.kvTokenLen = kvTokenLen_;
        task.qLen = qLen_;
        task.cuKStart = cuKStart_;
        task.sparseMode = sparseMode_;
        task.initBlocks = initBlocks_;
        task.localBlocks = localBlocks_;
        task.mStart = mStart;
        task.mActual = mActual;
        task.sChunkIdx = sChunkIdx;
        task.sChunks = (sChunks < 1U) ? 1U : sChunks;
        task.globalRowBase = cuQStart_ * numQHeads_ + task.mStart;

        const uint32_t tLo = (mActual == 0U) ? 0U : (task.mStart / numQHeads_);
        const int32_t visibleKeyEndLo = VisibleKeyEndOf(static_cast<int32_t>(tLo));
        const uint32_t visibleEndBlk = VisibleEndBlkOf(mStart, mActual);
        const uint32_t blkRows = (compressFactor_ == 1U) ? BLOCK_SIZE : rowsPerBlock_;
        const uint32_t causalFull = static_cast<uint32_t>(visibleKeyEndLo) / blkRows;
        const uint32_t seqFull = (kvLen_ < 0 ? 0U : static_cast<uint32_t>(kvLen_)) / blkRows;
        const uint32_t fullEndBlk = MsaMinU32(causalFull, seqFull);

        task.visibleEndBlk = visibleEndBlk;
        task.fullEndBlk = MsaMinU32(fullEndBlk, visibleEndBlk);
        task.numSTiles = MsaCeilDiv(scoreBlockStride_, blocksPerStile_);
        task.numComputeSTiles = MsaCeilDiv(task.visibleEndBlk, blocksPerStile_);
        if (task.numComputeSTiles > task.numSTiles) {
            task.numComputeSTiles = task.numSTiles;
        }
        AssignSRange(visibleEndBlk, sChunkIdx, sChunks, task);
    }

    __aicore__ inline uint32_t CountBatchTasks() const
    {
        const uint32_t mTiles = MsaCeilDiv(curRows_, MSA_ROW_TILE_M);
        if (kvChunks_ <= 1U) {
            return mTiles;
        }
        uint32_t n = 0;
        uint32_t mStart = 0;
        uint32_t mActual = 0;
        for (uint32_t t = 0; t < mTiles; ++t) {
            GetMTile(t, mStart, mActual);
            n += KvChunksOf(VisibleEndBlkOf(mStart, mActual));
        }
        return n;
    }
    /// actual_seq_klen 记的是压缩前 token 数。key 行数 = ceil(token / cf)。
    /// TND 每段单独向上取整再累加，不能用前缀和直接除 cf。
    __aicore__ inline void LoadCompressedBatch(uint32_t b)
    {
        const int32_t cf = static_cast<int32_t>(compressFactor_);
        int32_t tokenLen = 0;
        if (MsaIsSeqKeyLayout(keyLayout_)) {
            tokenLen = static_cast<int32_t>(gActualSeqKlen_.GetValue(b + 1)) -
                       static_cast<int32_t>(gActualSeqKlen_.GetValue(b));
        } else {
            tokenLen = gActualSeqKlen_.GetValue(b);
        }
        if (tokenLen < 0) {
            tokenLen = 0;
        }
        kvTokenLen_ = tokenLen;
        int32_t rows = (tokenLen + cf - 1) / cf;
        // BSND/BNSD 的 S 是压缩行槽。行数超过 S 时截断，避免读进下一个 batch。
        if (kMaxSeq_ > 0U && rows > static_cast<int32_t>(kMaxSeq_)) {
            rows = static_cast<int32_t>(kMaxSeq_);
        }
        kvLen_ = rows;
        cuKStart_ = 0;
        if (keyLayout_ == MSA_KEY_LAYOUT_TND) {
            uint32_t cu = 0;
            for (uint32_t i = 0; i < b; ++i) {
                int32_t prev = static_cast<int32_t>(gActualSeqKlen_.GetValue(i + 1)) -
                               static_cast<int32_t>(gActualSeqKlen_.GetValue(i));
                if (prev < 0) {
                    prev = 0;
                }
                cu += static_cast<uint32_t>((prev + cf - 1) / cf);
            }
            cuKStart_ = cu;
        }
    }

#if MSA_TASK_USE_LEGACY_FLOAT_SCHEDULER
    __aicore__ inline void LoadBatchMeta(uint32_t b)
#else
    __aicore__ inline void LoadBatchMeta(uint32_t b)
    {
        LoadBatchMetadata(b, true);
    }

    __aicore__ inline void LoadBatchMetadata(uint32_t b, bool loadStart)
#endif
    {
        if (b >= batch_) {
            curRows_ = 0;
            cuQStart_ = 0;
            kvLen_ = 0;
            kvTokenLen_ = 0;
            qLen_ = 0;
            startLoc_ = 0;
            cuKStart_ = 0;
            return;
        }
        cuQStart_ = static_cast<uint32_t>(gActualSeqQlen_.GetValue(b));
        qLen_ = static_cast<int32_t>(gActualSeqQlen_.GetValue(b + 1)) - static_cast<int32_t>(cuQStart_);
        if (qLen_ < 0) {
            qLen_ = 0;
        }
        if constexpr (BLOCK_SIZE == MSA_BLOCK16_SIZE) {
            if (qMaxSeq_ > 0U && qLen_ > static_cast<int32_t>(qMaxSeq_)) {
                qLen_ = static_cast<int32_t>(qMaxSeq_);
            }
        } else if (qMaxSeq_ > 0U) {
            // Packed BSND/BNSD output rows can be smaller than B*S. Device
            // lengths must not create tasks past that extent. TND output
            // capacity is checked against the query shape by Host.
            if (static_cast<int32_t>(cuQStart_) < 0 || cuQStart_ >= totalQ_) {
                qLen_ = 0;
            } else {
                const uint32_t available = MsaMinU32(qMaxSeq_, totalQ_ - cuQStart_);
                qLen_ = static_cast<int32_t>(MsaMinU32(static_cast<uint32_t>(qLen_), available));
            }
        }
        curRows_ = static_cast<uint32_t>(qLen_) * numQHeads_;
        if (compressFactor_ == 1U) {
            // 不压缩：actual_seq_klen 的差分就是 key 行数，TND 前缀和直接是打包起点。
            kvTokenLen_ = 0;
            if (MsaIsSeqKeyLayout(keyLayout_)) {
                cuKStart_ = static_cast<uint32_t>(gActualSeqKlen_.GetValue(b));
                kvLen_ = static_cast<int32_t>(gActualSeqKlen_.GetValue(b + 1)) - static_cast<int32_t>(cuKStart_);
                if (kvLen_ < 0) {
                    kvLen_ = 0;
                }
                // BSND/BNSD 的 S 是定长槽。长度超过 S 时截断，避免读进下一个 batch。
                if (kMaxSeq_ > 0U && kvLen_ > static_cast<int32_t>(kMaxSeq_)) {
                    kvLen_ = static_cast<int32_t>(kMaxSeq_);
                }
            } else {
                cuKStart_ = 0;
                kvLen_ = gActualSeqKlen_.GetValue(b);
            }
        } else {
            LoadCompressedBatch(b);
        }
#if MSA_TASK_USE_LEGACY_FLOAT_SCHEDULER
        startLoc_ = gStartLoc_.GetValue(b);
#else
        if (loadStart) {
            startLoc_ = gStartLoc_.GetValue(b);
        }
#endif
    }

#if MSA_TASK_USE_LEGACY_FLOAT_SCHEDULER
    __aicore__ inline void LoadBatchMetadata(uint32_t b, bool /* loadStart */)
    {
        LoadBatchMeta(b);
    }
#endif

#if MSA_TASK_USE_LEGACY_FLOAT_SCHEDULER
    __aicore__ inline void LoadBatch(uint32_t b)
    {
        if (b >= batch_) {
            curTasks_ = 0;
            LoadBatchMeta(b);
            return;
        }
        LoadBatchMeta(b);
        curTasks_ = CountBatchTasks();
    }
#else
    __aicore__ inline void LoadBatch(uint32_t b)
    {
        LoadBatchMeta(b);
        curTasks_ = b >= batch_ ? 0U : CountBatchTasks();
    }
#endif

    AscendC::GlobalTensor<int32_t> gActualSeqQlen_;
    AscendC::GlobalTensor<int32_t> gActualSeqKlen_;
    AscendC::GlobalTensor<int32_t> gStartLoc_;

    uint32_t totalQ_ = 0;
    uint32_t batch_ = 0;
    uint32_t numQHeads_ = 1;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t scoreBlockStride_ = 0;
    uint32_t sparseMode_ = MSA_SPARSE_MODE_RIGHT_DOWN;
    uint32_t initBlocks_ = MSA_DEFAULT_INIT_BLOCKS;
    uint32_t localBlocks_ = MSA_DEFAULT_LOCAL_BLOCKS;
    uint32_t keyLayout_ = MSA_KEY_LAYOUT_BBND;
    uint32_t kvChunks_ = 1;
    bool rotateChunk_ = false;
    uint32_t qMaxSeq_ = 0;
    uint32_t kMaxSeq_ = 0;
    uint32_t tableWidth_ = 0;
    uint32_t pageSize_ = MSA_BLOCK_SIZE;
    uint32_t compressFactor_ = 1;
    uint32_t rowsPerBlock_ = BLOCK_SIZE;
    uint32_t blocksPerStile_ = MSA_BLOCKS_PER_STILE;

    uint32_t tailWaveStart_ = 0U;
    uint32_t tailWaveFactor_ = 1U;
    bool useRequestPlan_ = false;
    bool batchPacketRows_ = false;
    uint32_t totalTasks_ = 0;
    uint32_t curBatch_ = 0;
    uint32_t taskBase_ = 0;
    uint32_t curTasks_ = 0;
    uint32_t curRows_ = 0;
    uint32_t cuQStart_ = 0;
    int32_t kvLen_ = 0;
    int32_t kvTokenLen_ = 0;
    int32_t qLen_ = 0;
    int32_t startLoc_ = 0;
    uint32_t cuKStart_ = 0;
};

} // namespace MsaIndexScoreNs

#undef MSA_TASK_USE_LEGACY_FLOAT_SCHEDULER

#endif // MSA_INDEX_SCORE_TASK_H
