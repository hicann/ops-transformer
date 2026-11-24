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
 * \file msa_index_score_tiling.cpp
 * \brief MsaIndexScore Tiling 实现（A2/A3 / 950：PA BBND/BNBD + TND packed key；sparse_mode 0/3；
 *        PA key 允许 dim0 非连续，strideKvBlock 来自 GetInputStride）。
 */

#include "msa_index_score_tiling.h"
#include "msa_index_score_output.h"
#include "../op_kernel/msa_index_score_common.h"

#include <cstring>
#include <limits>
#include <string>
#include "graph/utils/type_utils.h"

using namespace ge;
using namespace MsaIndexScoreNs;

namespace optiling {
namespace {
constexpr uint32_t MSA_QUERY_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_TND_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_PA_DIM_NUM = 4;
constexpr uint32_t MSA_BLOCK_TABLE_DIM_NUM = 2;
constexpr uint32_t MSA_SCALE_PA_DIM_NUM = 3;
constexpr uint32_t MSA_SCALE_TND_DIM_NUM = 2;
constexpr uint32_t MSA_ATTEN_MASK_DIM_NUM = 2;
constexpr uint32_t MSA_DIM_0 = 0;
constexpr uint32_t MSA_DIM_1 = 1;
constexpr uint32_t MSA_DIM_2 = 2;
constexpr uint32_t MSA_DIM_3 = 3;
constexpr uint32_t MSA_RANK_1D = 1;
constexpr uint32_t MSA_QLEN_MIN_DIM0 = 2; // prefix-sum 长度至少为 B+1

inline bool UseBalancedFp8Schedule(uint32_t totalQ, uint32_t scoreStride, uint32_t cores)
{
    if (cores == 0U || totalQ == 0U) {
        return false;
    }
    const uint64_t groups = (static_cast<uint64_t>(totalQ) + 7U) / 8U;
    const uint64_t tiles = (static_cast<uint64_t>(scoreStride) + 63U) / 64U;
    if (tiles < 32U) {
        return false;
    }
    const uint64_t oldMax =
        groups < cores ? (tiles + cores / groups - 1U) / (cores / groups) : ((groups + cores - 1U) / cores) * tiles;
    const uint64_t newMax = (groups * tiles + cores - 1U) / cores;
    // With an exact two-way (or deeper) split, adjacent cores can share a
    // request without extra Q reloads. Keep this mapping for long segments.
    const bool evenLocality = groups < cores && cores % groups == 0U && oldMax == newMax && newMax >= 64U;
    return evenLocality || (oldMax >= newMax + 8U && newMax * 5U <= oldMax * 4U);
}

inline bool IsFp8ComputeDtype(ge::DataType dt)
{
    return dt == ge::DT_HIFLOAT8 || dt == ge::DT_FLOAT8_E5M2 || dt == ge::DT_FLOAT8_E4M3FN;
}

inline bool IsNonQuantQueryDtype(ge::DataType dt, bool isAscend950)
{
    if (dt == ge::DT_BF16 || dt == ge::DT_FLOAT16) {
        return true;
    }
    return isAscend950 && IsFp8ComputeDtype(dt);
}

inline uint32_t RoundUpU32(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

// MIX：按估计 M-task 数启动 AIC，避免短 decode 打满空核（scheduler.Init GetValue / MODE4 VC 空转）。
// Host 读不到 per-request q_len。kernel TotalTasks = Σ CeilDiv(qLen_b * Hq, MSA_ROW_TILE_M)。
// Σ ceil(x_i) <= ceil(Σ x_i) + B；B=1 时 packed == actual。B>1 用 packed+B 上界，避免少估把
// 多 tile 请求串到已启动核上。多 M-tile / 大 batch 仍截到全量 AIC。
inline uint32_t EstPackedMTasks(const MsaIndexScoreInfo& info, uint32_t aicNum)
{
    if (info.totalQ == 0U || aicNum == 0U) {
        return 1U;
    }
    const uint64_t packedRows = static_cast<uint64_t>(info.totalQ) * static_cast<uint64_t>(info.numQHeads);
    const uint32_t packedM =
        static_cast<uint32_t>((packedRows + static_cast<uint64_t>(MSA_ROW_TILE_M) - 1U) / MSA_ROW_TILE_M);
    uint32_t est = packedM;
    if (info.batch > 1U) {
        const uint32_t add = packedM + info.batch;
        est = (add < packedM) ? aicNum : add;
    }
    if (est < 1U) {
        est = 1U;
    }
    if (est > aicNum) {
        est = aicNum;
    }
    return est;
}

// 可见 KV page 上界：优先 host 可读的 actual_seq_klen；否则 min(表宽, PA numPages)。
// GetData() 对 device tensor 也返回非空指针，解引用会段错误；仅 kOnHost/kFollowing 可读。
inline bool IsHostVisibleTensor(const gert::Tensor* tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto placement = tensor->GetPlacement();
    return (placement == gert::kOnHost) || (placement == gert::kFollowing);
}

// 与 device MsaC2UbHalfSplit 一致：偶数对半，供 host 判定 MXFP4 宽路径。
inline void HostC2UbHalfSplit(uint32_t mActual, uint32_t& m0, uint32_t& m1)
{
    const uint32_t mEven = (mActual + 1U) & ~1U;
    m0 = mEven / 2U;
    if ((m0 & 1U) != 0U) {
        m0 += 1U;
    }
    m1 = mEven - m0;
}

// 与 device MxFp4WideNFits 同公式：最大 AIV 行 × 4 页 × 128 × sizeof(half) ≤ 32KB ping。
inline bool HostMxFp4GroupFitsRows(uint32_t qTok, uint32_t numQHeads)
{
    const uint32_t heads = numQHeads == 0U ? 1U : numQHeads;
    const uint32_t rows = qTok * heads;
    const uint32_t mTile = rows > MSA_ROW_TILE_M ? MSA_ROW_TILE_M : rows;
    uint32_t m0 = 0U;
    uint32_t m1 = 0U;
    HostC2UbHalfSplit(mTile, m0, m1);
    const uint32_t aivRows = m0 > m1 ? m0 : m1;
    const uint64_t bytes =
        static_cast<uint64_t>(aivRows) * 4U * static_cast<uint64_t>(MSA_BLOCK_SIZE) * sizeof(uint16_t);
    return bytes <= static_cast<uint64_t>(MSA_A5_S_PING_BYTES);
}

// Host-only specialization choice: never pull a device length tensor to CPU.
inline bool UseMxFp4FusedStage(gert::TilingContext* context, const MsaIndexScoreInfo& info)
{
    if (!info.isMxFp4 || info.poolBlockSize != MSA_BLOCK_SIZE || info.totalQ == 0U || info.batch == 0U) {
        return false;
    }
    const auto* qlenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_QLEN);
    if (IsHostVisibleTensor(qlenTensor)) {
        const int32_t* data = qlenTensor->GetData<int32_t>();
        if (data != nullptr) {
            for (uint32_t b = 0U; b < info.batch; ++b) {
                int64_t qLen = static_cast<int64_t>(data[b + 1U]) - static_cast<int64_t>(data[b]);
                if (qLen < 0) {
                    qLen = 0;
                }
                if (info.qMaxSeq > 0U && static_cast<uint64_t>(qLen) > info.qMaxSeq) {
                    qLen = info.qMaxSeq;
                }
                if (static_cast<uint64_t>(qLen) * info.numQHeads >= 64U) {
                    return true;
                }
            }
            return false;
        }
    }
    return static_cast<uint64_t>(info.totalQ) * info.numQHeads >= static_cast<uint64_t>(info.batch) * 64U;
}

// 可读 qlen 时逐请求判定；否则小 packed Q 可用宽模板，较大多 batch 输入仍按满 M-tile 保守。
inline uint32_t EstMxFp4WideN(gert::TilingContext* context, const MsaIndexScoreInfo& info)
{
    if (!info.isMxFp4 || info.poolBlockSize == MSA_BLOCK16_SIZE) {
        return 0U;
    }
    auto qlenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_QLEN);
    if (IsHostVisibleTensor(qlenTensor)) {
        const int32_t* data = qlenTensor->GetData<int32_t>();
        if (data != nullptr) {
            for (uint32_t b = 0; b < info.batch; ++b) {
                int32_t q = data[b + 1U] - data[b];
                if (q < 0) {
                    q = 0;
                }
                if (info.qMaxSeq > 0U && q > static_cast<int32_t>(info.qMaxSeq)) {
                    q = static_cast<int32_t>(info.qMaxSeq);
                }
                if (!HostMxFp4GroupFitsRows(static_cast<uint32_t>(q), info.numQHeads)) {
                    return 0U;
                }
            }
            return 1U;
        }
    }
    // Packed or average-small score queries can use the wide template without
    // synchronizing lengths. The average is a hint: device checks each tile's UB capacity.
    const uint64_t packedRows = static_cast<uint64_t>(info.totalQ) * info.numQHeads;
    if (info.batch > 1U &&
        (packedRows <= MSA_ROW_TILE_M || (packedRows <= static_cast<uint64_t>(info.batch) * (MSA_ROW_TILE_M / 2U)))) {
        return 1U;
    }
    if (info.batch <= 1U) {
        return HostMxFp4GroupFitsRows(info.totalQ, info.numQHeads) ? 1U : 0U;
    }
    // 多 batch 且 qlen 不在 host：按满 M-tile 保守判定（与「任一请求很长」同结果）。
    return HostMxFp4GroupFitsRows(MSA_ROW_TILE_M, 1U) ? 1U : 0U;
}

// 手写 op_api 对未传的可选输入给 AllocTensor 空占位（0 维）。占位视同未传。
inline uint32_t EstExactTotalTasks(gert::TilingContext* context, const MsaIndexScoreInfo& info, uint32_t kvChunks)
{
    if (kvChunks > 1U || info.totalQ == 0U) {
        return 0U;
    }
    const uint32_t heads = info.numQHeads == 0U ? 1U : info.numQHeads;
    auto qlenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_QLEN);
    if (IsHostVisibleTensor(qlenTensor)) {
        const int32_t* data = qlenTensor->GetData<int32_t>();
        if (data != nullptr) {
            uint32_t total = 0U;
            for (uint32_t b = 0; b < info.batch; ++b) {
                int32_t q = data[b + 1U] - data[b];
                if (q < 0) {
                    q = 0;
                }
                if (info.qMaxSeq > 0U && q > static_cast<int32_t>(info.qMaxSeq)) {
                    q = static_cast<int32_t>(info.qMaxSeq);
                }
                const uint32_t rows = static_cast<uint32_t>(q) * heads;
                total += (rows + MSA_ROW_TILE_M - 1U) / MSA_ROW_TILE_M;
            }
            return total;
        }
    }
    if (info.batch <= 1U) {
        const uint32_t rows = info.totalQ * heads;
        return (rows + MSA_ROW_TILE_M - 1U) / MSA_ROW_TILE_M;
    }
    return 0U;
}

// 手写 op_api 对未传的可选输入给 AllocTensor 空占位（0 维）。占位视同未传。
inline bool IsOptionalProvided(const gert::StorageShape* shape)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() > 0;
}

// 可选输出未实例化时，运行时下标会前移。按 IR 下标取实例起点。
struct MsaIrOutput {
    const gert::StorageShape* shape = nullptr;
    const gert::CompileTimeTensorDesc* desc = nullptr;
};

inline MsaIrOutput IrOutputAt(gert::TilingContext* context, size_t irIndex)
{
    MsaIrOutput out;
    const size_t runtime = MsaIndexScoreOutput::RuntimeIndex(context, irIndex);
    if (runtime == MsaIndexScoreOutput::ABSENT) {
        return out;
    }
    out.shape = context->GetOutputShape(runtime);
    out.desc = context->GetOutputDesc(runtime);
    return out;
}

// 分页逻辑块数是页表宽度；非分页没有页表，上界是 score 末维。
// score 末维是对齐后的输出宽度，分页时只作不超过已分配 score 的钳位。
inline uint32_t LogicalBlockLimit(const MsaIndexScoreInfo& info)
{
    if (info.poolBlockSize == MSA_BLOCK16_SIZE && MsaIsPaKeyLayout(info.keyLayout) && info.tableWidth > 0U) {
        const uint64_t blocks = static_cast<uint64_t>(info.tableWidth) * (info.pageSize / info.rowsPerBlock);
        return blocks < info.maxBlocksPerBatch ? static_cast<uint32_t>(blocks) : info.maxBlocksPerBatch;
    }
    if (MsaIsPaKeyLayout(info.keyLayout) && info.tableWidth > 0U && info.tableWidth < info.maxBlocksPerBatch) {
        return info.tableWidth;
    }
    return info.maxBlocksPerBatch;
}

inline uint32_t EstMaxVisibleBlocks(gert::TilingContext* context, const MsaIndexScoreInfo& info)
{
    // 打分块数（pool_block_size 粒度）上界；klen host 可见时用真实可见 token 收紧。
    uint32_t vis = LogicalBlockLimit(info);
    auto klenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_KLEN);
    if (!IsHostVisibleTensor(klenTensor)) {
        return vis;
    }
    const int32_t* data = klenTensor->GetData<int32_t>();
    if (data == nullptr) {
        return vis;
    }
    uint32_t maxTok = 0U;
    // 顺序布局统一使用 [B+1] 前缀和；PA 使用 [B] 可见长度。
    if (MsaIsSeqKeyLayout(info.keyLayout)) {
        for (uint32_t b = 0; b < info.batch; ++b) {
            const int32_t delta = data[b + 1U] - data[b];
            if (delta > 0 && static_cast<uint32_t>(delta) > maxTok) {
                maxTok = static_cast<uint32_t>(delta);
            }
        }
    } else {
        for (uint32_t b = 0; b < info.batch; ++b) {
            if (data[b] > 0 && static_cast<uint32_t>(data[b]) > maxTok) {
                maxTok = static_cast<uint32_t>(data[b]);
            }
        }
    }
    const uint32_t blkLen = info.poolBlockSize == 0U ? MSA_BLOCK_SIZE : info.poolBlockSize;
    const uint32_t fromLen = (maxTok + blkLen - 1U) / blkLen;
    if (fromLen < vis) {
        vis = fromLen;
    }
    return vis;
}

inline uint32_t GcdU32(uint32_t a, uint32_t b)
{
    while (b != 0U) {
        const uint32_t t = a % b;
        a = b;
        b = t;
    }
    return (a == 0U) ? 1U : a;
}

// A2/A3：M-task 少于 AIC 时，kvChunks 取 aic/gcd(mTasks, aic)，让总 task 能铺满核。
// 仅当估算 makespan（每核 stile 数）下降才切开，避免短 KV 被握手开销拖慢。
inline uint32_t EstKvChunksA2(uint32_t aicNum, uint32_t packedM, uint32_t visBlocks)
{
    if (packedM == 0U || aicNum == 0U || packedM >= aicNum) {
        return 1U;
    }
    const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
    if (nStiles <= 1U) {
        return 1U;
    }
    uint32_t chunks = aicNum / GcdU32(packedM, aicNum);
    if (chunks > nStiles) {
        chunks = nStiles;
    }
    if (chunks <= 1U) {
        return 1U;
    }
    const uint32_t oldSpan = ((packedM + aicNum - 1U) / aicNum) * nStiles;
    const uint32_t newTasks = packedM * chunks;
    const uint32_t tasksPerCore = (newTasks + aicNum - 1U) / aicNum;
    const uint32_t stilePerChunk = (nStiles + chunks - 1U) / chunks;
    if (tasksPerCore * stilePerChunk >= oldSpan) {
        return 1U;
    }
    return chunks;
}

// A2/A3：batch 已超过 AIC、但每个请求仍是一个 M-tile（短 decode）时，按最长 KV 的 stile
// 再切 S。否则 32 个长短不一的请求会按任务号跨步，最慢核拿到约 2 倍平均页数。
// 仅当估算 makespan 下降才切。prefill（单请求行数 > M-tile）和 950 不走这里。
inline uint32_t EstKvChunksA2BatchBalance(const MsaIndexScoreInfo& info, uint32_t aicNum, uint32_t visBlocks)
{
    if (info.batch <= aicNum || aicNum == 0U) {
        return 1U;
    }
    const uint64_t rows = static_cast<uint64_t>(info.totalQ) * static_cast<uint64_t>(info.numQHeads);
    if (rows > static_cast<uint64_t>(info.batch) * MSA_ROW_TILE_M) {
        return 1U;
    }
    const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
    if (nStiles <= 1U) {
        return 1U;
    }
    const uint32_t oldSpan = ((info.batch + aicNum - 1U) / aicNum) * nStiles;
    uint32_t bestChunks = 1U;
    uint32_t bestSpan = oldSpan;
    for (uint32_t chunks = 2U; chunks <= nStiles && chunks <= 16U; chunks *= 2U) {
        const uint32_t newTasks = info.batch * chunks;
        const uint32_t tasksPerCore = (newTasks + aicNum - 1U) / aicNum;
        const uint32_t stilePerChunk = (nStiles + chunks - 1U) / chunks;
        const uint32_t span = tasksPerCore * stilePerChunk;
        // 与当前最优 span 相同时取更大 chunks：stile 估算打平，但请求长短不一时
        // 更细的 S 切分能把长请求拆开。span 不小于初始 makespan 时不切，避免空切。
        if (span < bestSpan || (span == bestSpan && bestChunks > 1U)) {
            bestSpan = span;
            bestChunks = chunks;
        }
    }
    return bestChunks;
}

// 每请求只有一个 M-tile 时，packed+B 上界会让 chunks 整除不了真实 batch。
// B=4、上界 packedM=5 时 chunks=4，实际只有 16 个 task，4 个 AIC 空转。
// 这里改用 batch 本身。makespan 不差过约 1/6 时取更大的 chunks，
// 避免长短不一的请求只被切成两三份、长请求堆在少数核上。
inline uint32_t ChunkSpan(uint32_t mTasks, uint32_t chunks, uint32_t aicNum, uint32_t nStiles)
{
    const uint32_t tasks = mTasks * chunks;
    const uint32_t tasksPerCore = (tasks + aicNum - 1U) / aicNum;
    const uint32_t stilePerChunk = (nStiles + chunks - 1U) / chunks;
    return tasksPerCore * stilePerChunk;
}

inline uint32_t EstKvChunksA2ByBatch(uint32_t aicNum, uint32_t mTasks, uint32_t nStiles)
{
    if (mTasks == 0U || aicNum == 0U || mTasks >= aicNum || nStiles <= 1U) {
        return 1U;
    }
    uint32_t base = aicNum / GcdU32(mTasks, aicNum);
    if (base > nStiles) {
        base = nStiles;
    }
    if (base <= 1U) {
        return 1U;
    }
    uint32_t bestChunks = base;
    uint32_t bestSpan = ChunkSpan(mTasks, base, aicNum, nStiles);
    const uint32_t spanLimit = bestSpan + ((bestSpan / 6U) > 1U ? (bestSpan / 6U) : 1U);
    for (uint32_t chunks = base + 1U; chunks <= nStiles && chunks <= aicNum; ++chunks) {
        const uint32_t tasks = mTasks * chunks;
        if (tasks / chunks != mTasks) {
            break;
        }
        if ((tasks % aicNum) != 0U) {
            continue;
        }
        const uint32_t span = ChunkSpan(mTasks, chunks, aicNum, nStiles);
        if (span < bestSpan || (span <= spanLimit && chunks > bestChunks)) {
            if (span < bestSpan) {
                bestSpan = span;
            }
            bestChunks = chunks;
        }
    }
    const uint32_t oldSpan = ((mTasks + aicNum - 1U) / aicNum) * nStiles;
    if (bestSpan >= oldSpan) {
        return 1U;
    }
    return bestChunks;
}

// 每请求只有一个 M-tile（短 decode）。packed+B 上界会让 chunks 对不齐真实 batch，950 上同样空核。
inline bool OneTilePerRequest(const MsaIndexScoreInfo& info, uint32_t aicNum)
{
    if (info.batch < 1U || info.batch >= aicNum) {
        return false;
    }
    const uint64_t rows = static_cast<uint64_t>(info.totalQ) * static_cast<uint64_t>(info.numQHeads);
    return (rows > 0U) && (rows <= static_cast<uint64_t>(info.batch) * MSA_ROW_TILE_M);
}

// 950：M-task 填不满 AIC 且可见 KV stile 足够时，沿 S 切开打满核。A1 类 packedM≈aicNum 时 kvChunks=1。
// 短 decode（每请求一个 M-tile）与 A2/A3 一样按真实 batch 取 chunks，避免 packed+B 上界留下空核。
// 其余 950 形状仍用 chunks=aic/packedM。A2/A3 分支不改。
constexpr uint32_t MSA_FD_BASE_STILES = 4U;
constexpr uint32_t MSA_FD_TOLERANCE = 2U;
constexpr uint32_t MSA_FD_LEAST_STILE = 4U;

inline uint32_t EstMxFp4KvChunks(const MsaIndexScoreInfo& info, uint32_t aicNum, uint32_t packedM, uint32_t visBlocks)
{
    const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
    if (nStiles <= 1U) {
        return 1U;
    }
    const uint64_t rows = static_cast<uint64_t>(info.totalQ) * static_cast<uint64_t>(info.numQHeads);
    // A single mid-size M tile benefits from distributing score reduction over KV stiles.
    if (info.isAscend950 && info.poolBlockSize == MSA_BLOCK_SIZE && info.batch == 1U &&
        info.queryLayout == MSA_QUERY_LAYOUT_TND && rows >= MSA_ROW_TILE_M / 2U && rows <= MSA_ROW_TILE_M &&
        nStiles >= MSA_FD_LEAST_STILE && nStiles <= MSA_FD_LEAST_STILE * 2U) {
        return std::min(nStiles, aicNum);
    }
    // Aligned BNSD avoids per-task scale repacking; retain the original policy for D64 large-M long-K.
    if (info.isAscend950 && info.poolBlockSize == MSA_BLOCK_SIZE && info.batch == 1U &&
        info.queryLayout == MSA_QUERY_LAYOUT_BNSD && info.numQHeads % 16U == 0U && rows >= 16U &&
        rows <= MSA_ROW_TILE_M && nStiles >= 2U &&
        (info.headDim != 64U || rows <= MSA_ROW_TILE_M / 2U || nStiles < aicNum)) {
        return std::min(nStiles, aicNum);
    }
    const bool oneTilePerReq = (info.batch >= 1U) && (info.batch < aicNum) && (rows > 0U) &&
                               (rows <= static_cast<uint64_t>(info.batch) * MSA_ROW_TILE_M);
    const uint32_t mTasks = oneTilePerReq ? info.batch : packedM;
    if (mTasks >= aicNum) {
        return 1U;
    }
    const uint32_t noFdCost = ((mTasks + aicNum - 1U) / aicNum) * nStiles;
    if (noFdCost <= MSA_FD_LEAST_STILE) {
        return 1U;
    }
    const uint32_t maxChunks = nStiles / MSA_FD_BASE_STILES;
    if (maxChunks <= 1U) {
        return 1U;
    }
    uint32_t best = 1U;
    uint32_t bestCost = noFdCost;
    bool bestAligned = false;
    for (uint32_t c = 2U; c <= maxChunks; ++c) {
        const uint32_t tasks = mTasks * c;
        if (tasks / c != mTasks) { // 溢出保护，与 EstLaunchAic 同口径
            break;
        }
        const uint32_t cost = ((tasks + aicNum - 1U) / aicNum) * ((nStiles + c - 1U) / c);
        // 代价打平时取任务数整除核数的那个，与 EstKvChunksA2ByBatch 的 tasks % aicNum 约束同口径：
        // ceil 掩盖了「部分核多扛一个任务」的差别。L-decode（mTasks=1、nStiles=128）下 c=26 与
        // c=28 估算同为 5 个 stile，取 26 实测 18.67 µs、取 28 为 17.76 µs，差 5%。
        const bool aligned = ((tasks % aicNum) == 0U);
        if ((cost < bestCost) || ((cost == bestCost) && aligned && !bestAligned && (best > 1U))) {
            bestCost = cost;
            best = c;
            bestAligned = aligned;
        }
    }
    if (best <= 1U || bestCost + MSA_FD_TOLERANCE > noFdCost) {
        return 1U;
    }
    return best;
}

// Estimate task startup as the cost of one S-tile.
inline uint32_t EstStartupAwareKvChunks(uint32_t aicNum, uint32_t batch, uint32_t nStiles)
{
    const uint32_t maxChunks = nStiles < aicNum ? nStiles : aicNum;
    uint32_t bestChunks = 1U;
    uint64_t bestCost = std::numeric_limits<uint64_t>::max();
    for (uint32_t chunks = 1U; chunks <= maxChunks; ++chunks) {
        const uint64_t tasks = static_cast<uint64_t>(batch) * chunks;
        const uint64_t tasksPerCore = (tasks + aicNum - 1U) / aicNum;
        const uint32_t stilesPerChunk = (nStiles + chunks - 1U) / chunks;
        const uint64_t cost = tasksPerCore * (stilesPerChunk + 1U);
        // Ascending candidates keep fewer owners when the modeled cost ties.
        if (cost < bestCost) {
            bestCost = cost;
            bestChunks = chunks;
        }
    }
    return bestChunks;
}

inline uint32_t EstKvChunks(const MsaIndexScoreInfo& info, uint32_t aicNum, uint32_t packedM, uint32_t visBlocks)
{
    if (packedM == 0U || aicNum == 0U) {
        return 1U;
    }
    if (packedM >= aicNum) {
        if (!info.isAscend950) {
            return EstKvChunksA2BatchBalance(info, aicNum, visBlocks);
        }
        return 1U;
    }
    if (!info.isAscend950) {
        if (OneTilePerRequest(info, aicNum)) {
            const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
            return EstKvChunksA2ByBatch(aicNum, info.batch, nStiles);
        }
        return EstKvChunksA2(aicNum, packedM, visBlocks);
    }
    const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
    if (nStiles <= 1U) {
        return 1U;
    }
    if (OneTilePerRequest(info, aicNum)) {
        return EstKvChunksA2ByBatch(aicNum, info.batch, nStiles);
    }
    uint32_t chunks = aicNum / packedM;
    if (chunks > nStiles) {
        chunks = nStiles;
    }
    if (chunks < 1U) {
        chunks = 1U;
    }
    return chunks;
}

// 950：M-task 填不满 AIC 且可见 KV stile 足够时，沿 S 切开打满核。
// 取 2 的幂里 makespan 最小的 chunks，上限 32，独立于最终选择器。
// A2/A3 仍走 EstKvChunksA2。
inline uint32_t EstBlock16KvChunks(const MsaIndexScoreInfo& info, uint32_t aicNum, uint32_t packedM, uint32_t visBlocks)
{
    if (packedM == 0U || aicNum == 0U) {
        return 1U;
    }
    const bool oneTilePerReq = OneTilePerRequest(info, aicNum);
    if (packedM >= aicNum && !oneTilePerReq) {
        return 1U;
    }
    const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
    if (nStiles <= 1U) {
        return 1U;
    }
    uint32_t mTasks = oneTilePerReq ? info.batch : packedM;
    if (mTasks == 0U || mTasks >= aicNum) {
        return 1U;
    }
    // Score production uses the same makespan policy for score and IRS selection outputs.
    uint32_t bestChunks = 1U;
    uint32_t bestSpan = ((mTasks + aicNum - 1U) / aicNum) * nStiles;
    for (uint32_t chunks = 2U; chunks <= nStiles && chunks <= 32U; chunks *= 2U) {
        const uint32_t span = ChunkSpan(mTasks, chunks, aicNum, nStiles);
        if (span < bestSpan) {
            bestSpan = span;
            bestChunks = chunks;
        }
    }
    return bestChunks;
}

inline uint32_t EstLaunchAic(uint32_t aicNum, uint32_t packedM, uint32_t kvChunks)
{
    if (aicNum == 0U) {
        return 1U;
    }
    uint32_t est = packedM;
    if (kvChunks > 1U) {
        const uint32_t prod = packedM * kvChunks;
        est = (prod / kvChunks != packedM) ? aicNum : prod;
    }
    if (est < 1U) {
        est = 1U;
    }
    if (est > aicNum) {
        est = aicNum;
    }
    return est;
}

inline uint64_t GetDefaultStride0(const gert::Shape& shape)
{
    uint64_t stride = 1;
    for (size_t dim = 1; dim < shape.GetDimNum(); ++dim) {
        stride *= static_cast<uint64_t>(shape.GetDim(dim));
    }
    return stride;
}

// 逻辑 view：优先 OriginShape（aclCreateTensor 的 viewDims / torch sizes）。
// StorageShape 在 dim0 插空时可能是 [NP*gap, ...]，不能当 numPages。
inline const gert::Shape& KeyLogicalShape(const gert::StorageShape* keyShape, uint32_t keyLayout)
{
    const size_t expectRank = (keyLayout == MSA_KEY_LAYOUT_TND) ? MSA_KEY_TND_DIM_NUM : MSA_KEY_PA_DIM_NUM;
    const gert::Shape& origin = keyShape->GetOriginShape();
    if (origin.GetDimNum() == expectRank) {
        return origin;
    }
    return keyShape->GetStorageShape();
}

// aclnn TensorV2 优先 GetRequiredInputStride；
// 文档接口 GetInputStride 作为兼容回退。
inline const gert::Stride* GetKeyStrideDesc(gert::TilingContext* context)
{
    const gert::Stride* stride = context->GetRequiredInputStride(MSA_IDX_KEY);
    if (stride != nullptr) {
        return stride;
    }
    return context->GetInputStride(MSA_IDX_KEY);
}

// 对齐 MlaProlog GetCacheStride0：dim1…末维须等于紧凑 stride，仅 dim0 可非连续。
// size-1 轴不参与寻址：PyTorch / torch_npu 对其 stride 不做连续约束（BBND↔BNBD 且 N2=1
// 时 permute().contiguous() 仍可能是 dim1 stride=P 而非 N2*P*D），跳过该轴比较。
inline ge::graphStatus ValidateKeyInnerAxesContiguous(gert::TilingContext* context, const gert::Stride* stride,
                                                      const gert::Shape& shape)
{
    if (stride == nullptr || stride->GetDimNum() != shape.GetDimNum()) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t expectedStride = 1U;
    for (int64_t dim = static_cast<int64_t>(shape.GetDimNum()) - 1; dim >= 1; --dim) {
        const int64_t dimSize = shape.GetDim(static_cast<size_t>(dim));
        OP_CHECK_IF(dimSize < 0, OP_LOGE(context, "key dim%ld size is negative.", dim), return ge::GRAPH_FAILED);
        if (dimSize > 1) {
            const uint64_t actualStride = static_cast<uint64_t>(stride->GetStride(static_cast<size_t>(dim)));
            OP_CHECK_IF(actualStride != expectedStride,
                        OP_LOGE_FOR_INVALID_STRIDE(
                            context->GetNodeName(), (std::string("key.dim") + std::to_string(dim)).c_str(),
                            std::to_string(actualStride).c_str(), std::to_string(expectedStride).c_str()),
                        return ge::GRAPH_FAILED);
        }
        expectedStride *= static_cast<uint64_t>(dimSize);
    }
    return ge::GRAPH_SUCCESS;
}

// PA：strideKvBlock 取 key dim0 元素 stride。连续输入写入值与 shape 推算相同。
inline ge::graphStatus ResolveStrideKvBlock(gert::TilingContext* context, const MsaIndexScoreInfo& info,
                                            uint32_t defaultStrideKvBlock, uint32_t& strideKvBlock)
{
    strideKvBlock = defaultStrideKvBlock;

    const gert::StorageShape* keyShape = context->GetInputShape(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape& kc = KeyLogicalShape(keyShape, info.keyLayout);
    if (kc.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t defaultStride0 = GetDefaultStride0(kc);
    const gert::Stride* stride = GetKeyStrideDesc(context);
    if (stride == nullptr || stride->GetDimNum() != kc.GetDimNum()) {
        OP_LOGD(context->GetNodeName(), "key has no stride desc, treat as contiguous, strideKvBlock=%u.",
                strideKvBlock);
        return ge::GRAPH_SUCCESS;
    }
    if (ValidateKeyInnerAxesContiguous(context, stride, kc) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t actualStride0 = stride->GetStride(MSA_DIM_0);
    if (MsaIsSeqKeyLayout(info.keyLayout)) {
        OP_CHECK_IF(
            actualStride0 < 0 || static_cast<uint64_t>(actualStride0) != defaultStride0,
            OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "key.dim0", std::to_string(actualStride0).c_str(),
                                       std::to_string(defaultStride0).c_str()),
            return ge::GRAPH_FAILED);
        strideKvBlock = 0;
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(actualStride0 < 0 || static_cast<uint64_t>(actualStride0) < defaultStride0,
                OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "key.dim0", std::to_string(actualStride0).c_str(),
                                           (std::string(">= ") + std::to_string(defaultStride0)).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(actualStride0) > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "key.dim0 stride",
                                                      std::to_string(actualStride0).c_str(),
                                                      "the stride must not exceed UINT32_MAX"),
                return ge::GRAPH_FAILED);
    strideKvBlock = static_cast<uint32_t>(actualStride0);
    OP_LOGI(context->GetNodeName(), "PA key strideKvBlock=%u (contiguous default=%lu).", strideKvBlock, defaultStride0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseLayoutKeyAttr(gert::TilingContext* context, const std::string& layoutKey, uint32_t& keyLayout)
{
    const std::string s = layoutKey.empty() ? "BBND" : layoutKey;
    if (s == "TND") {
        keyLayout = MSA_KEY_LAYOUT_TND;
    } else if (s == "BSND") {
        keyLayout = MSA_KEY_LAYOUT_BSND;
    } else if (s == "BNSD") {
        keyLayout = MSA_KEY_LAYOUT_BNSD;
    } else if (s == "BBND") {
        keyLayout = MSA_KEY_LAYOUT_BBND;
    } else if (s == "BNBD") {
        keyLayout = MSA_KEY_LAYOUT_BNBD;
    } else {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "layout_key", s.c_str(),
                                               "the supported values are TND, BSND, BNSD, BBND and BNBD");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseLayoutQueryAttr(gert::TilingContext* context, const char* layoutQuery, uint32_t& queryLayout)
{
    const char* s = (layoutQuery == nullptr || layoutQuery[0] == '\0') ? "TND" : layoutQuery;
    if (std::strcmp(s, "TND") == 0) {
        queryLayout = MSA_QUERY_LAYOUT_TND;
    } else if (std::strcmp(s, "BSND") == 0) {
        queryLayout = MSA_QUERY_LAYOUT_BSND;
    } else if (std::strcmp(s, "BNSD") == 0) {
        queryLayout = MSA_QUERY_LAYOUT_BNSD;
    } else {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "layout_query", s,
                                               "the supported values are TND, BSND and BNSD");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 块长 16 的 score 末维固定为 TopK 宽度，不能再拿来当扫描上界。
// 这里按 KV / 页表算出对齐后的逻辑块数，与改 shape 之前的 score 末维一致。
// actual_seq_klen 是原始 token 数，块数按 pool_block_size 切。读不到长度时退回 key 的压缩行容量。
inline uint32_t ResolveBlock16ScanStride(gert::TilingContext* context, const MsaIndexScoreInfo& info)
{
    uint64_t maxBlocks = 0;
    const uint32_t rowsPerBlock = info.rowsPerBlock;
    const uint32_t tokenBlock = info.poolBlockSize == 0U ? rowsPerBlock : info.poolBlockSize;
    if (MsaIsPaKeyLayout(info.keyLayout)) {
        if (rowsPerBlock != 0U) {
            maxBlocks = static_cast<uint64_t>(info.tableWidth) * (info.pageSize / rowsPerBlock);
        }
    } else {
        const gert::Tensor* klenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_KLEN);
        const int32_t* data = nullptr;
        if (IsHostVisibleTensor(klenTensor)) {
            data = klenTensor->GetData<int32_t>();
        }
        if (data != nullptr) {
            for (uint32_t b = 0; b < info.batch; ++b) {
                const int32_t kv = data[b + 1U] - data[b];
                if (kv <= 0 || tokenBlock == 0U) {
                    continue;
                }
                const uint64_t blocks = (static_cast<uint64_t>(kv) + tokenBlock - 1U) / tokenBlock;
                if (blocks > maxBlocks) {
                    maxBlocks = blocks;
                }
            }
        } else if (info.keyLayout == MSA_KEY_LAYOUT_TND && rowsPerBlock != 0U) {
            maxBlocks = (static_cast<uint64_t>(info.totalK) + rowsPerBlock - 1U) / rowsPerBlock;
        } else if (rowsPerBlock != 0U) {
            maxBlocks = (static_cast<uint64_t>(info.kMaxSeq) + rowsPerBlock - 1U) / rowsPerBlock;
        }
    }
    if (maxBlocks == 0U) {
        maxBlocks = 1U;
    }
    if (maxBlocks > std::numeric_limits<uint32_t>::max()) {
        maxBlocks = std::numeric_limits<uint32_t>::max();
    }
    return RoundUpU32(static_cast<uint32_t>(maxBlocks), MSA_SCORE_STRIDE_ALIGN);
}

// return_choice!=0 时没有用户 score。末维与 InferShape 相同：逻辑块数 16 对齐。
uint32_t DeriveAlignedMaxBlocks(gert::TilingContext* context, const MsaIndexScoreInfo& info)
{
    if (info.poolBlockSize == MSA_BLOCK16_SIZE) {
        return ResolveBlock16ScanStride(context, info);
    }
    uint32_t raw = 0;
    const uint32_t pool = (info.poolBlockSize == 0U) ? MSA_BLOCK_SIZE : info.poolBlockSize;
    if (MsaIsPaKeyLayout(info.keyLayout)) {
        const uint64_t blocks = static_cast<uint64_t>(info.tableWidth) * (info.pageSize / pool);
        if (blocks > UINT32_MAX - (MSA_SCORE_STRIDE_ALIGN - 1U)) {
            return 0U;
        }
        raw = static_cast<uint32_t>(blocks);
    } else {
        const gert::Tensor* klenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_KLEN);
        const int32_t* klenData = nullptr;
        if (IsHostVisibleTensor(klenTensor)) {
            klenData = klenTensor->GetData<int32_t>();
        }
        if (klenData != nullptr && info.batch > 0U) {
            for (uint32_t b = 0; b < info.batch; ++b) {
                int32_t kv = klenData[b + 1U] - klenData[b];
                if (kv < 0) {
                    kv = 0;
                }
                if (info.kMaxSeq > 0U && static_cast<uint32_t>(kv) > info.kMaxSeq) {
                    kv = static_cast<int32_t>(info.kMaxSeq);
                }
                const uint32_t blocks = (kv <= 0) ? 0U : (static_cast<uint32_t>(kv) + pool - 1U) / pool;
                if (blocks > raw) {
                    raw = blocks;
                }
            }
        } else if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
            raw = (pool == 0U) ? 0U : (info.totalK + pool - 1U) / pool;
        } else {
            raw = (pool == 0U) ? 0U : (info.kMaxSeq + pool - 1U) / pool;
        }
    }
    if (raw == 0U) {
        raw = 1U;
    }
    return RoundUpU32(raw, MSA_SCORE_STRIDE_ALIGN);
}

ge::graphStatus ParseAndCheck(gert::TilingContext* context, MsaIndexScoreInfo& info)
{
    info.platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(info.platformInfo == nullptr, OP_LOGE(context, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatformEarly = platform_ascendc::PlatformAscendC(info.platformInfo);
    info.isAscend950 = (ascendcPlatformEarly.GetCurNpuArch() == NpuArch::DAV_3510);

    const gert::StorageShape* queryShape = context->GetInputShape(MSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::StorageShape* keyShape = context->GetInputShape(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::StorageShape* blockTableShape = context->GetOptionalInputShape(MSA_IDX_BLOCK_TABLE);
    const gert::StorageShape* actualSeqQlenShape = context->GetOptionalInputShape(MSA_IDX_ACTUAL_SEQ_QLEN);
    OP_CHECK_IF(!IsOptionalProvided(actualSeqQlenShape), OP_LOGE(context, "TND query requires actual_seq_qlen."),
                return ge::GRAPH_FAILED);
    const gert::StorageShape* actualSeqKlenShape = context->GetOptionalInputShape(MSA_IDX_ACTUAL_SEQ_KLEN);
    OP_CHECK_IF(!IsOptionalProvided(actualSeqKlenShape), OP_LOGE(context, "actual_seq_klen is required."),
                return ge::GRAPH_FAILED);
    // start_loc 是必选输入，排在 query、key 之后。
    const gert::StorageShape* startLocShape = context->GetRequiredInputShape(MSA_IDX_START_LOC);
    OP_CHECK_NULL_WITH_CONTEXT(context, startLocShape);

    const gert::Shape& q = queryShape->GetStorageShape();
    const gert::Shape& qlen = actualSeqQlenShape->GetStorageShape();
    const gert::Shape& klen = actualSeqKlenShape->GetStorageShape();

    const auto* keyDesc = context->GetInputDesc(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDesc);
    info.keyDtype = keyDesc->GetDataType();
    OP_CHECK_IF(static_cast<ge::Format>(ge::GetPrimaryFormat(keyDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context, "FRACTAL_NZ key is not supported."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(qlen.GetDimNum() != MSA_RANK_1D || qlen.GetDim(MSA_DIM_0) < MSA_QLEN_MIN_DIM0,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "actual_seq_qlen",
                                                         (std::to_string(qlen.GetDimNum()) + "D").c_str(),
                                                         "actual_seq_qlen must be 1D [B+1]"),
                return ge::GRAPH_FAILED);
    info.batch = static_cast<uint32_t>(qlen.GetDim(MSA_DIM_0) - 1);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* layoutKeyPtr = attrs->GetStr(MSA_ATTR_LAYOUT_KEY);
    const std::string layoutKey = (layoutKeyPtr == nullptr) ? "" : layoutKeyPtr;
    if (ParseLayoutKeyAttr(context, layoutKey, info.keyLayout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseLayoutQueryAttr(context, attrs->GetStr(MSA_ATTR_LAYOUT_QUERY), info.queryLayout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // pool_block_size：缺省 128。只允许 16 / 128。顺序布局的 klen 统一为 [B+1] 前缀和。
    // V2（query 与 key 同布局）的 BSND/BNSD 固定 16，仅 950。
    // 非 V2（layout_query=TND、pool=128）允许 key 为 BSND/BNSD，A2/A3 与 950 都支持。
    const int64_t* poolBlockSizeAttrPtr = attrs->GetInt(MSA_ATTR_POOL_BLOCK_SIZE);
    const uint32_t poolBlockSizeAttr = (poolBlockSizeAttrPtr == nullptr || *poolBlockSizeAttrPtr <= 0) ?
                                           MSA_BLOCK_SIZE :
                                           static_cast<uint32_t>(*poolBlockSizeAttrPtr);
    OP_CHECK_IF(poolBlockSizeAttr != MSA_BLOCK_SIZE && poolBlockSizeAttr != MSA_BLOCK16_SIZE,
                OP_LOGE(context, "pool_block_size must be %u or %u, got %u.", MSA_BLOCK_SIZE, MSA_BLOCK16_SIZE,
                        poolBlockSizeAttr),
                return ge::GRAPH_FAILED);
    const bool logicalBlock16 = poolBlockSizeAttr == MSA_BLOCK16_SIZE;
    const int64_t* returnChoicePtr = attrs->GetInt(MSA_ATTR_RETURN_CHOICE);
    const int64_t returnChoice = (returnChoicePtr == nullptr) ? 0 : *returnChoicePtr;
    OP_CHECK_IF(returnChoice != static_cast<int64_t>(MSA_RETURN_SCORE) &&
                    returnChoice != static_cast<int64_t>(MSA_RETURN_INDICES) &&
                    returnChoice != static_cast<int64_t>(MSA_RETURN_INDICES_VALUES),
                OP_LOGE(context, "return_choice must be 0, 1 or 2, got %ld.", returnChoice), return ge::GRAPH_FAILED);
    info.returnChoice = static_cast<uint32_t>(returnChoice);
    if (info.returnChoice == MSA_RETURN_SCORE) {
        info.topk = 0;
    } else {
        const int64_t* topkPtr = attrs->GetInt(MSA_ATTR_TOPK);
        const int64_t topkVal = (topkPtr == nullptr) ? static_cast<int64_t>(MSA_TOPK_DEFAULT) : *topkPtr;
        OP_CHECK_IF(topkVal < 1 || topkVal > static_cast<int64_t>(MSA_TOPK_MAX),
                    OP_LOGE(context, "topk must be in [1, 128], got %ld.", topkVal), return ge::GRAPH_FAILED);
        OP_CHECK_IF(logicalBlock16 && topkVal != MSA_BLOCK16_TOPK,
                    OP_LOGE(context, "pool_block_size=16 requires topk=%u.", MSA_BLOCK16_TOPK),
                    return ge::GRAPH_FAILED);
        info.topk = static_cast<uint32_t>(topkVal);
    }
    const int64_t* compressFactorPtr = attrs->GetInt(MSA_ATTR_COMPRESS_FACTOR);
    const int64_t compressFactor = (compressFactorPtr == nullptr) ? 1 : *compressFactorPtr;
    if (logicalBlock16) {
        OP_CHECK_IF(compressFactor < 1 || compressFactor > MSA_BLOCK16_SIZE || MSA_BLOCK16_SIZE % compressFactor != 0,
                    OP_LOGE(context, "compress_factor(%ld) must divide pool_block_size=16.", compressFactor),
                    return ge::GRAPH_FAILED);
        info.compressFactor = static_cast<uint32_t>(compressFactor);
    } else {
        OP_CHECK_IF(compressFactor != 1, OP_LOGE(context, "compress_factor only supports 1, got %ld.", compressFactor),
                    return ge::GRAPH_FAILED);
    }
    info.rowsPerBlock = poolBlockSizeAttr / info.compressFactor;
    const bool batchKey128 = (info.keyLayout == MSA_KEY_LAYOUT_BSND || info.keyLayout == MSA_KEY_LAYOUT_BNSD) &&
                             poolBlockSizeAttr == MSA_BLOCK_SIZE;
    if (MsaIsSeqKeyLayout(info.keyLayout)) {
        if (batchKey128) {
            OP_CHECK_IF(info.queryLayout != MSA_QUERY_LAYOUT_TND,
                        OP_LOGE(context, "pool_block_size=%u BSND/BNSD requires layout_query=TND.", MSA_BLOCK_SIZE),
                        return ge::GRAPH_FAILED);
        } else {
            const uint32_t queryOfKey = (info.keyLayout == MSA_KEY_LAYOUT_BSND) ? MSA_QUERY_LAYOUT_BSND :
                                        (info.keyLayout == MSA_KEY_LAYOUT_BNSD) ? MSA_QUERY_LAYOUT_BNSD :
                                                                                  MSA_QUERY_LAYOUT_TND;
            OP_CHECK_IF(
                info.queryLayout != queryOfKey,
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                    context->GetNodeName(), "layout_key", layoutKey.c_str(),
                    (std::string("the non-paged key layout must match layout_query=") +
                     (attrs->GetStr(MSA_ATTR_LAYOUT_QUERY) == nullptr ? "TND" : attrs->GetStr(MSA_ATTR_LAYOUT_QUERY)))
                        .c_str()),
                return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(info.queryLayout > MSA_QUERY_LAYOUT_BNSD,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "layout_query",
                                                           std::to_string(info.queryLayout).c_str(),
                                                           "the layout_query value is invalid"),
                    return ge::GRAPH_FAILED);
    }
    if (info.queryLayout == MSA_QUERY_LAYOUT_TND) {
        OP_CHECK_IF(q.GetDimNum() != MSA_QUERY_DIM_NUM,
                    OP_LOGE(context, "layout_query=TND requires query rank 3 [T1,N1,D], got %zu.", q.GetDimNum()),
                    return ge::GRAPH_FAILED);
        info.totalQ = static_cast<uint32_t>(q.GetDim(MSA_DIM_0));
        info.numQHeads = static_cast<uint32_t>(q.GetDim(MSA_DIM_1));
        info.headDim = static_cast<uint32_t>(q.GetDim(MSA_DIM_2));
        info.qMaxSeq = 0;
    } else {
        OP_CHECK_IF(q.GetDimNum() != MSA_KEY_PA_DIM_NUM,
                    OP_LOGE(context, "layout_query=BSND/BNSD requires query rank 4, got %zu.", q.GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            static_cast<uint32_t>(q.GetDim(MSA_DIM_0)) != info.batch,
            OP_LOGE(context, "query batch(%ld) must equal actual_seq_qlen batch(%u).", q.GetDim(MSA_DIM_0), info.batch),
            return ge::GRAPH_FAILED);
        if (info.queryLayout == MSA_QUERY_LAYOUT_BSND) {
            info.qMaxSeq = static_cast<uint32_t>(q.GetDim(MSA_DIM_1));
            info.numQHeads = static_cast<uint32_t>(q.GetDim(MSA_DIM_2));
        } else {
            info.numQHeads = static_cast<uint32_t>(q.GetDim(MSA_DIM_1));
            info.qMaxSeq = static_cast<uint32_t>(q.GetDim(MSA_DIM_2));
        }
        info.headDim = static_cast<uint32_t>(q.GetDim(MSA_DIM_3));
        info.totalQ = info.batch * info.qMaxSeq;
        // GetData() 对 device tensor 返回非空指针，host 解引用会段错误。只在 kOnHost/kFollowing 时读前缀。
        bool qPrefixApplied = false;
        const gert::Tensor* qlenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_QLEN);
        if (IsHostVisibleTensor(qlenTensor)) {
            const int32_t* qlenData = qlenTensor->GetData<int32_t>();
            if (qlenData != nullptr) {
                info.totalQ = static_cast<uint32_t>(qlenData[info.batch]);
                qPrefixApplied = true;
                for (uint32_t b = 0; b < info.batch; ++b) {
                    const int32_t qlenB = qlenData[b + 1] - qlenData[b];
                    OP_CHECK_IF(qlenB < 0 || static_cast<uint32_t>(qlenB) > info.qMaxSeq,
                                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                    context->GetNodeName(), "actual_seq_qlen", std::to_string(qlenB).c_str(),
                                    (std::string("the query length must be non-negative and must not exceed the "
                                                 "layout S=") +
                                     std::to_string(info.qMaxSeq))
                                        .c_str()),
                                return ge::GRAPH_FAILED);
                }
            }
        }
        // score 的 T1 决定 head 步长。return_choice!=0 时没有 score，totalQ 保持前缀和或 B*S。
        if (info.returnChoice != MSA_RETURN_SCORE) {
            if (logicalBlock16 && !qPrefixApplied) {
                const auto* indices = IrOutputAt(context, MsaIndexScoreOutput::INDICES).shape;
                if (indices != nullptr && indices->GetStorageShape().GetDimNum() == MSA_QUERY_DIM_NUM) {
                    const int64_t tokens = indices->GetStorageShape().GetDim(MSA_DIM_1);
                    OP_CHECK_IF(tokens < 0 || tokens > info.totalQ,
                                OP_LOGE(context, "sparse_indices T1(%ld) exceeds batch*S(%u).", tokens, info.totalQ),
                                return ge::GRAPH_FAILED);
                    info.totalQ = static_cast<uint32_t>(tokens);
                }
            }
        } else {
            const gert::StorageShape* scoreShapeEarly = IrOutputAt(context, MsaIndexScoreOutput::SCORE).shape;
            OP_CHECK_NULL_WITH_CONTEXT(context, scoreShapeEarly);
            const gert::Shape& scoreEarly = scoreShapeEarly->GetStorageShape();
            OP_CHECK_IF(scoreEarly.GetDimNum() != MSA_QUERY_DIM_NUM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "score",
                                                                 (std::to_string(scoreEarly.GetDimNum()) + "D").c_str(),
                                                                 "score must have 3 dims [N1, T1, stride]"),
                        return ge::GRAPH_FAILED);
            // pool_block_size=16 的 head 间 max 把 score 收到 1 个 head；块长 128 仍是每个 query head 一份。
            const uint32_t expectScoreHeads = (poolBlockSizeAttr == MSA_BLOCK16_SIZE) ? 1U : info.numQHeads;
            OP_CHECK_IF(
                static_cast<uint32_t>(scoreEarly.GetDim(MSA_DIM_0)) != expectScoreHeads,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "score.shape[0]", std::to_string(scoreEarly.GetDim(MSA_DIM_0)).c_str(),
                    (std::string("the head dim must be ") + std::to_string(expectScoreHeads)).c_str()),
                return ge::GRAPH_FAILED);
            const int64_t scoreTokens = scoreEarly.GetDim(MSA_DIM_1);
            OP_CHECK_IF(scoreTokens < 0, OP_LOGE(context, "score T1 must be concrete, got %ld.", scoreTokens),
                        return ge::GRAPH_FAILED);
            if (qPrefixApplied) {
                OP_CHECK_IF(
                    static_cast<uint32_t>(scoreTokens) != info.totalQ,
                    OP_LOGE(context, "score T1(%ld) must equal actual_seq_qlen prefix(%u).", scoreTokens, info.totalQ),
                    return ge::GRAPH_FAILED);
            } else {
                const uint32_t cap = info.batch * info.qMaxSeq;
                OP_CHECK_IF(static_cast<uint32_t>(scoreTokens) > cap,
                            OP_LOGE(context, "score T1(%ld) exceeds batch*S(%u).", scoreTokens, cap),
                            return ge::GRAPH_FAILED);
                info.totalQ = static_cast<uint32_t>(scoreTokens);
            }
        }
    }
    // 块长 16 的 BSND/BNSD（及 query BSND/BNSD）仅 950。非 V2 的 pool=128 key BSND/BNSD 不在此列。
    const bool m3Layout =
        info.queryLayout == MSA_QUERY_LAYOUT_BSND || info.queryLayout == MSA_QUERY_LAYOUT_BNSD ||
        ((info.keyLayout == MSA_KEY_LAYOUT_BSND || info.keyLayout == MSA_KEY_LAYOUT_BNSD) && !batchKey128);
    OP_CHECK_IF(m3Layout && !info.isAscend950, OP_LOGE(context, "BSND/BNSD is only supported on Ascend 950."),
                return ge::GRAPH_FAILED);

    const gert::Shape& kc = KeyLogicalShape(keyShape, info.keyLayout);

    if (MsaIsSeqKeyLayout(info.keyLayout)) {
        // 块长来自 pool_block_size。TND 与非 V2 的 BSND/BNSD 为 128；V2 的 BSND/BNSD 为 16。
        info.poolBlockSize = poolBlockSizeAttr;
        info.pageSize = info.rowsPerBlock;
        info.numPages = 0;
        info.kMaxSeq = 0;
        OP_CHECK_IF(IsOptionalProvided(blockTableShape), OP_LOGE(context, "non-paged key must not pass block_table."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(klen.GetDimNum() != MSA_RANK_1D || static_cast<uint32_t>(klen.GetDim(MSA_DIM_0)) != info.batch + 1,
                    OP_LOGE(context, "non-paged actual_seq_klen must be [B+1] prefix-sum."), return ge::GRAPH_FAILED);
        if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
            OP_CHECK_IF(kc.GetDimNum() != MSA_KEY_TND_DIM_NUM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "key",
                                                                 (std::to_string(kc.GetDimNum()) + "D").c_str(),
                                                                 "layout_key=TND requires key rank 3 [T2, N2, D]"),
                        return ge::GRAPH_FAILED);
            info.totalK = static_cast<uint32_t>(kc.GetDim(MSA_DIM_0));
            info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
            OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(MSA_DIM_2)) != info.headDim,
                        OP_LOGE(context, "TND key headDim(%ld) must equal query headDim(%u).", kc.GetDim(MSA_DIM_2),
                                info.headDim),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(kc.GetDimNum() != MSA_KEY_PA_DIM_NUM,
                        OP_LOGE(context, "layout_key=BSND/BNSD requires key rank 4, got %zu.", kc.GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                static_cast<uint32_t>(kc.GetDim(MSA_DIM_0)) != info.batch,
                OP_LOGE(context, "key batch(%ld) must equal query batch(%u).", kc.GetDim(MSA_DIM_0), info.batch),
                return ge::GRAPH_FAILED);
            if (info.keyLayout == MSA_KEY_LAYOUT_BSND) {
                info.kMaxSeq = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
                info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_2));
            } else {
                info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
                info.kMaxSeq = static_cast<uint32_t>(kc.GetDim(MSA_DIM_2));
            }
            info.totalK = info.batch * info.kMaxSeq;
            const gert::Tensor* klenTensor = context->GetOptionalInputTensor(MSA_IDX_ACTUAL_SEQ_KLEN);
            if (IsHostVisibleTensor(klenTensor)) {
                const int32_t* klenData = klenTensor->GetData<int32_t>();
                if (klenData != nullptr) {
                    for (uint32_t b = 0; b < info.batch; ++b) {
                        const int32_t kvB = klenData[b + 1U] - klenData[b];
                        OP_CHECK_IF(kvB < 0 || static_cast<uint32_t>(kvB) > info.kMaxSeq,
                                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                        context->GetNodeName(), "actual_seq_klen", std::to_string(kvB).c_str(),
                                        (std::string("the key length must be non-negative and must not exceed the "
                                                     "layout S=") +
                                         std::to_string(info.kMaxSeq))
                                            .c_str()),
                                    return ge::GRAPH_FAILED);
                    }
                }
            }
            OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(MSA_DIM_3)) != info.headDim,
                        OP_LOGE(context, "BSND/BNSD key headDim(%ld) must equal query headDim(%u).",
                                kc.GetDim(MSA_DIM_3), info.headDim),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(!IsOptionalProvided(blockTableShape),
                    OP_LOGE(context, "PageAttention (BBND/BNBD) requires block_table."), return ge::GRAPH_FAILED);
        const gert::Shape& bt = blockTableShape->GetStorageShape();
        OP_CHECK_IF(bt.GetDimNum() != MSA_BLOCK_TABLE_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "block_table",
                                                             (std::to_string(bt.GetDimNum()) + "D").c_str(),
                                                             "block_table must have 2 dims"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint32_t>(bt.GetDim(MSA_DIM_0)) != info.batch,
                    OP_LOGE(context, "block_table batch(%ld) must equal actual_seq_qlen batch(%u).",
                            bt.GetDim(MSA_DIM_0), info.batch),
                    return ge::GRAPH_FAILED);
        // 页表按存储页粒度：宽度 = 存储页数，与打分块数（pool_block_size attr）解耦。
        info.tableWidth = static_cast<uint32_t>(bt.GetDim(MSA_DIM_1));
        OP_CHECK_IF(kc.GetDimNum() != MSA_KEY_PA_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "key", (std::to_string(kc.GetDimNum()) + "D").c_str(),
                        (std::string("layout_key=") + (info.keyLayout == MSA_KEY_LAYOUT_BNBD ? "BNBD" : "BBND") +
                         " requires key rank 4")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        info.numPages = static_cast<uint32_t>(kc.GetDim(MSA_DIM_0));
        // 存储页长 = key 的页维度；打分块长 = pool_block_size attr。两者解耦：打分块必须放得进存储页。
        uint32_t pageSize = 0;
        if (info.keyLayout == MSA_KEY_LAYOUT_BBND) {
            pageSize = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
            info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_2));
            OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(MSA_DIM_3)) != info.headDim,
                        OP_LOGE(context, "BBND key headDim(%ld) must equal query headDim(%u).", kc.GetDim(MSA_DIM_3),
                                info.headDim),
                        return ge::GRAPH_FAILED);
        } else {
            info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
            pageSize = static_cast<uint32_t>(kc.GetDim(MSA_DIM_2));
            OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(MSA_DIM_3)) != info.headDim,
                        OP_LOGE(context, "BNBD key headDim(%ld) must equal query headDim(%u).", kc.GetDim(MSA_DIM_3),
                                info.headDim),
                        return ge::GRAPH_FAILED);
        }
        // The key shape describes stored compressed rows, not original tokens.
        // A model page of 128 tokens occupies 128/compressFactor key rows.
        const bool compressedPage =
            logicalBlock16 && info.compressFactor > 1U && pageSize == MSA_BLOCK_SIZE / info.compressFactor;
        OP_CHECK_IF(pageSize != MSA_BLOCK_SIZE && pageSize != MSA_BLOCK16_SIZE && !compressedPage,
                    OP_LOGE(context, "BBND/BNBD page size must be %u, %u, or 128/compress_factor for pool16, got %u.",
                            MSA_BLOCK_SIZE, MSA_BLOCK16_SIZE, pageSize),
                    return ge::GRAPH_FAILED);
        // 页长按 key 行计，与打分块对齐的是 rowsPerBlock 而非 pool_block_size。
        OP_CHECK_IF(
            info.rowsPerBlock > pageSize || (pageSize % info.rowsPerBlock) != 0U,
            OP_LOGE(context,
                    "pool_block_size/compress_factor(%u) must divide BBND/BNBD page size(%u) and not exceed it.",
                    info.rowsPerBlock, pageSize),
            return ge::GRAPH_FAILED);
        info.poolBlockSize = poolBlockSizeAttr;
        info.pageSize = pageSize;
        OP_CHECK_IF(klen.GetDimNum() != MSA_RANK_1D || static_cast<uint32_t>(klen.GetDim(MSA_DIM_0)) != info.batch,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "actual_seq_klen", Ops::Base::ToString(klen).c_str(),
                        (std::string("PA actual_seq_klen must be 1D [B] with leading dim equal to batch=") +
                         std::to_string(info.batch))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }

    // score 末维 = 打分块数（16 对齐）。return_choice=0 时取自用户 score；否则按输入推导，与 InferShape 一致。
    const MsaIrOutput scoreOutIr = IrOutputAt(context, 0);
    const MsaIrOutput indicesOutIr = IrOutputAt(context, 1);
    const MsaIrOutput valuesOutIr = IrOutputAt(context, 2);
    const gert::StorageShape* scoreShape = scoreOutIr.shape;
    const gert::StorageShape* indicesShape = indicesOutIr.shape;
    const gert::StorageShape* valuesShape = valuesOutIr.shape;
    const bool hasScore = MsaIndexScoreOutput::IsPresent(scoreShape);
    const bool hasIndices = MsaIndexScoreOutput::IsPresent(indicesShape);
    const bool hasValues = MsaIndexScoreOutput::IsPresent(valuesShape);
    if (info.returnChoice == MSA_RETURN_SCORE) {
        OP_CHECK_IF(!hasScore, OP_LOGE(context, "outputs do not match return_choice=%u.", info.returnChoice),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hasIndices || hasValues,
                    OP_LOGE(context, "outputs do not match return_choice=%u.", info.returnChoice),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(logicalBlock16 && (scoreOutIr.desc == nullptr || scoreOutIr.desc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGE(context, "pool_block_size=16 score dtype must be float16."), return ge::GRAPH_FAILED);
        const gert::Shape& scOut = scoreShape->GetStorageShape();
        OP_CHECK_IF(scOut.GetDimNum() != MSA_QUERY_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "score",
                                                             (std::to_string(scOut.GetDimNum()) + "D").c_str(),
                                                             "score must have 3 dims [N1, T1, stride]"),
                    return ge::GRAPH_FAILED);
        if (info.queryLayout == MSA_QUERY_LAYOUT_TND) {
            const int64_t expectHeads = poolBlockSizeAttr == MSA_BLOCK16_SIZE ? 1 : info.numQHeads;
            OP_CHECK_IF(
                scOut.GetDim(MSA_DIM_0) != expectHeads || scOut.GetDim(MSA_DIM_1) != static_cast<int64_t>(info.totalQ),
                OP_LOGE(context, "score shape [%ld,%ld] must match [%ld,%u].", scOut.GetDim(MSA_DIM_0),
                        scOut.GetDim(MSA_DIM_1), expectHeads, info.totalQ),
                return ge::GRAPH_FAILED);
        }
        info.maxBlocksPerBatch = static_cast<uint32_t>(scOut.GetDim(MSA_DIM_2));
        OP_CHECK_IF(info.maxBlocksPerBatch == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "score.shape[2]",
                                                          std::to_string(info.maxBlocksPerBatch).c_str(),
                                                          "the score last dim must be positive"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (info.maxBlocksPerBatch % MSA_SCORE_STRIDE_ALIGN) != 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "score.shape[2]", std::to_string(info.maxBlocksPerBatch).c_str(),
                (std::string("the score last dim must be a multiple of ") + std::to_string(MSA_SCORE_STRIDE_ALIGN))
                    .c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_NULL_WITH_CONTEXT(context, scoreOutIr.desc);
        OP_CHECK_IF(!logicalBlock16 && scoreOutIr.desc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context, "fp16 score is reserved and not supported."), return ge::GRAPH_FAILED);
    } else {
        const bool outputsOk = !hasScore && hasIndices &&
                               ((info.returnChoice == MSA_RETURN_INDICES && !hasValues) ||
                                (info.returnChoice == MSA_RETURN_INDICES_VALUES && hasValues));
        if (!outputsOk) {
            OP_LOGE(context, "outputs do not match return_choice=%u.", info.returnChoice);
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& idxOut = indicesShape->GetStorageShape();
        const auto* idxDesc = indicesOutIr.desc;
        OP_CHECK_NULL_WITH_CONTEXT(context, idxDesc);
        const uint32_t outputHeads = logicalBlock16 ? 1U : info.numQHeads;
        const bool idxOk =
            idxOut.GetDimNum() == MSA_QUERY_DIM_NUM && idxDesc->GetDataType() == ge::DT_INT32 &&
            static_cast<ge::Format>(ge::GetPrimaryFormat(idxDesc->GetStorageFormat())) == ge::FORMAT_ND &&
            static_cast<uint32_t>(idxOut.GetDim(MSA_DIM_0)) == outputHeads &&
            static_cast<uint32_t>(idxOut.GetDim(MSA_DIM_1)) == info.totalQ &&
            static_cast<uint32_t>(idxOut.GetDim(MSA_DIM_2)) == info.topk;
        if (!idxOk) {
            const int dtype = (idxDesc == nullptr) ? -1 : static_cast<int>(idxDesc->GetDataType());
            const int fmt =
                (idxDesc == nullptr) ? -1 : static_cast<int>(ge::GetPrimaryFormat(idxDesc->GetStorageFormat()));
            OP_LOGE(context,
                    "sparse_indices must be [N1, T1, topk] int32 ND, got rank=%zu dims=[%ld,%ld,%ld] dtype=%d "
                    "format=%d heads=%u totalQ=%u topk=%u.",
                    idxOut.GetDimNum(), idxOut.GetDimNum() > 0 ? idxOut.GetDim(0) : -1,
                    idxOut.GetDimNum() > 1 ? idxOut.GetDim(1) : -1, idxOut.GetDimNum() > 2 ? idxOut.GetDim(2) : -1,
                    dtype, fmt, outputHeads, info.totalQ, info.topk);
            return ge::GRAPH_FAILED;
        }
        if (info.returnChoice == MSA_RETURN_INDICES_VALUES) {
            const gert::Shape& valOut = valuesShape->GetStorageShape();
            const auto* valDesc = valuesOutIr.desc;
            OP_CHECK_NULL_WITH_CONTEXT(context, valDesc);
            const bool valOk = valDesc->GetDataType() == ge::DT_FLOAT && valOut.GetDimNum() == idxOut.GetDimNum() &&
                               valOut.GetDim(MSA_DIM_0) == idxOut.GetDim(MSA_DIM_0) &&
                               valOut.GetDim(MSA_DIM_1) == idxOut.GetDim(MSA_DIM_1) &&
                               valOut.GetDim(MSA_DIM_2) == idxOut.GetDim(MSA_DIM_2);
            OP_CHECK_IF(!valOk,
                        OP_LOGE(context, "sparse_values must match sparse_indices and be float32. fp16 values are "
                                         "reserved and not supported."),
                        return ge::GRAPH_FAILED);
        }
        info.maxBlocksPerBatch = DeriveAlignedMaxBlocks(context, info);
    }
    // 块长 16：TND 16、BSND/BNSD，以及 BBND/BNBD 页长 16，仅 Ascend 950。
    // 取值白名单与 BSND/BNSD 组合已在 attr 解析处校验，PA 页长已与 attr 交叉校验。
    OP_CHECK_IF(logicalBlock16 && !info.isAscend950,
                OP_LOGE(context, "pool_block_size=%u is only supported on Ascend 950.", MSA_BLOCK16_SIZE),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(logicalBlock16 && (info.numQHeads == 0 || (MSA_ROW_TILE_M % info.numQHeads) != 0),
                OP_LOGE(context, "pool_block_size=%u requires numQHeads(%u) to divide %u.", MSA_BLOCK16_SIZE,
                        info.numQHeads, MSA_ROW_TILE_M),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        info.headDim == 0 || (info.headDim % MSA_SCORE_STRIDE_ALIGN) != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "headDim", std::to_string(info.headDim).c_str(),
            (std::string("the headDim must be a positive multiple of ") + std::to_string(MSA_SCORE_STRIDE_ALIGN))
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.headDim > MSA_K_TILE,
                OP_LOGE(context, "headDim(%u) larger than %u is not supported yet.", info.headDim, MSA_K_TILE),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.numKvHeads == 0 || (info.numQHeads % info.numKvHeads) != 0,
                OP_LOGE(context, "numQHeads(%u) must be divisible by numKvHeads(%u).", info.numQHeads, info.numKvHeads),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.batch == 0, OP_LOGE(context, "batch must be positive."), return ge::GRAPH_FAILED);
    // q_len/kv_len 全 0 跳过计算。InferShape 已把空 KV 的 score 末维钳到 16。
    OP_CHECK_IF(info.maxBlocksPerBatch == 0,
                OP_LOGE(context, "maxBlocksPerBatch must be positive (empty kv uses aligned score stride)."),
                return ge::GRAPH_FAILED);
    // 仅 PA 有 page 池；TND / BSND 直接按 token 寻址，numPages 恒为 0。
    OP_CHECK_IF(
        (info.keyLayout == MSA_KEY_LAYOUT_BBND || info.keyLayout == MSA_KEY_LAYOUT_BNBD) && info.numPages == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "numPages", std::to_string(info.numPages).c_str(),
                                              "PA numPages must be positive"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint32_t>(startLocShape->GetStorageShape().GetDim(MSA_DIM_0)) != info.batch,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "start_loc",
                    std::to_string(startLocShape->GetStorageShape().GetDim(MSA_DIM_0)).c_str(),
                    (std::string("the start_loc size must equal batch=") + std::to_string(info.batch)).c_str()),
                return ge::GRAPH_FAILED);

    const auto* queryDesc = context->GetInputDesc(MSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    info.queryDtype = queryDesc->GetDataType();
    if (info.queryDtype == ge::DT_INT8) {
        if (CheckMsaIndexScoreQuantInputs(context, info) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        info.isMxFp4 = info.isAscend950 && info.queryDtype == ge::DT_FLOAT4_E2M1 && info.keyDtype == ge::DT_FLOAT4_E2M1;
        OP_CHECK_IF(!info.isMxFp4 && !IsNonQuantQueryDtype(info.queryDtype, info.isAscend950),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        context->GetNodeName(), "query", Ops::Base::ToString(info.queryDtype).c_str(),
                        "bf16 or fp16 is supported on A2/A3, and hifloat8, float8_e5m2 or float8_e4m3fn is also "
                        "supported on Ascend 950; int8 query and float4 are reserved"),
                    return ge::GRAPH_FAILED);

        info.isQuant = (info.keyDtype == ge::DT_INT8);
        // 预留int8场景，当前拦截
        OP_CHECK_IF(
            info.queryDtype == ge::DT_FLOAT16 && info.keyDtype == ge::DT_INT8,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "query/key",
                                                  (std::string("query=") + Ops::Base::ToString(info.queryDtype) +
                                                   ", key=" + Ops::Base::ToString(info.keyDtype))
                                                      .c_str(),
                                                  "float16 query with int8 key is not supported yet"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            info.isAscend950 && info.queryDtype == ge::DT_HIFLOAT8 && info.keyDtype == ge::DT_HIFLOAT8,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "query/key",
                                                  (std::string("query=") + Ops::Base::ToString(info.queryDtype) +
                                                   ", key=" + Ops::Base::ToString(info.keyDtype))
                                                      .c_str(),
                                                  "hifloat8 query with hifloat8 key is not supported yet"),
            return ge::GRAPH_FAILED);
        // Block16 keeps upstream FP16/BF16/FP8/MXFP4 support; native A8C8 remains block128.
        OP_CHECK_IF(
            info.poolBlockSize == MSA_BLOCK16_SIZE && !info.isMxFp4 &&
                (info.isQuant || info.queryDtype == ge::DT_HIFLOAT8 || info.keyDtype == ge::DT_HIFLOAT8),
            OP_LOGE(context, "pool_block_size=%u supports fp16 / bf16 / float8_e4m3fn / float8_e5m2 / MXFP4 only.",
                    MSA_BLOCK16_SIZE),
            return ge::GRAPH_FAILED);
        // PA 存储页长 16（M3 短页存储）同样仅 950 上的 fp16 / bf16。
        OP_CHECK_IF(
            info.pageSize == MSA_BLOCK16_SIZE && !info.isMxFp4 &&
                (info.isQuant || info.queryDtype == ge::DT_HIFLOAT8 || info.keyDtype == ge::DT_HIFLOAT8),
            OP_LOGE(context, "BBND/BNBD page size %u supports fp16 / bf16 / float8_e4m3fn / float8_e5m2 / MXFP4 only.",
                    MSA_BLOCK16_SIZE),
            return ge::GRAPH_FAILED);
        if (info.isMxFp4) {
            OP_CHECK_IF(info.numKvHeads != 1, OP_LOGE(context, "MXFP4 x MXFP4 requires numKvHeads == 1."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((info.headDim % 64U) != 0,
                        OP_LOGE(context, "MXFP4 headDim(%u) must be a multiple of 64.", info.headDim),
                        return ge::GRAPH_FAILED);
        } else if (info.isQuant) {
            OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT16,
                        OP_LOGE(context, "ND int8 key currently requires fp16 query."), return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                info.keyDtype != info.queryDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "query, key",
                                                       (std::string("query=") + Ops::Base::ToString(info.queryDtype) +
                                                        ", key=" + Ops::Base::ToString(info.keyDtype))
                                                           .c_str(),
                                                       "the non-quant key dtype must match the query dtype"),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(IsFp8ComputeDtype(info.queryDtype) && !info.isAscend950,
                        OP_LOGE(context, "FP8 query/key is only supported on Ascend 950."), return ge::GRAPH_FAILED);
        }

        const gert::StorageShape* scaleShape = context->GetOptionalInputShape(MSA_IDX_KEY_SCALE);
        const gert::StorageShape* queryScaleShape = context->GetOptionalInputShape(MSA_IDX_QUERY_SCALE);
        if (info.isMxFp4) {
            const uint32_t groups = info.headDim / MSA_MX_GROUP;
            OP_CHECK_IF(!IsOptionalProvided(queryScaleShape) || !IsOptionalProvided(scaleShape),
                        OP_LOGE(context, "MXFP4 x MXFP4 requires query_scale and key_scale (e8m0, group 32)."),
                        return ge::GRAPH_FAILED);
            const auto* queryScaleDesc = context->GetOptionalInputDesc(MSA_IDX_QUERY_SCALE);
            const auto* scaleDesc = context->GetOptionalInputDesc(MSA_IDX_KEY_SCALE);
            OP_CHECK_NULL_WITH_CONTEXT(context, queryScaleDesc);
            OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
            OP_CHECK_IF(
                queryScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0 || scaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                OP_LOGE(context, "MXFP4 query_scale and key_scale must be float8_e8m0."), return ge::GRAPH_FAILED);
            const gert::Shape& qs = queryScaleShape->GetStorageShape();
            const gert::Shape& ks = scaleShape->GetStorageShape();
            bool qOk =
                (qs.GetDimNum() == q.GetDimNum()) && (static_cast<uint32_t>(qs.GetDim(qs.GetDimNum() - 1U)) == groups);
            for (size_t d = 0; qOk && d + 1U < q.GetDimNum(); ++d) {
                qOk = (qs.GetDim(d) == q.GetDim(d));
            }
            OP_CHECK_IF(!qOk, OP_LOGE(context, "query_scale must mirror query layout with last dim D/32."),
                        return ge::GRAPH_FAILED);
            bool kOk =
                (ks.GetDimNum() == kc.GetDimNum()) && (static_cast<uint32_t>(ks.GetDim(ks.GetDimNum() - 1U)) == groups);
            for (size_t d = 0; kOk && d + 1U < kc.GetDimNum(); ++d) {
                kOk = (ks.GetDim(d) == kc.GetDim(d));
            }
            OP_CHECK_IF(!kOk, OP_LOGE(context, "MXFP4 key_scale must mirror key layout with last dim D/32."),
                        return ge::GRAPH_FAILED);
        } else if (info.isQuant) {
            OP_CHECK_IF(!IsOptionalProvided(scaleShape),
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "key_scale",
                                                                 "an int8 key requires the dequant scale input"),
                        return ge::GRAPH_FAILED);
            const gert::Shape& sc = scaleShape->GetStorageShape();
            if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
                const bool ok2d = (sc.GetDimNum() == MSA_SCALE_TND_DIM_NUM &&
                                   static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) == info.totalK &&
                                   static_cast<uint32_t>(sc.GetDim(MSA_DIM_1)) == info.numKvHeads);
                const bool ok1d = (sc.GetDimNum() == MSA_RANK_1D && info.numKvHeads == 1 &&
                                   static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) == info.totalK);
                OP_CHECK_IF(
                    !ok2d && !ok1d,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "key_scale", Ops::Base::ToString(sc).c_str(),
                        (std::string("the TND dequant scale must be [") + std::to_string(info.totalK) + ", " +
                         std::to_string(info.numKvHeads) + "] or [" + std::to_string(info.totalK) + "] when N2=1")
                            .c_str()),
                    return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(sc.GetDimNum() != MSA_SCALE_PA_DIM_NUM,
                            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "key_scale",
                                                                     (std::to_string(sc.GetDimNum()) + "D").c_str(),
                                                                     "PA dequant scale must have 3 dims [NP, N_kv, P]"),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) != info.numPages ||
                                static_cast<uint32_t>(sc.GetDim(MSA_DIM_1)) != info.numKvHeads ||
                                static_cast<uint32_t>(sc.GetDim(MSA_DIM_2)) != info.pageSize,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                context->GetNodeName(), "key_scale", Ops::Base::ToString(sc).c_str(),
                                (std::string("the dequant scale shape must be [") + std::to_string(info.numPages) +
                                 ", " + std::to_string(info.numKvHeads) + ", " + std::to_string(info.pageSize) + "]")
                                    .c_str()),
                            return ge::GRAPH_FAILED);
            }
            const auto* scaleDesc = context->GetOptionalInputDesc(MSA_IDX_KEY_SCALE);
            if (scaleDesc == nullptr) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "key_scale",
                                                         "the dequant scale must be provided for an int8 key");
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(scaleDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "key_scale",
                                                              Ops::Base::ToString(scaleDesc->GetDataType()).c_str(),
                                                              "the dequant scale dtype must be float32"),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(IsOptionalProvided(scaleShape),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "key_scale", "provided",
                                                              "the non-quant path must not pass key_scale"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(!info.isMxFp4 && IsOptionalProvided(queryScaleShape),
                    OP_LOGE(context, "query_scale is only supported for MXFP4."), return ge::GRAPH_FAILED);
    }

    const int64_t* sparseModeAttr = attrs->GetInt(MSA_ATTR_SPARSE_MODE);
    info.sparseMode = (sparseModeAttr == nullptr) ? MSA_SPARSE_MODE_RIGHT_DOWN : static_cast<uint32_t>(*sparseModeAttr);
    OP_CHECK_IF(info.sparseMode != MSA_SPARSE_MODE_DEFAULT && info.sparseMode != MSA_SPARSE_MODE_RIGHT_DOWN,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "sparse_mode", std::to_string(info.sparseMode).c_str(),
                    (std::string("the supported values are ") + std::to_string(MSA_SPARSE_MODE_DEFAULT) + " and " +
                     std::to_string(MSA_SPARSE_MODE_RIGHT_DOWN))
                        .c_str()),
                return ge::GRAPH_FAILED);

    const gert::StorageShape* attenMaskShape = context->GetOptionalInputShape(MSA_IDX_ATTEN_MASK);
    if (info.sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN) {
        OP_CHECK_IF(!IsOptionalProvided(attenMaskShape),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "atten_mask",
                        (std::string("sparse_mode=") + std::to_string(MSA_SPARSE_MODE_RIGHT_DOWN) +
                         " requires atten_mask of shape [" + std::to_string(MSA_ATTEN_MASK_SIZE) + ", " +
                         std::to_string(MSA_ATTEN_MASK_SIZE) + "]")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        const gert::Shape& am = attenMaskShape->GetStorageShape();
        OP_CHECK_IF(am.GetDimNum() != MSA_ATTEN_MASK_DIM_NUM ||
                        static_cast<uint32_t>(am.GetDim(MSA_DIM_0)) != MSA_ATTEN_MASK_SIZE ||
                        static_cast<uint32_t>(am.GetDim(MSA_DIM_1)) != MSA_ATTEN_MASK_SIZE,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "atten_mask", Ops::Base::ToString(am).c_str(),
                        (std::string("atten_mask must be [") + std::to_string(MSA_ATTEN_MASK_SIZE) + ", " +
                         std::to_string(MSA_ATTEN_MASK_SIZE) + "]")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(IsOptionalProvided(attenMaskShape),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "sparse_mode", std::to_string(MSA_SPARSE_MODE_DEFAULT).c_str(),
                        "the atten_mask input must not be passed for this sparse_mode"),
                    return ge::GRAPH_FAILED);
    }

    const int64_t* initBlocksAttr = attrs->GetInt(MSA_ATTR_INIT_BLOCKS);
    const int64_t* localBlocksAttr = attrs->GetInt(MSA_ATTR_LOCAL_BLOCKS);
    info.initBlocks = (initBlocksAttr == nullptr) ? MSA_DEFAULT_INIT_BLOCKS : static_cast<uint32_t>(*initBlocksAttr);
    info.localBlocks =
        (localBlocksAttr == nullptr) ? MSA_DEFAULT_LOCAL_BLOCKS : static_cast<uint32_t>(*localBlocksAttr);
    const uint32_t logicalBlocks = LogicalBlockLimit(info);
    OP_CHECK_IF(info.initBlocks > logicalBlocks || info.localBlocks > logicalBlocks,
                OP_LOGE(context, "init_blocks/local_blocks must be <= logical blocks(%u).", logicalBlocks),
                return ge::GRAPH_FAILED);
    // 参考 select_blocks：可见块超过 topk 时，前缀加局部必须放进 128，否则报错。
    if (logicalBlock16 && info.returnChoice != MSA_RETURN_SCORE &&
        info.initBlocks + info.localBlocks > MSA_BLOCK16_TOPK) {
        const uint32_t vis = EstMaxVisibleBlocks(context, info);
        OP_CHECK_IF(
            vis > MSA_BLOCK16_TOPK,
            OP_LOGE(context,
                    "init_blocks(%u)+local_blocks(%u) exceed topk %u while some token can see more than %u blocks.",
                    info.initBlocks, info.localBlocks, MSA_BLOCK16_TOPK, MSA_BLOCK16_TOPK),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DoTiling(gert::TilingContext* context, const MsaIndexScoreInfo& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(info.platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "aicNum", "0",
                                                      "the platform must report at least one AI core"),
                return ge::GRAPH_FAILED);

    const bool logicalBlock16 = info.poolBlockSize == MSA_BLOCK16_SIZE;
    const bool block16Mx = logicalBlock16 && info.isMxFp4;
    const bool fixedScale = info.isQueryQuant && !info.isAscend950;
    if (info.isQueryQuant) {
        uint64_t ubBytes = 0, l1Bytes = 0, l0aBytes = 0, l0bBytes = 0, l0cBytes = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Bytes);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aBytes);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bBytes);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cBytes);
        const uint64_t requiredUbBytes = fixedScale ? MSA_FIXED_SCALE_UB_BYTES : 191488U;
        OP_CHECK_IF(ubBytes < requiredUbBytes || l1Bytes < 65536U || l0aBytes < 65536U || l0bBytes < 65536U ||
                        l0cBytes < 131072U,
                    OP_LOGE(context, "insufficient platform memory for native A8C8 fixed tile."),
                    return ge::GRAPH_FAILED);
    }
    const uint32_t paddedHeadDim = info.isQueryQuant ? RoundUpU32(info.headDim, 32U) : info.headDim;
    const uint32_t scoreBlockStride = RoundUpU32(info.maxBlocksPerBatch, MSA_SCORE_STRIDE_ALIGN);
    // 950 C_to_UB：S 不进 GM；int8/TND 尾页仍要一页 K scratch。A2 与 950 回退（MSA_A5_USE_C2UB=0）走 8-stage S。
    const uint32_t sElemBytes = (logicalBlock16 || (info.isQuant && !fixedScale)) ? sizeof(float) : sizeof(uint16_t);
    const bool c2ub = info.isAscend950 && (MSA_A5_USE_C2UB != 0);
    const uint64_t sWsBytes =
        c2ub ? 0U : (static_cast<uint64_t>(aicNum) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM * sElemBytes);
    // BSND 的尾窗可能越过 key 末尾（仅最后一个 batch），与 TND 尾页一样走 scratch 补零。
    const bool paBlock16 = (info.keyLayout == MSA_KEY_LAYOUT_BBND || info.keyLayout == MSA_KEY_LAYOUT_BNBD) &&
                           info.poolBlockSize == MSA_BLOCK16_SIZE;
    // 非 MX 块长 16 的 8 个 page 不连续，先收到 scratch 再做 128 列 GEMM。
    const bool useKScratch =
        block16Mx || (!info.isMxFp4 && (info.isQuant || MsaIsSeqKeyLayout(info.keyLayout) || paBlock16));
    uint32_t scratchElemBytes = info.isQueryQuant ? sizeof(int8_t) : sizeof(uint16_t);
    if (info.isMxFp4 || (info.isAscend950 && !info.isQuant && IsFp8ComputeDtype(info.queryDtype))) {
        scratchElemBytes = 1U;
    }
    uint32_t kScratchElems = c2ub ? MSA_A5_K_SCRATCH_ELEM_NUM : MSA_K_SCRATCH_ELEM_NUM;
    // A2/A3 int8 四槽 K scratch；950 C2UB / 非量化 TND 尾页保持单槽。
    // 块长 16 的 C2UB 用两槽：Cube 读当前窗时 Vector 预取下一窗。
    if (!info.isAscend950 && info.isQuant) {
        kScratchElems *= MSA_K_SCRATCH_STAGES_A2;
    }
    const uint64_t kScratchBytes =
        useKScratch ? (static_cast<uint64_t>(aicNum) * kScratchElems * scratchElemBytes) : 0U;
    const uint32_t qScratchBytesPerCore =
        info.isQueryQuant && paddedHeadDim != info.headDim ? MSA_ROW_TILE_M * paddedHeadDim : 0U;
    // Native Q padding and legacy layout gathering use disjoint, explicitly sized regions.
    const uint64_t qScratchBytes =
        info.isQueryQuant ? static_cast<uint64_t>(aicNum) * qScratchBytesPerCore :
                            (((!info.isMxFp4 || block16Mx) && info.queryLayout != MSA_QUERY_LAYOUT_TND) ?
                                 static_cast<uint64_t>(aicNum) * MSA_ROW_TILE_M * info.headDim * scratchElemBytes :
                                 0U);
    const uint64_t qScratchOffsetBytes = sWsBytes + kScratchBytes;
    const uint64_t kScratchOffsetWords = sWsBytes / sizeof(float);
    const uint64_t qScratchElemOffset = qScratchBytes == 0U ? 0U : qScratchOffsetBytes / scratchElemBytes;
    const uint64_t userWorkspaceBytes = qScratchOffsetBytes + qScratchBytes;
    const uint64_t libWorkspaceBytes = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(kScratchOffsetWords > std::numeric_limits<uint32_t>::max() ||
                    qScratchElemOffset > std::numeric_limits<uint32_t>::max() ||
                    userWorkspaceBytes > std::numeric_limits<size_t>::max() - libWorkspaceBytes,
                OP_LOGE(context, "workspace offsets exceed supported range."), return ge::GRAPH_FAILED);
    const uint32_t kScratchOffsetElems = static_cast<uint32_t>(kScratchOffsetWords);
    const uint32_t qScratchElemOff = static_cast<uint32_t>(qScratchElemOffset);

    uint32_t launchAic = aicNum;
    uint32_t kvChunks = 1U;
    uint32_t chunkBudget = 0U;
    if (info.totalQ == 0U) {
        launchAic = 1U;
    } else {
        const uint32_t packedM = EstPackedMTasks(info, aicNum);
        const uint32_t visBlocks = EstMaxVisibleBlocks(context, info);
        if (logicalBlock16) {
            kvChunks = EstBlock16KvChunks(info, aicNum, packedM, visBlocks);
        } else {
            kvChunks = info.isMxFp4 ? EstMxFp4KvChunks(info, aicNum, packedM, visBlocks) :
                                      EstKvChunks(info, aicNum, packedM, visBlocks);
        }
        launchAic = EstLaunchAic(aicNum, packedM, kvChunks);
        // Use the shape-level average to identify small batched requests.
        // Device-only per-request lengths can still differ within that bound.
        const uint64_t packedRows = static_cast<uint64_t>(info.totalQ) * info.numQHeads;
        if (info.isQueryQuant && !info.isAscend950 && info.batch > 1U && aicNum > 0U && chunkBudget == 0U &&
            packedRows <= static_cast<uint64_t>(info.batch) * 32U) {
            const uint32_t nStiles = (visBlocks + MSA_BLOCKS_PER_STILE - 1U) / MSA_BLOCKS_PER_STILE;
            kvChunks = EstStartupAwareKvChunks(aicNum, info.batch, nStiles);
            launchAic = EstLaunchAic(aicNum, packedM, kvChunks);
        }
    }

    const bool fixedPagedScore = fixedScale && info.numKvHeads == 1U && info.queryLayout == MSA_QUERY_LAYOUT_TND &&
                                 info.keyLayout == MSA_KEY_LAYOUT_BBND && info.pageSize == MSA_BLOCK_SIZE &&
                                 info.poolBlockSize == MSA_BLOCK_SIZE &&
                                 info.sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN && info.initBlocks == 0U &&
                                 info.localBlocks == 0U;
    const bool pagedA8c8Score = fixedPagedScore && info.headDim == MSA_K_TILE;
    constexpr uint32_t shortMaxPages = 64U;
    constexpr uint32_t wideMinPages = 512U;
    const bool singleQuery =
        pagedA8c8Score && info.returnChoice == MSA_RETURN_SCORE && info.batch == 1U && info.totalQ == 1U;
    const bool singleQueryD64 = fixedPagedScore && info.returnChoice == MSA_RETURN_SCORE && info.headDim == 64U &&
                                info.batch == 1U && info.totalQ == 1U;
    const bool mixOneDecode =
        (singleQuery && (info.numQHeads == 1U || info.numQHeads == 2U) && info.tableWidth > shortMaxPages) ||
        ((singleQuery || singleQueryD64) && info.numQHeads == 4U && info.tableWidth >= wideMinPages);
    const uint64_t packedRows = static_cast<uint64_t>(info.totalQ) * info.numQHeads;
    // Every request fits the single AIV's Qscale and score buffers because
    // the total TND packed rows are an upper bound for each ragged request.
    const uint32_t mixOneRowLimit = info.headDim == 96U ? 80U : MSA_ROW_TILE_M / MSA_AIV_PER_AIC;
    const bool mixOneBatch = fixedPagedScore && info.returnChoice == MSA_RETURN_SCORE && info.batch > 1U &&
                             info.tableWidth > shortMaxPages &&
                             (info.headDim == 64U || info.headDim == 96U || info.headDim == MSA_K_TILE) &&
                             packedRows > 0U && packedRows <= mixOneRowLimit;

    constexpr uint32_t shortChunks = 2U;
    // Each TND request contributes at most M-1 rows to the sum of rounded M tiles.
    const uint64_t taskUpper =
        (packedRows + static_cast<uint64_t>(info.batch) * (MSA_ROW_TILE_M - 1U)) / MSA_ROW_TILE_M * shortChunks;
    const bool shortC2 = pagedA8c8Score && info.batch > 0U && info.tableWidth > MSA_BLOCKS_PER_STILE &&
                         info.tableWidth <= shortMaxPages && launchAic == aicNum && aicNum == 24U &&
                         chunkBudget == 0U && packedRows > static_cast<uint64_t>(info.batch) * MSA_ROW_TILE_M &&
                         taskUpper < aicNum;
    if (shortC2) {
        kvChunks = shortChunks;
        launchAic = static_cast<uint32_t>(taskUpper);
    }

    // These page capacities bound the two-wave S-tile workload, and
    // retain the tested capacity of the physical key allocation.
    constexpr uint32_t ownerSmallPages = 193U;
    constexpr uint32_t ownerLargePages = 256U;
    constexpr uint32_t ownerTailPages = 257U;
    const bool reduceOwners = singleQuery && info.numQHeads == 4U && info.numPages == info.tableWidth &&
                              (info.tableWidth == ownerSmallPages || info.tableWidth == ownerLargePages ||
                               info.tableWidth == ownerTailPages) &&
                              aicNum == 24U && launchAic == aicNum && chunkBudget == 0U;
    if (reduceOwners) {
        const uint32_t stiles = RoundUpU32(info.tableWidth, MSA_BLOCKS_PER_STILE) / MSA_BLOCKS_PER_STILE;
        constexpr uint32_t stilesPerOwner = 2U;
        const uint32_t owners = RoundUpU32(stiles, stilesPerOwner) / stilesPerOwner;
        kvChunks = owners;
        launchAic = owners;
    }
    const bool lowDimOwners = singleQueryD64 && info.numQHeads == 4U && info.tableWidth == 256U &&
                              info.numPages == info.tableWidth && aicNum == 24U && launchAic == aicNum &&
                              chunkBudget == 0U;
    if (lowDimOwners) {
        kvChunks = 16U;
        launchAic = 16U;
    }
    const bool mixOne = mixOneDecode || mixOneBatch || reduceOwners || lowDimOwners;

    MsaIndexScoreTilingData tilingData;
    tilingData.set_batch(info.batch);
    tilingData.set_totalQ(info.totalQ);
    tilingData.set_numQHeads(info.numQHeads);
    tilingData.set_numKvHeads(info.numKvHeads);
    // Average packed rows is only a cost gate. Each actual request is checked on AIC.
    const bool packetCandidate = fixedPagedScore && !mixOne && info.batch > 1U && info.tableWidth > shortMaxPages &&
                                 (info.headDim == 64U || info.headDim == 96U || info.headDim == MSA_K_TILE) &&
                                 packedRows >= static_cast<uint64_t>(info.batch) * MSA_PACKET_MIN_M &&
                                 packedRows <= static_cast<uint64_t>(info.batch) * MSA_PACKET_MAX_M;
    const uint32_t gqaGroup = info.numQHeads / info.numKvHeads;
    tilingData.set_gqaGroup(gqaGroup | (packetCandidate ? MSA_PACKET_CANDIDATE : 0U));
    tilingData.set_headDim(info.headDim);
    tilingData.set_poolBlockSize(info.poolBlockSize);
    tilingData.set_pageSize(info.pageSize);
    tilingData.set_tableWidth(info.tableWidth);
    tilingData.set_maxBlocksPerBatch(info.maxBlocksPerBatch);
    tilingData.set_scoreBlockStride(scoreBlockStride);
    tilingData.set_usedCoreNum(launchAic);
    tilingData.set_isQuant(info.isQuant ? 1U : 0U);
    tilingData.set_sparseMode(info.sparseMode);
    tilingData.set_initBlocks(info.initBlocks);
    tilingData.set_localBlocks(info.localBlocks);
    tilingData.set_numPages(info.numPages);
    tilingData.set_keyLayout(info.keyLayout);
    tilingData.set_totalK(info.totalK);
    tilingData.set_kvChunks(kvChunks);
    tilingData.set_chunkBudget(chunkBudget);
    tilingData.set_compressFactor(info.compressFactor);
    tilingData.set_rowsPerBlock(info.rowsPerBlock);
    tilingData.set_totalTasks(logicalBlock16 ? EstExactTotalTasks(context, info, kvChunks) : 0U);
    tilingData.set_mxFp4WideN(block16Mx ? 0U : EstMxFp4WideN(context, info));
    tilingData.set_queryLayout(info.queryLayout);
    tilingData.set_qMaxSeq(info.qMaxSeq);
    tilingData.set_kMaxSeq(info.kMaxSeq);
    tilingData.set_qScratchElems(qScratchElemOff);
    tilingData.set_paddedHeadDim(paddedHeadDim);
    tilingData.set_kScratchOffsetBytes(sWsBytes);
    tilingData.set_qScratchOffsetBytes(qScratchOffsetBytes);
    tilingData.set_qScratchBytesPerCore(qScratchBytesPerCore);

    tilingData.set_strideQt(info.numQHeads * info.headDim);
    tilingData.set_strideQn(info.headDim);
    uint32_t defaultStrideKvBlock = 0;
    if (MsaIsSeqKeyLayout(info.keyLayout)) {
        defaultStrideKvBlock = 0;
        // BNSD 的 head0 沿 S 连续，token 步长是 D；BSND/TND 的 head0 步长是 N2*D。
        const uint32_t seqTokenStride =
            (info.keyLayout == MSA_KEY_LAYOUT_BNSD) ? info.headDim : (info.numKvHeads * info.headDim);
        tilingData.set_strideKvToken(seqTokenStride);
        tilingData.set_strideScalePage(info.numKvHeads);
        tilingData.set_strideScaleHead(1);
    } else if (info.keyLayout == MSA_KEY_LAYOUT_BNBD) {
        // PA 的页内步长按存储页长（key 的页维度）算，与打分块长解耦。
        defaultStrideKvBlock = info.numKvHeads * info.pageSize * info.headDim;
        tilingData.set_strideKvToken(info.headDim);
        tilingData.set_strideScalePage(info.numKvHeads * info.pageSize);
        tilingData.set_strideScaleHead(info.pageSize);
    } else {
        defaultStrideKvBlock = info.pageSize * info.numKvHeads * info.headDim;
        tilingData.set_strideKvToken(info.numKvHeads * info.headDim);
        tilingData.set_strideScalePage(info.numKvHeads * info.pageSize);
        tilingData.set_strideScaleHead(info.pageSize);
    }
    uint32_t strideKvBlock = defaultStrideKvBlock;
    if (ResolveStrideKvBlock(context, info, defaultStrideKvBlock, strideKvBlock) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tilingData.set_strideKvBlock(strideKvBlock);
    const uint32_t outStride =
        logicalBlock16 && info.returnChoice != MSA_RETURN_SCORE ? MSA_BLOCK16_TOPK : scoreBlockStride;
    tilingData.set_strideOutHead(info.totalQ * outStride);
    tilingData.set_strideOutToken(outStride);
    tilingData.set_kScratchOffsetElems(kScratchOffsetElems);

    size_t userBytes = static_cast<size_t>(userWorkspaceBytes);
    if (info.isAscend950 && sWsBytes == 0U && kScratchBytes == 0U) {
        userBytes += 64U;
    }
    tilingData.set_mxScaleScratchOffset(block16Mx ? userBytes : 0U);
    if (block16Mx) {
        userBytes += MSA_BLOCK16_MX_SCALE_SCRATCH_BYTES;
    }
    uint32_t scoreWorkspaceElems = 0;
    size_t scoreBytes = 0;
    if (info.returnChoice != MSA_RETURN_SCORE) {
        constexpr size_t align = 256U;
        const size_t maxSize = std::numeric_limits<size_t>::max();
        OP_CHECK_IF(userBytes > maxSize - (align - 1U), OP_LOGE(context, "TopK workspace alignment overflows."),
                    return ge::GRAPH_FAILED);
        const size_t aligned = (userBytes + align - 1U) & ~(align - 1U);
        OP_CHECK_IF(aligned / sizeof(float) > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(context, "TopK workspace offset exceeds uint32 range."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(scoreBlockStride != 0U && info.totalQ > std::numeric_limits<uint32_t>::max() / scoreBlockStride,
                    OP_LOGE(context, "TopK score head stride exceeds uint32 range."), return ge::GRAPH_FAILED);
        const size_t headElems = static_cast<size_t>(info.totalQ) * scoreBlockStride;
        const uint32_t scoreHeads = logicalBlock16 ? 1U : info.numQHeads;
        OP_CHECK_IF(scoreHeads != 0U && headElems > (maxSize - aligned) / sizeof(float) / scoreHeads,
                    OP_LOGE(context, "TopK score workspace size overflows."), return ge::GRAPH_FAILED);
        scoreWorkspaceElems = static_cast<uint32_t>(aligned / sizeof(float));
        scoreBytes = headElems * scoreHeads * sizeof(float);
        userBytes = aligned + scoreBytes;
    }
    tilingData.set_returnChoice(info.returnChoice);
    tilingData.set_topk(info.topk);
    tilingData.set_scoreWorkspaceElems(scoreWorkspaceElems);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    // MIX 1AIC:2AIV：CalcTschBlockDim 的 sliceNum 按 AIV 计数，内部再 / (aiv/aic)。
    // 传入 aicNum 会再除一次得到 blockDim=aic/2，只能打一半 Cube。
    // 整 batch q_len=0：queryS==0 → BlockDim=1，避免 totalTaskNum=0。
    // sliceNum = launchAic * 2：短 decode 只起实际 M-task（及 950 的 S-chunk）对应的 MIX；
    // 多 M-tile / 大 batch 仍打满 AIC。
    if (info.totalQ == 0U) {
        context->SetBlockDim(1);
    } else {
        uint32_t sliceAiv = launchAic * MSA_AIV_PER_AIC;
        if (sliceAiv > aivNum) {
            sliceAiv = aivNum;
        }
        context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(sliceAiv, aicNum, aivNum));
    }

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    OP_CHECK_IF(userBytes > std::numeric_limits<size_t>::max() - libWorkspaceBytes,
                OP_LOGE(context, "Workspace size overflows."), return ge::GRAPH_FAILED);
    workspaces[0] = libWorkspaceBytes + userBytes;

    uint64_t tilingKey = MSA_TILING_KEY_FP16;
    if (info.isMxFp4) {
        // block16 使用独立 key，选择对应的编译期实例。
        if (info.poolBlockSize == MSA_BLOCK16_SIZE) {
            tilingKey = MSA_TILING_KEY_MXFP4_BLK16;
        } else if (info.queryLayout == MSA_QUERY_LAYOUT_TND && info.keyLayout == MSA_KEY_LAYOUT_TND &&
                   info.totalK > 0U && info.totalK <= MSA_BLOCK_SIZE && info.totalQ > 0U &&
                   static_cast<uint64_t>(info.totalQ) * info.numQHeads < 64U) {
            // Static storage bounds prove every request fits one page; never read device length tensors here.
            tilingKey = MSA_TILING_KEY_MXFP4_LATENCY;
        } else if (UseMxFp4FusedStage(context, info)) {
            tilingKey = MSA_TILING_KEY_MXFP4_PREFILL;
        } else {
            tilingKey = MSA_TILING_KEY_MXFP4;
        }
    } else if (info.poolBlockSize == MSA_BLOCK16_SIZE) {
        // 块长 16（TND 16 / BSND/BNSD / 页长 16，两级 maxpool）独占 TilingKey，128 路径保持原样。
        if (info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
            tilingKey = MSA_TILING_KEY_FP8_E4M3FN_BLK16;
        } else if (info.queryDtype == ge::DT_FLOAT8_E5M2) {
            tilingKey = MSA_TILING_KEY_FP8_E5M2_BLK16;
        } else {
            tilingKey = (info.queryDtype == ge::DT_BF16) ? MSA_TILING_KEY_BF16_BLK16 : MSA_TILING_KEY_FP16_BLK16;
        }
        // Keep legacy page128/sequence kernels free of compressed-page checks.
        if (MsaIsPaKeyLayout(info.keyLayout) && info.compressFactor > 1U && info.pageSize < MSA_BLOCK_SIZE &&
            info.pageSize == MSA_BLOCK_SIZE / info.compressFactor) {
            if (info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
                tilingKey = MSA_TILING_KEY_FP8_E4M3FN_COMPRESSED_PAGE;
            } else if (info.queryDtype == ge::DT_FLOAT8_E5M2) {
                tilingKey = MSA_TILING_KEY_FP8_E5M2_COMPRESSED_PAGE;
            } else {
                tilingKey = info.queryDtype == ge::DT_BF16 ? MSA_TILING_KEY_BF16_COMPRESSED_PAGE :
                                                             MSA_TILING_KEY_FP16_COMPRESSED_PAGE;
            }
            if (info.compressFactor == 4U && info.numQHeads == 16U && info.numKvHeads == 1U && info.headDim == 128U &&
                info.queryLayout == MSA_QUERY_LAYOUT_TND) {
                const bool balanced = UseBalancedFp8Schedule(info.totalQ, scoreBlockStride, launchAic);
                // q8 decode: streaming K bypasses allocation on the legacy
                // schedule or beyond L2 capacity. Preserve q1/short-K/prefill
                // reuse and use literal cache hints in independent instances.
                const uint64_t keyBytes = static_cast<uint64_t>(info.batch) * scoreBlockStride * 4U * 128U;
                const bool bypass = info.totalQ == static_cast<uint64_t>(info.batch) * 8U &&
                                    scoreBlockStride >= 2048U && keyBytes >= 96ULL * 1024U * 1024U &&
                                    (!balanced || keyBytes >= 128ULL * 1024U * 1024U);
                if (info.queryDtype == ge::DT_FLOAT8_E4M3FN || info.queryDtype == ge::DT_FLOAT8_E5M2) {
                    // Key layout: dtype + fused(2) + balanced(4) + bypass(8).
                    tilingKey = (info.queryDtype == ge::DT_FLOAT8_E4M3FN ? MSA_TILING_KEY_FP8_E4M3FN_PIPELINED :
                                                                           MSA_TILING_KEY_FP8_E5M2_PIPELINED) +
                                (info.returnChoice == MSA_RETURN_SCORE ? 0U : 2U) + (balanced ? 4U : 0U) +
                                (bypass ? 8U : 0U);
                }
            }
        }
        // Keep established full-window binaries unchanged. The new N128 loader
        // serves compressed PA pages and shape-visible sequential tails.
        if ((info.queryDtype == ge::DT_BF16 || info.queryDtype == ge::DT_FLOAT16) && info.headDim <= MSA_K_TILE) {
            const bool smallPage = MsaIsPaKeyLayout(info.keyLayout) && info.pageSize < MSA_BLOCK_SIZE;
            const uint32_t seqRows = info.keyLayout == MSA_KEY_LAYOUT_TND ? info.totalK : info.kMaxSeq;
            const bool tail = MsaIsSeqKeyLayout(info.keyLayout) && seqRows % MSA_BLOCK_SIZE != 0U;
            const bool fastBnsd = info.numQHeads == 16U && info.headDim == 128U && info.qMaxSeq == 8U;
            const bool qGather = info.queryLayout == MSA_QUERY_LAYOUT_BNSD && !fastBnsd;
            if (smallPage || tail || qGather) {
                tilingKey = info.queryDtype == ge::DT_BF16 ? MSA_TILING_KEY_BF16_DIRECT_FLOAT :
                                                             MSA_TILING_KEY_FP16_DIRECT_FLOAT;
            }
        }
    } else if (info.isQueryQuant) {
        const bool paired = info.headDim == MSA_K_TILE;
        const uint64_t packedRows = static_cast<uint64_t>(info.totalQ) * info.numQHeads;
        const bool groupMask = info.batch == 1U && info.totalQ > 1U && info.sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN &&
                               packedRows >= MSA_ROW_TILE_M;
        // Shape-only coarse gate: never infer prefill from Hq (decode may have
        // many heads). Device additionally checks each request's true qLen and
        // mSub before enabling the 64-column window, including mixed batches.
        const bool wideScore =
            !info.isAscend950 && info.isQuant && info.queryLayout == MSA_QUERY_LAYOUT_TND &&
            (info.keyLayout == MSA_KEY_LAYOUT_TND || info.keyLayout == MSA_KEY_LAYOUT_BBND) &&
            info.pageSize == MSA_BLOCK_SIZE && info.sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN &&
            (info.headDim == 64U || info.headDim == 96U) && info.batch > 1U && scoreBlockStride > 64U &&
            static_cast<uint64_t>(info.totalQ) >= static_cast<uint64_t>(info.batch) * MSA_WIDE_SCORE_MIN_Q_TOKENS;
        if (wideScore) {
            tilingKey = MSA_TILING_KEY_INT8_INT8_FIXED_WIDE_SCORE;
        } else if (groupMask) {
            tilingKey =
                paired ? MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED_GROUP_MASK : MSA_TILING_KEY_INT8_INT8_FIXED_GROUP_MASK;
        } else {
            tilingKey = paired ? MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED : MSA_TILING_KEY_INT8_INT8_FIXED;
        }
    } else if (info.isQuant) {
        tilingKey = MSA_TILING_KEY_FP16_INT8;
    } else if (info.queryDtype == ge::DT_HIFLOAT8) {
        tilingKey = MSA_TILING_KEY_HIFLOAT8;
    } else if (info.queryDtype == ge::DT_FLOAT8_E5M2) {
        tilingKey = MSA_TILING_KEY_FP8_E5M2;
    } else if (info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
        tilingKey = MSA_TILING_KEY_FP8_E4M3FN;
    } else {
        tilingKey = (info.queryDtype == ge::DT_BF16) ? MSA_TILING_KEY_BF16 : MSA_TILING_KEY_FP16;
    }
    if (!logicalBlock16 && info.returnChoice != MSA_RETURN_SCORE && !info.isQueryQuant) {
        if (info.isMxFp4) {
            tilingKey = MSA_TILING_KEY_MXFP4_TOPK;
        } else if (!info.isAscend950) {
            // A2/A3 没有 fp8，也没有 L0C→UB。只发 fp16/bf16 的独立 key，950 的 9/10 保持不变。
            if (info.isQuant || (info.queryDtype != ge::DT_BF16 && info.queryDtype != ge::DT_FLOAT16)) {
                OP_LOGE(context, "return_choice=%u on A2/A3 supports fp16 and bf16 only.", info.returnChoice);
                return ge::GRAPH_FAILED;
            }
            tilingKey = (info.queryDtype == ge::DT_BF16) ? MSA_TILING_KEY_BF16_TOPK_A2 : MSA_TILING_KEY_FP16_TOPK_A2;
        } else if (info.queryDtype == ge::DT_BF16) {
            tilingKey = MSA_TILING_KEY_BF16_TOPK;
        } else if (info.queryDtype == ge::DT_FLOAT16) {
            tilingKey = MSA_TILING_KEY_FP16_TOPK;
        } else if (info.queryDtype == ge::DT_FLOAT8_E5M2) {
            tilingKey = MSA_TILING_KEY_FP8_E5M2_TOPK;
        } else if (info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
            tilingKey = MSA_TILING_KEY_FP8_E4M3FN_TOPK;
        } else {
            OP_LOGE(context, "return_choice=%u does not support this dtype.", info.returnChoice);
            return ge::GRAPH_FAILED;
        }
    }
    const bool batchedPrefill =
        pagedA8c8Score && info.batch > 1U &&
        static_cast<uint64_t>(info.totalQ) >= static_cast<uint64_t>(info.batch) * MSA_WIDE_SCORE_MIN_Q_TOKENS;
    if (mixOne) {
        tilingKey = MSA_TILING_KEY_INT8_INT8_FIXED_MIX_ONE;
    } else if (shortC2 || (tilingKey == MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED && batchedPrefill)) {
        tilingKey = MSA_TILING_KEY_INT8_INT8_FIXED_PAIRED_GROUP_MASK;
    }
    if (info.isQueryQuant && info.returnChoice != MSA_RETURN_SCORE) {
        tilingKey += MSA_TILING_KEY_INT8_INT8_FIXED_TOPK - MSA_TILING_KEY_INT8_INT8_FIXED;
    }
    context->SetTilingKey(tilingKey);
    OP_LOGI(context->GetNodeName(),
            "MsaIndexScore tilingKey=%lu headDim=%u layout=%u strideKvBlock=%u launchAic=%u kvChunks=%u "
            "aicNum=%u",
            tilingKey, info.headDim, info.keyLayout, strideKvBlock, launchAic, kvChunks, aicNum);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus CheckMsaIndexScoreQuantInputs(gert::TilingContext* context, MsaIndexScoreInfo& info)
{
    OP_CHECK_IF(!IsNonQuantQueryDtype(info.queryDtype, info.isAscend950) && info.queryDtype != ge::DT_INT8,
                OP_LOGE(context, "query dtype must be bf16, fp16 or int8 on A2/A3; Ascend 950 also allows hifloat8 / "
                                 "float8_e5m2 / float8_e4m3fn."),
                return ge::GRAPH_FAILED);

    info.isQuant = (info.keyDtype == ge::DT_INT8);
    info.isQueryQuant = (info.queryDtype == ge::DT_INT8);
    OP_CHECK_IF(info.isQuant && !info.isQueryQuant,
                OP_LOGE(context, "int8 key requires native int8 query; FP16 query with INT8 key is reserved."),
                return ge::GRAPH_FAILED);
    const auto* queryScaleShape = context->GetOptionalInputShape(MSA_IDX_QUERY_SCALE);
    if (info.isQueryQuant) {
        OP_CHECK_IF(info.isAscend950 || !info.isQuant, OP_LOGE(context, "int8 query requires int8 key on A2/A3."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            info.queryLayout != MSA_QUERY_LAYOUT_TND || info.poolBlockSize != MSA_BLOCK_SIZE || info.numKvHeads != 1,
            OP_LOGE(context, "int8 query requires TND query, pool_block_size=128 and N2=1."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(info.keyLayout != MSA_KEY_LAYOUT_TND && info.keyLayout != MSA_KEY_LAYOUT_BBND &&
                        info.keyLayout != MSA_KEY_LAYOUT_BNBD,
                    OP_LOGE(context, "native int8 query/key requires TND, BBND or BNBD key layout."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!IsOptionalProvided(queryScaleShape), OP_LOGE(context, "int8 query requires query_scale [Tq, Hq]."),
                    return ge::GRAPH_FAILED);
        const auto& qs = queryScaleShape->GetStorageShape();
        OP_CHECK_IF(
            qs.GetDimNum() != MSA_SCALE_TND_DIM_NUM || qs.GetDim(MSA_DIM_0) != static_cast<int64_t>(info.totalQ) ||
                qs.GetDim(MSA_DIM_1) != static_cast<int64_t>(info.numQHeads),
            OP_LOGE(context, "query_scale must be [%u, %u].", info.totalQ, info.numQHeads), return ge::GRAPH_FAILED);
        const auto* queryScaleDesc = context->GetOptionalInputDesc(MSA_IDX_QUERY_SCALE);
        OP_CHECK_NULL_WITH_CONTEXT(context, queryScaleDesc);
        OP_CHECK_IF(queryScaleDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context, "query_scale dtype must be float32."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(IsOptionalProvided(queryScaleShape),
                    OP_LOGE(context, "query_scale is only supported for int8 query."), return ge::GRAPH_FAILED);
    }
    // Block16 keeps upstream FP16/BF16/FP8/MXFP4 support; native A8C8 remains block128.
    OP_CHECK_IF(info.poolBlockSize == MSA_BLOCK16_SIZE &&
                    (info.isQuant || info.queryDtype == ge::DT_HIFLOAT8 || info.keyDtype == ge::DT_HIFLOAT8),
                OP_LOGE(context, "pool_block_size=%u supports fp16 / bf16 / float8_e4m3fn / float8_e5m2 / MXFP4 only.",
                        MSA_BLOCK16_SIZE),
                return ge::GRAPH_FAILED);
    // PA 存储页长 16（M3 短页存储）同样仅 950 上的 fp16 / bf16。
    OP_CHECK_IF(
        info.pageSize == MSA_BLOCK16_SIZE &&
            (info.isQuant || info.queryDtype == ge::DT_HIFLOAT8 || info.keyDtype == ge::DT_HIFLOAT8),
        OP_LOGE(context, "BBND/BNBD page size %u supports fp16 / bf16 / float8_e4m3fn / float8_e5m2 / MXFP4 only.",
                MSA_BLOCK16_SIZE),
        return ge::GRAPH_FAILED);
    if (info.isQuant) {
        OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT16 && !info.isQueryQuant,
                    OP_LOGE(context, "ND int8 key requires fp16 or int8 query."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(info.keyDtype != info.queryDtype, OP_LOGE(context, "non-quant key dtype must match query dtype."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(IsFp8ComputeDtype(info.queryDtype) && !info.isAscend950,
                    OP_LOGE(context, "FP8 query/key is only supported on Ascend 950."), return ge::GRAPH_FAILED);
    }

    const gert::StorageShape* scaleShape = context->GetOptionalInputShape(MSA_IDX_KEY_SCALE);
    if (info.isQuant) {
        OP_CHECK_IF(!IsOptionalProvided(scaleShape), OP_LOGE(context, "int8 key requires dequant scale."),
                    return ge::GRAPH_FAILED);
        const gert::Shape& sc = scaleShape->GetStorageShape();
        if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
            const bool ok2d = (sc.GetDimNum() == MSA_SCALE_TND_DIM_NUM &&
                               static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) == info.totalK &&
                               static_cast<uint32_t>(sc.GetDim(MSA_DIM_1)) == info.numKvHeads);
            const bool ok1d = (sc.GetDimNum() == MSA_RANK_1D && info.numKvHeads == 1 &&
                               static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) == info.totalK);
            OP_CHECK_IF(!ok2d && !ok1d,
                        OP_LOGE(context, "TND dequant scale must be [%u, %u] or [%u] (N2=1).", info.totalK,
                                info.numKvHeads, info.totalK),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(sc.GetDimNum() != MSA_SCALE_PA_DIM_NUM,
                        OP_LOGE(context, "PA dequant scale must be 3D [NP, N_kv, P], got rank %zu.", sc.GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) != info.numPages ||
                            static_cast<uint32_t>(sc.GetDim(MSA_DIM_1)) != info.numKvHeads ||
                            static_cast<uint32_t>(sc.GetDim(MSA_DIM_2)) != info.pageSize,
                        OP_LOGE(context, "dequant scale shape must be [%u, %u, %u], got [%ld, %ld, %ld].",
                                info.numPages, info.numKvHeads, info.pageSize, sc.GetDim(MSA_DIM_0),
                                sc.GetDim(MSA_DIM_1), sc.GetDim(MSA_DIM_2)),
                        return ge::GRAPH_FAILED);
        }
        const auto* scaleDesc = context->GetOptionalInputDesc(MSA_IDX_KEY_SCALE);
        OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
        OP_CHECK_IF(scaleDesc->GetDataType() != ge::DT_FLOAT, OP_LOGE(context, "dequant scale dtype must be float32."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(IsOptionalProvided(scaleShape), OP_LOGE(context, "non-quant path must not pass scale."),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMsaIndexScore(gert::TilingParseContext* /* context */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForMsaIndexScore(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("MsaIndexScore", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    MsaIndexScoreInfo info;
    if (ParseAndCheck(context, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return DoTiling(context, info);
}

IMPL_OP_OPTILING(MsaIndexScore)
    .Tiling(TilingForMsaIndexScore)
    .TilingParse<MsaIndexScoreCompileInfo>(TilingPrepareForMsaIndexScore);

} // namespace optiling
