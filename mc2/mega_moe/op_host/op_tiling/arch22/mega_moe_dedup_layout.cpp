/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>

#include "mc2_log.h"
#include "mega_moe_dedup_layout.h"

namespace MegaMoeDedup {

namespace {
const char* DEDUP_LAYOUT_TAG = "MegaMoeDedupLayout";
constexpr int64_t DEDUP_MB_BYTES = 1LL << 20; // 日志建议值的 MB 取整粒度

int64_t AlignUp(int64_t value, int64_t align)
{
    return (value + align - 1) / align * align;
}
} // namespace

// host 内自洽：行宽须覆盖 kernel 步长（同源约定见 const_args.hpp DEDUP_ROWMETA_ROW_INT32S）
static_assert(DEDUP_ROWMETA_ROW_BYTES >= DEDUP_ROWMETA_ROW_INT32S * sizeof(int32_t),
              "rowMeta host row width must cover kernel 32B/row footprint (8 x int32 stride)");

DedupPeermemLayout ComputePeermemLayout(int64_t chunkTokens, int64_t k, int64_t topK, int64_t ep)
{
    DedupPeermemLayout layout;

    // A 区原始序行数不乘 topK——跨卡去重后同一 token 只发一次，此即 Dedup 的 peermem 收益；
    // 行格式与基线 A 区逐字节一致（[k int8 | scale@+k]，行步长 k+512）
    const int64_t aSize = chunkTokens * (k + DEDUP_ALIGN_512);
    // 打包区每 rank 段 [slots i32 | probs f32] 等长两段
    const int64_t packSize = ep * chunkTokens * topK * static_cast<int64_t>(sizeof(int32_t)) * DEDUP_PACK_SECTION_COUNT;

    layout.offsetA = DEDUP_RESERVED_SPACE_SIZE;
    // 无 PeerPerTokenScale 保留区（基线该偏移只算不用，scale 内嵌 A 区行尾）
    layout.offsetExpandedDstToSrcRow = layout.offsetA + aSize;
    layout.offsetExpProb =
        layout.offsetExpandedDstToSrcRow + chunkTokens * topK * static_cast<int64_t>(sizeof(int32_t));
    // D 基址 128B 对齐：unpermute 对 D 区做非 Pad 整行 DataCopy，基址不齐 MTE 直接报错；
    // chunkTokens=1024 时 packSize 恒为 128 倍数（AlignUp 恒等），bs<1024 的单 chunk 形状
    // 可能触发 64B 垫
    layout.offsetD = AlignUp(layout.offsetExpandedDstToSrcRow + packSize, DEDUP_ALIGN_128);
    // D 区 token 块：每 chunk 内 token 独占 K̂ = min(topK, EP) 行（块内槽位 = 目标卡首现序）。
    // token 的 topK 个互异专家至多落 K̂ 张卡，故 K̂·chunkTokens 行即覆盖最坏路由；EP > topK 时
    // 较按发送卡分段（EP·chunkTokens 行）缩小 EP/K̂ 倍，EP ≤ topK 时两种口径等容量。
    // +1 行为池尾防御 pad（寻址越界时不踩尾部倒排区）
    const int64_t cardsPerToken = std::min(topK, ep);
    layout.dSize = (cardsPerToken * chunkTokens + 1) * k * DEDUP_D2_BYTES;
    layout.headTotal = layout.offsetD + layout.dSize;

    // 尾部与基线倒排区同源（TPE 含 +1 哨兵槽；tailReserved 取 self state 512KB 下限）
    layout.tokenPerExpertSize =
        ep * AlignUp(ep * DEDUP_MAX_EXPERTS_PER_RANK + 1, DEDUP_ALIGN_128) * static_cast<int64_t>(sizeof(int32_t));
    layout.tailReservedSize = std::max(ep * DEDUP_STATE_OFFSET, DEDUP_SYNC_STATE_RESERVED_SIZE);

    return layout;
}

DedupWorkspaceAddon ComputeWorkspaceAddon(int64_t chunkTokens, int64_t ep, int64_t recvRoundBudget)
{
    DedupWorkspaceAddon addon;
    // 对齐 kernel 清零块粒度，防 chunkTokens·EP 非 8 倍数时清零越段写入
    const int64_t tableSize = AlignUp(ep * chunkTokens * static_cast<int64_t>(sizeof(int32_t)), DEDUP_ALIGN_32);
    addon.distTokenFirstPosSize = tableSize; // FP 表
    addon.distTokenLastPosSize = tableSize;  // LP 表
    addon.rowMetaSize = recvRoundBudget * DEDUP_ROWMETA_ROW_BYTES;
    return addon;
}

int64_t CalcLeastCclBufferSizeDedup(int64_t chunkTokens, int64_t k, int64_t topK, int64_t ep)
{
    // 对齐基线 CalcLeastCclBufferSize 的 winIn 模型：least = TPE + tensor + flag + RESERVED
    const DedupPeermemLayout layout = ComputePeermemLayout(chunkTokens, k, topK, ep);

    const int64_t offsetTokenPerExpert = layout.tokenPerExpertSize;
    const int64_t offsetTensor = layout.headTotal - layout.offsetA; // A + 打包区 + D（含 D 对齐）
    const int64_t offsetFlag = layout.tailReservedSize;

    return offsetTokenPerExpert + offsetTensor + offsetFlag + DEDUP_RESERVED_SPACE_SIZE;
}

DedupFeasibility CheckDedupFeasible(int64_t chunkTokens, int64_t k, int64_t topK, int64_t ep, int64_t expertPerRank,
                                    int64_t recvRoundBudget, int64_t cclBufferSize)
{
    // 多轮接收会破坏「本地副本覆盖同一 token 全部出现」的前提（见文件头），Dedup 仅单轮
    // 可行；budget ≥ chunkRecvMax = chunkTokens(=min(M,1024))·EP·min(topK,epr) 即恒单轮，
    // 无需预知路由（固定全窗的 recvRoundsMax 只是保守轮表份数上限，非判据）。
    if (expertPerRank <= 0) {
        OP_LOGW(DEDUP_LAYOUT_TAG, "[DEDUP] infeasible: expertPerRank=%ld <= 0", static_cast<long>(expertPerRank));
        return DedupFeasibility::kInfeasible;
    }
    // rankOrdinal 内联编码位宽门（fail-fast，静默错位不可接受）：
    //   slots 元素 = srcIdx（低 24 bit）| j（高 8 bit）；rowMeta 槽 1 = cot（低 24 bit）| j（高 8 bit）
    if (chunkTokens * topK > 0xFFFFFF) {
        OP_LOGW(DEDUP_LAYOUT_TAG, "[DEDUP] infeasible: chunkTokens*topK=%ld exceeds 24-bit srcIdx field",
                static_cast<long>(chunkTokens * topK));
        return DedupFeasibility::kInfeasible;
    }
    if (ep > 256) {
        OP_LOGW(DEDUP_LAYOUT_TAG, "[DEDUP] infeasible: ep=%ld exceeds 8-bit rankOrdinal field", static_cast<long>(ep));
        return DedupFeasibility::kInfeasible;
    }
    if (std::min(topK, ep) > 0xFF) {
        // K̂ 上界保证 rankOrdinal ≤ K̂−1 ≤ 254，与打包段哨兵编码（rankOrdinal 段 0xFF）
        // 恒不撞（EP=256 且 topK≥256 时 rankOrdinal 可达 255，正撞哨兵——故取 0xFF 而非 0xFFFF）
        OP_LOGW(DEDUP_LAYOUT_TAG,
                "[DEDUP] infeasible: K_hat=min(topK,ep)=%ld exceeds 8-bit rankOrdinal field "
                "(sentinel 0xFF non-collision)",
                static_cast<long>(std::min(topK, ep)));
        return DedupFeasibility::kInfeasible;
    }
    const int64_t chunkRecvMax = chunkTokens * ep * std::min(topK, expertPerRank);
    const int64_t trueRecvRoundsMax = (chunkRecvMax + recvRoundBudget - 1) / recvRoundBudget;
    if (trueRecvRoundsMax != 1) {
        OP_LOGW(DEDUP_LAYOUT_TAG,
                "[DEDUP] fallback: multi-round receive (recv budget=%ld < chunk recv upper bound = "
                "chunkTokens(%ld)·EP(%ld)·min(topK,epr)=%ld); combine dedup disabled, baseline multi-round "
                "path used. Set max_recv_token_num >= the bound or combine_comm_mode=0.",
                static_cast<long>(recvRoundBudget), static_cast<long>(chunkTokens), static_cast<long>(ep),
                static_cast<long>(chunkRecvMax));
        return DedupFeasibility::kMultiRound;
    }
    // winIn 预算：头部与倒排尾部不碰撞
    const int64_t winInSize = cclBufferSize / 2;
    const int64_t leastCclBufferSize = CalcLeastCclBufferSizeDedup(chunkTokens, k, topK, ep);
    if (winInSize < leastCclBufferSize) {
        OP_LOGW(DEDUP_LAYOUT_TAG,
                "[DEDUP] infeasible: cclBufferSize(%ldB)/2=%ldB < Dedup least=%ldB (chunkTokens=%ld K=%ld topK=%ld "
                "EP=%ld). Suggestion: set ccl_buffer_size to at least %ld MB (winIn+winOut).",
                static_cast<long>(cclBufferSize), static_cast<long>(winInSize), static_cast<long>(leastCclBufferSize),
                static_cast<long>(chunkTokens), static_cast<long>(k), static_cast<long>(topK), static_cast<long>(ep),
                static_cast<long>((leastCclBufferSize * 2 + DEDUP_MB_BYTES - 1) / DEDUP_MB_BYTES));
        return DedupFeasibility::kInfeasible;
    }
    return DedupFeasibility::kFeasible;
}

ge::graphStatus CalcDedupTiling(int64_t chunkTokens, int64_t k, int64_t topK, int64_t ep, int64_t recvRoundBudget,
                                DedupTilingOutput& out)
{
    const DedupPeermemLayout layout = ComputePeermemLayout(chunkTokens, k, topK, ep);
    const DedupWorkspaceAddon addon = ComputeWorkspaceAddon(chunkTokens, ep, recvRoundBudget);

    out.offsetExpandedDstToSrcRow = layout.offsetExpandedDstToSrcRow;
    out.offsetD = layout.offsetD;
    out.distTokenFirstPosSize = addon.distTokenFirstPosSize;
    out.distTokenLastPosSize = addon.distTokenLastPosSize;
    out.rowMetaSize = addon.rowMetaSize;

    OP_LOGI(DEDUP_LAYOUT_TAG,
            "[DEDUP] peermem layout (chunkTokens=%ld K=%ld topK=%ld EP=%ld): "
            "offsetA=%ld offsetExpandedDstToSrcRow=%ld offsetExpProb=%ld offsetD=%ld DSize=%ld headTotal=%ld "
            "TPEsize=%ld tailReserved=%ld leastCcl=%ld",
            static_cast<long>(chunkTokens), static_cast<long>(k), static_cast<long>(topK), static_cast<long>(ep),
            static_cast<long>(layout.offsetA), static_cast<long>(layout.offsetExpandedDstToSrcRow),
            static_cast<long>(layout.offsetExpProb), static_cast<long>(layout.offsetD), static_cast<long>(layout.dSize),
            static_cast<long>(layout.headTotal), static_cast<long>(layout.tokenPerExpertSize),
            static_cast<long>(layout.tailReservedSize),
            static_cast<long>(CalcLeastCclBufferSizeDedup(chunkTokens, k, topK, ep)));
    OP_LOGI(DEDUP_LAYOUT_TAG,
            "[DEDUP] workspace addon: distTokenFirstPos=%ldB distTokenLastPos=%ldB rowMeta=%ldB total=%ldB",
            static_cast<long>(addon.distTokenFirstPosSize), static_cast<long>(addon.distTokenLastPosSize),
            static_cast<long>(addon.rowMetaSize),
            static_cast<long>(addon.distTokenFirstPosSize + addon.distTokenLastPosSize + addon.rowMetaSize));

    return ge::GRAPH_SUCCESS;
}

double CalcUniformDupRatio(int64_t worldSize, int64_t topK)
{
    if (worldSize <= 1 || topK <= 0) {
        return 0.0;
    }
    // 按 token 均匀命中各卡估算（由期望线性性推得，不要求各卡命中独立）
    const double w = static_cast<double>(worldSize);
    const double k = static_cast<double>(topK);
    const double missProb = std::pow(1.0 - 1.0 / w, k); // 给定卡未被某 token 任何专家命中的概率
    return 1.0 - w * (1.0 - missProb) / k;
}

double CalcDedupGateGBytes(int64_t m, int64_t hidden, int64_t topK, int64_t expertPerRank, int64_t worldSize)
{
    if (expertPerRank <= 0) {
        return 0.0;
    }
    const double dupRatio = CalcUniformDupRatio(worldSize, topK);
    return static_cast<double>(m) * static_cast<double>(topK) * dupRatio * static_cast<double>(hidden) *
           static_cast<double>(DEDUP_GATE_W_SIGMA_BYTES) / static_cast<double>(expertPerRank);
}

} // namespace MegaMoeDedup
