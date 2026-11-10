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
 * \file mega_moe_dedup_layout.h
 * \brief 通信零冗余（Dedup）host 侧 peermem/workspace 布局计算单一模块：偏移只由本模块按
 *        单一口径计算，经 tilingData 透传 kernel。仅 A3 + W4A8 路径。
 *        单轮可行性判别按形状界 recvRoundBudget ≥ chunkTokens·EP·min(topK,epr)，不可行时
 *        对显式请求（ccm=1）fail-fast。不 include mega_moe_tiling_a2a3.h（其传递包含的
 *        prologue tiling 头含非 inline 定义，多 TU 会重复符号）。
 */

#ifndef MEGA_MOE_DEDUP_LAYOUT_H
#define MEGA_MOE_DEDUP_LAYOUT_H

#include <cstdint>

#include "tiling/tiling_api.h"

namespace MegaMoeDedup {

// ===== 布局常量（与基线同源：mega_moe_tiling_arch22.cpp 常量节 / kernel const_args）=====
constexpr int64_t DEDUP_RESERVED_SPACE_SIZE = 10 * 1024 * 1024; // 头部 10MB 预留
constexpr int64_t DEDUP_ALIGN_128 = 128;
constexpr int64_t DEDUP_ALIGN_32 = 32;   // FP/LP 表对齐粒度，同 kernel 清零块（DEDUP_ALIGN32_INT32S）
constexpr int64_t DEDUP_ALIGN_512 = 512; // A 区行内 scale 槽宽度
constexpr int64_t DEDUP_MAX_EXPERTS_PER_RANK = 128;
constexpr int64_t DEDUP_STATE_OFFSET = 512; // 每 rank 尾留 512B（CrossRankSync）
// 与基线 tiling SYNC_STATE_RESERVED_SIZE 一致：前 256KB per-rank 状态 + 后 256KB self state
constexpr int64_t DEDUP_SYNC_STATE_RESERVED_SIZE = 512 * 1024LL;
constexpr int64_t DEDUP_D2_BYTES = 2; // ElementD2 = bfloat16_t

// ===== rowMeta 段行宽（分配侧单一口径）=====
// 32B/行 = 8×int32 [anchorRow|cot|probBits|lastRow|pad×4]，与 kernel 步长（row*8）锚定；
// 步长 32B 是 UB→GM DataCopy 源对齐硬约束（不足 32B 对齐 MTE 直接报错）。
constexpr int64_t DEDUP_ROWMETA_ROW_INT32S = 8;
constexpr int64_t DEDUP_ROWMETA_ROW_BYTES = 32;

// 打包区每 rank 段 = [slots i32 | probs f32] 等长两段（kernel 侧同源常量在 const_args.hpp）。
// slots 元素高 8 bit 内联 rankOrdinal = token 去重卡序列（topK 展开序首现）中目标卡的序数，
// 即 D 区 token 块槽位因子
constexpr int64_t DEDUP_PACK_SECTION_COUNT = 2; // 打包区每 rank 段 = [slots|probs] 两段

// Dedup peermem 布局（chunk 尺度；chunkTokens = min(M, PERMUTE_CHUNK=1024)）。
// 头部 RESERVED → A（原始序）→ 打包区（rank 段 [slots|probs]）→ D（token 块：每 token 独占
// K̂=min(topK,EP) 行，块内槽位 = 目标卡首现序）；尾部与基线倒排区同源。
struct DedupPeermemLayout {
    int64_t offsetA = 0;
    int64_t offsetExpandedDstToSrcRow = 0;
    int64_t offsetExpProb = 0;
    int64_t offsetD = 0;
    int64_t dSize = 0;
    int64_t headTotal = 0;
    int64_t tokenPerExpertSize = 0;
    int64_t tailReservedSize = 0;
};

// Dedup workspace 增量（追加于基线现有段之后）
struct DedupWorkspaceAddon {
    int64_t distTokenFirstPosSize = 0; // FP 表（cot 坐标）
    int64_t distTokenLastPosSize = 0;  // LP 表
    int64_t rowMetaSize = 0;
};

// Dedup tiling 输出（写入 MegaMoeA2A3TilingData.common 的字段集合；
// probs 打包段起点由 kernel 侧按 offsetExpandedDstToSrcRow+layoutChunkTokens·topK 现场推导，不透传）
struct DedupTilingOutput {
    int64_t offsetExpandedDstToSrcRow = 0;
    int64_t offsetD = 0;
    int64_t distTokenFirstPosSize = 0;
    int64_t distTokenLastPosSize = 0;
    int64_t rowMetaSize = 0;
};

// 计算 Dedup peermem 布局。
// chunkTokens=单 chunk token 数（min(M,1024)）, k=hidden 维(=n₂), topK=路由专家数, ep=EP 卡数
DedupPeermemLayout ComputePeermemLayout(int64_t chunkTokens, int64_t k, int64_t topK, int64_t ep);

// 计算 Dedup workspace 增量（FP/LP + rowMeta 独立段）。
DedupWorkspaceAddon ComputeWorkspaceAddon(int64_t chunkTokens, int64_t ep, int64_t recvRoundBudget);

// Dedup 布局下 winIn 最小需求（对齐基线 CalcLeastCclBufferSize 模型，无 perTokenScale 区）。
// 校验语义：cclBufferSize/2 >= least ⟺ 头部与倒排尾部不碰撞。
int64_t CalcLeastCclBufferSizeDedup(int64_t chunkTokens, int64_t k, int64_t topK, int64_t ep);

// 可行性三态判别（单轮约束 + winIn 预算）。单轮判别按形状界：
// recvRoundBudget ≥ chunkRecvMax = chunkTokens(=min(M,1024))·EP·min(topK,epr)
// ⟺ 实际接收行数（≤ 界）恒单轮，无需预知路由。
// kMultiRound：仅单轮不满足——基线按同预算多轮跑，ccm=1 由调用方自动回基线（日志留痕）；
// kInfeasible：配置须人工调整（路径/位宽/窗口），ccm=1 报错。
enum class DedupFeasibility {
    kFeasible,
    kMultiRound,
    kInfeasible,
};
DedupFeasibility CheckDedupFeasible(int64_t chunkTokens, int64_t k, int64_t topK, int64_t ep, int64_t expertPerRank,
                                    int64_t recvRoundBudget, int64_t cclBufferSize);

// Dedup tiling 分支入口（调用前须 CheckDedupFeasible 通过）：计算布局/增量并写 out
ge::graphStatus CalcDedupTiling(int64_t chunkTokens, int64_t k, int64_t topK, int64_t ep, int64_t recvRoundBudget,
                                DedupTilingOutput& out);

// ===== 收益门控 =====
// 开启 Dedup 还须 G > 门限（DEDUP_GATE_G_THRESHOLD_BYTES）：G = m·topK·dupRatio_unif·hidden·
// 的去重节省字节（w_Σ = dispatch int8 + combine bf16，每元素 3 字节）；dupRatio_unif 按
// w_Σ/epr，即平均到每个本卡专家的去重节省字节（w_Σ = dispatch int8 + combine bf16，每元素 3
// 字节）；dupRatio_unif 按 token 均匀命中各卡估算，tiling 可直接算出。G 不达标则静默走基线
// 并打一条 decision 日志（收益判断，不影响正确性，不报错）；单轮可行性校验不变，不可行仍报
// 错。m 取本卡 bs，各卡 BS 一致是 Dedup 的使用前提（README），因此各卡判定结果一致。门限由
// EP8/EP16 实测标定；更换 EP 档位或修改 kernel 后需重新标定。
constexpr int64_t DEDUP_GATE_W_SIGMA_BYTES = 3;
constexpr int64_t DEDUP_GATE_G_THRESHOLD_BYTES = 568660;

// 均匀路由闭式 dupRatio（W≤1 或 K≤0 → 0.0，即无跨卡冗余收益）
double CalcUniformDupRatio(int64_t worldSize, int64_t topK);

// G 门判别量（字节）；epr≤0 → 0.0 防除零
double CalcDedupGateGBytes(int64_t m, int64_t hidden, int64_t topK, int64_t expertPerRank, int64_t worldSize);

} // namespace MegaMoeDedup

#endif // MEGA_MOE_DEDUP_LAYOUT_H
