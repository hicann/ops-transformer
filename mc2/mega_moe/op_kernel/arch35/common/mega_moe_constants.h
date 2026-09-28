/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_CONSTANTS_H
#define MEGA_MOE_CONSTANTS_H

#include <cstdint>

namespace MegaMoeImpl {

constexpr uint32_t RANK_SYNC_COUNTER_OFFSET_BYTES = 48U * 1024U;
constexpr uint32_t RANK_SYNC_COUNTER_SLOT_BYTES = 64U;
// mode 0：本卡本次 kernel 的同类核全核同步，AIC 调用时同步全部 AIC，AIV 调用时同步全部 AIV。
constexpr uint8_t ALL_AICORE_SYNC_MODE = 0;
// mode 4：同一 AI Core 内 AIC 与单个 AIV 的同步，支持双向通知。
constexpr uint8_t AIC_SINGLE_AIV_SYNC_MODE = 4;
// reset、MoE 及共享量化写回完成事件；与 GMM mode 4 独立，避开 SyncAll 的 11--14。
constexpr uint16_t MTE_QUANT_READY_FLAG = 8;
// count 表读取完成事件编号：count 表清零前等待所有读取结束。
constexpr uint16_t COUNT_TABLE_READ_DONE_FLAG = 9;
// 输入就绪事件编号：AIV0 通知配对 AIC，本卡全部 AIV 已完成 reset 和量化。
constexpr uint16_t INPUT_READY_FLAG = 2;
// UB 释放事件编号：AIV0 通知配对 AIC，可以通过 FIX 写入输入准备阶段占用的 UB。
constexpr uint16_t INPUT_UB_FREE_FLAG = 15;

constexpr int64_t MAX_INT16_TOPK_INDEX_COUNT = 1LL << 15;

enum class MegaMoeActMode : uint8_t {
    SWIGLU = 0U,
    SITU = 1U,
    SWIGLU_STEP = 2U,
    SWIGLU_OAI = 3U,
};

enum class MegaMoeActSubMode : uint8_t {
    DEFAULT = 0U,
    LINEAR = 1U,
};

enum class MegaMoeGmmMode : uint8_t {
    A8W8_ND = 0U,
    A8W4_NZ = 1U,
    A8W8_NZ = 2U,
    A4W4_ND = 3U,
    A4W4_NZ = 4U,
};

constexpr uint8_t GMM_MODE_A8W8_ND = static_cast<uint8_t>(MegaMoeGmmMode::A8W8_ND);
constexpr uint8_t GMM_MODE_A8W4_NZ = static_cast<uint8_t>(MegaMoeGmmMode::A8W4_NZ);
constexpr uint8_t GMM_MODE_A8W8_NZ = static_cast<uint8_t>(MegaMoeGmmMode::A8W8_NZ);
constexpr uint8_t GMM_MODE_A4W4_ND = static_cast<uint8_t>(MegaMoeGmmMode::A4W4_ND);
constexpr uint8_t GMM_MODE_A4W4_NZ = static_cast<uint8_t>(MegaMoeGmmMode::A4W4_NZ);

constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint64_t IDX_GMM1_OFFSET = 12UL;
constexpr uint64_t IDX_GMM2_OFFSET = 13UL;

constexpr int32_t INT_CACHELINE = 16;
constexpr uint64_t INT32_PER_256B = 8U;
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t DOUBLE_BUFFER = 2U;
constexpr uint32_t RANK_ID = 0U;
constexpr uint32_t TOKEN_ID = 1U;
constexpr uint32_t TOPK_INDEX = 2U;
constexpr uint32_t WEIGHT_INDEX = 3U;
constexpr uint32_t SYNC_EVENT_ID0 = 0;
constexpr uint32_t SYNC_EVENT_ID1 = 1;
constexpr uint32_t SYNC_EVENT_ID2 = 2;
constexpr uint32_t SYNC_EVENT_ID3 = 3;
constexpr uint32_t SYNC_EVENT_ID4 = 4;
constexpr uint32_t SYNC_EVENT_ID5 = 5;
constexpr uint32_t SYNC_EVENT_ID6 = 6;
constexpr int64_t GM_FLAG_POLL_BACKOFF_CYCLES = 200;
constexpr int64_t OVERFLOW_MODE_CTRL = 60U;
constexpr uint8_t COMBINE_NO_QUANT = 0;
constexpr uint8_t MXFP8_E4M3_COMM_QUANT = 4;
constexpr uint32_t META_INFO_SIZE = 8U;
constexpr uint8_t GROUPED_MATMUL_MODE_GENERAL = 0U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W4 = 1U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W8_NZ = 2U;
// a4w4 混合场景：GMM1 走 generic a4w4，GMM2 走 A8W4。
constexpr uint8_t GROUPED_MATMUL_MODE_A4W4 = 3U;
constexpr uint8_t GROUPED_MATMUL_MODE_A4W4_NZ = 4U;

constexpr uint32_t L1_TILE_M_256 = 256U;
constexpr uint32_t L1_TILE_M_128 = 128U;
constexpr uint32_t L1_TILE_N = 256U;
constexpr uint32_t L0_TILE_K = 128U;
constexpr uint32_t ACTIVATION_N_HALF = 2U;

// ============ MTE 通信去重 ============
// 开关的唯一真源是 tilingData->dedupMode（0=全关 1=仅combine开 2=仅dispatch开 3=双开），
// 判断统一用 mega_moe_peermem.h 的 IsDispatchDedupOn/IsCombineDedupOn。
// prepass 工作项 chunk 数上限：flag 区按 worldSize*MAX_CHUNKS 上界分配；
// CalcDedupPrepassChunkTokens 的 floor-clamp 保证实际 chunkCount 不超此值。
constexpr uint32_t DEDUP_PREPASS_MAX_CHUNKS = 128U;
// AIV0 参与 prepass 建表只在通信主导档（与收益门的 hiddenDim 分档同轴）开启：该档 prepass 在关键
// 路径上、AIV0 分担净赢（k16 族/B1025/B1394 实测 -5%~-10%）；计算主导档 prepass 被 GMM 遮住，
// "发布前等全体 AIV0"的 gate 反而把 dispatch 首发布从"首 chunk 就绪"推迟到"半程建表完成"
// （B8192 实测劣化 ~200us），保持 AIV1 单方建表。
constexpr uint32_t DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM = 2048U;
// prepass UB 三区容量（kernel CalcDedupPrepassChunkTokens 与 host 门 floorChunk 护栏共用，单一真源）。
constexpr uint32_t DEDUP_ROUTE_LOAD_UB_BYTES = 4U * 1024U;
constexpr uint32_t DEDUP_COUNT_UB_BYTES = 4U * 1024U;
constexpr uint32_t DEDUP_MEMBER_UB_BYTES =
    54U * 1024U; // combine 生产卡预加权合并后单发（可选能力，强制 prefetch 语义）
/*
 * rowDesc：去重 prepass 为每个接收行生成的定长描述符（布局尺寸见 peermem.h CalcDedupRowDescStrideInt32）。
 * [0] flags | memberCount：0 表示该 (srcRank, token) 在本卡只出现一次，走原路径；
 * [1] 组内最小 topkIdx（combine 合并行发送的目标槽，与归属卡 unpermute 的 canonical 槽约定一致）；
 * [2+3i] / [3+3i] / [4+3i] 组内第 i 个成员的（全局接收行号 / 该成员 topkIndex / dispatch→GMM1
 * flag 槽偏移），成员按行号升序排列、含本行。flag 槽偏移在 prepass 预先算好，使扇出发布
 * 不依赖 cumsum UB（W4 路径的 Activation 会覆盖 AIV1 cumsum、分片只恢复本片区间）。
 * 组内每个成员行都持有整组的完整拷贝，读一行描述符即可拿到全组信息。
 */
constexpr int32_t DEDUP_ROW_FLAG_GROUP = 0x40000000; // 本行属于某个重复组
constexpr int32_t DEDUP_ROW_FLAG_FIRST = 0x20000000; // 组内首行（dispatch 扇出驱动行：拉一次远端、写全组）
constexpr int32_t DEDUP_ROW_FLAG_LAST = 0x10000000; // 组内末行（combine 合并驱动行：读全组、加和、单发）
constexpr int32_t DEDUP_ROW_MEMBER_COUNT_MASK = 0xFFFF;
constexpr uint32_t DEDUP_ROW_DESC_HEADER_INT32 = 2U;
// dispatch 去重的批内 UB 暂存（desc 装载 48K + 扇出 meta 8K + 活跃表 8K）从 dispatch 固定区尾动态起址；
// host 在 dedupMode!=0 时从 dispatch 自适应预算中预留本值，保证暂存区恒有落位（含对齐余量）。
constexpr uint32_t DEDUP_DISPATCH_UB_RESERVE_BYTES = 72U * 1024U;

// Hcomm 批量句柄中每条普通读写描述符占 64B。
// Layered 的数据批次仍按最多 256 个 token 组织，64 KiB 可容纳同批 token 的附加 WQE。
constexpr uint32_t HCOMM_BATCH_WQE_BYTES = 64U;
constexpr uint32_t LAYERED_USABLE_UB_BYTES = 248U * 1024U;
constexpr uint32_t LAYERED_HCOMM_BATCH_UB_BYTES = 64U * 1024U;
constexpr uint32_t LAYERED_HCOMM_BATCH_UB_OFFSET = LAYERED_USABLE_UB_BYTES - LAYERED_HCOMM_BATCH_UB_BYTES;
constexpr uint32_t LAYERED_HCOMM_BATCH_WQE_CAPACITY = LAYERED_HCOMM_BATCH_UB_BYTES / HCOMM_BATCH_WQE_BYTES;
} // namespace MegaMoeImpl

#endif // MEGA_MOE_CONSTANTS_H
