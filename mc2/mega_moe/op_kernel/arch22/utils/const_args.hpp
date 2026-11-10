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
 * \file const_args.hpp
 * \brief
 */

#ifndef CONST_ARGS_HPP
#define CONST_ARGS_HPP
constexpr static uint64_t MB_SIZE = 1024 * 1024UL;
constexpr static int32_t NUMS_PER_FLAG = 16;
constexpr static int32_t CACHE_LINE = 512;
constexpr static int32_t RESET_VAL = 0xffff;
constexpr static int32_t FLAGSTRIDE = 16;
constexpr static int32_t UB_ALIGN = 32;
constexpr static int32_t ALIGN_128 = 128;
constexpr static int32_t ALIGN_512 = 512;
constexpr uint16_t CROSS_CORE_FLAG_MAX_SET_COUNT = 15;
constexpr uint32_t MAX_EXPERTS_PER_RANK = 128;
constexpr uint32_t MAX_RANK_PER_CORE = 32;
constexpr static uint16_t SYNCFLAGC2V = 9;
constexpr static uint16_t SYNCFLAGV2C = 10;
constexpr static uint32_t SERVER_RANK_SIZE_A2 = 8;
constexpr static uint32_t kMaxDequantSyncGroups = 16;
constexpr static uint64_t RESERVED_SPACE_SIZE = 10 * 1024 * 1024;
// 发送侧逐 chunk 的 token 数（A2/A3 发送侧 prologue chunk 语义；A3 kernel 内另有同名 namespace 级定义）
constexpr static int64_t PERMUTE_CHUNK = 1024;
// prologue 每核计数区核数上限（host 侧 workspace 须按实际 aivNum 分配且 <= 此值）
constexpr static int64_t PERMUTE_MAX_CORES = 128;
// Flag 同步 magic 值（精确匹配，避免上一轮残留误判）
// 三处同步（SendTokensV3 / RecvTokensV3 / V2 allgather）统一使用该值，需配合
// ResetTokenPerExpert 每轮清零 flag 区，确保上一轮残留不会撞上本值
constexpr static int32_t FLAG_VALUE_MAGIC = 123456789; // 0x075BCD15
// A2/A3 每个 Cube 核附带 1 个 Vector 核，核数 / 2 即 Cube（matmul tile）数
constexpr static int32_t AIV_PER_AIC = 2;
// AIC 内 m 维拆分给两个 subblock 处理
constexpr static int32_t SUBBLOCK_NUM_PER_AIC = 2;
// W4 SwiGLU 场景 GMM/Combine 的 M 维展开倍数（gate+up 双行）
constexpr static uint32_t GATE_UP_ROWS_PER_TOKEN = 2;
// int4 打包密度：每字节 2 个 nibble
constexpr static int32_t INT4_NUM_PER_BYTE = 2;
// icache 预取行数
constexpr static int32_t ICACHE_PRELOAD_LINES = 8;
// rdma 描述符区在 UB 中的起始偏移（128KB）
constexpr static uint32_t RDMA_UB_OFFSET_BYTES = 128 * 1024;
// MoePermutePrologue UB 流水级数（双缓冲）
constexpr static uint32_t PROLOGUE_UB_STAGES = 2;
// GetSumPreRank 每核槽位步长（元素），16 元素 = 64B，避免跨核伪共享
constexpr static int32_t PRE_SUM_SLOT_STRIDE = 16;
// Dedup 打包段 AllGather 推送的 UB 单次搬运粒度（元素，int32 视图；8192×4B = 32KB，
// 与 CopyGMToGM 内部 32KB pingpong 缓冲匹配）
constexpr static int32_t DEDUP_PACK_UB_MOVE_ELEMS = 8192;
// Dedup 布局 kernel 侧常量（与 op_host/op_tiling/arch22/mega_moe_dedup_layout.h 同源，改动须双侧同步）
constexpr static int32_t DEDUP_PACK_SECTION_COUNT = 2;      // 打包区每 rank 段 = [slots|probs] 等长两段
constexpr static int32_t DEDUP_ROWMETA_ROW_INT32S = 8;      // rowMeta 行宽 8×int32（同源 dedup_layout.h）
constexpr static uint32_t DEDUP_ROWMETA_PAYLOAD_BYTES = 16; // rowMeta 有效载荷 16B/行（32B 行内）
constexpr static int32_t DEDUP_ROWMETA_LASTROW_SLOT = 3;    // lastRow 槽位（int32 槽 3，字节偏移 12B）
// combine 切片批读：每片行数与奇偶双缓冲总字节数（每行 rowMeta 8×i32 + scale2 1×f32）
constexpr static uint32_t DEDUP_COMBINE_SLICE_ROWS = 64;
constexpr static uint32_t DEDUP_COMBINE_SLICE_BYTES =
    2 *
    (DEDUP_COMBINE_SLICE_ROWS * DEDUP_ROWMETA_ROW_INT32S * sizeof(int32_t) + DEDUP_COMBINE_SLICE_ROWS * sizeof(float));
// rankOrdinal 内联编码：不占独立平面/独立槽——slots 元素与 rowMeta 槽 1 均为低 24 bit 载
// 数据（srcIdx / cot；cot 上界 ep·chunkTokens ≤ 256·1024 < 2^18，24 bit 恒容纳）、高 8 bit
// 载 rankOrdinal；打包段哨兵以 rankOrdinal 段 0xFF 标记（有效值 < EP ≤ 256 恒不撞）。
// 位宽由 host 能力门 fail-fast 保证（dedup_layout.cpp CheckDedupFeasible）
constexpr static int32_t DEDUP_SLOT_RANK_ORDINAL_SHIFT = 24;  // slots 元素内 rankOrdinal 的起始 bit
constexpr static int32_t DEDUP_SLOT_SRCIDX_MASK = 0x00FFFFFF; // slots 元素 srcIdx 掩码（低 24 bit）
constexpr static int32_t DEDUP_SLOT_RANK_ORDINAL_MASK = 0xFF; // rankOrdinal 段掩码
constexpr static int32_t DEDUP_SLOT_SENTINEL = static_cast<int32_t>(0xFF) << 24; // 打包段哨兵（rankOrdinal 段 0xFF）
constexpr static int32_t DEDUP_ROWMETA_RANK_ORDINAL_SHIFT = 24;                  // 槽 1 内 rankOrdinal 的起始 bit
constexpr static int32_t DEDUP_ROWMETA_COT_MASK = 0xFFFFFF;                      // 槽 1 cot 掩码（低 24 bit）
constexpr static uint32_t DEDUP_SCALAR_AREA_MARGIN = 1024; // 路由缓存 UB 高址预留 1KB 标量区余量
constexpr static int32_t DEDUP_ALIGN32_INT32S = 8;         // 32B 对齐粒度（int32 计），FP/LP 清零块粒度
#endif
