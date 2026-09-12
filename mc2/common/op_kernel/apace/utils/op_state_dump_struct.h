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
 * \file op_state_dump_struct.h
 * \brief
 */
#ifndef OP_STATE_DUMP_STRUCT_H
#define OP_STATE_DUMP_STRUCT_H

#include <cstdint>

#ifndef MC2_DFX_ENABLE
#define MC2_DFX_ENABLE 1
#endif

namespace Utils {

// ==================== Workspace 段布局信息 ====================

constexpr uint32_t MAX_WORKSPACE_SEGMENTS = 8U;

// Workspace 段类型
enum WorkspaceSegType : uint8_t {
    WS_SEG_LIB_API = 0,        // 系统库预留空间
    WS_SEG_ND2NZ = 1,          // ND2NZ 格式转换空间
    WS_SEG_GATHER = 2,         // AllGather 通信结果空间
    WS_SEG_GATHER_SCALE1 = 3,  // AllGather scale1 空间 (FP8 perblock)
    WS_SEG_GATHER_SCALE = 4,   // AllGather scale 空间
    WS_SEG_COMM_OUT = 5,       // AlltoAll 通信结果空间
    WS_SEG_PERMUTE_OUT = 6,    // AlltoAll 重排空间
    WS_SEG_BIAS = 7,           // bias 空间
    WS_SEG_MM_RESULT = 8,      // matmul 结果暂存空间
    WS_SEG_RECV_BUF = 9,       // ReduceScatter 接收缓冲 (a2a 路径)
    WS_SEG_COMM_INT8 = 10,     // int8 通信 workspace
    WS_SEG_DYNAMIC_QUANT = 11, // 动态量化临时空间
    WS_SEG_STATE_DUMP = 12,    // 状态打点段 (64KB)
    WS_SEG_MM_WORKSPACE = 13,  // matmul 计算临时 workspace
    WS_SEG_RESERVED = 255,
};

// 单个段描述 (24 bytes)
struct WorkspaceSegInfo {
    uint64_t offset;     // 相对 workspace 基地址的偏移 (字节)
    uint64_t size;       // 该段大小 (字节)
    uint8_t type;        // WorkspaceSegType
    uint8_t reserved[7]; // 对齐
};

// Workspace 布局信息 (208 bytes)，用于异常时 DFX dump 解析 workspace 分段
struct DfxWorkspaceLayoutInfo {
    uint64_t totalSize;                                // workspace 总大小
    uint32_t segCount;                                 // 实际段数
    uint32_t reserved;                                 // 对齐
    WorkspaceSegInfo segments[MAX_WORKSPACE_SEGMENTS]; // 段数组 (按 offset 升序)
};

// ==================== Peermem 段布局信息 ====================

constexpr uint32_t MAX_PEERMEM_SEGMENTS = 8U;

// Peermem 段类型
enum PeermemSegType : uint8_t {
    PEERMEM_SEG_DATA = 0,  // A data 区
    PEERMEM_SEG_SCALE = 1, // A scale 区
    PEERMEM_SEG_BIAS = 2,  // bias 区 (预留)
    PEERMEM_SEG_FLAG = 3,  // flag/counter 区 (预留)
    PEERMEM_SEG_RESERVED = 255,
};

// 单个段描述 (24 bytes)
struct PeermemSegInfo {
    uint64_t offset;     // 相对 peermem 窗口基地址的偏移 (字节)
    uint64_t size;       // 该段大小 (字节)
    uint8_t type;        // PeermemSegType
    uint8_t reserved[7]; // 对齐
};

// Peermem 布局信息 (208 bytes)
// peermem: 本卡通信窗口内存，供对端 rank 通过 URMA/UDMA (RDMA) 直接读取
struct DfxPeermemLayoutInfo {
    uint64_t totalSize;                            // peermem 总大小 (== peermemDataSize)
    uint32_t segCount;                             // 实际段数
    uint32_t reserved;                             // 对齐
    PeermemSegInfo segments[MAX_PEERMEM_SEGMENTS]; // 段数组 (按 offset 升序)
};

// ==================== DFX 统一头部 ====================

// 未开启 MC2_DFX_ENABLE 时退化为空结构体
struct DfxDumpInfo {
#if MC2_DFX_ENABLE
    DfxWorkspaceLayoutInfo workspaceLayout; // workspace 分段布局
    uint64_t peermemDataSize;               // peermem 窗口总大小
    DfxPeermemLayoutInfo peermemLayout;     // peermem 窗口分段布局
#endif                                      // MC2_DFX_ENABLE
};

// ==================== State Dump ====================

constexpr uint32_t STATE_DUMP_CORE_MAX = 128U;                                             // 最大核数
constexpr uint32_t STATE_DUMP_PER_CORE_SIZE = 512U;                                        // 每核 512 字节
constexpr uint32_t STATE_DUMP_TOTAL_SIZE = STATE_DUMP_CORE_MAX * STATE_DUMP_PER_CORE_SIZE; // 64KB

// 通信阶段枚举：仅 HCCL 五态状态机
enum RuntimePhase : uint8_t {
    RT_PHASE_COMM_INIT = 0,     // HCCL Init: 通信初始化
    RT_PHASE_COMM_PREPARE = 1,  // HCCL Prepare: 通信准备 (AllGather/AlltoAll prepare)
    RT_PHASE_COMM_COMMIT = 2,   // HCCL Commit: 通信提交 (下发描述符并激活)
    RT_PHASE_COMM_WAIT = 3,     // HCCL Wait: 等待通信完成 (hccl.Wait)
    RT_PHASE_COMM_FINALIZE = 4, // HCCL Finalize: 结束清理
    RT_PHASE_RESERVED = 255,
};

// 每个 core 的状态打点信息 (512 字节)
struct alignas(512) StateDumpPerCore {
    uint32_t magicNum;       // 4B  魔数 0x5A5A5A5A
    uint16_t coreId;         // 2B  core ID (C核: 0~aicCoreNum-1, V核: aicCoreNum~total-1)
    uint8_t execTurn;        // 1B  执行轮次: scale=1轮, 每 tile=1轮, 连续递增 (0=pipeline未启动)
    uint8_t execPosition;    // 1B  position tag: 0=COMM_BEFORE, 1=SCALE_COMM_BEFORE,
                             //     2=COMP_CUBE_MATMUL_BEFORE, 3=COMP_VEC_PERMUTE_BEFORE
    uint8_t commPhase;       // 1B  当前通信阶段 (RuntimePhase, 仅 HCCL 五态)
    uint8_t commCommitCount; // 1B  HCCL Commit 次数 (B层 communicator 递增)
    uint8_t commWaitCount;   // 1B  HCCL Wait 次数 (B层 communicator 递增)
    uint8_t reserved[501];   // 501B 预留
}; // 512 bytes
} // namespace Utils
#endif // OP_STATE_DUMP_STRUCT_H
