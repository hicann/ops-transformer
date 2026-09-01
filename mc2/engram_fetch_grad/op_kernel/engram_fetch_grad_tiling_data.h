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
 * \file engram_fetch_grad_tiling_data.h
 * \brief kernel侧tiling data结构 — EngramFetchGrad
 */

#ifndef ASCENDC_ENGRAM_FETCH_GRAD_TILING_H
#define ASCENDC_ENGRAM_FETCH_GRAD_TILING_H

#include "kernel_tiling/kernel_tiling.h"

// Host/Kernel 共享布局常量（单一权威定义）。
// Host 侧 op_tiling 与 Kernel 侧 utils/arch35/unique 均从此处引用，禁止在其它文件重复定义；
// 修改任一常量必须同步评估两侧 UB/GM 预算（TIL-2/SIMT-UB-01/CG-4.3 根治措施）。
namespace Mc2Kernel {
constexpr uint32_t MAX_QP_SIZE = 1024U;           // 通信域 rank 上界（EngramCommContext.commBuffer 容量）
constexpr uint32_t UB_ALIGN = 32U;                // UB 32B 对齐
constexpr uint32_t TILE_BYTES = 64U * 1024U;      // grad 单 tile 整缓冲容量
constexpr uint32_t GRAD_PING_BYTES = 32U * 1024U; // grad ping/pong 半缓冲容量
constexpr uint32_t ENTRY_BUF_BYTES = 64U * 1024U;
constexpr uint32_t GRAD_BUF_BYTES = 64U * 1024U;
constexpr uint32_t IDX_BUF_BYTES = 4U * 1024U;
constexpr uint32_t COMM_BUF_BYTES = 2U * TILE_BYTES; // entryBuf + gradBuf
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t STATE_OFFSET = 32U;         // 通信窗口状态槽步长
constexpr uint32_t ENTRY_BATCH_CAP = 1024U;    // entry 批容量（int32 个数）
constexpr uint32_t ENTRY_BUF_INT32_SLOTS = 5U; // entryBuf 头部被 int32 槽位占用的批数
constexpr uint32_t ACCUM_BUF_COPIES = 2U;      // fp32 累加行双缓冲份数（AccumBufBytes）
constexpr uint32_t FLUSH_CAST_HEAD_BYTES =
    static_cast<uint32_t>(ENTRY_BUF_INT32_SLOTS * ENTRY_BATCH_CAP * sizeof(int32_t));
} // namespace Mc2Kernel

struct EngramFetchGradTilingData {
    int64_t numTokens;         // gradFetched / perm dim0
    int32_t numEntriesPerRank; // 每 rank entry 数
    int64_t hiddenDim;         // hidden 维度（从 gradFetched dim1 获取）
    int64_t hiddenBytes;       // hiddenDim * sizeof(inputDtype)（a2a 交换 stride）
    uint32_t aivNum;           // AIV 核数
    uint64_t ubSize;           // UB 空间
    uint32_t rankSize;         // 通信域 rank 数（从 sendCounts dim0 获取）
    int64_t totalRecv;         // recvLocalEntry dim0（a2a 接收上界）
    int64_t commBufferSize;    // a2a GM 收发缓冲大小（即 commBuffer 200MB）
    int32_t inputDtype;        // gradFetched 的 dtype（ge::DataType）
    int32_t outputDtype;       // gradUniqueOut 的 dtype（ge::DataType）
    uint32_t gradSubBatch;     // unique 阶段每批处理的 grad 数量
};
#endif
