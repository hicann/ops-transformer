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

#include <cstdint>
#include "arch35/railaligned/engram_fetch_grad_route.h"

namespace Mc2Kernel {

constexpr uint32_t UB_ALIGN = 32U;                   // UB 32B 对齐
constexpr uint32_t TILE_BYTES = 64U * 1024U;         // grad 单 tile 整缓冲容量
constexpr uint32_t GRAD_PING_BYTES = 32U * 1024U;    // grad ping/pong 半缓冲容量
constexpr uint32_t COMM_BUF_BYTES = 2U * TILE_BYTES; // entryBuf + gradBuf
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t STATE_OFFSET = 32U;         // 通信窗口状态槽步长
constexpr uint32_t ENTRY_BUF_INT32_SLOTS = 5U; // entryBuf 头部被 int32 槽位占用的批数
constexpr uint32_t ACCUM_BUF_COPIES = 2U;      // fp32 累加行双缓冲份数（AccumBufBytes）
constexpr uint32_t FLUSH_CAST_HEAD_BYTES =
    static_cast<uint32_t>(ENTRY_BUF_INT32_SLOTS * ENTRY_BATCH_CAP * sizeof(int32_t));
constexpr uint32_t HIDDEN_CHUNK_ELEMS = 2048U;
constexpr uint32_t A2A_COPY_HALF_BYTES = 32U * 1024U;
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
    uint32_t sortNumTileData;  // SortLib: 每 tile 元素数
    uint32_t sortTileCount;    // SortLib: tile 数
    uint32_t sortTmpUbSize;    // SortLib: AscendC::Sort 临时 UB 字节
    uint32_t sortCoreCnt;      // sort 参与核数
    uint32_t chunkElems;
    uint32_t ranksPerSupernode; // 每超节点卡数（感知透传）
    uint32_t numRows;           // N：收件区行格数（按 commBufferSize 分满反解，非 2 的幂）
    uint32_t batchRows;         // 发送批行数 ≈ 64KB / P
    uint32_t sendCoreCnt;       // S 发送组核数
    uint32_t fwdCoreCnt;        // F 转发组核数（M/R 合并后的统一转发循环）
    uint32_t ownerCoreCnt;      // O 收集组核数
};

namespace Mc2Kernel {

constexpr uint32_t CTX_RAIL_ELEMS = RAIL_CTX_TOTAL_BYTES / 4U; // 平铺 rail ctx int32 数（头部 + node 表两张）
constexpr uint32_t CTX_LEGACY_ELEMS = 16400U / 4U;             // 4100

} // namespace Mc2Kernel
#endif
