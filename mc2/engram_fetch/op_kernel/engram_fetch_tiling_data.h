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
 * \file engram_fetch_tiling_data.h
 * \brief kernel侧tiling data结构
 */

#ifndef ASCENDC_ENGRAM_FETCH_TILING_H
#define ASCENDC_ENGRAM_FETCH_TILING_H

#include <cstdint>

struct EngramFetchTilingData {
    int64_t numTokens;
    int32_t numEntriesPerRank;
    int64_t hiddenDim;
    int64_t hiddenBytes;
    uint32_t aivNum;
    uint64_t ubSize;
    uint32_t rankSize;
    int64_t numMaxTokensPerRank;
    int64_t totalRecv;
    int64_t commBufferSize;
    int64_t numSfPacks;
    int64_t sfElemSize;
    uint32_t sortNumTileData;
    uint32_t sortTileCount;
    uint32_t sortTmpUbSize;
    // ═══ 分轨模型追加字段（append-only：老字段偏移零变化，老模板只读前缀）═══
    // 感知只取一数：ranksPerSupernode=每超节点卡数（直连域），host HCCL topo 感知经可选 attr
    // 自动透传；超节点数=world÷ranksPerSupernode、轨道对端数=world÷ranksPerSupernode−1 全推导。
    // 跨超节点统一轨道腿（dst 超节点的同 L 卡）+ 超节点内 CTP 直投，恒两跳。
    uint32_t ranksPerSupernode; // 每超节点卡数（感知透传）
    uint32_t numRows;           // N：两族共用收件区行格数（commBufferSize 分满反解，非 2 的幂）
    uint32_t batchRowsReq;      // req 族发送批行数 = min(64KB/P_req, N/2)
    uint32_t batchRowsResp;     // resp 族发送批行数 = min(64KB/P_resp, N/2)
    uint32_t sendCoreCnt;       // S 统一发送组核数
    uint32_t fwdCoreCnt;        // F 消费+暂存组核数
    uint32_t ownerCoreCnt;      // O 收集组核数
};
#endif
