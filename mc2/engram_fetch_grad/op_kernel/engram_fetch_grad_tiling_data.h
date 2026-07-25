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
};
#endif
