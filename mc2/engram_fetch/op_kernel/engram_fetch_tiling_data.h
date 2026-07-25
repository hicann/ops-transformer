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

#include "kernel_tiling/kernel_tiling.h"

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
};
#endif
