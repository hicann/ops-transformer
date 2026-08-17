/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __LIGHTNING_INDEXER_KL_LOSS_TILLING_DATA_H__
#define __LIGHTNING_INDEXER_KL_LOSS_TILLING_DATA_H__

struct LightningIndexerKLLossTilingData {
    int64_t totalLength;   // 总行数 M = B*S (BSK) 或 T (TK)
    int64_t K;             // hidden size（最后一维长度）
    int64_t formerNum;     // 多处理一块的 Core 数量（formerCore）
    int64_t formerTileNum; // formerCore 处理的 tile 块数 = (totalTileNum + coreNum - 1) / coreNum
    int64_t tailTileNum;   // tailCore 处理的 tile 块数 = totalTileNum / coreNum
    int64_t tileLength;    // 每 tile 处理行数
    float eps;             // 数值稳定常数
    int64_t coreNum;       // AI Core 总数
    int64_t KAligned;      // K 向 8 对齐
};
#endif
