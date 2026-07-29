/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STEM_OAM_PREP_VARLEN_Q_TILING_DATA_H
#define STEM_OAM_PREP_VARLEN_Q_TILING_DATA_H

#include <cstdint>

struct StemPrepQTilingData {
    uint32_t batchSize;
    uint32_t numQHeads;
    uint32_t dimQk;
    uint32_t stemBlockSize;
    uint32_t stemStride;
    uint32_t rVal;
    uint32_t kflatDim;
    uint32_t maxQb;
    uint32_t totalTokens;
    uint32_t usedCoreNum;
    uint32_t blocksPerCoreBase;
    uint32_t blocksRemainder;
    uint32_t ubFactor;
};

#endif
