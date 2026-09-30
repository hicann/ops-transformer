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
    int64_t batchSize = 0;
    int64_t numQHeads = 0;
    int64_t dimQk = 0;
    int64_t stemBlockSize = 0;
    int64_t stemStride = 0;
    int64_t rVal = 0;
    int64_t kflatDim = 0;
    int64_t maxQb = 0;
    int64_t totalTokens = 0;
    int64_t usedCoreNum = 0;
    int64_t blocksPerCoreBase = 0;
    int64_t blocksRemainder = 0;
    int64_t ubFactor = 0;
};

#endif
