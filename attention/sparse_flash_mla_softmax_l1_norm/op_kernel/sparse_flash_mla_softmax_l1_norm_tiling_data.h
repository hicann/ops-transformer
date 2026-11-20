/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Kernel-side POD mirror of SmlaSoftmaxL1NormTilingData. The field order and
// widths must stay synchronized with the host tiling definition.
#ifndef SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_DATA_H
#define SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_DATA_H

#include <cstdint>

namespace SmlaL1Norm {
struct SmlaTilingPod {
    int64_t b;
    int64_t sq;
    int64_t sk;
    int64_t g;
    int64_t d;
    int64_t t1;
    int64_t t2;
    int64_t maxSeqlenK;
    int64_t kLength;
    int64_t cmpRatio;
    int64_t initPerCoreNum;
    int64_t initTotalNum;
    float softmaxScale;
    int64_t hasSequsedQ;
    int64_t hasSequsedK;
    int64_t hasTopkLength;
    int64_t totalRows;
    int64_t numCores;
    int64_t outRowLen;
    int64_t gHalfFirst;
    int64_t wsSlotFloats;
    int64_t maskMode;
    int64_t isTnd;
    int64_t isSparse;
    int64_t dtype;
    int64_t packRows;
};
} // namespace SmlaL1Norm

#endif // SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_DATA_H
