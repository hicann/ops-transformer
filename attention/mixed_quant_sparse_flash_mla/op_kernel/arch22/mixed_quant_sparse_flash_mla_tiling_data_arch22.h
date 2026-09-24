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
 * \file mixed_quant_sparse_flash_mla_tiling_data_arch22.h
 * \brief
 */

#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_TILING_DATA_ARCH22_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_TILING_DATA_ARCH22_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
struct MixedQuantSparseFlashMlaTqBaseParams {
    uint32_t batchSize;
    uint32_t qSeqSize;
    uint32_t kvSeqSize;
    int64_t paBlockSize;
    int64_t oriBlockSize;
    int64_t cmpBlockSize;
    uint32_t oriMaxBlockNumPerBatch;
    uint32_t nNumOfQInOneGroup;
    uint32_t actualLenDimsQ;
    uint32_t actualLenDimsKV;
    float softmaxScale;
    uint32_t outputLayout;
    uint64_t oriMaskMode;
    int64_t oriKvStride0;
    int64_t oriWinLeft;
    int64_t oriWinRight;
    int64_t sparseBlockSize;
    uint32_t usedCoreNum;
    uint32_t mmResUbSize;
    uint32_t bmm2ResUbSize;
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
    bool returnSoftmaxLse;
    uint32_t kvQuantMode;
};

struct MixedQuantSparseFlashMlaTqCmpParams {
    uint32_t cmpMaxBlockNumPerBatch;
    uint32_t sparseBlockCount;
    int64_t cmpRatio;
    uint64_t cmpMaskMode;
    int64_t cmpKvStride0;
};

struct MixedQuantSparseFlashMlaTqTilingData {
    MixedQuantSparseFlashMlaTqBaseParams tqBaseParams;
    MixedQuantSparseFlashMlaTqCmpParams tqCmpParams;
};
} // namespace optiling

#endif // MIXED_QUANT_SPARSE_FLASH_MLA_TILING_DATA_ARCH22_H
