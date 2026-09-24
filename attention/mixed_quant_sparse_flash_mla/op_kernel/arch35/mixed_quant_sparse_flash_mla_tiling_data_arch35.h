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
 * \file mixed_quant_sparse_flash_mla_tiling_data_arch35.h
 * \brief
 */

#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_ARCH35_TILING_DATA_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_ARCH35_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
struct MixedQuantSparseFlashMlaBaseParams {
    uint32_t batchSize;
    uint32_t qSeqSize;
    uint32_t kvSeqSize;
    uint32_t cmpKvSeqSize;
    uint32_t paOriBlockSize;
    uint32_t paCmpBlockSize;
    uint32_t oriMaxBlockNumPerBatch;
    uint32_t cmpMaxBlockNumPerBatch;
    uint32_t nNumOfQInOneGroup;
    uint32_t oriSparseBlockCount;
    uint32_t cmpSparseBlockCount;
    float softmaxScale;
    int32_t oriKvStride;
    int32_t cmpKvStride;
    uint32_t tileSize;
    uint32_t ropeHeadDim;
    uint32_t cmpRatio;
    uint32_t oriMaskMode;
    uint32_t cmpMaskMode;
    int32_t oriWinLeft;
    int32_t oriWinRight;
    uint32_t sparseBlockSize;
    uint32_t dSize;
    uint32_t dSizeVInput;
    uint32_t returnSoftmaxLse;
};

struct MixedQuantSparseFlashMlaTilingData {
    MixedQuantSparseFlashMlaBaseParams baseParams;
};
} // namespace optiling

#endif // MIXED_QUANT_SPARSE_FLASH_MLA_ARCH35_TILING_DATA_H
