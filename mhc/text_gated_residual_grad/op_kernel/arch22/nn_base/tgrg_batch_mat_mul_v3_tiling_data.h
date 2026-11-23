/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file tgrg_batch_mat_mul_v3_tiling_data.h
 * \brief
 */

#ifndef TGRG_ARCH22_BATCH_MAT_MUL_V3_TILING_DATA_H
#define TGRG_ARCH22_BATCH_MAT_MUL_V3_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#include "tgrg_mat_mul_v3_tiling_data.h"

constexpr uint32_t kBATCH_MATMUL_TILING_ALIGN = 8;

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) MultiBatchInfo {
    uint32_t batchUsedCoreNum;
    uint32_t aBatchDimAll;
    uint32_t bBatchDimAll;
    uint32_t cBatchDimAll;
    // a0..a5 = tensor batch axes left-to-right (dim0=a0, nearest-M=a5). Missing axes pad 1.
    // aBatchDim0..5 are blob SLOT names, not axis numbers. Do not assign DimN = aN.
    //   tensor axis  a0  a1  a2  a3  a4  a5
    //   blob slot    D5  D4  D0  D1  D2  D3
    // Dim0..3 keep 6D offsets (8D's dim2..dim5). Dim4/5 are appended (8D dim1/dim0).
    uint32_t aBatchDim0;
    uint32_t bBatchDim0;
    uint32_t cBatchDim0;
    uint32_t aBatchDim1;
    uint32_t bBatchDim1;
    uint32_t cBatchDim1;
    uint32_t aBatchDim2;
    uint32_t bBatchDim2;
    uint32_t cBatchDim2;
    uint32_t aBatchDim3;
    uint32_t bBatchDim3;
    uint32_t cBatchDim3;
    uint32_t iterBatch;
    uint32_t biasWithBatch;
    uint32_t mOri;
    uint32_t batchTileBlock;
    uint32_t aBatch;
    uint32_t bBatch;
    // Appended 8D slots: D4=a1, D5=a0. See table above aBatchDim0.
    uint32_t aBatchDim4;
    uint32_t bBatchDim4;
    uint32_t cBatchDim4;
    uint32_t aBatchDim5;
    uint32_t bBatchDim5;
    uint32_t cBatchDim5;
};
#pragma pack(pop)

#pragma pack(push, 8)
constexpr uint32_t kVECTOR_TILING_ALIGN = 8;
struct alignas(kVECTOR_TILING_ALIGN) VectorTilingInfo {
    uint64_t coreNumber;
    uint64_t coreData;
    uint64_t copyLoop;
    uint64_t copyTail;
    uint64_t lastCopyLoop;
    uint64_t lastCopyTail;
    uint64_t rowsPerCore;
    uint64_t aTotalSize;
    uint64_t bTotalSize;
    uint64_t cTotalSize;
    uint64_t dimSizeSecondLast;
    uint64_t dimSizeLast;
    uint64_t alignedDimSizeLast;
    uint64_t bRowsPerBatch;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(kBATCH_MATMUL_TILING_ALIGN) BatchMatmulTilingData {
    MatmulTilingData matmulTiling;
    MultiBatchInfo multiBatchInfo;
    VectorTilingInfo vectorTilingInfo;
};
#pragma pack(pop)

#endif
