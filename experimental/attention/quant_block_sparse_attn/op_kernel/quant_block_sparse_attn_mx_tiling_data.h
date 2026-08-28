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
 * \file quant_block_sparse_attn_mx_tiling_data.h
 * \brief host tiling 与 MX kernel 共享的 MXFP8 全量化 tiling data。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_TILING_DATA_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_TILING_DATA_H_

#include <cstdint>

namespace optiling {
// MX 使用独立 struct tiling data，与 FP8 路径隔离。
constexpr uint8_t MX_PSCALE_DTYPE_E8M0 = 0U;
constexpr uint8_t MX_PSCALE_DTYPE_FP32 = 1U;
struct QuantBlockSparseAttnMxStrideParams {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;
};

struct QuantBlockSparseAttnMxAttrParams {
    uint32_t layoutQ = 0;
    uint32_t layoutKv = 0;
    uint32_t layoutSparseIndices = 0;
    uint32_t quantMode = 0;
    uint32_t maskMode = 0;
    uint32_t returnSoftmaxLse = 0;
};

struct QuantBlockSparseAttnMxBaseParams {
    uint32_t bSize = 0;
    uint32_t t1Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dSizeRope = 0;
    uint32_t actualSeqLengthsQSize = 0;
    uint32_t actualSeqLengthsKVSize = 0;
    float scaleValue = 0.0F;
    uint8_t isKvContinuous = 0;
    uint8_t isActualSeqLengthsNull = 0;
    uint8_t isActualSeqLengthsKVNull = 0;
    uint32_t coreNum = 0;
    uint32_t outputLayout = 0;
    // PA BNBD stride，单位为元素。K/V 数据和 K/V scale 均为 [block,N,row,lastDim]。
    QuantBlockSparseAttnMxStrideParams keyStrides;
    QuantBlockSparseAttnMxStrideParams valueStrides;
    QuantBlockSparseAttnMxStrideParams kScaleStrides;
    QuantBlockSparseAttnMxStrideParams vScaleStrides;
};

struct QuantBlockSparseAttnMxAttenMaskParams {
    uint8_t sparseMode = 0;
    uint8_t attenMaskDataType = 0;
    uint8_t attenMaskCompressMode = 0;
    uint8_t isRowInvalidOpen = 0;
    int32_t preTokens = 0;
    int32_t nextTokens = 0;
    uint32_t attenMaskBatch = 0;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    uint8_t isExistRowInvalid = 0;
    uint8_t reserve[3] = {0, 0, 0};
};

struct QuantBlockSparseAttnMxPageAttentionParams {
    uint8_t paLayoutType = 0;
    uint8_t reserve[3] = {0, 0, 0};
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t paBlockNumSum = 0;
    uint32_t paBlockStride = 0;
    uint32_t qBlockSize = 0;
    uint32_t kvBlockSize = 0;
};

struct QuantBlockSparseAttnMxSparseParams {
    uint32_t gS1OuterSize = 0;
    uint32_t sparseSeqLenStride = 0;
    uint32_t sparseIndicesStride = 0;
    uint32_t maxQb = 0;
    uint32_t maxKb = 0;
    uint32_t sparseCount = 0;
};

struct QuantBlockSparseAttnMxWorkspaceParams {
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

struct QuantBlockSparseAttnMxScaleParams {
    // PScale:[1]；QScale:[T,N,D/64,2]；
    // KScale:[blockNum,N,blockSize,D/64,2]；VScale:[blockNum,N,blockSize/64,DV,2]。
    uint32_t scaleGroupSize = 0;
    uint32_t scaleLastDim = 0;
    uint32_t queryScaleDSize = 0;
    uint32_t keyScaleDSize = 0;
    uint32_t valueScaleBlockSize = 0;
    uint32_t valueScaleDSize = 0;
    uint32_t pScaleShapeSize = 0;
    uint8_t pScaleDtype = 0; // 0=E8M0, 1=FP32
    uint8_t reserve[3] = {0, 0, 0};
    uint32_t queryQuantMode = 0;
    uint32_t keyAntiquantMode = 0;
    uint32_t valueAntiquantMode = 0;
};

struct QuantBlockSparseAttnMxEmptyTensorParams {
    uint64_t totalOutputSize = 0;
    uint64_t totalSoftMaxLseOutputSize = 0;
};

struct QuantBlockSparseAttnMxTilingData {
    // 仅由 MXFullQuantMode 填充。
    QuantBlockSparseAttnMxAttrParams attrParams;
    QuantBlockSparseAttnMxBaseParams baseParams;
    QuantBlockSparseAttnMxAttenMaskParams attenMaskParams;
    QuantBlockSparseAttnMxPageAttentionParams pageAttentionParams;
    QuantBlockSparseAttnMxSparseParams sparseParams;
    QuantBlockSparseAttnMxWorkspaceParams workspaceParams;
    QuantBlockSparseAttnMxScaleParams scaleParams;
    QuantBlockSparseAttnMxEmptyTensorParams emptyTensorParams;
};
} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_TILING_DATA_H_
