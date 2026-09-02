/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_tiling_data_apt.h
 * \brief
 */
#ifndef GROUPED_MATMUL_TILING_DATA_H
#define GROUPED_MATMUL_TILING_DATA_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace GroupedMatmulTilingData {
#pragma pack(push, 8)
struct GMMArray {
    // GroupedMatmul::MAX_TENSOR_CONT
    int32_t mList[128] = {0};
    int32_t kList[128] = {0};
    int32_t nList[128] = {0};
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMNoQuantBaseParams {
    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint32_t singleWeight = 0;
    uint32_t singleX = 0;
    uint32_t singleY = 0;
    int32_t groupType = 0;
    uint32_t groupListType = 0;
    uint32_t hasBias = 0;
    uint32_t mTailCnt = 0;
    uint32_t nTailCnt = 0;
    uint32_t weightNoL2Cache = 0;
    uint32_t placeHolder = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMQuantParams {
    uint32_t groupNum = 0;
    uint32_t activeType = 0;
    uint32_t aQuantMode = 0;
    uint32_t bQuantMode = 0;
    uint8_t singleX = 0;
    uint8_t singleW = 0;
    uint8_t singleY = 0;
    int8_t groupType = 0;
    uint8_t groupListType = 0;
    uint8_t hasBias = 0;
    uint16_t reserved = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct QuantBasicApiMMTiling {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t kAL1 = 0;
    uint32_t kBL1 = 0;
    uint32_t scaleKAL1 = 0;
    uint32_t scaleKBL1 = 0;
    uint8_t isBias = 0;
    uint8_t dbL0C = 0;
    uint8_t l1BufferStage = 0;
    uint8_t reserved1 = 0;
    uint32_t reserved2 = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMWeightQuantParam {
    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint8_t singleX = 0;
    uint8_t singleWeight = 0;
    uint8_t singleY = 0;
    int8_t groupType = 0;
    uint8_t groupListType = 0;
    uint8_t hasBias = 0;
    uint8_t cubeNumBlocksN = 0;
    uint8_t reserved = 0;
    uint32_t groupSize = 0;
    uint32_t mainBlockSize = 0;
    uint64_t mainBlockCount = 0;
    uint16_t firstTailBlockSize = 0;
    uint16_t secondTailBlockSize = 0;
    uint16_t firstTailBlockCount = 0;
    uint16_t secondTailBlockCount = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMQuantTilingData {
    GMMQuantParams gmmQuantParams;
    GMMArray gmmArray;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMQuantBasicApiTilingData {
    GMMQuantParams gmmQuantParams;
    QuantBasicApiMMTiling mmTilingData;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMS8S4BasicApiParams {
    uint32_t quantGroupSize = 0;
    uint32_t quantGroupNum = 0;
    uint32_t expectedTokenNum = 0;
    uint32_t coreNum = 0;
    uint8_t dequantMode = 0; // 0: symmetric per-group, 1: asymmetric per-channel
    uint8_t hasOffset = 0;
    uint8_t specialWeightFormat = 0;
    uint8_t weightPackedInt32 = 0;
    uint8_t enableWeightPreprocess = 0;
    uint8_t reserved[3] = {0};
    uint64_t expandedWeightOffsetBytes = 0;
    uint64_t expandedWeightSizeBytes = 0;
    uint64_t expandedWeightStrideBytes = 0;
    uint64_t tileWorkspaceOffsetBytes = 0;
    uint64_t tileWorkspaceSizeBytes = 0;
    uint64_t tileWorkspaceStrideBytes = 0;
    uint64_t rowSumOffsetBytes = 0;
    uint64_t rowSumSizeBytes = 0;
    uint64_t userWorkspaceSizeBytes = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMS8S4BasicApiTilingData {
    GMMQuantParams gmmQuantParams;
    GMMS8S4BasicApiParams s8s4Params;
    QuantBasicApiMMTiling mmTilingData;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMNoQuantTilingData {
    GMMNoQuantBaseParams gmmNoQuantParam;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMWeightQuantTilingData {
    GMMWeightQuantParam gmmWeightQuantParam;
    GMMArray gmmArray;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

// A5 S4S4 (INT4×INT4) mix-core tiling：AIV prologue(int4->int8) + AIC int8×int8 Mmad + AIV epilogue
#pragma pack(push, 8)
struct GMMBaseParamsS4S4 {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t ubCalSize = 0;       // epilogue 单次计算元素数
    uint32_t ubRestBytes = 0;     // 留给 prologue 的 UB (byte)
    uint32_t quantGroupNum = 1;   // perchannel=1, pergroup=k/256
    uint32_t singleN = 0;         // perchannel 动态分块
    uint32_t isPerTokenQuant = 0; // perTokenScale 是否存在
    uint32_t isS4S4Optimize = 0;
    int32_t groupType = 0;
    uint32_t groupListType = 0;
    uint64_t reserved = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMS4S4IntQuantTilingData {
    GMMBaseParamsS4S4 gmmS4S4Params;
    GMMArray gmmArray;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

} // namespace GroupedMatmulTilingData
#endif
