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
 * \file all_gather_matmul_v2_tiling.h
 * \brief
 */

#ifndef __ALL_GATHER_MATMUL_AIV_MODE_TILING_H__
#define __ALL_GATHER_MATMUL_AIV_MODE_TILING_H__

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"

constexpr static int32_t MAX_BLOCK_COUNT = 2;

enum DequantType : int {
    DEQUANT_TYPE_UNDEFINED = -1,
    PER_CHANNEL = 0,
    PER_TOKEN = 1,
    DEQUANT_TYPE_MAX = 2,
};

#pragma once
struct AllGatherMatmulAIVModeInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t worldSize;
    uint32_t rankId;
    uint32_t aivNum;
    uint32_t totalUbSize;
    uint64_t aAlignSize;
    uint64_t bAlignSize;
    bool isTransposeX1;
    bool isTransposeX2;
    bool hasAAlign;
    bool hasBAlign;
    bool quantFlag;
    bool is910C;
    bool isX2ScaleTypeInt64;
    bool accumWorkSpacePingPong;
    DequantType dequantType;
};

struct CoCTiling {
    int32_t m = -1;
    int32_t k = -1;
    int32_t n = -1;
    int32_t numBlocks = -1;
    int32_t rankSize = -1;

    int32_t m0 = -1;
    int32_t k0 = -1;
    int32_t n0 = -1;
    int32_t mLoop = -1;
    int32_t kLoop = -1;
    int32_t nLoop = -1;
    int32_t swizzlCount = -1;
    int32_t swizzlDirect = -1;
    int32_t pValue = -1;
    int32_t ubMoveNum = -1;
    int32_t commNpuSplit = -1;
    int32_t commDataSplit = -1;
    int32_t commDirect = -1;
    int32_t lenPerLoop = -1;
    int32_t extraUbMoveNum = -1;
    int32_t extraCommNpuSplit = -1;  // 2dtp使用
    int32_t extraCommDataSplit = -1; // 2dtp使用
    int32_t extraCommDirect = -1;    // 2dtp使用
    int32_t extraLenPerLoop = -1;    // 2dtp使用
    int32_t splitK = -1;
    int32_t write2OtherRank = -1;
    int32_t withSerialMode = -1;

    int32_t is91093 = -1;
};

static_assert(std::is_standard_layout_v<CoCTiling>);
static_assert(offsetof(CoCTiling, m0) == 5 * sizeof(int32_t));
static_assert(offsetof(CoCTiling, commDataSplit) == 16 * sizeof(int32_t));
static_assert(offsetof(CoCTiling, is91093) == 27 * sizeof(int32_t));
static_assert(sizeof(CoCTiling) == 28 * sizeof(int32_t));

struct AllGatherMatmulAIVModeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    AllGatherMatmulAIVModeInfo allGatherMatmulInfo;
    CoCTiling cocTiling;
};

#endif //__ALL_GATHER_MATMUL_AIV_MODE_TILING_H__
