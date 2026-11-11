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
 * \file matmul_tiling_common.h
 * \brief Shared constants and runtime state for MATMUL tiling generation.
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "tiling/platform/platform_ascendc.h"

struct MatmulPlatformInfo {
    uint32_t aicNum{0};
    uint32_t aivNum{0};
    uint64_t ubSize{0};
    uint64_t l1Size{0};
    uint64_t l0aSize{0};
    uint64_t l0bSize{0};
    uint64_t l0cSize{0};
    uint64_t l2Size{0};
    uint64_t btSize{0};
    platform_ascendc::SocVersion socVersion{0};
};

struct MatmulArgs {
    uint64_t m{0};
    uint64_t n{0};
    uint64_t k{0};
    uint64_t dataTypeSize{0};
    bool hasBias{false};
    bool isATrans{false};
    bool isBTrans{false};
    bool isHf32{false};
};

struct MatMulV3TailInfo {
    uint64_t mCnt = 1UL;
    uint64_t nCnt = 1UL;
    uint64_t kCnt = 1UL;
    uint64_t mTailMain = 0UL;
    uint64_t nTailMain = 0UL;
};

struct MatmulRunInfo {
    uint64_t baseM{1};
    uint64_t baseN{1};
    uint64_t baseK{1};
    uint64_t singleCoreM{1};
    uint64_t singleCoreN{1};
    uint64_t singleCoreK{1};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint64_t usedCoreNum{1};
    uint64_t depthA1{1};
    uint64_t depthB1{1};
    uint64_t stepKa{1};
    uint64_t stepKb{1};
    uint64_t stepM{1};
    uint64_t stepN{1};
    uint64_t iterateOrder{0};
    uint64_t dbL0c{0};
    uint64_t l1BufferNum{2};
    MatMulV3TailInfo tailInfo;
    double defaultBalance{0.0};
    double redundantData{0.0};
};
