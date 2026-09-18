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
 * \file matmul_performance_arch35.h
 * \brief
 */
#ifndef __MATMUL_PERFORMANCE_ARCH35_H__
#define __MATMUL_PERFORMANCE_ARCH35_H__
#pragma once
#include "matmul_performance.h"

class MatmulPerformanceArch35 : public MatmulPerformanceModel {
public:
    // A5专用拟合模型:仅服务于DAV_3510平台,A5内蕴于类本身,不再接收档位key
    static constexpr uint64_t A5_BASE_BLOCK_M = 256; // A5最优base block
    static constexpr uint64_t A5_BASE_BLOCK_N = 256;
    static constexpr uint64_t A5_BASE_BLOCK_K = 128;

    explicit MatmulPerformanceArch35(const mc2tiling::TilingArgs &args)
        : MatmulPerformanceModel(args, Ops::Base::DAV_3510)
    {
        mmShapeInfo_.baseM = A5_BASE_BLOCK_M;
        mmShapeInfo_.baseN = A5_BASE_BLOCK_N;
        mmShapeInfo_.baseK = A5_BASE_BLOCK_K;
    }

    void FindCubeUtil(uint64_t rankTileNum)
    {
        double mnharmonicMean = static_cast<double>(mmShapeInfo_.mValue * rankTileNum * mmShapeInfo_.nValue) /
                                static_cast<double>(mmShapeInfo_.mValue * rankTileNum + mmShapeInfo_.nValue);
        double kExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(static_cast<double>(mmShapeInfo_.kValue), MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::KVALUE_THRESHOLD, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double mnExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(mnharmonicMean, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::MNVALUE_THRESHOLD, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double result = MatmulPerformance::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
        cubeUtil_ = std::min(MatmulPerformance::MAX_CUBE_UTIL, result);
    }
};

#endif // __MATMUL_PERFORMANCE_ARCH35_H__
