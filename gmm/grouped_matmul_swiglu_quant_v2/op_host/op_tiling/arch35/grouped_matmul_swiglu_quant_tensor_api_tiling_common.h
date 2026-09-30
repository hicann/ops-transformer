/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_TENSOR_API_TILING_COMMON_H
#define GROUPED_MATMUL_SWIGLU_QUANT_TENSOR_API_TILING_COMMON_H

#include <algorithm>
#include <cstdint>
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"

#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"

namespace optiling {
namespace GroupedMatmulSwigluQuantTensorApiTiling {

constexpr uint64_t K_LOWER_LIMIT = 64UL;
constexpr uint64_t FP8_N_ALIGN = 128UL;
constexpr uint64_t K_ALIGN = 128UL;
constexpr uint64_t TRANS_B_M_PER_GROUP_LOWER_LIMIT = 128UL;
constexpr uint64_t NOT_TRANS_B_M_PER_GROUP_LOWER_LIMIT = 512UL;
constexpr uint64_t NOT_TRANS_B_M_LOWER_LIMIT = 512UL;
constexpr uint64_t AIC_AIV_CORE_RATIO = 2UL;
constexpr uint64_t BASE_M = 256UL;
constexpr uint64_t BASE_N = 128UL;
constexpr uint64_t SWIGLU_SPLIT = 2UL;

struct CapabilityParams {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t groupNum;
    uint64_t aicNum;
    uint64_t aivNum;
    bool transB;
    bool platformMemoryReady;
};

inline bool IsShapeAndPlatformCapable(const CapabilityParams& params)
{
    if (!params.platformMemoryReady || params.groupNum == 0UL || params.aicNum == 0UL ||
        params.aivNum != AIC_AIV_CORE_RATIO * params.aicNum) {
        return false;
    }
    const uint64_t averageM = params.m / params.groupNum;
    const uint64_t mPerGroupLimit =
        params.transB ? TRANS_B_M_PER_GROUP_LOWER_LIMIT : NOT_TRANS_B_M_PER_GROUP_LOWER_LIMIT;
    return averageM > mPerGroupLimit && (params.transB || params.m > NOT_TRANS_B_M_LOWER_LIMIT) &&
           params.n % FP8_N_ALIGN == 0UL && params.k > K_LOWER_LIMIT && params.k % K_ALIGN == 0UL;
}

inline void PrepareBasicBlockForL1(GQmmBasicTiling& tiling)
{
    tiling.baseM = GroupedMatmul::CeilAlign(std::min(tiling.baseM, BASE_M), GmmConstant::CUBE_BLOCK);
    tiling.baseN = GroupedMatmul::CeilAlign(std::min(tiling.baseN, BASE_N), GmmConstant::CUBE_BLOCK);
    tiling.baseN *= SWIGLU_SPLIT;
}

inline void RestoreBasicBlockAfterL1(GQmmBasicTiling& tiling)
{
    tiling.baseN /= SWIGLU_SPLIT;
}

inline void FillQuantParams(GroupedMatmulSwigluQuantV2TensorApi::GMMTensorApiQuantParams& params,
                            const GQmmInputInfo& input)
{
    params = {};
    params.groupNum = static_cast<uint32_t>(input.groupNum);
    params.activeType = static_cast<uint32_t>(input.actType);
    params.aQuantMode = static_cast<uint32_t>(input.aQuantMode);
    params.bQuantMode = static_cast<uint32_t>(input.bQuantMode);
    params.singleX = static_cast<uint8_t>(input.isSingleX);
    params.singleW = static_cast<uint8_t>(input.isSingleW);
    params.singleY = static_cast<uint8_t>(input.isSingleY);
    params.groupType = static_cast<int8_t>(input.groupType);
    params.groupListType = static_cast<uint8_t>(input.groupListType);
}

inline void FillMatmulTiling(GroupedMatmulSwigluQuantV2TensorApi::GMMTensorApiMMTiling& mm, const GQmmInputInfo& input,
                             const GQmmBasicTiling& basic, uint64_t scaleFactorA, uint64_t scaleFactorB)
{
    mm = {};
    mm.m = static_cast<uint32_t>(input.mSize);
    mm.n = static_cast<uint32_t>(input.nSize);
    mm.k = static_cast<uint32_t>(input.kSize);
    mm.baseM = static_cast<uint32_t>(basic.baseM);
    mm.baseN = static_cast<uint32_t>(basic.baseN);
    mm.baseK = static_cast<uint32_t>(basic.baseK);
    mm.kAL1 = static_cast<uint32_t>(basic.stepKa * basic.baseK);
    mm.kBL1 = static_cast<uint32_t>(basic.stepKb * basic.baseK);
    const uint64_t scaleKL1 =
        std::min(std::max(scaleFactorA * basic.stepKa, scaleFactorB * basic.stepKb) * basic.baseK, input.kSize);
    mm.scaleKAL1 = static_cast<uint32_t>(scaleKL1);
    mm.scaleKBL1 = static_cast<uint32_t>(scaleKL1);
    mm.dbL0C = static_cast<uint8_t>(basic.dbL0c);
}

} // namespace GroupedMatmulSwigluQuantTensorApiTiling
} // namespace optiling

#endif // GROUPED_MATMUL_SWIGLU_QUANT_TENSOR_API_TILING_COMMON_H
