/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_tensor_api_swiglu_quant_v3_a8w4_kernel.h
 * \brief V3 MX A8W4 WeightNZ TensorAPI entry with optional expert-specific smooth scale.
 */
#pragma once

#include "kernel_tiling/kernel_tiling.h"
#include "blaze/gemm/block/block_mmad_weight_prologue_mx.h"
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "../grouped_matmul_swiglu_quant_v2_utils_kernel.h"
#include "grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"

namespace GroupedMatmulSwigluQuantV3A8W4 {
using TilingData = GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData;

template <typename Params>
__aicore__ inline void SetInputParams(Params& params, const TilingData& tilingData, GM_ADDR x, GM_ADDR xScale,
                                      GM_ADDR weight, GM_ADDR weightScale, GM_ADDR groupList)
{
    params.x = x;
    params.xScale = xScale;
    params.weight =
        tilingData.gmmQuantParams.singleW == 1 ?
            reinterpret_cast<GM_ADDR>(GroupedMatmulDequantSwigluQuant::GetTensorAddr<DTYPE_WEIGHT>(0, weight)) :
            weight;
    params.weightScale = tilingData.gmmQuantParams.singleW == 1 ?
                             reinterpret_cast<GM_ADDR>(
                                 GroupedMatmulDequantSwigluQuant::GetTensorAddr<AscendC::fp8_e8m0_t>(0, weightScale)) :
                             weightScale;
    params.groupList = groupList;
    params.m = tilingData.mmTilingData.m;
    params.n = tilingData.mmTilingData.n;
    params.k = tilingData.mmTilingData.k;
    params.groupNum = tilingData.gmmQuantParams.groupNum;
    params.groupListType = tilingData.gmmQuantParams.groupListType;
    params.scheduler.baseM = tilingData.mmTilingData.baseM;
    params.scheduler.baseN = tilingData.mmTilingData.baseN;
}

template <typename Params>
__aicore__ inline void SetEpilogueParams(Params& params, const TilingData& tilingData, GM_ADDR y, GM_ADDR yScale,
                                         GM_ADDR smoothScale)
{
    params.epilogue.yGmAddr = y;
    params.epilogue.yScaleGmAddr = yScale;
    params.epilogue.swigluMode = tilingData.swigluParams.swigluMode;
    params.epilogue.clampLimit = tilingData.swigluParams.clampLimit;
    params.epilogue.gluAlpha = tilingData.swigluParams.gluAlpha;
    params.epilogue.gluBias = tilingData.swigluParams.gluBias;
    params.epilogue.scaleAlg = tilingData.swigluParams.scaleAlg;
    params.epilogue.dstTypeMax = tilingData.swigluParams.dstTypeMax;
    params.epilogue.smoothScaleGmAddr = smoothScale;
}
} // namespace GroupedMatmulSwigluQuantV3A8W4

template <bool TransposeWeight>
__aicore__ inline void GmmTensorApiSwigluQuantV3A8W4Kernel(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                                           GM_ADDR xScale, GM_ADDR smoothScale, GM_ADDR groupList,
                                                           GM_ADDR y, GM_ADDR yScale, GM_ADDR tiling)
{
    static_assert(TransposeWeight, "MX A8W4 SwiGLU only supports transposed WeightNZ");
    GET_TILING_DATA_WITH_STRUCT(GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData, tilingData,
                                tiling);
    using ScaleType = AscendC::fp8_e8m0_t;
    using WeightLayout = asc::te::zn_layout_ptn;
    using WeightScaleLayout = asc::te::scaleb_dn_layout_ptn;
    using Mmad = Blaze::Gemm::Block::BlockMmad<
        Blaze::Gemm::GroupedMatmulSwigluWithWeightQuantMx, AscendC::Std::tuple<DTYPE_X, ScaleType>,
        AscendC::Std::tuple<asc::te::nd_ext_layout_ptn, asc::te::scalea_nd_layout_ptn>,
        AscendC::Std::tuple<DTYPE_WEIGHT, ScaleType>, AscendC::Std::tuple<WeightLayout, WeightScaleLayout>, float,
        asc::te::nd_ext_layout_ptn, bfloat16_t, asc::te::nd_ext_layout_ptn>;
    using Prologue = Blaze::Gemm::Kernel::GroupedWeightPrologueMx<DTYPE_X, DTYPE_WEIGHT, bfloat16_t, true>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<DTYPE_Y, float, ScaleType, false, true, true>;
    using Kernel = Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<asc::te::shape<int64_t, int64_t, int64_t, int64_t>, Mmad,
                                                               Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit,
                                                               Epilogue, Prologue, true>;
    typename Kernel::Params params;
    GroupedMatmulSwigluQuantV3A8W4::SetInputParams(params, tilingData, x, xScale, weight, weightScale, groupList);
    GroupedMatmulSwigluQuantV3A8W4::SetEpilogueParams(params, tilingData, y, yScale, smoothScale);
    Kernel kernel;
    kernel(params);
}
