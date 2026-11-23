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
 * \file qgmm_inplace_add_pertensor_blaze.h
 * \brief HiFloat8 split-K grouped matmul with per-group scalar scales and FP32 inplace add.
 */
#pragma once

#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qgmm_cube.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "quant_grouped_matmul_inplace_add_tiling_data.h"

__aicore__ inline void QGmmInplaceAddPerTensorTensorApi(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale2, GM_ADDR groupList,
                                                        GM_ADDR scale1, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddBasicApiTilingData, gmmQuantParams, groupParams,
                           tiling);
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddBasicApiTilingData, mmTilingData, mmTiling,
                           tiling);
    if (groupParams.groupNum == 0 || mmTiling.m == 0 || mmTiling.n == 0 || mmTiling.k == 0) {
        return;
    }

    using Shape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::dn_ext_layout_ptn;
    using LayoutBC = asc::te::nd_ext_layout_ptn;
    using Policy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<
        Blaze::Gemm::NONE_FULL_LOAD_MODE, true, Blaze::Gemm::KernelGroupedMmadFixpipeQuant, __gm__ int32_t*,
        static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_NONE), false>;
    using BTypes = AscendC::Std::tuple<hifloat8_t, float>;
    using Mmad =
        Blaze::Gemm::Block::BlockMmad<Policy, hifloat8_t, LayoutA, BTypes, LayoutBC, float, LayoutBC, float, LayoutBC>;
    using QGmmKernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Blaze::Epilogue::Block::BlockEpilogueEmpty,
                                                          Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;

    constexpr uint32_t PER_TENSOR = static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERTENSOR_MODE);
    constexpr int8_t SPLIT_K = 2;
    constexpr uint8_t SINGLE_TENSOR = 1;
    QGmmKernel::Params params{};
    params.mmadParams.aGmAddr = x1;
    params.mmadParams.bGmAddr = x2;
    params.mmadParams.cGmAddr = y;
    params.mmadParams.scaleAGmAddr = scale1;
    params.mmadParams.scaleBGmAddr = scale2;
    params.groupListGmAddr = groupList;
    params.gmmParams = {groupParams.groupNum,
                        mmTiling.m,
                        mmTiling.n,
                        mmTiling.k,
                        mmTiling.baseM,
                        mmTiling.baseN,
                        mmTiling.baseK,
                        mmTiling.kAL1,
                        mmTiling.kBL1,
                        PER_TENSOR,
                        PER_TENSOR,
                        0,
                        static_cast<uint8_t>(mmTiling.dbL0C),
                        SPLIT_K,
                        groupParams.groupListType,
                        SINGLE_TENSOR,
                        SINGLE_TENSOR,
                        SINGLE_TENSOR};

    QGmmKernel op;
    op(params);
}
