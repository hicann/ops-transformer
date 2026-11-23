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
 * \file qgmm_inplace_add_mx_blaze.h
 * \brief MXFP8 and MXFP4 grouped matmul with FP32 inplace accumulation.
 */
#pragma once

#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"
#include "quant_grouped_matmul_inplace_add_tiling_data.h"

template <typename AType, typename BType>
__aicore__ inline void QGmmInplaceAddTensorApi(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale2, GM_ADDR groupList,
                                               GM_ADDR scale1, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddBasicApiTilingData, gmmQuantParams, groupParams,
                           tiling);
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddBasicApiTilingData, mmTilingData, mmTiling,
                           tiling);
    const uint32_t m = mmTiling.m;
    const uint32_t n = mmTiling.n;
    const uint32_t k = mmTiling.k;
    const uint32_t kAL1 = mmTiling.kAL1;
    const uint32_t kBL1 = mmTiling.kBL1;
    const uint32_t scaleKL1 = mmTiling.scaleKAL1;
    if (groupParams.groupNum == 0 || m == 0 || n == 0 || k == 0) {
        return;
    }
    constexpr uint64_t FULL_LOAD_DISABLED = 0UL;
    constexpr uint8_t NO_BIAS = 0;
    constexpr int8_t SPLIT_K = 2;
    constexpr uint8_t SINGLE_WEIGHT = 1;
    constexpr int64_t NO_BATCH = 0;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<FULL_LOAD_DISABLED, true>;
    using LayoutA = asc::te::dn_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, float, LayoutC, float, LayoutC>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Epilogue, Scheduler>;

    typename Kernel::Params params{};
    params.problemShape = {static_cast<int64_t>(m), static_cast<int64_t>(n), static_cast<int64_t>(k), NO_BATCH};
    params.mmadParams = {x1, x2, y, nullptr, scale1, scale2};
    params.groupListGmAddr = groupList;
    params.gmmParams = {groupParams.groupNum,
                        static_cast<int64_t>(m),
                        static_cast<int64_t>(n),
                        static_cast<int64_t>(k),
                        mmTiling.baseM,
                        mmTiling.baseN,
                        mmTiling.baseK,
                        kAL1,
                        kBL1,
                        scaleKL1,
                        scaleKL1,
                        NO_BIAS,
                        mmTiling.dbL0C,
                        Blaze::Gemm::DOUBLE_BUFFER_COUNT,
                        SPLIT_K,
                        groupParams.groupListType,
                        SINGLE_WEIGHT};
    Kernel kernel;
    kernel(params);
}
