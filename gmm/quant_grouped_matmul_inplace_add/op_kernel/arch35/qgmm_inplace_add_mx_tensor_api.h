/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QGMM_INPLACE_ADD_MX_TENSOR_API_H
#define QGMM_INPLACE_ADD_MX_TENSOR_API_H

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/epilogue/block/block_epilogue_inplace_add.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "quant_grouped_matmul_inplace_add_tiling_data.h"

template <bool AtomicAdd>
__aicore__ inline void QGmmInplaceAddMxTensorApi(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale2, GM_ADDR groupList,
                                                 GM_ADDR scale1, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams, quantGmmInplaceAddParams,
                           gmmParams, tiling);
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams, mmTilingData, mmTiling,
                           tiling);
    // An empty reduction leaves y unchanged and has no L1 K window to divide by.
    if (mmTiling.Ka == 0) {
        return;
    }

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, AtomicAdd>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<Policy, DTYPE_X1, asc::te::dn_ext_layout_ptn, DTYPE_X2,
                                                    asc::te::nd_ext_layout_ptn, DTYPE_Y, asc::te::nd_ext_layout_ptn,
                                                    float, asc::te::nd_ext_layout_ptn>;
    using Epilogue = AscendC::Std::conditional_t<AtomicAdd, Blaze::Gemm::Block::BlockEpilogueEmpty,
                                                 Blaze::Epilogue::Block::BlockEpilogueInplaceAdd>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, Epilogue,
                                                      Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;

    // Each packed scale-factor field occupies one byte.
    constexpr uint32_t scaleFactorMask = 0xffU;
    constexpr uint32_t scaleFactorBShift = 8;
    constexpr uint8_t l1BufferStages = 2;
    constexpr uint8_t splitKGroupType = 2;
    uint32_t kAL1 = static_cast<uint32_t>(mmTiling.stepKa * mmTiling.baseK);
    uint32_t kBL1 = static_cast<uint32_t>(mmTiling.stepKb * mmTiling.baseK);
    const uint32_t scaleKAL1 = (mmTiling.mxTypePara & scaleFactorMask) * kAL1;
    const uint32_t scaleKBL1 = ((mmTiling.mxTypePara >> scaleFactorBShift) & scaleFactorMask) * kBL1;
    // The shared scale window must fit both independently budgeted L1 scale buffers.
    uint32_t scaleKL1 = scaleKAL1 < scaleKBL1 ? scaleKAL1 : scaleKBL1;
    uint32_t outerKL1 = kAL1 > kBL1 ? kAL1 : kBL1;
    if (scaleKL1 < outerKL1) {
        // Shorten the data strips so one scale window covers a complete outer K strip.
        outerKL1 = kAL1 < kBL1 ? kAL1 : kBL1;
        kAL1 = outerKL1;
        kBL1 = outerKL1;
    }
    scaleKL1 = scaleKL1 / outerKL1 * outerKL1;

    typename Kernel::GMMTiling tensorTiling{};
    tensorTiling.groupNum = gmmParams.groupNum;
    tensorTiling.m = mmTiling.M;
    tensorTiling.n = mmTiling.N;
    tensorTiling.k = mmTiling.Ka;
    tensorTiling.baseM = static_cast<uint32_t>(mmTiling.baseM);
    tensorTiling.baseN = static_cast<uint32_t>(mmTiling.baseN);
    tensorTiling.baseK = static_cast<uint32_t>(mmTiling.baseK);
    tensorTiling.kAL1 = kAL1;
    tensorTiling.kBL1 = kBL1;
    tensorTiling.scaleKAL1 = scaleKL1;
    tensorTiling.scaleKBL1 = scaleKL1;
    tensorTiling.isBias = 0;
    tensorTiling.dbL0C = static_cast<uint8_t>(mmTiling.dbL0C);
    tensorTiling.l1BufferStage = l1BufferStages;
    tensorTiling.groupType = splitKGroupType;
    tensorTiling.groupListType = gmmParams.groupListType;
    tensorTiling.singleW = 1;

    typename Kernel::Params params{
        {mmTiling.M, mmTiling.N, mmTiling.Ka, 0}, {x1, x2, y, nullptr, scale1, scale2}, {}, groupList, tensorTiling};
    if constexpr (!AtomicAdd) {
        params.epilogueParams.baseM = mmTiling.baseM;
        params.epilogueParams.baseN = mmTiling.baseN;
    }
    Kernel op;
    op(params);
}

#endif // QGMM_INPLACE_ADD_MX_TENSOR_API_H
