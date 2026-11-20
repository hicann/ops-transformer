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
 * \file gmm_tensor_api_mx_a8w4_kernel.h
 * \brief Adapt the A6 weight-quant ASW ABI to the shared MX Tensor API kernel.
 */
#pragma once

#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

namespace GROUPED_MATMUL {
template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void GmmTensorApiMxA8W4Kernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR antiquantScale,
                                                GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                const GroupedMatmulTilingData::GMMWeightQuantAswParams* gmmParams,
                                                const TCubeTiling* mmParams)
{
    if ASCEND_IS_AIV {
        return;
    }
    using LayoutA = asc::te::nd_ext_layout_ptn;
    // The API's transposed FRACTAL_NZ weight [N,K] is a logical B[K,N] ZN view.
    using LayoutB = asc::te::zn_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
    using BlockMmad =
        Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutC>;
    using Kernel =
        Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, Blaze::Epilogue::Block::BlockEpilogueEmpty,
                                           Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;

    typename Kernel::Params params{};
    params.problemShape = {mmParams->M, mmParams->N, mmParams->Ka, 1};
    params.mmadParams.aGmAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<AType>(0, x));
    params.mmadParams.bGmAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<BType>(0, weight));
    params.mmadParams.cGmAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<CType>(0, y));
    params.mmadParams.biasGmAddr =
        mmParams->isBias ? reinterpret_cast<GM_ADDR>(GetTensorAddr<BiasType>(0, bias)) : nullptr;
    params.mmadParams.scaleAGmAddr = perTokenScale;
    params.mmadParams.scaleBGmAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<fp8_e8m0_t>(0, antiquantScale));
    params.groupListGmAddr = groupList;

    constexpr uint32_t SCALE_K_L1_WINDOW = 4096U;
    constexpr uint64_t BYTES_PER_KIB = 1024UL;
    constexpr uint8_t SINGLE_BUFFER_COUNT = 1U;
    auto& tiling = params.gmmParams;
    tiling.groupNum = gmmParams->groupNum;
    tiling.m = mmParams->M;
    tiling.n = mmParams->N;
    tiling.k = mmParams->Ka;
    tiling.baseM = mmParams->baseM;
    tiling.baseN = mmParams->baseN;
    tiling.baseK = mmParams->baseK;
    tiling.kAL1 = mmParams->stepKa * mmParams->baseK;
    tiling.kBL1 = mmParams->stepKb * mmParams->baseK;
    // Preserve the ASW scale window and double-buffer tiling contract.
    tiling.scaleKAL1 = SCALE_K_L1_WINDOW;
    tiling.scaleKBL1 = tiling.scaleKAL1;
    tiling.isBias = mmParams->isBias;
    const uint64_t l0cTileBytes = Blaze::Gemm::CeilAlign<uint64_t>(tiling.baseM, Blaze::Gemm::C0_SIZE_L0C) *
                                  Blaze::Gemm::CeilAlign<uint64_t>(tiling.baseN, Blaze::Gemm::BLOCK_CUBE) *
                                  sizeof(float);
    const uint64_t l0cBytes = static_cast<uint64_t>(gmmParams->l0cSize) * BYTES_PER_KIB;
    tiling.dbL0C = static_cast<uint8_t>(l0cTileBytes * Blaze::Gemm::DOUBLE_BUFFER_COUNT <= l0cBytes ?
                                            Blaze::Gemm::DOUBLE_BUFFER_COUNT :
                                            SINGLE_BUFFER_COUNT);
    tiling.l1BufferStage = static_cast<uint8_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT);
    tiling.groupType = gmmParams->groupType;
    tiling.groupListType = gmmParams->groupListType;
    // A6 ASW checker requires one concatenated weight tensor; all list descriptors were unwrapped above.
    tiling.singleW = 1;

    Kernel kernel;
    kernel(params);
}
} // namespace GROUPED_MATMUL
