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
 * \file grouped_matmul_kernel.h
 * \brief Non-quant grouped matmul outer kernel implemented with Tensor API.
 */

#pragma once

#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_grouped_matmul.h"
#include "blaze/gemm/kernel/kernel_grouped_matmul.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMNoQuantTilingData = GroupedMatmulTilingData::GMMNoQuantTilingData;

namespace GROUPED_MATMUL {

template <typename LayoutA, typename LayoutB>
__aicore__ inline void GroupedMatMulKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR groupList, GM_ADDR y,
                                           GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GMMNoQuantTilingData, gmmNoQuantParam, gmmBaseParams, tiling);
    GET_TILING_DATA_MEMBER(GMMNoQuantTilingData, mmTilingData, mmTilingData, tiling);

    using AType = DTYPE_X;
    using BType = DTYPE_X;
    using CType = DTYPE_Y;
    using BiasType = DTYPE_BIAS;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelGroupedMmadNoQuant, 0,
                                                              Blaze::Gemm::MatmulOutputMode::OVERWRITE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmNoQuant;
    using GroupedMatmulKernel =
        Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename GroupedMatmulKernel::Params;
    using GMMTiling = typename GroupedMatmulKernel::GMMTiling;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    const uint64_t baseM = static_cast<uint64_t>(mmTilingData.baseM);
    const uint64_t baseN = static_cast<uint64_t>(mmTilingData.baseN);
    const uint64_t baseK = static_cast<uint64_t>(mmTilingData.baseK);
    const uint64_t sharedKStep =
        Blaze::Gemm::Min(static_cast<uint64_t>(mmTilingData.stepKa), static_cast<uint64_t>(mmTilingData.stepKb));
    const uint64_t kL1 = baseK * Blaze::Gemm::Max(sharedKStep, static_cast<uint64_t>(1));

    GMMTiling gmmParams{
        static_cast<uint32_t>(gmmBaseParams.groupNum),      static_cast<int32_t>(gmmBaseParams.groupType),
        static_cast<uint32_t>(gmmBaseParams.groupListType), static_cast<uint64_t>(gmmBaseParams.singleX),
        static_cast<uint64_t>(gmmBaseParams.singleWeight),  static_cast<uint64_t>(gmmBaseParams.singleY),
        static_cast<uint32_t>(gmmBaseParams.hasBias),       static_cast<uint32_t>(gmmBaseParams.weightNoL2Cache)};
    constexpr uint32_t nTailAlign =
        BlockMmad::WEIGHT_NZ_FORMAT ? static_cast<uint32_t>(AscendC::Te::C0_ELEMENT<BType>) : 1U;
    BlockSchedulerParams schedulerParams{static_cast<int32_t>(baseM),
                                         static_cast<int32_t>(baseN),
                                         static_cast<uint64_t>(gmmBaseParams.mTailCnt),
                                         static_cast<uint64_t>(gmmBaseParams.nTailCnt),
                                         1U, // mTailAlign
                                         nTailAlign,
                                         static_cast<int32_t>(gmmBaseParams.groupType),
                                         static_cast<uint32_t>(gmmBaseParams.groupNum),
                                         static_cast<int64_t>(mmTilingData.M),
                                         gmmBaseParams.singleX == 1,
                                         gmmBaseParams.singleWeight == 1,
                                         gmmBaseParams.singleY == 1,
                                         BlockMmad::TRANS_B,
                                         BlockMmad::WEIGHT_NZ_FORMAT,
                                         static_cast<uint32_t>(sizeof(BType))};

    typename BlockMmad::Params mmParams{x,
                                        weight,
                                        y,
                                        gmmBaseParams.hasBias == 0 ? nullptr : bias,
                                        groupList,
                                        nullptr, // workspaceGmAddr
                                        baseM,
                                        baseN,
                                        kL1,
                                        static_cast<uint32_t>(baseM),
                                        static_cast<uint32_t>(baseN),
                                        static_cast<uint32_t>(baseK),
                                        2U, // l1Stages
                                        static_cast<uint16_t>(mmTilingData.dbL0C),
                                        nullptr}; // scaleGmAddr

    ProblemShape problemShape{static_cast<int64_t>(mmTilingData.M), static_cast<int64_t>(mmTilingData.N),
                              static_cast<int64_t>(mmTilingData.Ka), static_cast<int64_t>(1)};
    Params params{problemShape, mmParams, {}, schedulerParams, gmmParams};
    GroupedMatmulKernel kernel;
    kernel(params);
}

} // namespace GROUPED_MATMUL
