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
 * \file gqmm_tensor_api_mx_kernel.h
 * \brief
 */

#ifndef GQMM_TENSOR_API_MX_KERNEL_H
#define GQMM_TENSOR_API_MX_KERNEL_H

#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using GMMQuantParams = GroupedMatmulTilingData::GMMQuantParams;
using QuantBasicApiMMTiling = GroupedMatmulTilingData::QuantBasicApiMMTiling;

template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType, class xLayout,
          class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmTensorApiMxKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                            GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                            const GMMQuantParams *gmmBaseParamsIn,
                                            const QuantBasicApiMMTiling *mmTilingDataIn, AscendC::TPipe *que)
{
    (void)workspace;
    (void)que;
    if ASCEND_IS_AIV {
        return;
    }
    using AType = xType;
    using BType = wType;
    using BiasType = biasType;
    using YType = yType;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockMmadPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGroupedMmadWithScaleMx>;
    using QgmmBlockMmad = Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, YType, LayoutC,
                                                        BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using QgmmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, QgmmBlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QgmmKernel::Params;
    using GMMTiling = typename QgmmKernel::GMMTiling;

    GMMTiling gmmParams{
        static_cast<uint32_t>(gmmBaseParamsIn->groupNum), static_cast<int64_t>(mmTilingDataIn->m),
        static_cast<int64_t>(mmTilingDataIn->n),          static_cast<int64_t>(mmTilingDataIn->k),
        static_cast<uint32_t>(mmTilingDataIn->baseM),     static_cast<uint32_t>(mmTilingDataIn->baseN),
        static_cast<uint32_t>(mmTilingDataIn->baseK),     static_cast<uint32_t>(mmTilingDataIn->kAL1),
        static_cast<uint32_t>(mmTilingDataIn->kBL1),      static_cast<uint32_t>(mmTilingDataIn->scaleKAL1),
        static_cast<uint32_t>(mmTilingDataIn->scaleKBL1), static_cast<uint8_t>(mmTilingDataIn->isBias),
        static_cast<uint8_t>(mmTilingDataIn->dbL0C),      static_cast<uint8_t>(mmTilingDataIn->l1BufferStage),
        static_cast<int8_t>(gmmBaseParamsIn->groupType),  static_cast<uint8_t>(gmmBaseParamsIn->groupListType),
        static_cast<uint8_t>(gmmBaseParamsIn->singleW)};

    GM_ADDR aDataAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<AType>(0, x));
    GM_ADDR bDataAddr = gmmBaseParamsIn->singleW == 1 ?
                            reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BType>(0, weight)) :
                            weight;
    GM_ADDR yDataAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
    GM_ADDR biasDataAddr = mmTilingDataIn->isBias == 1 ?
                               reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BiasType>(0, bias)) :
                               nullptr;
    GM_ADDR perTokenScaleDataAddr = perTokenScale;
    GM_ADDR scaleDataAddr =
        gmmBaseParamsIn->singleW == 1 ?
            reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<AscendC::fp8_e8m0_t>(0, scale)) :
            scale;
    GM_ADDR groupListDataAddr = groupList;

    Params params = {{gmmParams.m, gmmParams.n, gmmParams.k, static_cast<int64_t>(1)},
                     {aDataAddr, bDataAddr, yDataAddr, biasDataAddr, perTokenScaleDataAddr, scaleDataAddr},
                     {},
                     groupListDataAddr,
                     gmmParams};

    QgmmKernel gmm;
    gmm(params);
}

#endif
