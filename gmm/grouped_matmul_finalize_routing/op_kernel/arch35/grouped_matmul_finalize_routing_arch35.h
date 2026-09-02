/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_finalize_routing_arch35.h
 * \brief
 */

#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_ARCH35_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_ARCH35_H

#include "cgmct/kernel/kernel_gmm_finalize_routing.h"
#include "cgmct/block/block_mx_mm_aic_to_aiv_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

template <typename layoutA, typename layoutB>
__aicore__ inline void grouped_matmul_finalize_routing_mx(GM_ADDR x, GM_ADDR w, GM_ADDR w_scale, GM_ADDR bias,
                                                          GM_ADDR x_scale, GM_ADDR group_list, GM_ADDR share_input,
                                                          GM_ADDR logit, GM_ADDR row_index, GM_ADDR offset, GM_ADDR y,
                                                          GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    auto gmmFinalizeRoutingQuantParams_ = tilingData.gmmFinalizeRoutingDataParams;
    auto matmulTiling_ = tilingData.matmulTiling;

    using L1TileShape = AscendC::Shape<Cgmct::Gemm::_0, Cgmct::Gemm::_0, Cgmct::Gemm::_0>;
    using L0TileShape = AscendC::Shape<Cgmct::Gemm::_0, Cgmct::Gemm::_0, Cgmct::Gemm::_0>;

    using AType = DTYPE_X;
    using BType = DTYPE_W;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    using weightscaleType = AscendC::fp8_e8m0_t;
    using BiasType = bfloat16_t;

    using ProblemShape = Cgmct::Gemm::MatmulShape;

    using BlockScheduler = Cgmct::Gemm::GroupedMatmulAswtWithTailSplitScheduler;

    using BlockMmadPolicy = Cgmct::Gemm::GMMPerTile<>;

    using BlockMmadBuilder =
        Block::BlockMxMmAicToAivBuilder<AType, LayoutA, BType, LayoutB, BiasType, CType, LayoutC, L1TileShape,
                                        L0TileShape, BlockScheduler, QuantMatmulWithTileMultiBlock<>,
                                        Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>>;

    using BlockPrologue = Cgmct::Gemm::Block::BlockPrologueFinalizeRouting<CType, BiasType>;

    using BlockEpilogue = Cgmct::Gemm::Block::BlockEpilogueFinalizeRouting<CType>;

    using GmmKernel = Cgmct::Gemm::Kernel::KernelGmmFinalizeRouting<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                    BlockEpilogue, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;

    GMMTiling gmmParams{gmmFinalizeRoutingQuantParams_.groupNum,
                        gmmFinalizeRoutingQuantParams_.groupListType,
                        matmulTiling_.baseM,
                        matmulTiling_.baseN,
                        matmulTiling_.baseK,
                        gmmFinalizeRoutingQuantParams_.hasBias};

    gmmParams.matmulTiling = &matmulTiling_;
    Params params = {
        {1, 1, 1, 1}, // problem shape
        {x, w, w_scale, x_scale, y, group_list,
         bias}, // BlockMmadParams   {aGmAdd,bGmAddr,cGmAddr,biasGmAddr,groupListGmAddr,bias}
        {share_input, y, gmmFinalizeRoutingQuantParams_.sharedInputOffset,
         gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N, gmmFinalizeRoutingQuantParams_.batch,
         gmmFinalizeRoutingQuantParams_
             .residualScale}, // prologue
                              // params{sharedInputGmaddr,yGMAddr,shareInputOffset,shareInputLen,n,batch,residualScale}
        {y, w_scale, x_scale, bias, logit, row_index, matmulTiling_.baseM,
         matmulTiling_
             .baseN}, // epilogue params{ygmaddr，x2scalegmaddr，biasgmaddr，logitgmaddr，rowidxgmaddr，baseM，baseN}
        gmmParams};
    GmmKernel gmm;
    gmm(params);
}
#endif
