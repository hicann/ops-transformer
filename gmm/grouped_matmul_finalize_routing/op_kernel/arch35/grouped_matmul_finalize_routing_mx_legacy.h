/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_MX_LEGACY_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_MX_LEGACY_H

#include "cgmct/kernel/kernel_gmm_finalize_routing.h"
#include "cgmct/block/block_mx_mm_aic_to_aiv_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"

template <typename LayoutA, typename LayoutB>
__aicore__ inline void grouped_matmul_finalize_routing_mx_legacy(GM_ADDR x, GM_ADDR w, GM_ADDR wScale, GM_ADDR bias,
                                                                 GM_ADDR xScale, GM_ADDR groupList, GM_ADDR shareInput,
                                                                 GM_ADDR logit, GM_ADDR rowIndex, GM_ADDR offset,
                                                                 GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    auto gmmParams = tilingData.gmmFinalizeRoutingDataParams;
    auto matmulTiling = tilingData.matmulTiling;

    using L1TileShape = AscendC::Shape<Cgmct::Gemm::_0, Cgmct::Gemm::_0, Cgmct::Gemm::_0>;
    using L0TileShape = AscendC::Shape<Cgmct::Gemm::_0, Cgmct::Gemm::_0, Cgmct::Gemm::_0>;
    using AType = DTYPE_X;
    using BType = DTYPE_W;
    using CType = DTYPE_Y;
    using LayoutC = Cgmct::Gemm::layout::RowMajorAlign;
    using BiasType = bfloat16_t;
    using ProblemShape = Cgmct::Gemm::MatmulShape;
    using BlockScheduler = Cgmct::Gemm::GroupedMatmulAswtWithTailSplitScheduler;
    using BlockMmadBuilder = Cgmct::Gemm::Block::BlockMxMmAicToAivBuilder<
        AType, LayoutA, BType, LayoutB, BiasType, CType, LayoutC, L1TileShape, L0TileShape, BlockScheduler,
        Cgmct::Gemm::QuantMatmulWithTileMultiBlock<>,
        Cgmct::Gemm::Tile::TileCopy<Cgmct::Gemm::Arch::DAV3510, Cgmct::Gemm::Tile::CopyInAndCopyOutSplitMWithParams>>;
    using BlockPrologue = Cgmct::Gemm::Block::BlockPrologueFinalizeRouting<CType, BiasType>;
    using BlockEpilogue = Cgmct::Gemm::Block::BlockEpilogueFinalizeRouting<CType>;
    using GmmKernel = Cgmct::Gemm::Kernel::KernelGmmFinalizeRouting<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                    BlockEpilogue, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;

    GMMTiling gmmTiling{gmmParams.groupNum, gmmParams.groupListType, matmulTiling.baseM,
                        matmulTiling.baseN, matmulTiling.baseK,      gmmParams.hasBias};
    gmmTiling.matmulTiling = &matmulTiling;
    Params params = {{1, 1, 1, 1},
                     {x, w, wScale, xScale, y, groupList, bias},
                     {shareInput, y, gmmParams.sharedInputOffset, gmmParams.sharedInputLen, matmulTiling.N,
                      gmmParams.batch, gmmParams.residualScale},
                     {y, wScale, xScale, bias, logit, rowIndex, matmulTiling.baseM, matmulTiling.baseN},
                     gmmTiling};
    GmmKernel gmm;
    gmm(params);

    (void)offset;
    (void)workspaceGM;
}

#endif
