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
 * \file gqmm_cgmct_mx_kernel.h
 * \brief
 */
#ifndef GQMM_CGMCT_MX_KERNEL_H
#define GQMM_CGMCT_MX_KERNEL_H

#include "cgmct/block/block_mmad_mx.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "cgmct/block/block_scheduler_policy.h"
#include "cgmct/epilogue/cgmct_block_epilogue_empty.h"
#include "cgmct/kernel/cgmct_kernel_qgmm_mx.h"
#include "cgmct/policy/cgmct_dispatch_policy.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"
using GMMQuantParams = GroupedMatmulTilingData::GMMQuantParams;
using QuantBasicApiMMTiling = GroupedMatmulTilingData::QuantBasicApiMMTiling;
using namespace Cgmct::Gemm;
template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType, class xLayout,
          class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmCgmctMxKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                        GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                        const GMMQuantParams *gmmBaseParamsIn,
                                        const QuantBasicApiMMTiling *mmTilingDataIn, AscendC::TPipe *que)
{
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    // 定义矩阵的类型和布局
    using AType = xType;
    using BType = wType;
    using CType = l0cType;
    using BiasType = float;
    using ScaleType = scaleType;
    using PtScaleType = ptScaleType;
    using YType = yType;

    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using LayoutY = yLayout;
    using LayoutBias = yLayout;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;

    // 定义scheduler类型
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;

    // 定义MMAD类型
    using BlockMmadPolicy = MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, 0>;
    using BlockMmad = Block::BlockMmadMx<BlockMmadPolicy, L1TileShape, L0TileShape, AType, LayoutA, BType, LayoutB,
                                         YType, LayoutC, BiasType, LayoutBias, void>;
    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueEmpty;
    // 定义Kernel类型
    using GmmKernel = Kernel::KernelQGmmMx<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;
    GMMTiling gmmParams{gmmBaseParamsIn->groupNum, mmTilingDataIn->m,          mmTilingDataIn->n,
                        mmTilingDataIn->k,         mmTilingDataIn->baseM,      mmTilingDataIn->baseN,
                        mmTilingDataIn->baseK,     mmTilingDataIn->kAL1,       mmTilingDataIn->kBL1,
                        mmTilingDataIn->scaleKAL1, mmTilingDataIn->scaleKBL1,  mmTilingDataIn->isBias,
                        mmTilingDataIn->dbL0C,     gmmBaseParamsIn->groupType, gmmBaseParamsIn->groupListType};
    Params params = {{1, 1, 1, 1},                                          // shape
                     {x, weight, scale, perTokenScale, y, bias, groupList}, // gm addr
                     gmmParams};
    GmmKernel gmm;
    gmm(params);
}
#endif
