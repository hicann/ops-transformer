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
 * \file qgmm_inplace_add_mix_online_dynamic.h
 * \brief
 */

#ifndef QGMM_INPLACE_ADD_MIX_ONLNE_DYNAMIC_H
#define QGMM_INPLACE_ADD_MIX_ONLNE_DYNAMIC_H

#include "cgmct/block/block_mmad_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "cgmct/kernel/kernel_qgmm_inplace_add_mix_online_dynamic.h"
#include "qgmm_inplace_add_utils.h"
#include "quant_grouped_matmul_inplace_add_tiling_data.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

template <typename layoutA, typename layoutB>
__aicore__ inline void QGmmInplaceAddMixAswt(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale2, GM_ADDR groupList, GM_ADDR scale1,
                                             GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams, quantGmmInplaceAddParams,
                           gmmBaseParams_, tiling);
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams, mmTilingData, mmTilingData_,
                           tiling);
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    // 定义矩阵的类型和布局
    using AType = DTYPE_X1;
    using BType = DTYPE_X2;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    // 定义scheduler类型
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    // 定义MMAD类型
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, CType, LayoutC, float, layout::RowMajor,
                                              L1TileShape, L0TileShape, BlockScheduler, MatmulMultiBlock<>,
                                              Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>>;
    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueDequant<L0TileShape, DTYPE_Y, DTYPE_Y, DTYPE_SCALE2, float, void, false>;
    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;
    // 定义Kernel类型
    using QGmmKernel =
        Kernel::KernelQGmmInplaceAddMixOnlineDynamic<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    GMMTiling gmmParams{gmmBaseParams_.groupNum, gmmBaseParams_.groupListType, mmTilingData_.baseM, mmTilingData_.baseN,
                        mmTilingData_.baseK};
    gmmParams.matmulTiling = &mmTilingData_;
    using DequantTiling = typename Block::Qmm::DequantTiling;
    DequantTiling dequantTiling{static_cast<uint32_t>(mmTilingData_.baseM),
                                static_cast<uint32_t>(mmTilingData_.baseN),
                                Block::Qmm::QuantMode::PERTENSOR_MODE,
                                Block::Qmm::QuantMode::PERCHANNEL_MODE,
                                0,
                                false};
    Params params = {// template shape, gmm shape can not get now
                     {1, 1, 1, 1},
                     // mmad args
                     {x1, x2, y, nullptr, groupList},
                     // epilogue args
                     {y, scale2, scale1, nullptr, dequantTiling},
                     // gmm tiling data
                     gmmParams};
    QGmmKernel op;
    op(params);
}

#endif
