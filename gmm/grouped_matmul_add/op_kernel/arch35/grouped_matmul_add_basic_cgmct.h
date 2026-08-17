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
 * \file grouped_matmul_add_basic_cgmct.h
 * \brief
 */

#ifndef __GROUPED_MATMUL_ADD_NO_QUANT_KERNEL_CGMCT
#define __GROUPED_MATMUL_ADD_NO_QUANT_KERNEL_CGMCT

#include "cgmct/kernel/kernel_grouped_matmul_add.h"
#include "cgmct/block/block_scheduler_grouped_matmul_aswt.h"
#include "../grouped_matmul_add_kernel.h"
#include "grouped_matmul_add_tiling_data.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

namespace GroupedMatmulAdd {

template <typename layoutA, typename layoutB>
__aicore__ inline void GmmAddCgmct(GM_ADDR x, GM_ADDR weight, GM_ADDR groupList, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(GmmAddTilingDataParams, gmmAddParams, gmmBaseParams_, tiling);
    GET_TILING_DATA_MEMBER(GmmAddTilingDataParams, mmTilingData, mmTilingData_, tiling);
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    // 定义矩阵的类型和布局
    using AType = DTYPE_X;
    using BType = DTYPE_X;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajor;
    using LayoutBias = layout::RowMajor;
    // 定义scheduler类型
    using BlockScheduler = GroupedMatmulAswtScheduler;
    // 定义MMAD类型
    using BlockMmad =
        Block::BlockGroupedMatmulBuilder<AType, LayoutA, BType, LayoutB, CType, LayoutC, CType, LayoutBias, L1TileShape,
                                         L0TileShape, BlockScheduler, MatmulMultiBlockBias<>>;
    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueEmpty;
    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;
    // 定义Kernel类型
    using GmmAdd = Kernel::KernelGroupedMatmulAdd<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename GmmAdd::Params;
    using GMMAddTiling = typename GmmAdd::GMMAddTiling;
    GMMAddTiling gmmParams{gmmBaseParams_.groupNum, gmmBaseParams_.groupListType, gmmBaseParams_.mTailCnt,
                           gmmBaseParams_.nTailCnt};
    gmmParams.matmulTiling = &mmTilingData_;
    Params params = {// template shape, gmm shape can not get now
                     {1, 1, 1, 1},
                     // mmad args
                     {x, weight, y, nullptr, groupList},
                     // epilogue args
                     {},
                     // gmm tiling data
                     gmmParams};
    GmmAdd op;
    op(params);
}

} // namespace GroupedMatmulAdd
#endif