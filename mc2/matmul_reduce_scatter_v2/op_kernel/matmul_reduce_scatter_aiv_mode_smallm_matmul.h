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
 * \file matmul_reduce_scatter_aiv_mode_smallm_matmul.h
 * \brief Matrix multiplication implementation for small-M reduce scatter.
 */

#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_SMALLM_MATMUL_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_SMALLM_MATMUL_H

#include "matmul_reduce_scatter_aiv_mode_smallM.h"

namespace MatmulReduceScatterV2Impl {

template <TemplateMMReduceScatterV2Class>
__aicore__ inline void MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        if (m0 == TILE_SHAPE_128) {
            LaunchSmallMMatmul<TILE_SHAPE_128, TILE_SHAPE_256>();
        } else {
            LaunchSmallMMatmul<TILE_SHAPE_256, TILE_SHAPE_128>();
        }
    }
}

template <TemplateMMReduceScatterV2Class>
template <int32_t TileM, int32_t TileN>
__aicore__ inline void MatmulReduceScatterAivModeSmallM<TemplateMMReduceScatterV2Func>::LaunchSmallMMatmul()
{
    bool need_fixpipe = quantFlag && isX2ScaleTypeInt64 && std::is_same<cType, half>::value;
    using ArchTag = Arch::AtlasA2;
    constexpr bool ENABLE_UNIT_FLAG = false;
    constexpr bool ENABLE_SHUFFLE_K = true;
    constexpr bool aicCalBias = !quantFlag && hasBias; // 如果计算量化后的矩阵乘，bias不由CatlassMatmul负责
    using ElementA = AType;
    using ElementB = BType;
    using ElementC = typename std::conditional<quantFlag, int32_t, cType>::type;
    using ElementBias = std::conditional_t<needCastBias, float, biasType>;

    using LayoutA = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using LayoutBias = layout::VectorLayout;

    LayoutA layoutA{static_cast<uint32_t>(m), static_cast<uint32_t>(k_align)};
    LayoutBias layoutBias{static_cast<uint32_t>(n)};

    uint32_t layout_b_row = (TB && !weight_nz) ? static_cast<uint32_t>(k_align) : static_cast<uint32_t>(k);
    uint32_t layout_b_col = (TB || weight_nz) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(n_align);

    using LayoutB = std::conditional_t<weight_nz, std::conditional_t<TB, layout::nZ, layout::zN>,
                                       std::conditional_t<TB, layout::ColumnMajor, layout::RowMajor>>;
    LayoutB layoutB;
    if constexpr (weight_nz) {
        layoutB = LayoutB::template MakeLayout<ElementB>(layout_b_row, layout_b_col);
    } else {
        layoutB = LayoutB{layout_b_row, layout_b_col};
    }

    using DispatchPolicy = std::conditional_t<aicCalBias, Gemm::MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG>,
                                              Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>>;
    using AType_ = Gemm::GemmType<ElementA, LayoutA>;
    using BType_ = Gemm::GemmType<ElementB, LayoutB>;
    using CType_ = Gemm::GemmType<ElementC, LayoutC>;
    using BiasType_ = std::conditional_t<aicCalBias, Gemm::GemmType<ElementBias, LayoutBias>, void>;

    struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_> {
        using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_>;
        using ElementA = typename Base::ElementA;
        using ElementB = typename Base::ElementB;
        using ElementAccumulator = typename Base::ElementAccumulator;
        using CopyGmToL1A = typename Base::CopyGmToL1A;
        using CopyGmToL1B = typename Base::CopyGmToL1B;
        using CopyL1ToL0A = typename Base::CopyL1ToL0A;
        using CopyL1ToL0B = typename Base::CopyL1ToL0B;
        using CopyL0CToGm = typename Base::CopyL0CToGm;
        using BiasTypeSelector = typename Base::BiasTypeSelector;
        using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
        using CopyL1ToBT = typename Base::CopyL1ToBT;
    };
    using TileCopy = TileCopyOpt;
    GM_ADDR blockmat_output_ptr;
    if (quantFlag && (!need_fixpipe)) {
        blockmat_output_ptr = reinterpret_cast<GM_ADDR>(gm_accum);
    } else {
        blockmat_output_ptr = reinterpret_cast<GM_ADDR>(buff[rank]);
    }
    using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    GemmCoord processSize{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};
    using L1TileShape = GemmShape<TileM, TileN, L1TileShapeK>;
    using L0TileShape = GemmShape<TileM, TileN, L0TileShapeK>;
    using BlockMmadOpt =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
    using MatmulKernel =
        Gemm::Kernel::MatmulReduceScatterAivModeSmallM<void, void, BlockMmadOpt, void, BlockScheduler30, aicCalBias>;
    typename MatmulKernel::Params params{processSize,
                                         reinterpret_cast<GM_ADDR>(gm_a_src),
                                         layoutA,
                                         reinterpret_cast<GM_ADDR>(gm_b_src),
                                         layoutB,
                                         biasGM_,
                                         blockmat_output_ptr, // mte远端读，结果矩阵直接写在peermem，提供读取能力
                                         reinterpret_cast<GM_ADDR>(perChannelScaleGM_),
                                         p_value,
                                         swizzl_count,
                                         swizzl_direct,
                                         rank,
                                         rank_size,
                                         need_fixpipe};
    MatmulKernel matmul_op;
    matmul_op(params);
}

} // namespace MatmulReduceScatterV2Impl

#endif // MATMUL_REDUCE_SCATTER_AIV_MODE_SMALLM_MATMUL_H
