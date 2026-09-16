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
 * \file matmul_reduce_scatter_aiv_mode_matmul.h
 * \brief Matrix multiplication implementation for AIV-mode reduce scatter.
 */

#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_MATMUL_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_MATMUL_H

#include "matmul_reduce_scatter_aiv_mode.h"

namespace MatmulReduceScatterV2Impl {

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        uint32_t layout_b_row = (TB && !weight_nz) ? static_cast<uint32_t>(k_align) : static_cast<uint32_t>(k);
        uint32_t layout_b_col = (TB || weight_nz) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(n_align);
        if (weight_nz) {
            using LayoutNZ = typename std::conditional<TB, layout::nZ, layout::zN>::type;
            LayoutNZ layoutBNZ = LayoutNZ::template MakeLayout<BType>(layout_b_row, layout_b_col);
            DispatchMatmul(layoutBNZ);
        } else {
            using LayoutB = typename std::conditional<TB, layout::ColumnMajor, layout::RowMajor>::type;
            LayoutB layoutB{layout_b_row, layout_b_col};
            DispatchMatmul(layoutB);
        }
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
template <typename LayoutB>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::DispatchMatmul(
    const LayoutB &layoutB)
{
    if (m0 == TILE_SHAPE_128) {
        LaunchMatmul<LayoutB, TILE_SHAPE_128, TILE_SHAPE_256>(layoutB);
    } else {
        LaunchMatmul<LayoutB, TILE_SHAPE_256, TILE_SHAPE_128>(layoutB);
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
template <typename LayoutB, int32_t TileM, int32_t TileN>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::LaunchMatmul(
    const LayoutB &layoutB)
{
    bool need_fixpipe = quantFlag && isX2ScaleTypeInt64 && std::is_same<cType, half>::value;
    int32_t peer_mem_m = m0 * loop_num_per_comm * MAX_BLOCK_COUNT;
    using ArchTag = Arch::AtlasA2;
    constexpr bool ENABLE_UNIT_FLAG = false;
    constexpr bool ENABLE_SHUFFLE_K = true;
    constexpr bool aicCalBias = !quantFlag && hasBias; // 如果计算量化后的矩阵乘，bias不由CatlassMatmul负责
    using ElementA = AType;
    using ElementB = BType;
    using ElementC = typename std::conditional<quantFlag, int32_t, cType>::type;
    using ElementBias = std::conditional_t<needCastBias, float, biasType>;

    using LayoutA = typename std::conditional<TA, layout::ColumnMajor, layout::RowMajor>::type;
    using LayoutC = layout::RowMajor;
    using LayoutScale = layout::VectorLayout;
    using LayoutBias = layout::VectorLayout;

    LayoutA layoutA{TA ? static_cast<uint32_t>(m_align) : static_cast<uint32_t>(m),
                    TA ? static_cast<uint32_t>(k) : static_cast<uint32_t>(k_align)};
    LayoutC layoutC{static_cast<uint32_t>(m / rank_size), static_cast<uint32_t>(n)};
    LayoutC layoutPeerMem{static_cast<uint32_t>(peer_mem_m), static_cast<uint32_t>(n0)};
    LayoutScale layoutScale{static_cast<uint32_t>(n)};
    LayoutBias layoutBias{static_cast<uint32_t>(n)};
    GemmCoord processSize{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

    constexpr int32_t L1TileShapeK = quantFlag ? TILE_SHAPE_512 : TILE_SHAPE_256;
    constexpr int32_t L0TileShapeK = quantFlag ? TILE_SHAPE_128 : TILE_SHAPE_64;
    using DispatchPolicy = std::conditional_t<aicCalBias, Gemm::MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG>,
                                              Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>>;
    using AType_ = Gemm::GemmType<ElementA, LayoutA>;
    using CType_ = Gemm::GemmType<ElementC, LayoutC>;
    using BiasType_ = std::conditional_t<aicCalBias, Gemm::GemmType<ElementBias, LayoutBias>, void>;
    using BType_ = Gemm::GemmType<ElementB, LayoutB>;
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

    using L1TileShape = GemmShape<TileM, TileN, L1TileShapeK>;
    using L0TileShape = GemmShape<TileM, TileN, L0TileShapeK>;
    using BlockMmadOpt =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
    using MatmulKernel = Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt, void, void, aicCalBias>;
    typename MatmulKernel::Params params{processSize,
                                         reinterpret_cast<GM_ADDR>(gm_a_src),
                                         layoutA,
                                         reinterpret_cast<GM_ADDR>(gm_b_src),
                                         layoutB,
                                         biasGM_,
                                         reinterpret_cast<GM_ADDR>(cGM_),
                                         layoutC,
                                         reinterpret_cast<GM_ADDR>(perChannelScaleGM_),
                                         layoutScale,
                                         reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                         layoutPeerMem,
                                         reinterpret_cast<GM_ADDR>(gm_accum),
                                         p_value,
                                         swizzl_count,
                                         swizzl_direct,
                                         dequant_type,
                                         rank,
                                         rank_size,
                                         need_fixpipe};
    MatmulKernel matmul_op;
    matmul_op(params);
}

} // namespace MatmulReduceScatterV2Impl

#endif // MATMUL_REDUCE_SCATTER_AIV_MODE_MATMUL_H
