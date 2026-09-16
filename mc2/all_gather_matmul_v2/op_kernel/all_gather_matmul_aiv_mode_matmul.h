/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Matrix multiplication dispatch implementation for AllGatherMatmulAIVMode.
#pragma once

#include "all_gather_matmul_aiv_mode.h"

namespace AllGatherMatmulAIVModeImpl {

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        uint32_t layout_b_row = (TB && !weightNZ) ? static_cast<uint32_t>(k_align) : static_cast<uint32_t>(k);
        uint32_t layout_b_col = (TB || weightNZ) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(n_align);
        if (weightNZ) {
            using LayoutNZ = typename std::conditional<TB, layout::nZ, layout::zN>::type;
            LayoutNZ layoutBNZ = LayoutNZ::template MakeLayout<X2Type>(layout_b_row, layout_b_col);
            DispatchMatmul(layoutBNZ);
        } else {
            using LayoutB = typename std::conditional<TB, layout::ColumnMajor, layout::RowMajor>::type;
            LayoutB layoutB{layout_b_row, layout_b_col};
            DispatchMatmul(layoutB);
        }
    }
}

template <TemplateAGMMClass>
template <typename LayoutB>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::DispatchMatmul(const LayoutB &layoutB)
{
    if (m0 == TILE_SHAPE_128) {
        LaunchMatmul<LayoutB, TILE_SHAPE_128, TILE_SHAPE_256>(layoutB);
    } else {
        LaunchMatmul<LayoutB, TILE_SHAPE_256, TILE_SHAPE_128>(layoutB);
    }
}

template <TemplateAGMMClass>
template <typename LayoutB, int32_t TileM, int32_t TileN>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::LaunchMatmul(const LayoutB &layoutB)
{
    int64_t peer_mem_m = static_cast<int64_t>(m0) * pValue * worldSize;
    bool need_fixpipe = quantFlag && std::is_same<YType, half>::value && isX2ScaleTypeInt64 &&
                        (!std::is_same_v<X1Type, AscendC::int4b_t>);

    using ArchTag = Arch::AtlasA2;
    constexpr bool ENABLE_UNIT_FLAG = false;
    constexpr bool ENABLE_SHUFFLE_K = true;
    using ElementA = X1Type;
    using ElementB = X2Type;
    using ElementC = typename std::conditional<quantFlag, int32_t, YType>::type;
    using LayoutA = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using LayoutScale = layout::VectorLayout;
    LayoutA layoutA{static_cast<uint32_t>(m), static_cast<uint32_t>(k_align)};
    LayoutC layoutC{static_cast<uint32_t>(m * worldSize), static_cast<uint32_t>(n)};
    LayoutA layoutPeerMem{static_cast<uint32_t>(peer_mem_m * MAX_BLOCK_COUNT), static_cast<uint32_t>(k_align)};
    LayoutScale layoutScale{static_cast<uint32_t>(n)};
    GemmCoord processSize{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

    constexpr int32_t L1TileShapeK = TILE_SHAPE_K_512B<X1Type, int32_t>::value;
    constexpr int32_t L0TileShapeK = TILE_SHAPE_K_128B<X1Type, int32_t>::value;
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;
    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using CType = Gemm::GemmType<ElementC, LayoutC>;
    using BType = Gemm::GemmType<ElementB, LayoutB>;

    struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, void> {
        using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, void>;
        using ElementA = typename Base::ElementA;
        using ElementB = typename Base::ElementB;
        using ElementAccumulator = typename Base::ElementAccumulator;
        using CopyGmToL1A = typename Base::CopyGmToL1A;
        using CopyGmToL1B = typename Base::CopyGmToL1B;
        using CopyL1ToL0A = typename Base::CopyL1ToL0A;
        using CopyL1ToL0B = typename Base::CopyL1ToL0B;
        using CopyL0CToGm = typename Base::CopyL0CToGm;
    };
    using TileCopy = TileCopyOpt;

    using L1TileShape = GemmShape<TileM, TileN, L1TileShapeK>; // m n k
    using L0TileShape = GemmShape<TileM, TileN, L0TileShapeK>;
    using BlockMmadOpt =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
    using MatmulKernel = Gemm::Kernel::AllGatherMatmulV2<void, void, BlockMmadOpt>;
    typename MatmulKernel::Params params{processSize,
                                         reinterpret_cast<GM_ADDR>(gm_a_src),
                                         layoutA,
                                         reinterpret_cast<GM_ADDR>(gm_b_src),
                                         layoutB,
                                         reinterpret_cast<GM_ADDR>(cGM_),
                                         layoutC,
                                         reinterpret_cast<GM_ADDR>(x2ScaleGM_),
                                         layoutScale,
                                         reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                         layoutPeerMem,
                                         reinterpret_cast<GM_ADDR>(gm_accum),
                                         pValue,
                                         swizzlCount,
                                         swizzlDirect,
                                         rankId,
                                         worldSize,
                                         need_fixpipe,
                                         accumWorkSpacePingPong};
    MatmulKernel matmul_op;
    matmul_op(params);
}

} // namespace AllGatherMatmulAIVModeImpl
